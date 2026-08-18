#!/usr/bin/env python3
"""라벨링 툴 데이터를 vlm-dataset-maker 입력(택소노미 JSON)으로 변환한다.

  라벨링 툴  <dataRoot>/json/*.json  +  <dataRoot>/images/*
      ↓  (이 스크립트)
  택소노미   data/instruct/<source_dataset>/private/<name>_v0.0.0.json
      ↓  (vlm-dataset-maker/main.py)
  trainset   {"messages": [...], "images": [...]}      ← Qwen VLM SFT 입력

라벨링 툴의 parsing_res_list 를 block_order 순서로 읽어 마크다운 정답을 만든다.
검수 전 원본에도, 검수 후 교정본에도 똑같이 돌아간다 (같은 경로를 덮어쓰므로).

사용:
  python3 make_vlm_taxonomy.py --src /data/workspace/VLM/.../data/train --dry
  python3 make_vlm_taxonomy.py --src ... --out ~/vlm-dataset-maker/data --link-images
"""
import argparse
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

# statistic_calculate() 가 검증하는 값들. 여기 없는 값을 쓰면 main.py 가 exit(1) 한다.
TASK_C1 = "OCR"
TASK_C2 = "Document Understanding"   # Text/Table Extraction, Table Understanding 등도 가능
DOMAIN = "Other"

QUESTION = "이 영수증의 내용을 마크다운으로 정리해줘."

# 텍스트가 없는 블록. 마크다운에 넣지 않는다.
SKIP_LABELS = {"image", "seal", "header_image", "footer_image", "figure_title"}
H1_LABELS = {"doc_title"}
H2_LABELS = {"paragraph_title"}


FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?(.*?)\n?\s*```\s*$", re.S)


def strip_fence(s):
    """파서가 ```html … ``` 로 감싸 뱉는 경우가 있다. 껍질을 벗긴다."""
    m = FENCE.match(s)
    return m.group(1) if m else s


class TableParser(HTMLParser):
    """PP-Structure 가 뱉는 <table> 을 행렬로 되돌린다."""

    def __init__(self):
        super().__init__()
        self.rows, self.cur, self.cell, self.in_cell = [], [], [], False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.cur = []
        elif tag in ("td", "th"):
            self.in_cell, self.cell = True, []
            # colspan 만큼 뒤에 빈 칸을 채워 열을 맞춘다
            self.span = 1
            for k, v in attrs:
                if k == "colspan":
                    try:
                        self.span = max(1, int(v))
                    except (TypeError, ValueError):
                        pass
        elif tag == "br" and self.in_cell:
            self.cell.append(" ")     # 줄바꿈이 단어를 붙여버리지 않게

    def handle_startendtag(self, tag, attrs):
        if tag == "br" and self.in_cell:
            self.cell.append(" ")

    def handle_endtag(self, tag):
        if tag == "tr":
            if self.cur:
                self.rows.append(self.cur)
        elif tag in ("td", "th"):
            text = " ".join("".join(self.cell).split())
            self.cur.append(text)
            self.cur.extend([""] * (getattr(self, "span", 1) - 1))
            self.in_cell = False

    def handle_data(self, data):
        if self.in_cell:
            self.cell.append(data)


def html_table_to_md(html):
    p = TableParser()
    try:
        p.feed(strip_fence(html))
    except Exception:
        return None
    rows = [r for r in p.rows if any(c for c in r)]
    if not rows:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    def line(cells):
        # 셀 안의 파이프는 표를 깨뜨리므로 escape
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    head, *body = rows
    out = [line(head), "| " + " | ".join(["---"] * width) + " |"]
    out += [line(r) for r in body]
    return "\n".join(out)


def strip_tags(s):
    """표 변환이 실패했을 때의 안전망. 태그만 걷어내고 텍스트를 남긴다."""
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"</(tr|table|thead|tbody)>", "\n", s, flags=re.I)
    s = re.sub(r"</t[dh]>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    lines = [" ".join(l.split()) for l in s.split("\n")]
    return "\n".join(l for l in lines if l)


def detect_language(text):
    kor = len(re.findall(r"[가-힣]", text))
    eng = len(re.findall(r"[A-Za-z]", text))
    if kor and eng:
        # 한쪽이 압도적이면 그쪽으로, 아니면 mix
        ratio = kor / (kor + eng)
        return "kor" if ratio > 0.8 else ("eng" if ratio < 0.2 else "mix")
    if kor:
        return "kor"
    if eng:
        return "eng"
    return "other"


# ── 파싱 엔진 환각 블록 걸러내기 ────────────────────────────────
# 레이아웃 검출이 만든 세로 33px 짜리 납작한 띠를 VLM 이 못 읽으면,
# 전사 대신 영어로 이미지를 '설명'하거나 패턴 반복 루프에 빠진다.
# 그 블록만 버리고 나머지 정상 블록은 살린다.
META_PHRASES = (
    "the image displays", "the image shows", "the image appears",
    "without additional context", "for the purpose of this task",
    "it is challenging to determine", "the extracted text from the image",
    "further information would be necessary", "possibly forming a pattern",
    "appears to be a single, continuous",
)
SEQ_RUN = re.compile(r"(?:\d{3,4}\s*[-–]\s*\d{3,4}\s+){6,}")


def is_hallucinated(content, label=None):
    """환각 블록이면 사유를 돌려준다. 정상이면 None."""
    low = content.lower()
    for m in META_PHRASES:
        if m in low:
            return "메타 서술"
    if SEQ_RUN.search(content):
        return "증가 수열 반복"

    # 한 줄 안에서 같은 토큰 형태가 25회 이상 반복되면 루프
    for line in content.split("\n"):
        toks = line.split()
        if len(toks) >= 25:
            shapes = {}
            for t in toks:
                k = re.sub(r"\d+", "#", t)
                shapes[k] = shapes.get(k, 0) + 1
            if max(shapes.values()) >= 25:
                return "토큰 반복 루프"

    # 영어 서술문이 통째로 들어온 경우. 표 블록은 제외한다 — 해외 한인마트
    # 영수증처럼 품목마다 영문명이 함께 찍히면 정상인데도 걸린다 (gcse_00739).
    # 서술문은 문장부호가 따라오므로 '영단어 + 마침표' 를 함께 본다.
    if label != "table" and len(content) > 300:
        eng = len(re.findall(r"\b[A-Za-z]{4,}\b", content))
        sentences = len(re.findall(r"[a-z]{3,}[.,;] ", content))
        if eng >= 30 and sentences >= 5:
            return "영문 서술문"
    return None


def compact_html(s):
    """태그 사이 공백만 지운다. 셀 안 글자는 건드리지 않는다.

    라벨링 원본은 `<table>\\n<tr>\\n<td>` 처럼 태그마다 줄바꿈이 들어 있다.
    구조가 같으므로 지워도 TEDS 는 1.0 그대로이고, 정답 토큰만 줄어든다.
    """
    return re.sub(r">\s+<", "><", s).strip()


def build_markdown(blocks, dropped=None, keep_html_tables=False):
    """block_order 순서대로 마크다운을 조립한다.

    keep_html_tables=True 면 표를 마크다운으로 바꾸지 않고 HTML 그대로 둔다.
    마크다운 표 문법에는 rowspan/colspan 이 없어서, 변환하는 순간 병합 셀이
    빈 칸으로 펴진다(이 데이터셋 표의 53% 가 병합을 쓴다). 채점을 TEDS 로
    하고 서비스 프롬프트도 HTML 을 요구하므로, 그때는 변환하지 않는 게 맞다.
    """
    ordered = sorted(
        blocks,
        key=lambda b: (b.get("block_order") if b.get("block_order") is not None
                       else b.get("block_id", 0)),
    )
    parts = []
    for b in ordered:
        label = b.get("block_label") or "text"
        content = strip_fence((b.get("block_content") or "").strip()).strip()
        if not content or label in SKIP_LABELS:
            continue
        why = is_hallucinated(content, label)
        if why:
            if dropped is not None:
                dropped.append((label, len(content), why))
            continue
        # 라벨과 무관하게 HTML 표가 들어 있으면 표로 변환한다.
        # 변환에 실패하면 원문 대신 태그를 걷어낸 텍스트를 쓴다 (원문 HTML 이
        # 정답에 섞이면 모델이 <td> 를 뱉도록 학습된다).
        if "<table" in content or "<td" in content:
            if keep_html_tables:
                parts.append(compact_html(content))
            else:
                md = html_table_to_md(content)
                parts.append(md if md else strip_tags(content))
        elif label in H1_LABELS:
            parts.append("# " + content.replace("\n", " ").strip())
        elif label in H2_LABELS:
            parts.append("## " + content.replace("\n", " ").strip())
        else:
            parts.append(content)
    return "\n\n".join(parts).strip()


def convert(src, dataset_name, restrict=None):
    jdir, idir = src / "json", src / "images"
    if not jdir.is_dir():
        sys.exit(f"json 폴더가 없습니다: {jdir}")

    images = {p.stem: p for p in idir.iterdir()} if idir.is_dir() else {}

    # 기준 폴더가 주어지면 그 안에 있는 것만 만든다. 검수로 걸러낸 결과를
    # 그 폴더가 그대로 들고 있으므로, 재생성해도 지운 것이 되살아나지 않는다.
    allow = None
    if restrict:
        allow = {p.stem for p in Path(restrict).iterdir()
                 if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
        print(f"기준 폴더 {restrict}: {len(allow)}장만 대상으로 함")

    rows, skipped, dropped_log = [], [], []

    for jf in sorted(jdir.glob("*.json")):
        if allow is not None and jf.stem not in allow:
            continue
        img = images.get(jf.stem)
        if img is None:
            skipped.append((jf.stem, "이미지 없음"))
            continue
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            skipped.append((jf.stem, f"JSON 파싱 실패: {e}"))
            continue

        # 툴 업로드 형식(최상위)과 원본 파싱 형식(중첩) 둘 다 받는다
        blocks = d.get("parsing_res_list")
        if blocks is None:
            els = (d.get("result") or {}).get("elements") or []
            blocks = (els[0].get("json") or {}).get("parsing_res_list") if els else None
        if not blocks:
            skipped.append((jf.stem, "parsing_res_list 없음"))
            continue

        drops = []
        answer = build_markdown(blocks, drops)
        if drops:
            dropped_log.append((jf.stem, drops))
        if not answer:
            skipped.append((jf.stem, "텍스트 블록 없음"))
            continue

        rows.append({
            "dataset_name": dataset_name,
            "version": "0.0.0",
            "task": {"category_1": TASK_C1, "category_2": TASK_C2},
            "domain": {"category_1": DOMAIN},
            "metadata": {"image_cnt": 1, "source_file": jf.name},
            "prompt": {
                "question": QUESTION,
                "input": [],
                "image": [img.name],
                "conversation": [],
            },
            "completion": {"chosen": answer},
            "feature": {"completion": {"chosen": {"language": detect_language(answer)}}},
        })
    return rows, skipped, dropped_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="라벨링 툴 dataRoot (json/ 과 images/ 를 가진 폴더)")
    ap.add_argument("--out", help="vlm-dataset-maker 의 data/ 경로")
    ap.add_argument("--source-dataset", default="luxia3_instruct_v0.0.0")
    ap.add_argument("--dataset-name", default="luxia3_receipt")
    ap.add_argument("--link-images", action="store_true",
                    help="trainset/image/instruct/ 에 이미지 심볼릭 링크 생성")
    ap.add_argument("--restrict-to",
                    help="이 폴더에 있는 이미지만 대상으로 한다 (검수 결과 반영본)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    rows, skipped, dropped_log = convert(Path(a.src), a.dataset_name, a.restrict_to)
    print(f"변환 {len(rows)}건 / 건너뜀 {len(skipped)}건")
    if dropped_log:
        import collections as _c
        tot = sum(len(d) for _, d in dropped_log)
        why = _c.Counter(w for _, d in dropped_log for _, _, w in d)
        print(f"환각 블록 제거: {len(dropped_log)}개 파일에서 {tot}블록")
        for k, v in why.most_common():
            print(f"     {k:16s} {v:4d}")
    if skipped:
        for s in skipped[:10]:
            print(f"   ✗ {s[0]}: {s[1]}")
        if len(skipped) > 10:
            print(f"   … 외 {len(skipped)-10}건")

    langs = {}
    for r in rows:
        k = r["feature"]["completion"]["chosen"]["language"]
        langs[k] = langs.get(k, 0) + 1
    lens = sorted(len(r["completion"]["chosen"]) for r in rows)
    if lens:
        print(f"언어 {langs}")
        print(f"정답 길이  중앙값 {lens[len(lens)//2]}자  최소 {lens[0]}  최대 {lens[-1]}")

    if a.dry or not a.out:
        if rows:
            print("\n=== 샘플 1건 ===")
            s = dict(rows[0])
            s["completion"] = {"chosen": s["completion"]["chosen"][:300] + " …"}
            print(json.dumps(s, ensure_ascii=False, indent=2))
        if not a.out:
            print("\n(--out 없음: 파일을 쓰지 않았습니다)")
        return 0

    out = Path(a.out)
    tgt = out / "instruct" / a.source_dataset / "private" / f"{a.dataset_name}_v0.0.0.json"
    tgt.parent.mkdir(parents=True, exist_ok=True)
    tgt.write_text(json.dumps(rows, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"\n택소노미 저장 → {tgt}")

    if a.link_images:
        imgdir = out / "trainset" / "image" / "instruct"
        imgdir.mkdir(parents=True, exist_ok=True)
        n = 0
        for r in rows:
            name = r["prompt"]["image"][0]
            link, real = imgdir / name, Path(a.src) / "images" / name
            if not link.exists():
                link.symlink_to(real)
                n += 1
        print(f"이미지 링크 {n}개 → {imgdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
