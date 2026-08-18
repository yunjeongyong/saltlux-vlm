#!/usr/bin/env python3
"""택소노미 데이터셋이 학습에 쓸 만한지 검사한다.

두 층으로 본다.
  A. 구조 — 형식이 깨져 학습이 멈추거나 조용히 잘못 학습되는 것
  B. 내용 — 형식은 맞지만 정답이 쓰레기인 것 (파서 오류가 그대로 정답이 된 경우)

사용:
  python3 validate_vlm_taxonomy.py --file .../luxia3_receipt_v0.0.0.json \
      --images .../data/trainset/image/instruct [--report out.json]
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

TASK_OK = {
    "OCR": {"Table Extraction", "Text Extraction", "Table Understanding",
            "Document Understanding", "Multi-Choice", "VQA"},
    "Chart": {"Captioning", "VQA", "Multi-Choice"},
    "Diagram": {"Captioning", "VQA", "Multi-Choice"},
    "Reasoning": {"VQA", "Multi-Choice"},
    "General": {"VQA", "Multi-Choice", "Captioning", "Scene Understanding",
                "Object Detection"},
}
DOMAIN_OK = {"Math", "Science", "Coding", "Other"}
LANG_OK = {"kor", "eng", "mix", "other"}


def structural(rows, imgdir):
    """학습을 멈추게 하거나 조용히 망가뜨리는 문제."""
    bad = collections.defaultdict(list)
    seen_img = collections.Counter()

    for i, r in enumerate(rows):
        tag = r.get("metadata", {}).get("source_file") or f"#{i}"
        try:
            t1 = r["task"]["category_1"]
            t2 = r["task"]["category_2"]
            dom = r["domain"]["category_1"]
            lang = r["feature"]["completion"]["chosen"]["language"]
            cnt = r["metadata"]["image_cnt"]
            imgs = r["prompt"]["image"]
            ans = r["completion"]["chosen"]
            q = r["prompt"]["question"]
        except (KeyError, TypeError) as e:
            bad["필수 키 없음"].append((tag, str(e)))
            continue

        # main.py 가 exit(1) 하는 조건들
        if t1 not in TASK_OK or t2 not in TASK_OK.get(t1, ()):
            bad["정의되지 않은 task"].append((tag, f"{t1}/{t2}"))
        if dom not in DOMAIN_OK:
            bad["정의되지 않은 domain"].append((tag, dom))
        if not cnt:
            bad["image_cnt 가 0/None"].append((tag, cnt))
        if lang not in LANG_OK:
            bad["언어값 이상"].append((tag, lang))

        # 이미지 개수와 실제 파일
        if len(imgs) != cnt:
            bad["image_cnt 와 이미지 수 불일치"].append((tag, f"{cnt} vs {len(imgs)}"))
        for im in imgs:
            seen_img[im] += 1
            if imgdir:
                p = imgdir / im
                if not p.exists():
                    bad["이미지 파일 없음"].append((tag, im))
                elif p.is_symlink() and not p.resolve().exists():
                    bad["깨진 심볼릭 링크"].append((tag, im))

        if not ans or not ans.strip():
            bad["정답 비어 있음"].append((tag, ""))
        if not q or not q.strip():
            bad["질문 비어 있음"].append((tag, ""))

    for im, n in seen_img.items():
        if n > 1:
            bad["같은 이미지 중복 사용"].append((im, f"{n}회"))
    return bad


def content(rows):
    """형식은 맞지만 정답이 못 쓸 것들. 사람이 봐야 할 후보를 추린다."""
    flag = collections.defaultdict(list)
    answers = collections.Counter()

    for r in rows:
        tag = r.get("metadata", {}).get("source_file") or "?"
        ans = r["completion"]["chosen"]
        answers[ans] += 1
        n = len(ans)

        if n < 30:
            flag["정답이 지나치게 짧음(<30자)"].append((tag, f"{n}자: {ans[:40]!r}"))
        if n > 8000:
            flag["정답이 지나치게 긺(>8000자)"].append((tag, f"{n}자"))

        # 파서가 같은 줄을 반복해 뱉는 전형적 실패
        lines = [l.strip() for l in ans.split("\n") if l.strip()]
        if lines:
            top, c = collections.Counter(lines).most_common(1)[0]
            if c >= 4 and len(top) > 5:
                flag["같은 줄 4회 이상 반복"].append((tag, f"{c}회: {top[:40]!r}"))

        # OCR 실패 흔적
        if "�" in ans:
            flag["대체문자(�) 포함"].append((tag, ""))
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", ans):
            flag["제어문자 포함"].append((tag, ""))

        # 마크다운 표 열 수 검사
        for blk in re.findall(r"(?:^\|.*\|$\n?)+", ans, re.M):
            rws = [x for x in blk.strip().split("\n") if x.strip()]
            widths = {x.count("|") for x in rws}
            if len(widths) > 1:
                flag["표의 열 수가 행마다 다름"].append((tag, f"열수 {sorted(widths)}"))
                break

        # 표는 있는데 셀이 거의 비어 있음
        cells = re.findall(r"\|([^|\n]*)", ans)
        if len(cells) >= 12:
            empty = sum(1 for c in cells if not c.strip() or set(c.strip()) <= {"-"})
            if empty / len(cells) > 0.6:
                flag["표 셀 60% 이상이 빔"].append((tag, f"{empty}/{len(cells)}"))

    for a, n in answers.items():
        if n > 1:
            flag["정답이 완전히 동일한 건"].append((a[:50], f"{n}건"))
    return flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--images")
    ap.add_argument("--report")
    ap.add_argument("--show", type=int, default=3, help="문제 유형별 예시 개수")
    a = ap.parse_args()

    rows = json.loads(Path(a.file).read_text(encoding="utf-8"))
    imgdir = Path(a.images) if a.images else None
    print(f"검사 대상 {len(rows)}건\n")

    struct = structural(rows, imgdir)
    cont = content(rows)

    print("── A. 구조 (학습을 막거나 조용히 망가뜨림) " + "─" * 24)
    if not struct:
        print("   문제 없음")
    for k, v in sorted(struct.items(), key=lambda x: -len(x[1])):
        print(f"   ✗ {k}: {len(v)}건")
        for t in v[:a.show]:
            print(f"       {t[0]}  {t[1]}")

    print("\n── B. 내용 (사람이 확인해야 할 후보) " + "─" * 27)
    if not cont:
        print("   문제 없음")
    for k, v in sorted(cont.items(), key=lambda x: -len(x[1])):
        print(f"   ⚠ {k}: {len(v)}건")
        for t in v[:a.show]:
            print(f"       {t[0]}  {t[1]}")

    lens = sorted(len(r["completion"]["chosen"]) for r in rows)
    q = lambda p: lens[int(len(lens) * p)] if lens else 0
    print("\n── 분포 " + "─" * 55)
    print(f"   정답 길이  p10 {q(.1)}  중앙 {q(.5)}  p90 {q(.9)}  최대 {lens[-1]}")
    langs = collections.Counter(
        r["feature"]["completion"]["chosen"]["language"] for r in rows)
    print(f"   언어  {dict(langs)}")
    tbl = sum(1 for r in rows if "|" in r["completion"]["chosen"])
    print(f"   표 포함  {tbl}건 ({tbl/len(rows)*100:.1f}%)")

    n_struct = sum(len(v) for v in struct.values())
    n_cont = sum(len(v) for v in cont.values())
    print(f"\n구조 문제 {n_struct}건 · 내용 의심 {n_cont}건")

    if a.report:
        Path(a.report).write_text(json.dumps(
            {"structural": {k: v for k, v in struct.items()},
             "content": {k: v for k, v in cont.items()}},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"상세 → {a.report}")

    return 1 if n_struct else 0


if __name__ == "__main__":
    sys.exit(main())
