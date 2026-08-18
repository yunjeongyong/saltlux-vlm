"""
파트장 선별 기준에 맞춰 신규 3종에서 8천건을 추리고 upload_mix 와 합친다.

선별 기준 (2026-08-13 전달받은 것)
    kogovdoc-bench      294  ->  약   200   표·수식·계층 구조가 섞인 페이지
    multimodal-retrieval 14,208 -> 약 2,500~3,000  글자수로 단문/중문/중장문 층화, 단문 5% 미만
    pubtabnet-html     500,777 -> 약 4,000~5,000  간단한 표 제외, 복잡한 표 비중을 높임

출력은 학습 스크립트가 읽는 upload_mix 포맷(doc_id/task/images/messages)에 맞춘다.

주의 — 이 세 종의 **이미지가 서버에 없다**. 선별 자체는 정답 텍스트만으로 되므로
지금 만들어 두고, 이미지가 들어오면 --images-root 로 실경로만 붙이면 된다.
그 전에는 --check-images 로 존재 여부만 확인된다.

usage:
    # 선별만 (이미지 없이도 됨)
    python3 scripts/build_mix8k.py

    # 이미지가 들어온 뒤 — 실경로 확인까지
    python3 scripts/build_mix8k.py --images-root /workspace/vlm_data/images --check-images

    # upload_mix 와 합쳐 최종 학습셋 만들기
    python3 scripts/build_mix8k.py --merge-upload-mix --out receipt_data/mix8k
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAX = ROOT / "vlm_data/Distill-VLM/Qwen3.6-35B-A3B/01_taxonomy/document"
UPLOAD_MIX = ROOT / "receipt_data/upload_mix"

SRC = {
    "kogovdoc": TAX / "ocr/kogovdoc-bench/ocr_text_recognition.json",
    "multimodal": TAX / "ocr/multimodal-retrieval/ocr_text_recognition.json",
    "pubtabnet": TAX / "table/pubtabnet-html/table_extraction.json",
}

SEED = 42


def gt_of(rec):
    """assistant 턴이 정답이다."""
    for m in rec.get("messages", []):
        if m.get("role") == "assistant":
            return m.get("content") or ""
    return ""


# ────────────────────── kogovdoc: 구조가 있는 페이지 ──────────────────────

def kogov_score(gt):
    """표·수식·계층 제목이 섞일수록 높은 점수. 단순 본문 페이지는 0점."""
    s = 0
    if re.search(r"^\|.*\|", gt, re.M) or "<table" in gt.lower():
        s += 3                                   # 표
    if re.search(r"\$.+?\$|\\\(|\\begin\{(equation|align)", gt):
        s += 2                                   # 수식
    levels = {len(m.group(1)) for m in re.finditer(r"^(#{1,6})\s", gt, re.M)}
    if len(levels) >= 2:
        s += 2                                   # 제목 계층 2단 이상
    elif levels:
        s += 1
    if len(gt) > 800:
        s += 1                                   # 너무 짧은 페이지는 학습가치 낮음
    return s


def pick_kogov(rows, n):
    scored = [(kogov_score(gt_of(r)), r) for r in rows]
    scored = [(s, r) for s, r in scored if s > 0]        # 구조 없는 페이지 제외
    scored.sort(key=lambda x: -x[0])
    picked = [r for _, r in scored[:n]]
    return picked, Counter(s for s, _ in scored[:n])


# ────────────────── multimodal: 글자수 층화, 단문 5% 미만 ──────────────────

def length_bucket(gt):
    n = len(gt)
    if n < 200:
        return "단문"
    if n < 800:
        return "중문"
    return "중장문"


def pick_multimodal(rows, n, short_ratio=0.05):
    rnd = random.Random(SEED)
    by = {}
    for r in rows:
        by.setdefault(length_bucket(gt_of(r)), []).append(r)
    for v in by.values():
        rnd.shuffle(v)

    n_short = int(n * short_ratio)
    rest = n - n_short
    # 중문·중장문을 반반. 모자라면 있는 만큼만 가져오고 남은 몫은 다른 쪽에서 채운다.
    want = {"단문": n_short, "중문": rest // 2, "중장문": rest - rest // 2}
    picked, short_fall = [], 0
    for k in ("단문", "중문", "중장문"):
        take = by.get(k, [])[:want[k]]
        short_fall += want[k] - len(take)
        picked += take
    if short_fall:                                   # 부족분은 남은 것에서 보충
        pool = [r for k in ("중장문", "중문", "단문")
                for r in by.get(k, [])[want[k]:]]
        picked += pool[:short_fall]
    return picked, Counter(length_bucket(gt_of(r)) for r in picked)


# ─────────────── pubtabnet: 단순 표 제외, 복잡한 표 비중 확대 ───────────────

def table_complexity(gt):
    """셀 수·병합·서식·행수로 표 복잡도를 점수화."""
    cells = gt.count("<td")
    rows_ = gt.count("<tr")
    span = len(re.findall(r"(row|col)span", gt))
    fmt = len(re.findall(r"<(b|i|sup|sub|strong|em)\b", gt))
    return cells + rows_ + span * 3 + fmt * 2, cells, rows_, span


def pick_pubtabnet(rows, n, min_cells=12, min_rows=4):
    scored = []
    for r in rows:
        sc, cells, rows_, span = table_complexity(gt_of(r))
        if cells < min_cells or rows_ < min_rows:
            continue                                  # 간략한 표 제외
        scored.append((sc, r))
    scored.sort(key=lambda x: -x[0])
    picked = [r for _, r in scored[:n]]
    return picked, len(scored)


# ─────────────────────────── 공통 변환 ───────────────────────────

TASK_OF = {"kogovdoc": "doc_markdown", "multimodal": "page_ocr",
           "pubtabnet": "table_html"}


def to_train_row(rec, source, images_root=None):
    """taxonomy 레코드 -> upload_mix 포맷."""
    imgs = rec.get("images") or []
    if images_root:
        imgs = [str(Path(images_root) / p) for p in imgs]
    return {"doc_id": rec.get("doc_id"), "task": TASK_OF[source],
            "source": source, "images": imgs,
            "messages": rec.get("messages", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-kogov", type=int, default=200)
    ap.add_argument("--n-multimodal", type=int, default=2750)
    ap.add_argument("--n-pubtabnet", type=int, default=4500)
    ap.add_argument("--images-root", default=None,
                    help="이미지가 들어온 뒤 앞에 붙일 실경로")
    ap.add_argument("--check-images", action="store_true",
                    help="이미지 파일이 실제로 있는지 검사")
    ap.add_argument("--merge-upload-mix", action="store_true",
                    help="upload_mix/train.jsonl 을 앞에 붙여 최종 학습셋 생성")
    ap.add_argument("--out", default=str(ROOT / "receipt_data/mix8k"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    picked_all, report = [], []

    print("원본 적재 중 (pubtabnet 1.5GB 라 시간이 걸린다)", flush=True)
    for name in ("kogovdoc", "multimodal", "pubtabnet"):
        rows = json.loads(SRC[name].read_text(encoding="utf-8"))
        if name == "kogovdoc":
            picked, dist = pick_kogov(rows, args.n_kogov)
            detail = f"구조점수 분포 {dict(sorted(dist.items(), reverse=True))}"
        elif name == "multimodal":
            picked, dist = pick_multimodal(rows, args.n_multimodal)
            tot = sum(dist.values())
            detail = ("길이 분포 " + " / ".join(
                f"{k} {v}건({v/tot*100:.1f}%)" for k, v in dist.items()))
        else:
            picked, n_pass = pick_pubtabnet(rows, args.n_pubtabnet)
            detail = f"복잡도 조건 통과 {n_pass:,}건 중 상위 {len(picked):,}건"

        print(f"  {name:<12} {len(rows):>7,}행 -> {len(picked):>5,}건  {detail}")
        report.append({"source": name, "total": len(rows),
                       "picked": len(picked), "detail": detail})
        picked_all += [to_train_row(r, name, args.images_root) for r in picked]

    # 이미지 존재 검사
    missing = 0
    if args.check_images:
        for r in picked_all:
            for p in r["images"]:
                if not Path(p).exists():
                    missing += 1
        print(f"\n이미지 검사: 참조 {sum(len(r['images']) for r in picked_all):,}개 "
              f"중 없음 {missing:,}개")
    else:
        print("\n(이미지 존재 검사 안 함 — --check-images 로 확인)")

    sel = out / "mix8k.jsonl"
    sel.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                             for r in picked_all) + "\n", encoding="utf-8")
    print(f"\n신규 선별 {len(picked_all):,}건 -> {sel}")

    if args.merge_upload_mix:
        base = [json.loads(l) for l in
                (UPLOAD_MIX / "train.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
        merged = base + picked_all
        tp = out / "train.jsonl"
        tp.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                                for r in merged) + "\n", encoding="utf-8")
        # val/test 는 그대로 쓴다 (요청대로 동일 유지)
        for f in ("val.jsonl", "test.jsonl"):
            (out / f).write_text((UPLOAD_MIX / f).read_text(encoding="utf-8"),
                                 encoding="utf-8")
        print(f"upload_mix {len(base):,} + 신규 {len(picked_all):,} "
              f"= {len(merged):,}행 -> {tp}")
        print(f"val/test 는 upload_mix 그대로 복사 (각 98행)")

    (out / "selection_report.json").write_text(
        json.dumps({"criteria": report, "missing_images": missing},
                   ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
