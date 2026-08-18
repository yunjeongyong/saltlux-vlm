"""
학습셋 무작위 표본을 입력/출력 쌍으로 볼 수 있는 HTML 을 만든다.

학습을 걸기 전에 사람이 눈으로 확인하는 용도다. 숫자만으로는 좌표가 어긋났거나
정답이 엉뚱한 문서에 붙었거나 표가 깨진 것을 알 수 없다.

이미지는 외부 링크가 아니라 data URI 로 박아 넣는다(파일 하나로 열려야 한다).
원본 그대로면 수십 MB 가 되므로 긴 변을 줄이고 JPEG 로 다시 굽는다.

usage:
    python3 scripts/viz_dataset_samples.py
    python3 scripts/viz_dataset_samples.py --seed 7 --n-receipt 6
"""
import argparse
import base64
import html
import io
import json
import random
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def load(p):
    return [json.loads(l) for l in (ROOT / p).open(encoding="utf-8") if l.strip()]


def thumb(rel, max_side=680, quality=72):
    """긴 변 기준으로 줄여 base64 JPEG 로 만든다."""
    im = Image.open(ROOT / rel).convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}", w, h


def tables(s):
    """정답 안의 <table> 덩어리만 뽑는다. 렌더링해서 구조를 눈으로 보려고."""
    return re.findall(r"<table.*?</table>", s or "", flags=re.S | re.I)


def sample(rows, n, rng, key=None):
    pool = [r for r in rows if key is None or key(r)]
    return rng.sample(pool, min(n, len(pool)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-receipt", type=int, default=4)
    ap.add_argument("--n-doc", type=int, default=3)
    ap.add_argument("--n-val", type=int, default=3)
    ap.add_argument("--n-crop", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "receipt_data/review/dataset_samples.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tr = load("receipt_data/upload_mix/train.jsonl")
    va = load("receipt_data/upload_mix/val.jsonl")
    cr = load("receipt_data/eval_crops/manifest.jsonl")

    picks = []
    for r in sample(tr, args.n_receipt, rng, lambda x: x.get("task") == "receipt_markdown"):
        picks.append(("train", "receipt_markdown", r))
    for r in sample(tr, args.n_doc, rng, lambda x: x.get("task") == "page_ocr"):
        picks.append(("train", "page_ocr", r))
    for r in sample(va, args.n_val, rng):
        picks.append(("val/test", "receipt_markdown", r))

    out = []
    for split, task, r in picks:
        src, w, h = thumb(r["images"][0])
        ans = r["messages"][1]["content"]
        out.append({
            "split": split, "task": task, "doc_id": r.get("doc_id", "-"),
            "source": r.get("source", ""), "path": r["images"][0],
            "img": src, "w": w, "h": h,
            "prompt": r["messages"][0]["content"].replace("<image>", ""),
            "answer": ans, "tables": tables(ans),
            "chars": len(ans),
        })

    # 평가 크롭은 경로 기준이 다르다 (manifest 의 image 는 eval_crops 상대경로)
    base = "receipt_data/eval_crops/"
    for t in ("table", "text"):
        for r in sample(cr, args.n_crop, rng, lambda x, t=t: x["task"] == t):
            src, w, h = thumb(base + r["image"], max_side=760, quality=80)
            out.append({
                "split": "eval_crops", "task": f"crop_{t}", "doc_id": r["crop_id"],
                "source": r["label"], "path": base + r["image"],
                "img": src, "w": w, "h": h,
                "prompt": "(평가 시 OCR/TABLE 프롬프트 사용 — 4번 항목 참고)",
                "answer": r["gt"], "tables": tables(r["gt"]), "chars": len(r["gt"]),
            })

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    mb = p.stat().st_size / 2**20
    print(f"표본 {len(out)}건 -> {p} ({mb:.1f}MB)")
    for o in out:
        print(f"  [{o['split']:<10}] {o['task']:<18} {o['doc_id']:<28} "
              f"{o['w']}x{o['h']}px  정답 {o['chars']}자  표 {len(o['tables'])}개")


if __name__ == "__main__":
    main()
