"""학습 데이터의 실제 토큰 길이를 잰다 — max_len 을 정하기 전에 확인용.

train_vlm.py 는 max_len 초과분을 truncation=True 로 조용히 자른다. 잘리는 쪽은
정답(assistant) 뒤쪽이라, 모델은 "중간에서 끊는 법"을 학습하게 된다. 경고도
남지 않으므로 학습 전에 여기서 미리 세어 본다.

이미지 토큰은 프로세서를 한 번 태워 실측한 뒤, 나머지 행은 같은 공식으로 계산한다
(11,000장을 전부 프로세서에 넣으면 너무 느리다).

usage: python3 scripts/check_token_len.py <train.jsonl> [--max-len 8192] [--max-px 1000000]
"""
import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "ml/models/Qwen3.6-35B-A3B"

Image.MAX_IMAGE_PIXELS = None


def img_tokens(w, h, max_px, patch, merge, min_px, proc_max_px):
    """학습과 같은 순서로 이미지 토큰 수를 센다.

    ① train_vlm.py 가 max_px 로 먼저 축소하고
    ② 프로세서의 smart_resize 가 unit 배수로 맞추면서 min/max 픽셀을 강제한다.
       작은 크롭은 여기서 오히려 확대된다 — 얇은 띠 이미지가 토큰을 더 먹는 이유.
    """
    if w * h > max_px:
        s = (max_px / (w * h)) ** 0.5
        w, h = int(w * s), int(h * s)
    unit = patch * merge                      # 병합 후 토큰 하나가 덮는 픽셀
    hb = max(unit, round(h / unit) * unit)
    wb = max(unit, round(w / unit) * unit)
    if hb * wb > proc_max_px:
        beta = ((h * w) / proc_max_px) ** 0.5
        hb = max(unit, math.floor(h / beta / unit) * unit)
        wb = max(unit, math.floor(w / beta / unit) * unit)
    elif hb * wb < min_px:
        beta = (min_px / (h * w)) ** 0.5
        hb = math.ceil(h * beta / unit) * unit
        wb = math.ceil(w * beta / unit) * unit
    return (hb // unit) * (wb // unit), (wb, hb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--max-px", type=int, default=1_000_000)
    ap.add_argument("--verify", type=int, default=5,
                    help="프로세서로 실측 검증할 표본 수")
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.jsonl).read_text(
        encoding="utf-8").split("\n") if l.strip()]
    print(f"{a.jsonl} — {len(rows):,}행 · max_len {a.max_len:,} · "
          f"max_px {a.max_px:,}\n")

    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    ip = proc.image_processor
    patch = getattr(ip, "patch_size", 16)
    merge = getattr(ip, "merge_size", 2)
    size = getattr(ip, "size", {}) or {}
    min_px = size.get("shortest_edge", 65536)
    proc_max_px = size.get("longest_edge", 16_777_216)
    tok = proc.tokenizer

    # ── 공식 검증 ────────────────────────────────────────────────────
    print(f"[검증] 이미지 토큰 공식 (patch {patch} · merge {merge} · "
          f"min_px {min_px:,} · proc_max_px {proc_max_px:,})")
    ok = True
    for r in rows[:a.verify]:
        p = ROOT / r["images"][0]
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        w, h = im.size
        est, (rw, rh) = img_tokens(w, h, a.max_px, patch, merge, min_px, proc_max_px)
        if w * h > a.max_px:
            s = (a.max_px / (w * h)) ** 0.5
            im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
        enc = proc(text=["<image>"], images=[im], return_tensors="pt")
        real = int((enc["input_ids"][0] == tok.convert_tokens_to_ids(
            "<|image_pad|>")).sum())
        real = real or enc["pixel_values"].shape[0] // (merge * merge)
        flag = "" if abs(real - est) <= 1 else "  ← 불일치"
        ok &= abs(real - est) <= 1
        print(f"  {w}x{h} → {rw}x{rh}  추정 {est:,} / 실측 {real:,}{flag}")
    print(f"  {'공식 일치' if ok else '공식 불일치 — 추정치 신뢰 불가'}\n")

    # ── 전체 집계 ────────────────────────────────────────────────────
    stat = []
    for r in rows:
        p = ROOT / r["images"][0]
        if not p.exists():
            continue
        with Image.open(p) as im:
            w, h = im.size
        it, _ = img_tokens(w, h, a.max_px, patch, merge, min_px, proc_max_px)
        user, asst = r["messages"][0], r["messages"][1]
        pt = len(tok(user["content"].replace("<image>", ""))["input_ids"])
        at = len(tok(asst["content"])["input_ids"])
        stat.append({"task": r["task"], "img": it, "prompt": pt, "ans": at,
                     "total": it + pt + at + 20})      # 20 = 템플릿 특수토큰 여유

    stat.sort(key=lambda s: -s["total"])
    n = len(stat)
    over = [s for s in stat if s["total"] > a.max_len]
    print(f"[전체] {n:,}행")
    for q in (50, 90, 95, 99, 100):
        v = stat[min(n - 1, int(n * (100 - q) / 100))]["total"]
        print(f"  p{q:<3} {v:>7,} 토큰")
    print(f"\n  max_len {a.max_len:,} 초과: {len(over):,}행 "
          f"({100*len(over)/n:.2f}%)")
    if over:
        lost = sum(s["total"] - a.max_len for s in over)
        print(f"  잘려나가는 정답 토큰 합계 ≈ {lost:,}")
        from collections import Counter
        for t, c in Counter(s["task"] for s in over).most_common():
            tot = sum(1 for s in stat if s["task"] == t)
            print(f"    {t:24} {c:>5} / {tot:<6} ({100*c/tot:.1f}%)")

    print(f"\n[구성 평균] 이미지 {sum(s['img'] for s in stat)//n:,} · "
          f"프롬프트 {sum(s['prompt'] for s in stat)//n:,} · "
          f"정답 {sum(s['ans'] for s in stat)//n:,}")
    print(f"[이미지 토큰 최대] {max(s['img'] for s in stat):,}")
    print(f"[정답 토큰 최대]   {max(s['ans'] for s in stat):,}")

    print("\n[상위 10행]")
    print(f"  {'task':24} {'이미지':>7} {'프롬프트':>7} {'정답':>7} {'합계':>8}")
    for s in stat[:10]:
        print(f"  {s['task']:24} {s['img']:>7,} {s['prompt']:>7,} "
              f"{s['ans']:>7,} {s['total']:>8,}")


if __name__ == "__main__":
    main()
