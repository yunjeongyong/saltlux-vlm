"""
학습 시퀀스 길이 실측.

"토큰 길이 제한을 얼마로 잡을 것인가"는 추측할 문제가 아니다.
영수증은 세로로 긴 이미지라 Qwen 계열의 동적 해상도에서 비전 토큰이
폭발하고, 그 대부분이 시퀀스 길이를 차지한다. 실제 이미지 크기와
타겟 텍스트로 분포를 뽑아, max-px / max-len 을 근거 있게 정한다.

usage:
    python3 scripts/token_budget.py --model ml/models/Qwen3.6-35B-A3B
    python3 scripts/token_budget.py --model ... --max-px 1000000 4000000
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
Image.MAX_IMAGE_PIXELS = None


def smart_resize(h, w, factor, min_px, max_px):
    """Qwen 이미지 프로세서와 동일한 규칙 (factor 배수 정렬 + 픽셀 예산)."""
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_px:
        s = math.sqrt((h * w) / max_px)
        hb = max(factor, math.floor(h / s / factor) * factor)
        wb = max(factor, math.floor(w / s / factor) * factor)
    elif hb * wb < min_px:
        s = math.sqrt(min_px / (h * w))
        hb = math.ceil(h * s / factor) * factor
        wb = math.ceil(w * s / factor) * factor
    return hb, wb


def pct(v, q):
    if not v:
        return 0
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * q))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(ROOT / "receipt_data/unified/train.jsonl"))
    # 학습 스크립트의 --max-px 후보들. 사전 축소가 비전 토큰을 결정한다.
    ap.add_argument("--max-px", type=int, nargs="+",
                    default=[6_000_000, 4_000_000, 2_000_000, 1_000_000, 500_000])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    ip, tok = proc.image_processor, proc.tokenizer
    factor = ip.patch_size * ip.merge_size          # 토큰 1개가 덮는 변 길이
    px_per_tok = factor * factor
    proc_max = ip.size.longest_edge
    proc_min = ip.size.shortest_edge
    print(f"factor {factor}px  ({px_per_tok}px/token)  "
          f"processor 예산 {proc_min:,}~{proc_max:,}px "
          f"→ 최대 {proc_max // px_per_tok:,} 비전토큰\n")

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    # 태스크 구분: 지시문으로 나뉜다 (full_md 는 마크다운 표 요청)
    stat = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        u, a = r["messages"][0]["content"], r["messages"][1]["content"]
        task = "full_md" if "마크다운" in u else "field"
        with Image.open(ROOT / r["images"][0]) as im:
            w, h = im.size
        n_txt = len(tok(a)["input_ids"])
        n_usr = len(tok(u.replace("<image>", ""))["input_ids"])
        stat[task]["orig_px"].append(w * h)
        stat[task]["tgt"].append(n_txt)
        stat[task]["usr"].append(n_usr)
        for mp in args.max_px:
            ww, hh = w, h
            if ww * hh > mp:                        # 학습 스크립트의 사전 축소
                s = (mp / (ww * hh)) ** 0.5
                ww, hh = int(ww * s), int(hh * s)
            rh, rw = smart_resize(hh, ww, factor, proc_min, proc_max)
            stat[task][f"vis@{mp}"].append(rh * rw // px_per_tok)
        if i % 2000 == 0 and i:
            print(f"  {i}/{len(rows)}", flush=True)

    OVER = 40                                        # 채팅 템플릿 오버헤드 여유분
    for task, s in stat.items():
        n = len(s["tgt"])
        print(f"\n{'=' * 66}\n{task}  {n}건\n{'=' * 66}")
        print(f"원본 픽셀   중앙 {pct(s['orig_px'], .5):,}  "
              f"p95 {pct(s['orig_px'], .95):,}  최대 {max(s['orig_px']):,}")
        print(f"타겟 토큰   중앙 {pct(s['tgt'], .5)}  p95 {pct(s['tgt'], .95)}  "
              f"최대 {max(s['tgt'])}")
        print(f"지시 토큰   중앙 {pct(s['usr'], .5)}")
        print(f"\n{'max-px':>10} {'비전토큰 중앙':>14} {'p95':>8} {'최대':>8} "
              f"{'총길이 p95':>11} {'총길이 최대':>12}")
        for mp in args.max_px:
            v = s[f"vis@{mp}"]
            tot = [a + b for a, b in zip(sorted(v), sorted(s["tgt"]))]
            mx = max(v) + max(s["tgt"]) + pct(s["usr"], .5) + OVER
            print(f"{mp:>10,} {pct(v, .5):>14,} {pct(v, .95):>8,} {max(v):>8,} "
                  f"{pct(tot, .95) + pct(s['usr'], .5) + OVER:>11,} {mx:>12,}")


if __name__ == "__main__":
    main()
