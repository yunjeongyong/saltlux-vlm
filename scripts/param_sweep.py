"""
추론 파라미터 스윕 — 반복 생성/할루시네이션을 줄이는 설정을 고른다.

모델을 한 번만 올리고 설정을 바꿔가며 돌린다. 설정마다 따로 실행하면
65.5GiB 로딩에 매번 2분씩 버린다.

품질 등급(clean/light/heavy/...)별로 나눠 집계한다. 등급을 섞으면
"heavy 에서만 나는 문제"가 평균에 묻혀서 파라미터 효과가 안 보인다.

의심하는 것: 기본값 repetition_penalty=1.05 가 오히려 해로울 수 있다.
마크다운 표는 '|', '---', 쉼표, 숫자가 정당하게 반복되는 구조인데
거기에 패널티를 걸고 있다.

usage:
    CUDA_VISIBLE_DEVICES=2,3 python3 scripts/param_sweep.py --n 120
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from bench_receipt import cer                                  # noqa: E402
from infer_receipt import RECEIPT_CONTRACT, build_prompt, load_img   # noqa: E402

ROW = re.compile(r"^\s*\|\s*([^|]+?)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*$")

# (이름, generate kwargs)
CONFIGS = [
    ("AS-IS  greedy+rep1.05", dict(do_sample=False, repetition_penalty=1.05)),
    ("greedy  rep 1.00",      dict(do_sample=False, repetition_penalty=1.0)),
    ("greedy  rep 1.10",      dict(do_sample=False, repetition_penalty=1.10)),
    ("greedy  rep 1.20",      dict(do_sample=False, repetition_penalty=1.20)),
    ("greedy  no_repeat_3",   dict(do_sample=False, repetition_penalty=1.0,
                                   no_repeat_ngram_size=3)),
    ("sample  T=0.2 top_p.9", dict(do_sample=True, temperature=0.2, top_p=0.9,
                                   repetition_penalty=1.0)),
    ("sample  T=0.7 top_p.9", dict(do_sample=True, temperature=0.7, top_p=0.9,
                                   repetition_penalty=1.0)),
]


def load_tiers():
    d = ROOT / "receipt_data/unified/eval_by_quality"
    t = {}
    if d.is_dir():
        for f in os.listdir(d):
            if f.endswith(".jsonl"):
                for l in open(d / f, encoding="utf-8"):
                    if l.strip():
                        t[json.loads(l)["images"][0].split("/")[-1]] = f[:-6]
    return t


def loops(text):
    """반복 생성 감지 — 같은 줄이 3번 이상, 또는 같은 표 행이 반복."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 0
    from collections import Counter
    c = Counter(lines)
    return sum(v - 2 for v in c.values() if v >= 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "ml/models/Qwen3.6-35B-A3B"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--val", default=str(ROOT / "receipt_data/unified/test.jsonl"))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    tiers = load_tiers()
    rows = [json.loads(l) for l in open(args.val, encoding="utf-8") if l.strip()]
    random.seed(args.seed)
    items = [r for r in random.sample(rows, min(args.n, len(rows)))
             if "receipts3000" in r["images"][0]]

    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[1/2] 로드 · 영수증 {len(items)}장 x 설정 {len(CONFIGS)}개", flush=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device_map,
        trust_remote_code=True)
    model.config.use_cache = True
    model.eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    # 이미지는 설정마다 다시 안 읽는다
    cache = [(load_img(r["images"][0], 6_000_000), r) for r in items]
    pad = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id
    results = {}

    for ci, (name, kw) in enumerate(CONFIGS, 1):
        print(f"\n[2/2] ({ci}/{len(CONFIGS)}) {name}", flush=True)
        rec = []
        for img, r in cache:
            gt = r["messages"][1]["content"]
            enc = proc(text=[build_prompt(proc, RECEIPT_CONTRACT)],
                       images=[img], return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            n_in = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new,
                                     pad_token_id=pad, **kw)
            hyp = proc.tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
            hyp = hyp.split("</think>")[-1].strip()
            gi = [m.group(1).strip() for l in gt.splitlines()
                  if (m := ROW.match(l)) and not set(m.group(1)) <= set(":- ")]
            hi = [m.group(1).strip() for l in hyp.splitlines()
                  if (m := ROW.match(l)) and not set(m.group(1)) <= set(":- ")]
            rec.append({
                "tier": tiers.get(Path(r["images"][0]).name, "?"),
                "cer": cer(gt, hyp), "exact": gt.strip() == hyp.strip(),
                "lr": len(hyp) / len(gt) if gt else 0,
                "loop": loops(hyp), "trunc": out.shape[1] - n_in >= args.max_new,
                "name_ok": sum(1 for i in range(min(len(gi), len(hi)))
                               if gi[i] == hi[i]),
                "name_n": min(len(gi), len(hi)),
            })
        results[name] = rec
        n = len(rec)
        print(f"     완전일치 {sum(r['exact'] for r in rec)}/{n}  "
              f"CER {sum(r['cer'] for r in rec)/n:.4f}  "
              f"반복 {sum(r['loop'] for r in rec)}  "
              f"잘림 {sum(r['trunc'] for r in rec)}", flush=True)

    # ── 최종 표
    print("\n" + "=" * 96)
    print(f"{'설정':<24}{'완전일치':>10}{'평균CER':>10}{'길이배율':>10}"
          f"{'상품명':>10}{'반복줄':>8}{'잘림':>6}")
    print("=" * 96)
    for name, rec in results.items():
        n = len(rec)
        nn = sum(r["name_n"] for r in rec)
        print(f"{name:<24}{sum(r['exact'] for r in rec):>6}/{n:<3}"
              f"{sum(r['cer'] for r in rec)/n:>10.4f}"
              f"{sum(r['lr'] for r in rec)/n:>10.3f}"
              f"{sum(r['name_ok'] for r in rec)/nn*100:>9.1f}%"
              f"{sum(r['loop'] for r in rec):>8}"
              f"{sum(r['trunc'] for r in rec):>6}")

    order = ["clean", "light", "sharp_photo", "screenshot", "heavy", "extreme", "?"]
    present = [t for t in order if any(r["tier"] == t for rec in results.values()
                                       for r in rec)]
    print("\n품질 등급별 완전일치")
    print(f"{'설정':<24}" + "".join(f"{t:>13}" for t in present))
    for name, rec in results.items():
        line = f"{name:<24}"
        for t in present:
            sub = [r for r in rec if r["tier"] == t]
            line += f"{sum(r['exact'] for r in sub)}/{len(sub)}".rjust(13) if sub else "".rjust(13)
        print(line)


if __name__ == "__main__":
    main()
