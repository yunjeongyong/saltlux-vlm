"""
학습이 실제로 됐는지 검증.

loss 가 내려가는 것만으로는 부족하다. 같은 입력에 대해 학습 전/후 출력이
정답에 가까워졌는지를 문자 단위로 재야 한다. 그래서 두 가지를 본다:

  1. loss 추세 (train / eval)
  2. 학습 전 vs 후 생성 결과를 GT와 CER 비교

usage:
    python3 scripts/verify_train.py \
        --model /workspace/ml/models/Qwen3.5-2B \
        --adapter /workspace/ml/runs/qwen2b-block \
        --log /workspace/ml/train_qwen2b.log
"""
import argparse
import ast
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent


def parse_log(p):
    """Trainer 가 찍은 dict 로그에서 loss 계열을 뽑는다."""
    tr, ev = [], []
    for line in Path(p).read_text(errors="ignore").splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            d = ast.literal_eval(line)
        except Exception:
            continue
        f = lambda k: float(d[k]) if k in d else None
        if "loss" in d and "eval_loss" not in d:
            tr.append((f("epoch"), f("loss")))
        if "eval_loss" in d:
            ev.append((f("epoch"), f("eval_loss")))
    return tr, ev


def spark(vals, width=52):
    if len(vals) < 2:
        return "(데이터 부족)"
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    step = max(1, len(vals) // width)
    s = vals[::step]
    bars = "▁▂▃▄▅▆▇█"
    return "".join(bars[min(7, int((v - lo) / rng * 7.99))] for v in s)


def cer(ref, hyp):
    if not ref:
        return 1.0 if hyp else 0.0
    sm = SequenceMatcher(None, ref, hyp, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes()
               if t != "equal") / len(ref)


def chat_prompt(proc, instr):
    """enable_thinking=False 를 지원하면 끈다.

    Qwen3.5 는 기본값이 사고 ON 이라, 끄지 않으면 베이스 모델이 영어 사고 과정을
    길게 출력해 CER 이 수천 %로 부풀려진다. 그 상태로 파인튜닝 전후를 비교하면
    '사고를 안 하게 됐다'를 'OCR 이 좋아졌다'로 오독하게 된다.
    """
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": instr}]}]
    try:
        return proc.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=False, enable_thinking=False)
    except TypeError:
        return proc.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=False)


def strip_think(text):
    """혹시 남은 <think>…</think> 구간을 제거한다."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def gen(model, proc, img, instr, max_new=1024):
    prompt = chat_prompt(proc, instr)
    enc = proc(text=[prompt], images=[img], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    new = out[0][enc["input_ids"].shape[1]:]
    return strip_think(proc.decode(new, skip_special_tokens=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--val", default=str(ROOT / "eval_dataset/train/val_block.jsonl"))
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    # ---- 1. loss 추세
    if args.log and Path(args.log).exists():
        tr, ev = parse_log(args.log)
        print("=" * 70)
        print("[1] loss 추세")
        print("=" * 70)
        if tr:
            v = [x[1] for x in tr]
            print(f"  train  {len(v)}스텝  {v[0]:.4f} -> {v[-1]:.4f}   {spark(v)}")
            n = max(1, len(v) // 4)
            print(f"         앞 1/4 평균 {sum(v[:n])/n:.4f}  /  뒤 1/4 평균 {sum(v[-n:])/n:.4f}")
        if ev:
            v = [x[1] for x in ev]
            print(f"  eval   {len(v)}회    {v[0]:.4f} -> {v[-1]:.4f}   {spark(v)}")
        else:
            print("  eval   기록 없음")
        print()

    # ---- 2. 학습 전/후 생성 비교
    rows = [json.loads(l) for l in open(args.val, encoding="utf-8")][:args.n]
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    print("=" * 70)
    print("[2] 학습 전/후 생성 비교 (val 셋)")
    print("=" * 70)

    print("  베이스 모델 로드...", flush=True)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    base.eval()
    before = []
    for r in rows:
        img = Image.open(ROOT / r["images"][0]).convert("RGB")
        instr = r["messages"][0]["content"].replace("<image>", "")
        before.append(gen(base, proc, img, instr))
    del base
    torch.cuda.empty_cache()

    print("  어댑터 적용 모델 로드...", flush=True)
    from peft import PeftModel
    tuned = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    tuned = PeftModel.from_pretrained(tuned, args.adapter)
    tuned.eval()
    after = []
    for r in rows:
        img = Image.open(ROOT / r["images"][0]).convert("RGB")
        instr = r["messages"][0]["content"].replace("<image>", "")
        after.append(gen(tuned, proc, img, instr))

    print(f"\n{'샘플':<26}{'학습전 CER':>11}{'학습후 CER':>11}   판정")
    print("-" * 66)
    b_tot = a_tot = 0.0
    for r, b, a in zip(rows, before, after):
        gt = r["messages"][1]["content"]
        cb, ca = cer(gt, b), cer(gt, a)
        b_tot += cb; a_tot += ca
        mark = "개선" if ca < cb - 0.01 else ("악화" if ca > cb + 0.01 else "변화없음")
        print(f"{r['id'][:25]:<26}{cb:>10.2%}{ca:>11.2%}   {mark}")
    n = len(rows)
    print("-" * 66)
    print(f"{'평균':<26}{b_tot/n:>10.2%}{a_tot/n:>11.2%}   "
          f"{'학습 효과 있음' if a_tot < b_tot - 0.01*n else '학습 효과 불명확'}")

    out = Path(args.adapter) / "verify.json"
    out.write_text(json.dumps({
        "samples": [{"id": r["id"], "gt": r["messages"][1]["content"],
                     "before": b, "after": a,
                     "cer_before": cer(r["messages"][1]["content"], b),
                     "cer_after": cer(r["messages"][1]["content"], a)}
                    for r, b, a in zip(rows, before, after)],
        "cer_before_avg": b_tot / n, "cer_after_avg": a_tot / n,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n상세 -> {out}")


if __name__ == "__main__":
    main()
