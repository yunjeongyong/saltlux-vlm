"""
학습 전/후 출력을 같은 이미지에 대해 나란히 뽑아 본다.

loss 가 내려갔다는 것만으로는 무엇이 좋아졌는지 알 수 없다. 같은 검증 이미지에
베이스 모델과 LoRA 를 붙인 모델을 각각 돌려, 정답과 함께 세 벌을 늘어놓는다.

두 과제를 섞어 학습했으므로 둘 다 확인한다.
  영수증 → 마크다운 표
  공공문서 → 평문 전사

usage:
    python3 scripts/eval_upload_run.py --adapter ml/runs/upload-v1 -n 3
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path("/data/workspace/yjyong")


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def cer(ref, hyp):
    """정답 대비 편집거리 비율. 낮을수록 좋다."""
    import difflib
    a, b = norm(ref), norm(hyp)
    if not a:
        return 1.0
    return 1 - difflib.SequenceMatcher(None, a, b).ratio()


def gen(model, proc, img_path, prompt, max_new=512):
    img = Image.open(ROOT / img_path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": prompt}]}]
    try:
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=False)
    except TypeError:                      # 사고모드 인자를 안 받는 버전
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
    enc = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    gen_ids = out[0][enc["input_ids"].shape[1]:]
    return proc.decode(gen_ids, skip_special_tokens=True).strip(), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "ml/models/Qwen3.5-4B"))
    ap.add_argument("--adapter", default=str(ROOT / "ml/runs/upload-v1"))
    ap.add_argument("--val", default=str(ROOT / "receipt_data/upload_mix/val.jsonl"))
    ap.add_argument("-n", type=int, default=3, help="과제별 표본 수")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "receipt_data/review/upload_run_eval.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.val).read_text(encoding="utf-8").splitlines() if l.strip()]
    rcp = [r for r in rows if r.get("task") == "receipt_markdown"]
    doc = [r for r in rows if r.get("task") == "page_ocr"]
    rng = random.Random(args.seed)
    picks = rng.sample(rcp, min(args.n, len(rcp))) + rng.sample(doc, min(args.n, len(doc)))
    print(f"검증 표본 {len(picks)}건 (영수증 {min(args.n,len(rcp))} / 문서 {min(args.n,len(doc))})")

    proc = AutoProcessor.from_pretrained(args.model)
    print("베이스 모델 로드…")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    res = []
    for r in picks:
        prompt = r["messages"][0]["content"].replace("<image>", "")
        txt, dt = gen(model, proc, r["images"][0], prompt, args.max_new)
        res.append({"doc_id": r["doc_id"], "task": r["task"], "image": r["images"][0],
                    "prompt": prompt, "gt": r["messages"][1]["content"],
                    "base": txt, "base_sec": round(dt, 1)})
        print(f"  [base] {r['doc_id'][:28]:30} {dt:5.1f}s  CER {cer(r['messages'][1]['content'], txt):.3f}")

    print("LoRA 어댑터 붙이는 중…")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    for i, r in enumerate(picks):
        prompt = r["messages"][0]["content"].replace("<image>", "")
        txt, dt = gen(model, proc, r["images"][0], prompt, args.max_new)
        res[i]["lora"] = txt
        res[i]["lora_sec"] = round(dt, 1)
        print(f"  [lora] {r['doc_id'][:28]:30} {dt:5.1f}s  CER {cer(res[i]['gt'], txt):.3f}")

    for r in res:
        r["cer_base"] = round(cer(r["gt"], r["base"]), 4)
        r["cer_lora"] = round(cer(r["gt"], r["lora"]), 4)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n과제별 CER (낮을수록 좋음)")
    for task in ("receipt_markdown", "page_ocr"):
        g = [r for r in res if r["task"] == task]
        if not g:
            continue
        b = sum(r["cer_base"] for r in g) / len(g)
        l = sum(r["cer_lora"] for r in g) / len(g)
        arrow = "개선" if l < b else "악화"
        print(f"  {task:18} base {b:.3f} → lora {l:.3f}   {arrow} {abs(b-l):.3f}")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
