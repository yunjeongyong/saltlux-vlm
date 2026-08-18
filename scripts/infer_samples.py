"""주간보고 샘플 이미지에 exp_003 어댑터로 추론을 돌린다.

학습 중인 GPU 는 건드리지 않도록 CUDA_VISIBLE_DEVICES 로 범위를 좁혀 쓴다.
결과는 images/_pred.json 에 모아 시각화 단계에서 GT 와 나란히 그린다.

usage: CUDA_VISIBLE_DEVICES=0,1 python3 scripts/infer_samples.py <images폴더>
"""
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts_receipt import PROMPTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMG = Path(sys.argv[1])
MODEL = ROOT / "ml/models/Qwen3.6-35B-A3B"
ADAPTER = ROOT / "ml/runs/upload-mix-qwen36"
MAXPX = 1_000_000

# 태스크마다 학습 때 쓴 지시문이 다르다. 추론도 같은 것을 붙여야 비교가 된다.
PROMPT_BY_TASK = {
    "html_table": "이미지 속 표를 웹페이지에 넣을 수 있는 HTML 코드로 변환해줘. "
                  "이미지에 없는 내용을 추측해서 넣지 마라. "
                  "설명이나 인사말 없이 결과만 출력하라.",
    "receipt_crop_table": PROMPTS["table"],
    "receipt_crop_text": PROMPTS["text"],
    "receipt_markdown": "이 영수증의 내용을 마크다운으로 정리해줘.",
    "page_ocr": PROMPTS["text"],
    "parsing/전체텍스트": PROMPTS["text"],
    "doc_parsing/government": PROMPTS["text"],
    "doc_parsing/paper": PROMPTS["text"],
}


def gen(model, proc, path, prompt, max_new=1024):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w * h > MAXPX:
        s = (MAXPX / (w * h)) ** 0.5
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": prompt}]}]
    try:
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=False)
    except TypeError:
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
    enc = proc(text=[text], images=[im], return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    ids = out[0][enc["input_ids"].shape[1]:]
    return proc.decode(ids, skip_special_tokens=True).strip(), time.time() - t0


def main():
    meta = json.loads((IMG / "_gt.json").read_text(encoding="utf-8"))
    print(f"대상 {len(meta)}장", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()
    print("모델 로드 완료 (exp_003 어댑터)", flush=True)

    # 시각화는 원본 학습 이미지를 다시 읽어야 한다. 테두리를 두른 png 로 추론하면
    # 그 선까지 읽으려 들 수 있어서, 학습 jsonl 의 원본 경로를 되짚는다.
    rows = {}
    for line in (ROOT / "receipt_data/exp004c_260813/train.jsonl").read_text(
            encoding="utf-8").split("\n"):
        if line.strip():
            r = json.loads(line)
            rows.setdefault(r["task"], []).append(r)

    out = []
    for m in meta:
        # gt 가 같은 행을 찾아 원본 이미지 경로를 얻는다
        src = next((r for r in rows.get(m["task"], [])
                    if r["messages"][-1]["content"] == m["gt"]), None)
        p = (ROOT / src["images"][0]) if src else (IMG / m["file"])
        if not p.exists():
            p = IMG / m["file"]
        prompt = PROMPT_BY_TASK.get(m["task"], PROMPTS["text"])
        pred, sec = gen(model, proc, p, prompt)
        out.append({**m, "pred": pred, "sec": round(sec, 1),
                    "src": str(p.relative_to(ROOT))})
        print(f"  {m['file']:34} {sec:5.1f}s  pred {len(pred):,}자", flush=True)
    (IMG / "_pred.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {IMG/'_pred.json'}")


if __name__ == "__main__":
    main()
