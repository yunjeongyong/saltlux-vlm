"""미학습 샘플에 exp_003 어댑터로 추론을 돌린다.

infer_samples.py 와 다른 점은 하나다 — 원본 경로를 학습 jsonl 에서 되짚지 않고
_gt.json 의 src 를 그대로 쓴다. 평가셋 크롭은 애초에 학습 jsonl 에 없기 때문이다.

학습 중인 GPU(2,3)는 건드리지 않는다.
usage: CUDA_VISIBLE_DEVICES=0,1 python3 scripts/infer_unseen.py <폴더>
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


def load(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w * h > MAXPX:                       # 학습 때와 같은 상한을 건다
        s = (MAXPX / (w * h)) ** 0.5
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    return im


def gen(model, proc, path, prompt, max_new=1024):
    im = load(path)
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": prompt}]}]
    # 학습 때 사고모드를 끄고 태웠다. 평가(eval_crops.py)도 끄고 잰다.
    # 켜면 </think> 서두가 그대로 출력에 섞여 CER·TEDS 가 무의미해진다.
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

    out = []
    for m in meta:
        p = ROOT / m["src"]
        prompt = PROMPT_BY_TASK.get(m["task"], PROMPTS["text"])
        pred, sec = gen(model, proc, p, prompt)
        out.append({**m, "pred": pred, "sec": round(sec, 1)})
        print(f"  {m['file']:28} {sec:5.1f}s  pred {len(pred):,}자", flush=True)
        (IMG / "_pred.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {IMG/'_pred.json'}")


if __name__ == "__main__":
    main()
