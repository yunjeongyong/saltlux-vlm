"""
격리된 synthetic 608장에서 '실제 인쇄된 빈 서식'만 되살린다.

앞선 필터(filter_crawled_photos.py)에서 synthetic 정의에 '빈 양식'을 넣은 탓에,
한국식 영수증 서식처럼 실물 인쇄물이지만 값만 안 채워진 것들이 전부 목업·AI합성과
같은 통에 들어갔다. 성격이 전혀 다르므로 다시 가른다.

값이 없으니 VLM(값 추출) 학습에는 못 쓴다. 다만 표 괘선·칸 구조가 오히려 선명해서
Layout Detection 학습에는 쓸 만하다. 그래서 별도 셋으로 뺀다.

  form       실제 인쇄·발행되는 서식. 한국식 영수증/간이영수증/거래명세서/기부금영수증
             등. 괘선과 칸이 뚜렷하고 인쇄물 특유의 서식 번호·안내문구가 있다.
  mockup     디자인 목업·스톡·AI생성. Lorem Ipsum, SHOP NAME, freepik 등 워터마크.
  infographic 설명용 도해·광고·제품사진. 화살표 주석, 홍보 문구.
  other      위 어디에도 안 맞음.

form 만 forms_detection/images/ 로 복사한다. 원본은 _rejected/ 에 그대로 둔다.

usage:
    python3 scripts/reclassify_forms.py --limit 20
    python3 scripts/reclassify_forms.py --apply
"""
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

MODEL = "/workspace/ml/models/Qwen3.5-4B"
ROOT = Path("/workspace/receipt_data/crawl_google")

PROMPT = """이 이미지를 넷 중 하나로 분류해라. 아래 순서대로 판단하고, 먼저 걸리는 것으로 정한다.

1단계 — 서식 자체가 아니라 '무언가를 설명하거나 파는 이미지'인가? 그러면 infographic.
  다음 중 하나라도 있으면 무조건 infographic 이다.
  · 화살표나 지시선, 부위별 설명 라벨("상품 내역", "결제 내역", "바코드" 등)
  · 제목형 홍보 문구("~란 무엇인가요?", "~표준양식", "~하는 법", "이젠 ~하세요")
  · 광고 카피, 회사 로고, 앱 아이콘, 다운로드 버튼, 가격표
  · 캐릭터·일러스트·말풍선
  · 여러 장을 나란히 놓고 O/X 나 색으로 옳고 그름을 비교하는 구성
  · 상품을 판매하는 사진(스티커·용지 묶음 등)

2단계 — 실물 종이를 카메라로 찍은 사진인가? 그러면 other.
  책상·바구니·손 위에 놓인 종이를 촬영한 것. 스캔이나 평면 이미지가 아니다.

3단계 — 디자인 소재인가? 그러면 mockup.
  "Lorem Ipsum", "SHOP NAME", "Your Company", freepik 워터마크, 영문 더미 문구.
  앱 화면 UI 디자인.

4단계 — 위 어디에도 안 걸리고, 관공서·사업장에서 실제로 쓰는 인쇄 서식이면 form.
  한국식 영수증, 간이영수증, 거래명세서, 기부금영수증, 공급자용/공급받는자용 서식.
  표 괘선과 칸만 있고 설명·홍보 요소가 전혀 없다. 값이 비어 있어도 form 이다.

주의: 영수증 서식처럼 보인다고 바로 form 으로 정하지 마라. 1~3단계를 먼저 확인해라.

정확히 다음 형식으로만 답해라.
label: <form|mockup|infographic|other>
reason: <한 문장>"""

LABELS = ["form", "mockup", "infographic", "other"]
LABEL_RE = re.compile(r"label\s*:\s*(form|mockup|infographic|other)", re.I)


def prep(path, max_px=1_000_000):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w * h > max_px:
        s = (max_px / (w * h)) ** 0.5
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return im


def build_prompt(proc, text):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    try:
        return proc.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=False, enable_thinking=False)
    except TypeError:
        return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def classify(model, proc, img, max_new=64):
    enc = proc(text=[build_prompt(proc, PROMPT)], images=[img], return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    n_in = enc["input_ids"].shape[1]
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id)
    return proc.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).split("</think>")[-1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rejected", default=str(ROOT / "_rejected"))
    ap.add_argument("--filter-log", default=str(ROOT / "photo_filter.jsonl"))
    ap.add_argument("--out", default=str(ROOT.parent / "forms_detection"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--apply", action="store_true", help="form 을 별도 셋으로 복사")
    args = ap.parse_args()

    rej = Path(args.rejected)
    # synthetic 으로 격리된 것만 다시 본다. graphic/notreceipt 는 그대로 둔다.
    targets = [rej / json.loads(l)["file"] for l in open(args.filter_log, encoding="utf-8")
               if json.loads(l)["label"] == "synthetic"]
    targets = [p for p in targets if p.exists()]
    if args.limit:
        targets = targets[:args.limit]
    if not targets:
        raise SystemExit("대상이 없다.")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"로드: {Path(args.model).name}  |  대상 {len(targets)}장", flush=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = True
    model.eval()

    counts, results = Counter(), []
    for i, p in enumerate(targets, 1):
        try:
            raw = classify(model, proc, prep(p))
        except Exception as e:
            counts["error"] += 1
            results.append({"file": p.name, "label": "error", "raw": str(e)[:120]})
            continue
        m = LABEL_RE.search(raw)
        label = m.group(1).lower() if m else next((l for l in LABELS if l in raw.lower()), "unknown")
        counts[label] += 1
        results.append({"file": p.name, "label": label, "raw": raw.replace("\n", " ")[:200]})
        if i % 25 == 0 or i == len(targets):
            print(f"  [{i}/{len(targets)}] {dict(counts)}", flush=True)

    log = ROOT / "form_reclass.jsonl"
    with log.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n분포: {dict(counts.most_common())}\n로그: {log}")

    if args.apply:
        out = Path(args.out)
        img_dir = out / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for r in results:
            if r["label"] != "form":
                continue
            src = rej / r["file"]
            if not src.exists():
                continue
            # 원본은 _rejected/ 에 남긴다. 판단이 또 바뀔 수 있다.
            dst = img_dir / f"form_{len(rows) + 1:05d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            with Image.open(dst) as im:
                w, h = im.size
            rows.append({"doc_id": dst.stem,
                         "image": f"receipt_data/forms_detection/images/{dst.name}",
                         "width": w, "height": h, "source": "google-crawl-form",
                         "capture": "form", "split": "train", "gt_status": "none",
                         "src_file": r["file"], "n_fields": 0, "n_items": 0, "n_layout": 0})
        mf = out / "manifest.jsonl"
        with mf.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nDetection 전용 셋: {img_dir} ({len(rows)}장)\nmanifest: {mf}")
        print("값이 비어 있으므로 VLM(값 추출) 학습에는 쓰지 말 것.")
    else:
        print(f"\n--apply 없이 판정만 했다. 적용하면 form {counts['form']}장이 별도 셋으로 간다.")


if __name__ == "__main__":
    main()
