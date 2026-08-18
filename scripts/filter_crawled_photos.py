"""
크롤링한 이미지에서 '사람이 실제로 촬영한 영수증'만 남긴다.

구글 이미지에는 실사 촬영본보다 목업·스톡·일러스트가 훨씬 많이 섞인다.
문제는 그중 상당수가 얼핏 사진처럼 보인다는 것이다. 예를 들어 구겨진 종이
위에 글자를 합성한 스톡 이미지는 배경만 보면 사진이지만, 주름을 따라
글자가 전혀 휘지 않고 감열지 특유의 번짐·노이즈가 없다. VLM 에게 바로
그 지점을 보게 한다.

판정은 4분류로 받는다. 이진 판정보다 오분류 원인을 추적하기 쉽다.
  photo      실제 촬영 (조명·그림자·원근·종이질감이 일관)
  synthetic  합성/목업/AI생성 (글자가 표면 왜곡을 안 따라감, 인쇄 노이즈 없음)
  graphic    일러스트·아이콘·클립아트·스크린샷
  notreceipt 영수증이 아님

photo 만 남기고 나머지는 지운다. 다만 바로 rm 하지 않고 _rejected/ 로 옮긴다.
분류기가 틀렸을 때 되돌릴 수 있어야 하고, 기준을 눈으로 확인해야 하기 때문이다.

usage:
    python3 scripts/filter_crawled_photos.py --dir receipt_data/crawl_google/images --limit 20
    python3 scripts/filter_crawled_photos.py --dir receipt_data/crawl_google/images --apply
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

PROMPT = """이 이미지가 '실제로 발행된 진짜 영수증'을 담고 있는지 판정해라.

다음 다섯 중 하나만 골라라.
- photo: 카메라로 촬영한 실제 영수증. 조명·그림자·원근이 자연스럽고, 종이 표면의 굴곡을 따라 글자도 같이 휘거나 흐려진다. 손에 들고 있거나 바닥·책상 위에 놓여 있다.
- scan: 실제 영수증이지만 스캐너나 정면 촬영으로 평평하게 담긴 것. 원근 왜곡은 없지만 인쇄 상태가 실제 감열지답게 불균일하고, 상호·사업자번호·금액이 실제 값이다.
- synthetic: 합성·목업·AI생성·빈 양식. 종이는 구겨졌는데 글자는 완벽히 평평하고 선명하다. 인쇄 노이즈가 없다. 값이 비어 있거나 더미다(1234-1234, 홍길동, OOO, Your Company, EPIC STEAKHOUSE 등).
- graphic: 일러스트·아이콘·클립아트·벡터 그래픽·앱 화면 스크린샷·설명용 인포그래픽.
- notreceipt: 영수증이 아니다.

핵심 판별 순서.
1) 적힌 값이 실제 거래값인가, 아니면 빈칸·더미인가. 더미면 synthetic.
2) 실제 값이라면, 원근·굴곡이 있으면 photo, 평평하면 scan.
3) 종이가 구겨졌는데 글자만 완벽히 평평하면 실제 값이어도 synthetic.

정확히 다음 형식으로만 답해라.
label: <photo|scan|synthetic|graphic|notreceipt>
reason: <한 문장>"""

LABELS = ["photo", "scan", "synthetic", "graphic", "notreceipt"]
LABEL_RE = re.compile(r"label\s*:\s*(photo|scan|synthetic|graphic|notreceipt)", re.I)


def prep(path, max_px=1_000_000):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w * h > max_px:
        s = (max_px / (w * h)) ** 0.5
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return im


def build_prompt(proc, text):
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": text}]}]
    try:
        return proc.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=False, enable_thinking=False)
    except TypeError:
        return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def classify(model, proc, img, max_new=64):
    prompt = build_prompt(proc, PROMPT)
    enc = proc(text=[prompt], images=[img], return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    n_in = enc["input_ids"].shape[1]
    out = model.generate(
        **enc, max_new_tokens=max_new,
        do_sample=False,                 # 판정은 확정적이어야 한다
        pad_token_id=proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id)
    txt = proc.tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
    return txt.split("</think>")[-1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="분류할 이미지 폴더")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N장만 (확인용)")
    ap.add_argument("--apply", action="store_true",
                    help="keep 라벨이 아닌 것을 _rejected/ 로 옮긴다 (없으면 판정만)")
    ap.add_argument("--keep", default="photo,scan",
                    help="남길 라벨. 촬영본만 원하면 --keep photo")
    ap.add_argument("--out", default="", help="판정 결과 jsonl (기본: <dir>/../photo_filter.jsonl)")
    args = ap.parse_args()

    img_dir = Path(args.dir)
    files = sorted([p for p in img_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"{img_dir} 에 이미지가 없다.")

    out_path = Path(args.out) if args.out else img_dir.parent / "photo_filter.jsonl"

    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"로드: {Path(args.model).name}", flush=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = True
    model.eval()

    counts = Counter()
    results = []
    print(f"판정 시작: {len(files)}장\n", flush=True)
    for i, p in enumerate(files, 1):
        try:
            raw = classify(model, proc, prep(p))
        except Exception as e:
            counts["error"] += 1
            results.append({"file": p.name, "label": "error", "raw": f"{type(e).__name__}: {e}"})
            continue
        m = LABEL_RE.search(raw)
        # 형식을 안 지키면 본문에서 라벨 단어를 찾는다. 그래도 없으면 unknown.
        label = m.group(1).lower() if m else next(
            (l for l in LABELS if l in raw.lower()), "unknown")
        counts[label] += 1
        results.append({"file": p.name, "label": label, "raw": raw.replace("\n", " ")[:200]})
        if i % 25 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {dict(counts)}", flush=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n판정 분포: {dict(counts.most_common())}")
    print(f"판정 로그: {out_path}")

    keep_set = {s.strip() for s in args.keep.split(",") if s.strip()}
    n_keep = sum(counts[l] for l in keep_set)

    if args.apply:
        rej = img_dir.parent / "_rejected"
        rej.mkdir(exist_ok=True)
        moved = 0
        for r in results:
            if r["label"] in keep_set:
                continue
            src = img_dir / r["file"]
            if src.exists():
                # rm 대신 격리. 분류기가 틀렸을 때 되돌릴 수 있어야 한다.
                shutil.move(str(src), str(rej / r["file"]))
                moved += 1
        print(f"\n격리 {moved}장 -> {rej}  (남긴 라벨: {sorted(keep_set)})")
        print(f"남은 실사본: {len(list(img_dir.iterdir()))}장 ({img_dir})")
    else:
        print(f"\n--apply 없이 판정만 했다. --keep {args.keep} 로 적용하면 "
              f"{len(files) - n_keep}장이 격리되고 {n_keep}장이 남는다.")


if __name__ == "__main__":
    main()
