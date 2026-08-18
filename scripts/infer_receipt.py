"""
영수증 이미지 1장 -> 텍스트. 베이스 모델과 학습한 어댑터를 나란히 비교한다.

"학습이 됐다"가 아니라 "정답과 얼마나 같은가"를 본다. 그래서 GT 가 있는
val 샘플을 기본 입력으로 쓰고, 문자단위 오류율(CER)과 숫자열 일치를 함께 찍는다.
영수증은 숫자 하나 틀리면 무의미하므로 숫자만 뽑아 따로 비교한다.

베이스와 어댑터를 각각 로드하면 65.5GiB x 2 라 안 들어간다. 한 번만 올리고
peft 의 disable_adapter() 로 같은 가중치에서 두 출력을 뽑는다.

usage:
    # val 에서 3장 뽑아 베이스 vs 어댑터 비교
    CUDA_VISIBLE_DEVICES=2,3 python3 scripts/infer_receipt.py \
        --adapter ml/runs/receipt-qwen36-bal30 --n 3

    # 임의 이미지 1장 (GT 없음 -> 출력만)
    CUDA_VISIBLE_DEVICES=2,3 python3 scripts/infer_receipt.py \
        --adapter ml/runs/receipt-qwen36-bal30 --image path/to/receipt.jpg
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from bench_receipt import cer, digits          # noqa: E402  동일 채점 기준을 공유

DEFAULT_PROMPT = "이 이미지에 적힌 텍스트를 그대로 옮겨 적어줘."

# 영수증 -> 마크다운 표 계약. 이 문장 하나가 전체 영수증 과제에서
# 정확도 0.0% -> 99.6%, 완전일치 0/27 -> 21/27 을 만들었다 (학습 없음).
# 빠지면 안 되는 것 세 가지 — 각각 실측으로 확인됨:
#   단가(@) 금지      없으면 상품명 칸에 '@1,500' 이 붙는다 (147줄 중 79줄)
#   구분선 형식 지정   없으면 '| :--- |' 로 나온다
#   합계줄 표기 지정   없으면 콜론과 볼드가 빠진다
# 그리고 "결과만 출력" 이 없으면 서론·이모지·후기가 붙어 길이가 2.5배가 된다.
RECEIPT_CONTRACT = (
    "이 영수증을 마크다운으로 정리해줘. 형식은 정확히 다음과 같이 쓴다. "
    "첫 줄: '# 상호명'. 빈 줄. 그 다음 '| 상품명 | 수량 | 금액 |' 헤더와 "
    "'|---|---|---|' 구분선. 그 아래 품목 행들. 빈 줄. "
    "마지막 줄: '공급가액: N  부가세: N  **합계: N원**'. "
    "상품명 칸에는 이름만 쓰고 단가(@1,500 같은 것)는 절대 쓰지 마. "
    "주소·전화번호·사업자등록번호·날짜·시각·결제수단은 쓰지 마. "
    "설명이나 인사말 없이 결과만 출력해."
)


def load_img(rel, max_px, upscale=1.0):
    im = Image.open(rel if Path(rel).is_absolute() else ROOT / rel).convert("RGB")
    # 한글은 자모 조합이라 숫자보다 글자당 픽셀이 더 필요하다. 원본 560px 폭이면
    # 패치 16 x merge 2 기준 한 글자에 토큰 1개도 안 돌아간다. 업스케일은 정보를
    # 늘리지 않지만 비전 토큰 수를 늘려 글자당 해상도를 확보한다.
    if upscale != 1.0:
        im = im.resize((int(im.size[0] * upscale), int(im.size[1] * upscale)),
                       Image.LANCZOS)
    w, h = im.size
    if w * h > max_px:
        s = (max_px / (w * h)) ** 0.5
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return im


def build_prompt(proc, text):
    msgs = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": text.replace("<image>", "")}]}]
    # 학습 때와 동일하게 사고를 끈다. 여기서 어긋나면 프롬프트 분포가 달라져
    # 어댑터가 배운 것과 다른 조건에서 재는 셈이 된다.
    try:
        return proc.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=False, enable_thinking=False)
    except TypeError:
        return proc.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=False)


@torch.no_grad()
def gen(model, proc, img, prompt_text, max_new):
    prompt = build_prompt(proc, prompt_text)
    enc = proc(text=[prompt], images=[img], return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    n_in = enc["input_ids"].shape[1]
    out = model.generate(
        **enc, max_new_tokens=max_new,
        do_sample=False,                 # OCR 은 확정적이어야 한다. 샘플링은 환각을 키운다.
        repetition_penalty=1.05,         # 반복 폭주 억제
        pad_token_id=proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id)
    txt = proc.tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
    return txt.split("</think>")[-1].strip()


NUMTOK = re.compile(r"\d[\d,]*")


def score(gt, hyp):
    c = cer(gt, hyp)
    dg, dh = digits(gt), digits(hyp)
    # 정확도는 0 에서 클램프되므로 출력이 길어질수록 전부 0.0% 로 뭉개진다.
    # 계약이 먹히는지 보려면 클램프 안 되는 지표가 필요하다:
    #   len_ratio  출력이 GT 의 몇 배인가 (계약이 먹히면 1 로 수렴)
    #   digit_rec  GT 가 원하는 숫자를 읽었는가 (능력 지표 — 이미 99% 대)
    #   extra      GT 에 없는 여분 숫자 개수 (수다 지표 — 여기가 줄어야 한다)
    gn = [n.replace(",", "") for n in NUMTOK.findall(gt) if n.replace(",", "")]
    hn = [n.replace(",", "") for n in NUMTOK.findall(hyp) if n.replace(",", "")]
    hit = sum(1 for x in gn if x in hn)
    return {"cer": c, "acc": max(0.0, 1 - c),
            "exact": gt.strip() == hyp.strip(),
            "digit_exact": dg == dh, "gt_digits": dg, "hyp_digits": dh,
            "len_ratio": len(hyp) / len(gt) if gt else 0.0,
            "digit_rec": hit / len(gn) if gn else 1.0,
            "extra": max(0, len(hn) - hit)}


def show(tag, txt, sc):
    print(f"\n  ── {tag} " + "─" * (58 - len(tag)))
    body = txt if txt.strip() else "(빈 출력)"
    for line in body.splitlines() or [""]:
        print("    " + line)
    if sc:
        flag = "OK" if sc["exact"] else ("숫자일치" if sc["digit_exact"] else "불일치")
        print(f"    [{flag}]  문자정확도 {sc['acc']*100:5.1f}%  CER {sc['cer']:.3f}"
              f"  길이배율 {sc['len_ratio']:.2f}x"
              f"  숫자재현 {sc['digit_rec']*100:.0f}%  여분숫자 {sc['extra']}")
        if not sc["digit_exact"]:
            print(f"    숫자  정답 {sc['gt_digits'] or '(없음)'} / 출력 {sc['hyp_digits'] or '(없음)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "ml/models/Qwen3.6-35B-A3B"))
    ap.add_argument("--adapter", default=None, help="없으면 베이스만 실행")
    ap.add_argument("--image", default=None, help="임의 이미지 1장 (GT 없음)")
    ap.add_argument("--val", default=str(ROOT / "receipt_data/unified/val_bal.jsonl"))
    ap.add_argument("--n", type=int, default=3, help="val 에서 뽑을 장수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="--image 일 때만 사용")
    # 데이터의 지시문을 무시하고 이 문장으로 갈아끼운다. 출력 형식 계약을
    # 프롬프트에 명시했을 때 여분 출력이 줄어드는지 보는 실험용.
    ap.add_argument("--override-prompt", default=None)
    ap.add_argument("--contract", action="store_true",
                    help="전체 영수증 과제에 RECEIPT_CONTRACT 를 지시문으로 사용")
    ap.add_argument("--task", choices=["all", "crop", "receipt"], default="all",
                    help="표본 추출 후 과제로 거름 — 같은 seed 면 동일 샘플이 남는다")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--max-px", type=int, default=6_000_000)
    ap.add_argument("--upscale", type=float, default=1.0,
                    help="추론 전 이미지 확대 배율. 한글 글자당 비전 토큰을 늘린다")
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    if args.image:
        items = [{"img": args.image, "prompt": args.prompt, "gt": None}]
    else:
        rows = [json.loads(l) for l in open(args.val, encoding="utf-8") if l.strip()]
        random.seed(args.seed)
        items = [{"img": r["images"][0],
                  "prompt": r["messages"][0]["content"],
                  "gt": r["messages"][1]["content"]}
                 for r in random.sample(rows, min(args.n, len(rows)))]
        if args.task != "all":
            want = "receipts3000" if args.task == "receipt" else "korie"
            items = [x for x in items if want in x["img"]]
        if args.contract:
            for x in items:
                if "receipts3000" in x["img"]:
                    x["prompt"] = RECEIPT_CONTRACT
        if args.override_prompt:
            for x in items:
                x["prompt"] = args.override_prompt

    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[1/3] 로드: {Path(args.model).name}", flush=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device_map,
        trust_remote_code=True)
    model.config.use_cache = True          # 학습 때 껐던 것을 생성용으로 되돌린다
    model.eval()

    has_ad = False
    if args.adapter:
        from peft import PeftModel
        print(f"[2/3] 어댑터: {args.adapter}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        has_ad = True
    else:
        print("[2/3] 어댑터 없음 — 베이스만 실행", flush=True)

    print(f"[3/3] 추론 {len(items)}장\n", flush=True)
    tot = {"base": [], "adapter": []}
    for i, it in enumerate(items, 1):
        img = load_img(it["img"], args.max_px, args.upscale)
        print("=" * 66)
        print(f"[{i}/{len(items)}] {it['img']}  ({img.size[0]}x{img.size[1]})")
        if it["gt"] is not None:
            print(f"\n  ── 정답(GT) " + "─" * 50)
            for line in it["gt"].splitlines() or [""]:
                print("    " + line)

        if has_ad:
            with model.disable_adapter():
                t = gen(model, proc, img, it["prompt"], args.max_new)
            s = score(it["gt"], t) if it["gt"] is not None else None
            show("베이스 (어댑터 끔)", t, s)
            if s: tot["base"].append(s)

        t = gen(model, proc, img, it["prompt"], args.max_new)
        s = score(it["gt"], t) if it["gt"] is not None else None
        show("어댑터 적용" if has_ad else "베이스", t, s)
        if s: tot["adapter" if has_ad else "base"].append(s)
        print()

    for name, rs in tot.items():
        if not rs:
            continue
        n = len(rs)
        print(f"{name:<8} n={n}  평균 문자정확도 {sum(r['acc'] for r in rs)/n*100:5.1f}%"
              f"  완전일치 {sum(r['exact'] for r in rs)}/{n}"
              f"  숫자일치 {sum(r['digit_exact'] for r in rs)}/{n}")


if __name__ == "__main__":
    main()
