"""
영수증 1장에 난이도가 다른 지시를 여러 개 던져 '지시를 읽는가'를 본다.

"영수증을 아는가"와 "지시를 따르는가"는 다른 문제다. 베이스는 상호명·공급가액·
부가세를 정확히 라벨링하므로 영수증은 안다. 그런데 GT 가 원하지 않는 것까지 쓴다.
그래서 난이도가 다른 지시를 던져 어디서 무너지는지 본다:

  Q1 자유서술   — 형식 제약 없음. 통제군.
  Q2 값 하나    — "합계만 숫자로". 가장 좁은 제약. 이것도 못 지키면 지시를 못 읽는 것.
  Q3 표만       — GT 와 같은 형식. 실제로 원하는 것.
  Q4 표+금지    — Q3 에 금지 목록 추가. 명시가 도움이 되는지.

Q2/Q3 는 GT 에서 정답을 뽑아낼 수 있어 눈으로 보는 대신 채점한다.

usage:
    CUDA_VISIBLE_DEVICES=2,3 python3 scripts/probe_prompts.py \
        --adapter ml/runs/receipt-qwen36-bal30 --n 3
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
from infer_receipt import build_prompt, gen, load_img          # noqa: E402

PROBES = [
    ("Q1 자유서술", "이 이미지가 뭔지 설명해줘.", None),
    ("Q2 값 하나", "이 영수증의 합계 금액만 숫자로 답해줘.", "total"),
    ("Q3 표만", "이 영수증에 있는 항목을 상품명·수량·금액 3열 마크다운 표로만 "
                "만들어줘. 다른 정보는 쓰지 마.", "table"),
    ("Q4 표+금지", "이 영수증에 있는 항목을 상품명·수량·금액 3열 마크다운 표로만 "
                   "만들어줘. 상호명·주소·전화번호·사업자등록번호·날짜·시각·"
                   "결제수단·공급가액·부가세·합계는 쓰지 마. 표 외에는 아무것도 "
                   "쓰지 마.", "table"),
]

TOTAL = re.compile(r"\*\*합계:\s*([\d,]+)\s*원\*\*")
NUM = re.compile(r"\d[\d,]*")


def expected(kind, gt):
    """GT 에서 그 지시의 정답을 뽑는다."""
    if kind == "total":
        m = TOTAL.search(gt)
        return m.group(1).replace(",", "") if m else None
    if kind == "table":
        return "\n".join(l for l in gt.splitlines() if l.strip().startswith("|"))
    return None


def judge(kind, exp, out):
    if kind == "total":
        got = [n.replace(",", "") for n in NUM.findall(out)]
        if not got:
            return "숫자 없음"
        ok = got[0] == exp
        clean = len(got) == 1
        return ("정답" if ok else f"오답(기대 {exp})") + ("" if clean else f" · 숫자 {len(got)}개 출력")
    if kind == "table":
        rows_e = [l for l in exp.splitlines() if l.strip()]
        rows_o = [l for l in out.splitlines() if l.strip().startswith("|")]
        extra = [l for l in out.splitlines() if l.strip() and not l.strip().startswith("|")]
        return (f"표 {len(rows_o)}줄/기대 {len(rows_e)}줄 · 표 밖 {len(extra)}줄"
                + (" · 표만 출력" if not extra else ""))
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "ml/models/Qwen3.6-35B-A3B"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--val", default=str(ROOT / "receipt_data/unified/test.jsonl"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--image", default=None,
                    help="특정 이미지 파일명 (예: receipt_1249.jpg). GT 가 있어야 채점된다")
    # 임의 질문(VQA). GT 로 자동 채점이 안 되므로 출력만 보여준다.
    ap.add_argument("--ask", action="append", default=[],
                    help="자유 질문. 여러 번 줄 수 있다. 예: --ask '츄러스는 얼마야?'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--max-px", type=int, default=6_000_000)
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.val, encoding="utf-8") if l.strip()]
    if args.image:
        items = [r for r in rows if args.image in r["images"][0]]
        if not items:
            sys.exit(f"{args.image} 를 {args.val} 에서 못 찾음 — GT 가 있어야 채점된다")
    else:
        rows = [r for r in rows if "receipts3000" in r["images"][0]]
        random.seed(args.seed)
        items = random.sample(rows, min(args.n, len(rows)))

    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[1/2] 로드: {Path(args.model).name}", flush=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device_map,
        trust_remote_code=True)
    model.config.use_cache = True
    model.eval()
    arms = [("베이스", None)]
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        arms.append(("어댑터", True))

    probes = PROBES + [(f"Q{5+i} 자유질문", q, None)
                       for i, q in enumerate(args.ask)]
    print(f"[2/2] 프로브 {len(items)}장 x {len(probes)}지시 x {len(arms)}arm\n", flush=True)
    for i, r in enumerate(items, 1):
        img = load_img(r["images"][0], args.max_px)
        gt = r["messages"][1]["content"]
        print("=" * 70)
        print(f"[{i}/{len(items)}] {r['images'][0]}  ({img.size[0]}x{img.size[1]})")
        print(f"  GT {len(gt)}자 / {gt.count(chr(10))+1}줄   합계 {expected('total', gt)}")
        for name, text, kind in probes:
            exp = expected(kind, gt) if kind else None
            print(f"\n  ┌─ {name}  «{text[:44]}{'…' if len(text) > 44 else ''}»")
            for arm, use_ad in arms:
                if use_ad:
                    out = gen(model, proc, img, text, args.max_new)
                else:
                    ctx = model.disable_adapter() if args.adapter else None
                    if ctx:
                        with ctx:
                            out = gen(model, proc, img, text, args.max_new)
                    else:
                        out = gen(model, proc, img, text, args.max_new)
                head = out if len(out) <= 400 else out[:400] + f" …(총 {len(out)}자)"
                print(f"  │ [{arm}] {len(out)}자" +
                      (f"  → {judge(kind, exp, out)}" if kind else ""))
                for line in head.splitlines():
                    print("  │   " + line)
            print("  └" + "─" * 66)
        print()


if __name__ == "__main__":
    main()
