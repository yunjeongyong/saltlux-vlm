"""
생성 토큰의 log_prob 이 '틀린 곳'을 실제로 짚어내는지 검증한다.

GT 가 없는 실전에서는 CER 을 못 쓴다. 그때 쓸 수 있는 유일한 신호가
모델 자신의 확신도다. 다만 "확신 높은데 틀린다"면 무용지물이므로,
GT 가 있는 곳에서 상관을 먼저 확인해야 쓸 수 있다.

방법: 계약 프롬프트로 표를 생성하면서 토큰별 log_prob 를 받아두고,
표의 상품명 칸마다 그 칸을 이루는 토큰들의 최저 log_prob 를 구한다.
그 값이 '맞은 상품명'과 '틀린 상품명'을 가르면 신뢰도 장치로 쓸 수 있다.

usage:
    CUDA_VISIBLE_DEVICES=2,3 python3 scripts/logprob_conf.py --n 60
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from infer_receipt import RECEIPT_CONTRACT, build_prompt, load_img   # noqa: E402

ROW = re.compile(r"^\s*\|\s*([^|]+?)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*$")


@torch.no_grad()
def gen_with_conf(model, proc, img, text, max_new):
    """생성 결과와 (토큰문자열, logprob) 목록을 함께 돌려준다."""
    enc = proc(text=[build_prompt(proc, text)], images=[img], return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    n_in = enc["input_ids"].shape[1]
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         repetition_penalty=1.05,
                         output_scores=True, return_dict_in_generate=True,
                         pad_token_id=proc.tokenizer.pad_token_id
                         or proc.tokenizer.eos_token_id)
    ids = out.sequences[0][n_in:]
    # 토큰을 하나씩 decode 하면 안 된다. 한글은 UTF-8 3바이트인데 BPE 토큰이
    # 글자 중간을 가르므로 조각마다 U+FFFD 로 깨진다. 누적 디코딩으로 차이를
    # 취하면, 글자를 완성하는 토큰이 그 글자를 통째로 받고 앞 조각은 ""가 된다.
    dec, prev, toks = proc.tokenizer.decode, "", []
    pending = []                      # 아직 글자를 완성 못 한 조각들의 logprob
    for i, tid in enumerate(ids):
        lp = torch.log_softmax(out.scores[i][0].float(), dim=-1)[tid].item()
        # 접두사가 글자 중간에서 끊기면 decode 가 끝에 U+FFFD 를 붙인다.
        # 그 미완성 꼬리를 떼야 다음 단계 diff 의 기준 길이가 어긋나지 않는다.
        cur = dec(ids[:i + 1], skip_special_tokens=True).rstrip("�")
        piece = cur[len(prev):]
        prev = cur
        if not piece:                 # 글자 앞부분 — 다음 조각에 합친다
            pending.append(lp)
            continue
        toks.append((piece, min([lp] + pending)))   # 글자 안의 최저 확신도
        pending = []
    return prev, toks


def cell_conf(toks, text):
    """출력 텍스트의 각 표 행에 대해 (상품명, 그 구간 최저 logprob) 을 낸다."""
    # 토큰을 이어붙이며 문자 오프셋을 만든다
    pos, spans = 0, []
    for t, lp in toks:
        spans.append((pos, pos + len(t), lp))
        pos += len(t)
    out = []
    off = 0
    for line in text.splitlines(keepends=True):
        m = ROW.match(line.rstrip("\n"))
        if m and not set(m.group(1)) <= set(":- "):
            name = m.group(1).strip()
            s = off + line.index(name)
            e = s + len(name)
            lps = [lp for a, b, lp in spans if a < e and b > s]
            if lps:
                out.append((name, min(lps)))
        off += len(line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "ml/models/Qwen3.6-35B-A3B"))
    ap.add_argument("--val", default=str(ROOT / "receipt_data/unified/val_bal.jsonl"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.val, encoding="utf-8") if l.strip()]
    random.seed(args.seed)
    items = [r for r in random.sample(rows, min(args.n, len(rows)))
             if "receipts3000" in r["images"][0]]

    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[1/2] 로드 · 영수증 {len(items)}장", flush=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device_map,
        trust_remote_code=True)
    model.config.use_cache = True
    model.eval()

    ok, bad = [], []
    print("[2/2] 생성 + 확신도 수집\n", flush=True)
    for i, r in enumerate(items, 1):
        img = load_img(r["images"][0], 6_000_000)
        txt, toks = gen_with_conf(model, proc, img, RECEIPT_CONTRACT, args.max_new)
        gt_names = [m.group(1).strip() for l in r["messages"][1]["content"].splitlines()
                    if (m := ROW.match(l)) and not set(m.group(1)) <= set(":- ")]
        got = cell_conf(toks, txt)
        for j, (name, lp) in enumerate(got):
            if j < len(gt_names):
                (ok if name == gt_names[j] else bad).append((name, gt_names[j], lp))
        print(f"  [{i}/{len(items)}] {Path(r['images'][0]).name}  "
              f"품목 {len(got)}  틀림 {sum(1 for j,(n,_) in enumerate(got) if j<len(gt_names) and n!=gt_names[j])}",
              flush=True)

    import statistics as st
    print(f"\n{'':<12}{'n':>5}{'평균 logprob':>15}{'중앙값':>12}{'최소':>12}")
    for tag, arr in (("맞은 상품명", ok), ("틀린 상품명", bad)):
        if not arr:
            continue
        v = [x[2] for x in arr]
        print(f"{tag:<12}{len(v):>5}{st.mean(v):>15.4f}{st.median(v):>12.4f}{min(v):>12.4f}")

    if ok and bad:
        # 임계값을 훑으며 '틀린 것을 몇 % 잡고, 맞은 것을 몇 % 헛되이 잡는가'
        print(f"\n{'임계값':>10}{'틀린것 검출률':>16}{'맞은것 오탐률':>16}")
        for th in (-0.05, -0.1, -0.2, -0.5, -1.0, -2.0):
            tp = sum(1 for x in bad if x[2] < th) / len(bad)
            fp = sum(1 for x in ok if x[2] < th) / len(ok)
            print(f"{th:>10.2f}{tp*100:>15.1f}%{fp*100:>15.1f}%")
        print("\n틀린 상품명 상위 (확신 낮은 순):")
        for name, gt, lp in sorted(bad, key=lambda x: x[2])[:8]:
            print(f"   logprob {lp:7.3f}   출력 {name!r}  /  정답 {gt!r}")
        print("\n틀렸는데 확신은 높았던 것:")
        for name, gt, lp in sorted(bad, key=lambda x: -x[2])[:5]:
            print(f"   logprob {lp:7.3f}   출력 {name!r}  /  정답 {gt!r}")


if __name__ == "__main__":
    main()
