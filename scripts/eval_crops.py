"""
테이블/텍스트 크롭을 VLM 에 직접 호출해 CER / TEDS / AVG 를 낸다.

평가 단위는 페이지가 아니라 블록 크롭이다(build_eval_crops.py 산출물).
크롭의 task 에 따라 서비스에서 쓰는 프롬프트를 그대로 붙인다.

  text  -> OCR 프롬프트   -> CER  (낮을수록 좋음)
  table -> TABLE 프롬프트 -> TEDS (높을수록 좋음)

여러 모델을 한 번에 비교한다. 베이스는 한 번만 올리고 어댑터를 갈아끼우므로
35B 를 여러 벌 올리지 않는다.

usage:
    # 베이스만
    python3 scripts/eval_crops.py --model ml/models/Qwen3.6-35B-A3B
    # 베이스 + 학습중 체크포인트 비교
    python3 scripts/eval_crops.py --model ml/models/Qwen3.6-35B-A3B \
        --adapter ml/runs/receipt-v3-qwen36/checkpoint-17000
    # 빠른 확인
    python3 scripts/eval_crops.py --model ... --limit 20
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_doc import avg_score, cer, teds          # noqa: E402
from prompts_receipt import PROMPT_VERSION, PROMPTS   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_rows(manifest, tasks, limit, docs=None):
    rows = [json.loads(l) for l in Path(manifest).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["task"] in tasks]
    if docs:
        rows = [r for r in rows if r["doc_id"] in docs]
    if limit:
        # 앞에서 자르면 한 영수증에 몰린다. task 별로 고르게 자른다.
        out = []
        for t in tasks:
            out += [r for r in rows if r["task"] == t][:limit]
        rows = out
    return rows


def gen(model, proc, img, prompt, max_new, max_px):
    im = Image.open(img).convert("RGB")
    w, h = im.size
    if w * h > max_px:
        s = (max_px / (w * h)) ** 0.5
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": prompt}]}]
    # 학습 때 사고모드를 끄고 태웠다. 평가에서 켜면 조건이 달라져 비교가 안 된다.
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


def score(task, gt, pred):
    if task == "table":
        t = teds(gt, pred)
        return {"teds": t, "cer": None, "avg": avg_score(None, t)}
    c = cer(gt, pred)
    return {"teds": None, "cer": c, "avg": avg_score(c, None)}


def summarize(rows, key):
    """task 별 평균과 전체 AVG. 표/텍스트 건수가 9배 차이라 매크로 평균도 같이 낸다."""
    out = {}
    for task in ("text", "table"):
        g = [r for r in rows if r["task"] == task and r[key].get("avg") is not None]
        if not g:
            continue
        d = {"n": len(g)}
        if task == "text":
            d["CER"] = sum(r[key]["cer"] for r in g) / len(g)
        else:
            d["TEDS"] = sum(r[key]["teds"] for r in g) / len(g)
        d["정확도"] = sum(r[key]["avg"] for r in g) / len(g)
        out[task] = d
    if "text" in out and "table" in out:
        # 요청 지표 AVG(CER, TEDS). 두 과제 점수의 단순 평균 = 매크로.
        out["AVG"] = {"매크로": (out["text"]["정확도"] + out["table"]["정확도"]) / 2,
                      "마이크로": sum(r[key]["avg"] for r in rows
                                   if r[key].get("avg") is not None)
                      / max(1, len([r for r in rows if r[key].get("avg") is not None]))}
    return out


def show(name, s):
    print(f"\n  [{name}]")
    for task in ("text", "table"):
        if task not in s:
            continue
        d = s[task]
        m = f"CER {d['CER']:.4f}" if "CER" in d else f"TEDS {d['TEDS']:.4f}"
        print(f"    {task:<6} n={d['n']:<5} {m:<14} 정확도 {d['정확도']:.4f}")
    if "AVG" in s:
        print(f"    AVG(CER,TEDS)  매크로 {s['AVG']['매크로']:.4f}  "
              f"마이크로 {s['AVG']['마이크로']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="베이스 모델 경로")
    ap.add_argument("--adapter", nargs="*", default=[],
                    help="비교할 LoRA 어댑터/체크포인트 (여러 개 가능)")
    ap.add_argument("--manifest",
                    default=str(ROOT / "receipt_data/eval_crops/manifest.jsonl"))
    ap.add_argument("--tasks", nargs="+", default=["text", "table"])
    ap.add_argument("--limit", type=int, default=0, help="task 별 상한 (0=전량)")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--max-px", type=int, default=4_000_000)
    ap.add_argument("--device-map", default="auto")
    # 베이스를 이미 같은 조건(같은 manifest·프롬프트 버전)으로 잰 적이 있으면
    # 2시간짜리 재측정을 건너뛴다. 조건이 하나라도 다르면 켜지 말 것.
    ap.add_argument("--skip-base", action="store_true",
                    help="어댑터만 재고 베이스 측정은 건너뛴다")
    ap.add_argument("--out", default=str(ROOT / "receipt_data/review/eval_crops.json"))
    args = ap.parse_args()

    rows = load_rows(args.manifest, args.tasks, args.limit)
    base_dir = Path(args.manifest).parent
    n_doc = len({r["doc_id"] for r in rows})
    print(f"평가 대상 {len(rows):,}건 (영수증 {n_doc}장) "
          f"— text {sum(1 for r in rows if r['task']=='text')} / "
          f"table {sum(1 for r in rows if r['task']=='table')}")
    print(f"프롬프트 버전 {PROMPT_VERSION}")

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print(f"베이스 로드: {args.model}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device_map,
        trust_remote_code=True)
    model.eval()

    variants = ([] if args.skip_base else [("base", None)]) \
        + [(Path(a).name, a) for a in args.adapter]
    if not variants:
        raise SystemExit("잴 대상이 없다 — --skip-base 를 켰으면 --adapter 를 줘야 한다")
    for name, adapter in variants:
        if adapter:
            from peft import PeftModel
            print(f"\n어댑터 적용: {adapter}", flush=True)
            model = PeftModel.from_pretrained(model, adapter)
            model.eval()

        print(f"추론 시작 [{name}]", flush=True)
        t0 = time.time()
        for i, r in enumerate(rows, 1):
            try:
                pred, dt = gen(model, proc, base_dir / r["image"],
                               PROMPTS[r["task"]], args.max_new, args.max_px)
            except Exception as e:                    # 한 건 실패로 전체를 버리지 않는다
                pred, dt = "", 0.0
                r.setdefault("errors", {})[name] = f"{type(e).__name__}: {e}"[:200]
            r[name] = {"pred": pred, "sec": round(dt, 1), **score(r["task"], r["gt"], pred)}
            if i % 25 == 0 or i == len(rows):
                el = time.time() - t0
                print(f"    {i}/{len(rows)}  {el/60:.1f}분 경과 "
                      f"(잔여 {(el/i*(len(rows)-i))/60:.1f}분)", flush=True)

        if adapter:                                   # 다음 비교를 위해 어댑터를 떼어낸다
            model = model.unload()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result = {"prompt_version": PROMPT_VERSION, "model": args.model,
              "adapters": args.adapter, "n": len(rows),
              "summary": {n: summarize(rows, n) for n, _ in variants},
              "rows": rows}
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1),
                              encoding="utf-8")

    print("\n" + "=" * 62)
    print("결과 (CER 낮을수록 좋음 / TEDS·정확도·AVG 높을수록 좋음)")
    for name, _ in variants:
        show(name, result["summary"][name])
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
