"""여러 체크포인트를 같은 표본에 돌려 '몇 스텝이 최적인가'를 잰다.

체크포인트마다 eval_upload_run.py 를 따로 돌리면 베이스 추론을 매번 다시 하고
모델도 매번 새로 올린다. 여기서는 베이스를 한 번만 올려 한 번만 추론하고,
어댑터만 갈아 끼운다.

출력은 eval_upload_run.py 와 같은 형식이라 viz_infer_report.py 가 그대로 읽는다.

usage:
    python3 scripts/eval_ckpt_sweep.py --run ml/runs/upload-v3 --ckpt 250,500,1000,final
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_upload_run import cer, gen  # noqa: E402

ROOT = Path("/data/workspace/yjyong")


def resolve(run, tag):
    """'250' → run/checkpoint-250, 'final' → run 자체(마지막 저장 어댑터)."""
    p = Path(run) if tag == "final" else Path(run) / f"checkpoint-{tag}"
    return p if (p / "adapter_model.safetensors").exists() else None


def training_alive(run):
    """해당 run 을 쓰는 train_vlm.py 가 아직 살아 있나."""
    try:
        out = subprocess.run(["pgrep", "-af", "train_vlm.py"],
                             capture_output=True, text=True).stdout
    except OSError:
        return False
    return any(str(run) in line for line in out.splitlines())


def watch_loop(a, run_one):
    """새 체크포인트가 생기는 대로 순서대로 잰다. 학습이 끝나면 final 까지 재고 종료."""
    done, i = set(), 0
    while True:
        found = []
        for d in Path(a.run).glob("checkpoint-*"):
            m = re.fullmatch(r"checkpoint-(\d+)", d.name)
            if not m or not (d / "adapter_model.safetensors").exists():
                continue
            step = int(m.group(1))
            if step not in done and (not a.until or step <= a.until):
                found.append((step, d))
        for step, d in sorted(found):
            i += 1
            run_one(str(step), d, i, "?")
            done.add(step)

        # --until 을 채웠으면 학습이 계속 돌든 말든 끝낸다
        reached = a.until and done and max(done) >= a.until
        if not reached and training_alive(a.run):
            time.sleep(a.poll)
            continue

        # 학습 종료(또는 목표 도달) — 루트에 최종 어댑터가 있으면 그것까지 재고 끝낸다
        fin = resolve(a.run, "final")
        if fin is not None and "final" not in done:
            i += 1
            run_one("final", fin, i, "?")
            done.add("final")
        print(f"\nwatch 종료 — 잰 체크포인트 {len(done)}개")
        return


def rebuild_report(a, tag, ckpt, evjson):
    """라운드마다 HTML 리포트를 다시 만든다. 실패해도 평가는 계속한다."""
    cmd = [sys.executable, str(Path(__file__).with_name("viz_infer_report.py")),
           "--eval", str(Path(evjson).relative_to(ROOT)),
           "--label", f"{tag} · {ckpt}스텝",
           "--run-name", f"{tag} (ckpt {ckpt})",
           "--rounds", "receipt_data/review/upload_run_eval_holdout.json:v2 · 8x오버샘플 2000스텝",
           "--out", a.report]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(f"  리포트 {'갱신' if r.returncode == 0 else '실패'} {a.report}"
              + ("" if r.returncode == 0 else f"\n{r.stderr[-400:]}"), flush=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"  리포트 실패: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "ml/models/Qwen3.5-4B"))
    ap.add_argument("--run", default=str(ROOT / "ml/runs/upload-v3"))
    ap.add_argument("--ckpt", default="250,500,1000,final")
    ap.add_argument("--val", default=str(ROOT / "receipt_data/review/holdout_eval.jsonl"))
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--outdir", default=str(ROOT / "receipt_data/review"))
    ap.add_argument("--tag", default=None, help="출력 파일 접두사 (기본: run 폴더 이름)")
    ap.add_argument("--watch", action="store_true",
                    help="학습이 도는 동안 새 체크포인트가 생길 때마다 이어서 잰다")
    ap.add_argument("--until", type=int, default=0,
                    help="watch: 이 스텝까지 재면 종료 (0=학습 프로세스가 끝날 때까지)")
    ap.add_argument("--poll", type=int, default=60, help="watch: 확인 주기(초)")
    ap.add_argument("--report", default=None,
                    help="라운드마다 이 경로로 HTML 리포트를 다시 만든다")
    a = ap.parse_args()

    tag = a.tag or Path(a.run).name
    rows = [json.loads(l) for l in Path(a.val).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"표본 {len(rows)}건 · {a.val}")

    want = [t.strip() for t in a.ckpt.split(",") if t.strip()]
    if a.watch:
        print(f"  watch 모드 — {a.poll}초마다 새 체크포인트 확인"
              + (f" (스텝 {a.until} 까지)" if a.until else " (학습이 끝날 때까지)"))
        cks = []
    else:
        cks = [(t, resolve(a.run, t)) for t in want]
        missing = [t for t, p in cks if p is None]
        cks = [(t, p) for t, p in cks if p is not None]
        if missing:
            print(f"  ⚠ 어댑터 없음 — 건너뜀: {missing}")
        if not cks:
            print("평가할 체크포인트가 없다.")
            return
        print(f"  체크포인트 {[t for t, _ in cks]}")

    proc = AutoProcessor.from_pretrained(a.model)
    print("베이스 모델 로드…")
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    # 베이스는 체크포인트와 무관하므로 한 번만 돌린다
    base = []
    for r in rows:
        prompt = r["messages"][0]["content"].replace("<image>", "")
        txt, dt = gen(model, proc, r["images"][0], prompt, a.max_new)
        base.append({"doc_id": r["doc_id"], "task": r["task"], "image": r["images"][0],
                     "prompt": prompt, "gt": r["messages"][1]["content"],
                     "base": txt, "base_sec": round(dt, 1),
                     "cer_base": round(cer(r["messages"][1]["content"], txt), 4)})
        print(f"  [base] {r['doc_id'][:30]:32} {dt:5.1f}s  CER {base[-1]['cer_base']:.3f}", flush=True)

    from peft import PeftModel
    summary = []
    state = {"peft": None}
    sp = Path(a.outdir) / f"{tag}_ckpt_sweep.json"

    def run_one(t, p, i, n):
        name = f"ck{t}"
        if state["peft"] is None:
            state["peft"] = PeftModel.from_pretrained(model, str(p), adapter_name=name)
        else:
            state["peft"].load_adapter(str(p), adapter_name=name)
        pm = state["peft"]
        pm.set_adapter(name)
        pm.eval()
        print(f"\n[{i}/{n}] {t} — {p}", flush=True)

        res = []
        for b in base:
            txt, dt = gen(pm, proc, b["image"], b["prompt"], a.max_new)
            row = dict(b)
            row["lora"] = txt
            row["lora_sec"] = round(dt, 1)
            row["cer_lora"] = round(cer(b["gt"], txt), 4)
            res.append(row)
            print(f"  [{t}] {b['doc_id'][:30]:32} {dt:5.1f}s  CER {row['cer_lora']:.3f}", flush=True)

        out = Path(a.outdir) / f"{tag}_eval_ck{t}.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

        line = {"ckpt": t, "path": str(p)}
        for task in ("receipt_markdown", "page_ocr"):
            g = [r for r in res if r["task"] == task]
            if not g:
                continue
            line[task] = {
                "n": len(g),
                "base": round(sum(r["cer_base"] for r in g) / len(g), 4),
                "lora": round(sum(r["cer_lora"] for r in g) / len(g), 4),
                "win": sum(1 for r in g if r["cer_lora"] < r["cer_base"]),
            }
        summary.append(line)
        # 라운드마다 즉시 저장한다 — 중간에 끊겨도 여기까지는 남는다
        sp.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  저장 {out.name}")
        if a.report:
            rebuild_report(a, tag, t, out)

    if a.watch:
        watch_loop(a, run_one)
    else:
        for i, (t, p) in enumerate(cks):
            run_one(t, p, i + 1, len(cks))

    print("\n체크포인트별 CER (낮을수록 좋음)")
    print(f"  {'ckpt':>6}  {'영수증 base→lora':>24}  {'문서 base→lora':>24}")
    for s in summary:
        def cell(k):
            m = s.get(k)
            if not m:
                return " " * 24
            mark = "개선" if m["lora"] < m["base"] else "악화"
            return f"{m['base']:.3f} → {m['lora']:.3f} {mark} {m['win']}/{m['n']:<2}"
        print(f"  {s['ckpt']:>6}  {cell('receipt_markdown'):>24}  {cell('page_ocr'):>24}")
    print(f"\n저장: {sp}")


if __name__ == "__main__":
    main()
