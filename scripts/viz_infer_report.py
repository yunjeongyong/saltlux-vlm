"""추론 결과를 이미지·정답·예측 세 벌로 늘어놓은 HTML 리포트를 만든다.

CER 숫자만 보면 '나빠졌다'까지만 알 수 있다. 무엇이 나빠졌는지는 출력을 봐야 한다.
그래서 장마다 원본 이미지 · 정답 · 베이스 출력 · LoRA 출력을 나란히 붙인다.

이미지는 폭 460px JPEG 로 줄여 data URI 로 박는다 (외부 파일 참조가 막힌 환경 대비).

usage:
    python3 scripts/viz_infer_report.py
    python3 scripts/viz_infer_report.py --eval receipt_data/review/upload_run_eval_holdout.json
"""
import argparse
import base64
import glob
import html
import io
import json
import os
import re
from pathlib import Path

from PIL import Image

ROOT = Path("/data/workspace/yjyong")
RUN = ROOT / "ml/runs/receipt-v3-qwen36"
TASK_KO = {"receipt_markdown": "영수증 → 마크다운", "page_ocr": "공공문서 → 평문 전사"}


def thumb(rel, width=460, quality=72):
    """이미지를 줄여 data URI 로. 원본이 없으면 None."""
    p = ROOT / rel
    if not p.exists():
        return None
    im = Image.open(p).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def train_curve():
    """가장 최근 체크포인트의 loss 이력. 100스텝 평균으로 솎아낸다."""
    cks = sorted(glob.glob(str(RUN / "checkpoint-*")), key=os.path.getmtime)
    if not cks:
        return None
    s = json.loads(Path(cks[-1], "trainer_state.json").read_text())
    tr, ev, buck = [], [], {}
    for e in s["log_history"]:
        if "loss" in e:
            buck.setdefault(e["step"] // 100, []).append((e["step"], e["loss"]))
        if "eval_loss" in e:
            ev.append({"step": e["step"], "loss": round(e["eval_loss"], 4)})
    for k in sorted(buck):
        v = buck[k]
        tr.append({"step": v[-1][0], "loss": round(sum(x[1] for x in v) / len(v), 4)})
    return {"train": tr, "eval": ev, "step": s["global_step"],
            "max_steps": s["max_steps"], "epoch": round(s["epoch"], 3),
            "ckpt": Path(cks[-1]).name}


def live_step():
    """학습 로그 끝에서 현재 스텝과 남은 시간을 긁는다. tqdm 진행바 형식."""
    log = ROOT / "ml/train_v3_qwen36.log"
    if not log.exists():
        return None
    with open(log, "rb") as f:
        f.seek(max(0, log.stat().st_size - 4000))
        tail = f.read().decode("utf-8", "ignore").replace("\r", "\n")
    m = None
    for m in re.finditer(r"(\d+)/(\d+) \[([\d:]+)<([\d:]+),\s*([\d.]+)s/it\]", tail):
        pass
    if not m:
        return None
    return {"step": int(m.group(1)), "total": int(m.group(2)),
            "elapsed": m.group(3), "eta": m.group(4), "sec_it": float(m.group(5))}


def summarize(rows):
    out = {}
    for t in ("receipt_markdown", "page_ocr"):
        g = [r for r in rows if r["task"] == t]
        if not g:
            continue
        out[t] = {
            "n": len(g),
            "base": round(sum(r["cer_base"] for r in g) / len(g), 4),
            "lora": round(sum(r["cer_lora"] for r in g) / len(g), 4),
            "win": sum(1 for r in g if r["cer_lora"] < r["cer_base"]),
        }
    return out


def esc(s, limit=2400):
    s = str(s or "")
    cut = len(s) > limit
    return html.escape(s[:limit]) + ("\n…(이하 생략)" if cut else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="receipt_data/review/upload_run_eval_holdout.json",
                    help="본문에 펼칠 추론 결과 JSON")
    ap.add_argument("--rounds", default="receipt_data/review/upload_run_eval.json:v1 · 300스텝,"
                                        "receipt_data/review/upload_run_eval_ckpt1000.json:v2 · ckpt-1000,"
                                        "receipt_data/review/upload_run_eval_v2.json:v2 · 최종 2000스텝",
                    help="요약 비교에 쓸 라운드들 path:label,… (label 에 쉼표 금지)")
    ap.add_argument("--label", default=None, help="본문 결과의 라운드 이름")
    ap.add_argument("--headline", default=None, help="페이지 제목 (기본: 결과에서 자동 생성)")
    ap.add_argument("--subhead", default=None)
    ap.add_argument("--run-name", default=None, help="헤더/푸터에 쓸 학습 이름")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--out", default="/tmp/infer_report.html")
    a = ap.parse_args()

    main_rows = json.loads((ROOT / a.eval).read_text(encoding="utf-8"))
    rounds = []
    for spec in a.rounds.split(","):
        p, _, lab = spec.rpartition(":")
        f = ROOT / p.strip()
        if f.exists():
            rows = json.loads(f.read_text(encoding="utf-8"))
            rounds.append({"label": lab.strip(), "n": len(rows), "sum": summarize(rows)})
    rounds.append({"label": a.label or f"홀드아웃 · {len(main_rows)}건", "n": len(main_rows),
                   "sum": summarize(main_rows), "hi": True})

    docs = []
    for r in sorted(main_rows, key=lambda x: (x["task"], -x["cer_lora"])):
        docs.append({
            "id": r["doc_id"], "task": r["task"],
            "cer_base": r["cer_base"], "cer_lora": r["cer_lora"],
            "sec_base": r.get("base_sec"), "sec_lora": r.get("lora_sec"),
            "img": None if a.no_images else thumb(r["image"]),
            "gt": esc(r["gt"]), "base": esc(r["base"]), "lora": esc(r["lora"]),
        })

    S = summarize(main_rows)
    if a.headline:
        headline = a.headline
    else:
        wins = [t for t, m in S.items() if m["lora"] < m["base"]]
        if len(wins) == len(S):
            headline = "LoRA가 두 과제 모두에서 베이스보다 좋아졌습니다"
        elif wins:
            headline = f"{TASK_KO[wins[0]].split(' →')[0]} 쪽만 좋아졌습니다"
        else:
            headline = "학습은 loss를 내렸지만, 모델은 이미지를 덜 읽게 됐습니다"

    payload = {
        "curve": train_curve(), "live": live_step(),
        "rounds": rounds, "docs": docs, "summary": S,
        "task_ko": TASK_KO, "eval_src": a.eval,
        "headline": headline, "subhead": a.subhead,
        "run": a.run_name or Path(a.eval).stem,
    }
    tpl = Path(__file__).with_name("infer_report_template.html").read_text(encoding="utf-8")
    out = Path(a.out)
    out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False)),
                   encoding="utf-8")
    print(f"저장: {out}  ({out.stat().st_size/1e6:.1f} MB · 문서 {len(docs)}건)")


if __name__ == "__main__":
    main()
