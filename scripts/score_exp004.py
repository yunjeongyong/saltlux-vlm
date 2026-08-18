"""exp_004 를 포함해 현행 평가셋 978건 정답으로 재채점한다.

score_asis_tobe.py 와 같은 방식이다 — 기록된 eval_crops_*.json 은 정답 정정
(fix_gt.py) 전에 매긴 점수라 그대로 쓰면 안 되고, 예측 텍스트만 가져와 지금
manifest 의 정답으로 다시 잰다.

exp_004 결과 파일이 없으면(=평가 미실행) 그 열은 빠진 채로 나온다.

usage: python3 scripts/score_exp004.py [출력.json]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_doc import avg_score, cer, teds  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "receipt_data"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RD / "review/exp004_compare.json"

DROP_DOCS = {"receipt34", "receipt67"}
SRC = {
    "베이스(luxia)":  ("eval_crops_luxia.json",  "luxia-document-parsing-high"),
    "출발점(맨몸)":   ("eval_crops_base.json",   "base"),
    "exp_003 ck1000": ("eval_crops_exp003.json", "checkpoint-1000"),
    "exp_003 최종":   ("eval_crops_exp003.json", "upload-mix-qwen36"),
    # 재측정본이 있으면 이쪽이 exp_004 와 같은 크롭에서 잰 값이다
    "exp_003 재측정": ("eval_crops_exp004.json", "checkpoint-1000"),
    "exp_004":        ("eval_crops_exp004.json", "exp004-260813"),
}
BASELINE = "exp_003 최종"   # 주간보고 기준선 (TEDS 0.7286)


def score(task, gt, pred):
    if task == "table":
        t = teds(gt, pred)
        return {"teds": t, "cer": None, "avg": avg_score(None, t)}
    c = cer(gt, pred)
    return {"teds": None, "cer": c, "avg": avg_score(c, None)}


def main():
    man = [json.loads(l) for l in (RD / "eval_crops/manifest.jsonl")
           .read_text(encoding="utf-8").split("\n") if l.strip()]
    man = [r for r in man if r["doc_id"] not in DROP_DOCS]
    print(f"평가셋 {len(man)}건 / {len({r['doc_id'] for r in man})}장")

    preds = {}
    for name, (fn, key) in SRC.items():
        p = RD / "review" / fn
        if not p.exists():
            print(f"  {name:14} 건너뜀 — {fn} 없음 (평가 미실행)")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        got = {r["crop_id"]: r[key]["pred"] for r in d["rows"] if r.get(key)}
        if not got:
            print(f"  {name:14} 건너뜀 — {fn} 에 '{key}' 열이 없음")
            continue
        preds[name] = got
        print(f"  {name:14} 예측 {len(got):,}건  ({fn})")

    rows = []
    for r in man:
        rec = {k: r[k] for k in ("crop_id", "doc_id", "split", "task",
                                 "label", "bbox", "size", "image", "gt")}
        for name in preds:
            p = preds[name].get(r["crop_id"])
            if p is None:
                continue
            rec[name] = {"pred": p, **score(r["task"], r["gt"], p)}
        rows.append(rec)

    print("\n현행 정답 기준 재채점 (CER 낮을수록 / TEDS·AVG 높을수록 좋음)")
    summary = {}
    for name in preds:
        g = [r for r in rows if name in r]
        txt = [r for r in g if r["task"] == "text"]
        tbl = [r for r in g if r["task"] == "table"]
        s = {
            "n": len(g),
            "CER": sum(r[name]["cer"] for r in txt) / max(1, len(txt)),
            "TEDS": sum(r[name]["teds"] for r in tbl) / max(1, len(tbl)),
            "긴오답": sum(1 for r in txt if r[name]["cer"] > 1),
            "완전일치%": 100 * sum(1 for r in txt if r[name]["cer"] == 0)
                       / max(1, len(txt)),
            "AVG마이크로": sum(r[name]["avg"] for r in g) / max(1, len(g)),
        }
        # 표 97건이 텍스트 881건에 묻히지 않게 태스크별 평균을 다시 평균낸다
        s["AVG매크로"] = (sum(r[name]["avg"] for r in txt) / max(1, len(txt))
                        + sum(r[name]["avg"] for r in tbl) / max(1, len(tbl))) / 2
        summary[name] = s
        print(f"  {name:14} n={s['n']:<5} CER {s['CER']:.4f}  "
              f"TEDS {s['TEDS']:.4f}  긴오답 {s['긴오답']:>3}건  "
              f"완전일치 {s['완전일치%']:.1f}%  "
              f"AVG 마이크로 {s['AVG마이크로']:.4f} / 매크로 {s['AVG매크로']:.4f}")

    # 판정 — 주간보고 단기목표 1번 (TEDS 0.7286 초과 여부)
    if "exp_004" in summary and BASELINE in summary:
        a, b = summary["exp_004"], summary[BASELINE]
        print(f"\n[판정] exp_004 vs {BASELINE}")
        for k, better in (("TEDS", "up"), ("CER", "down"),
                          ("AVG마이크로", "up"), ("AVG매크로", "up")):
            d = a[k] - b[k]
            ok = (d > 0) if better == "up" else (d < 0)
            print(f"  {k:11} {b[k]:.4f} → {a[k]:.4f}  ({d:+.4f}) "
                  f"{'개선' if ok else '악화'}")
        print(f"  긴오답      {b['긴오답']}건 → {a['긴오답']}건")
    else:
        print("\n[판정] exp_004 결과가 없어 비교 생략 — "
              "bash scripts/run_exp004_eval.sh 먼저 실행할 것")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"n": len(rows), "baseline": BASELINE, "summary": summary, "rows": rows},
        ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
