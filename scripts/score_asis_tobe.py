"""현행 평가셋 978건 정답으로 네 모델을 다시 채점한다.

기록된 review/eval_crops_*.json 은 정답 정정(fix_gt.py) 전에 매긴 점수라 그대로
쓰면 안 된다. 예측 텍스트만 가져와 지금 manifest 의 정답으로 다시 잰다.

  luxia-document-parsing-high   베이스 (서빙)      = AS-IS
  base                          출발점 (맨몸)
  checkpoint-1000 / upload-mix-qwen36   학습 모델  = TO-BE 후보

usage: python3 scripts/score_asis_tobe.py [출력.json]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_doc import avg_score, cer, teds  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "receipt_data"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RD / "review/asis_tobe.json"

DROP_DOCS = {"receipt34", "receipt67"}
SRC = {
    "베이스(luxia)": ("eval_crops_luxia.json", "luxia-document-parsing-high"),
    "출발점(맨몸)": ("eval_crops_base.json", "base"),
    "학습 ckpt1000": ("eval_crops_exp003.json", "checkpoint-1000"),
    "학습 최종": ("eval_crops_exp003.json", "upload-mix-qwen36"),
}


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
        d = json.loads((RD / "review" / fn).read_text(encoding="utf-8"))
        preds[name] = {r["crop_id"]: r[key]["pred"] for r in d["rows"]
                       if r.get(key)}
        print(f"  {name:14} 예측 {len(preds[name]):,}건  ({fn})")

    rows = []
    for r in man:
        rec = {k: r[k] for k in ("crop_id", "doc_id", "split", "task",
                                 "label", "bbox", "size", "image", "gt")}
        for name in SRC:
            p = preds[name].get(r["crop_id"])
            if p is None:
                continue
            rec[name] = {"pred": p, **score(r["task"], r["gt"], p)}
        rows.append(rec)

    print("\n현행 정답 기준 재채점 (CER 낮을수록 / TEDS·AVG 높을수록 좋음)")
    summary = {}
    for name in SRC:
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
        # 매크로는 태스크별 평균을 다시 평균낸다. 표 97건이 텍스트 881건에
        # 묻히지 않게 하려는 것 — 표 성능을 볼 때는 이쪽을 봐야 한다.
        s["AVG매크로"] = (sum(r[name]["avg"] for r in txt) / max(1, len(txt))
                        + sum(r[name]["avg"] for r in tbl) / max(1, len(tbl))) / 2
        summary[name] = s
        print(f"  {name:14} n={s['n']:<5} CER {s['CER']:.4f}  "
              f"TEDS {s['TEDS']:.4f}  긴오답 {s['긴오답']:>3}건  "
              f"완전일치 {s['완전일치%']:.1f}%  "
              f"AVG 마이크로 {s['AVG마이크로']:.4f} / 매크로 {s['AVG매크로']:.4f}")

    best = max((k for k in SRC if k.startswith("학습")),
               key=lambda k: summary[k]["AVG마이크로"])
    print(f"\nTO-BE 기준 모델: {best} (AVG 마이크로 {summary[best]['AVG마이크로']:.4f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"n": len(rows), "summary": summary, "best": best, "rows": rows},
        ensure_ascii=False), encoding="utf-8")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
