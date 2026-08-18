"""실험로그 행에 넣을 지표를 계산한다. transpose 2장(receipt34/67) 제외 지원.

fill_exp_log.py 와 같은 정의를 쓴다. 다른 것은 크롭 부분집합을 고를 수 있다는 점뿐이다.
receipt34/67 은 이미지와 라벨 좌표계가 90° 어긋나 크롭 41건이 엉뚱한 영역을 잘랐다.
CER 은 위로 열린 값이라 이 41건이 평균을 지배한다 — 제외하고 재계산한다.

    정확도    = 크롭별 max(0, 1-CER) 의 평균 (CER 은 1 을 넘으므로 clip 한다)
    AVG_매크로 = (텍스트 정확도 + 표 TEDS) / 2
    AVG_마이크로 = 크롭 단위 가중평균 (text 933 : table 103)
    완전일치%  = CER == 0 인 텍스트 크롭 비율

usage:
    python3 scripts/exp_log_rows.py                    # 2장 제외 (기본)
    python3 scripts/exp_log_rows.py --keep-transposed  # 제외 없이 (기존 보고값 재현)
"""
import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "receipt_data/eval_crops/manifest.jsonl"
TRANSPOSED = ("receipt34", "receipt67")

RUNS = [
    ("exp_001", "LUXIA", "receipt_data/review/eval_crops_luxia.json",
     "luxia-document-parsing-high"),
    ("exp_002", "베이스 Qwen3.6-35B-A3B", "receipt_data/review/eval_crops_base.json",
     "base"),
    ("exp_003", "학습 ckpt-1000", "receipt_data/review/eval_crops_exp003.json", None),
]


def dropped_ids():
    with open(MANIFEST, encoding="utf-8") as f:
        return {json.loads(l)["crop_id"] for l in f if l.strip()
                and json.loads(l)["doc_id"] in TRANSPOSED}


def metrics(path, key, drop):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    key = key or list(d["summary"])[-1]
    rows = [r for r in d["rows"] if r["crop_id"] not in drop]
    tx = [r for r in rows if r["task"] == "text" and r[key].get("cer") is not None]
    tb = [r for r in rows if r["task"] == "table" and r[key].get("teds") is not None]

    cs = [r[key]["cer"] for r in tx]
    acc = st.mean(max(0.0, 1 - c) for c in cs)
    teds = st.mean(r[key]["teds"] for r in tb)
    return {
        "평가건수": len(rows),
        "text_n": len(tx), "table_n": len(tb),
        "CER_중앙값": round(st.median(cs), 4),
        "CER_평균": round(st.mean(cs), 4),
        "CER>1": sum(1 for c in cs if c > 1),
        "TEDS": round(teds, 4),
        "완전일치%": round(sum(1 for c in cs if c == 0) / len(cs) * 100, 1),
        "AVG_매크로": round((acc + teds) / 2, 4),
        "AVG_마이크로": round((acc * len(tx) + teds * len(tb)) / (len(tx) + len(tb)), 4),
        "_정확도": round(acc, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-transposed", action="store_true")
    args = ap.parse_args()

    drop = set() if args.keep_transposed else dropped_ids()
    print(f"제외 크롭 {len(drop)}건 "
          f"({'없음' if not drop else '/'.join(TRANSPOSED) + ' 소속 전량'})\n")

    base_macro = None
    for exp, label, path, key in RUNS:
        p = ROOT / path
        if not p.exists():
            print(f"{exp}  {label:<22} — 결과 JSON 없음 ({path})")
            continue
        m = metrics(p, key, drop)
        if exp == "exp_002":
            base_macro = m["AVG_매크로"]
        delta = ("" if base_macro is None or exp == "exp_002"
                 else f"  base대비Δ {m['AVG_매크로'] - base_macro:+.4f}")
        print(f"{exp}  {label}")
        print(f"   평가 {m['평가건수']}건 (text {m['text_n']} / table {m['table_n']})")
        print(f"   CER 중앙값 {m['CER_중앙값']}  평균 {m['CER_평균']}  "
              f"CER>1 {m['CER>1']}건  완전일치 {m['완전일치%']}%")
        print(f"   TEDS {m['TEDS']}  AVG 매크로 {m['AVG_매크로']} / "
              f"마이크로 {m['AVG_마이크로']}{delta}\n")


if __name__ == "__main__":
    main()
