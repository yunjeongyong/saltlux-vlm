"""
사람 검토 결과를 반영해 최종 GT를 확정한다.

입력: eval_dataset/gt_consensus/review_sheet.csv 의 verdict / correct 열
      verdict: parser | model | other | (빈칸 = 미판정)
      other 인 경우 correct 열의 값을 정답으로 쓴다.

출력: eval_dataset/gt_final/gt.jsonl        확정 GT (학습·평가 공용)
      eval_dataset/gt_final/blocked.jsonl   판정 불가 (원본 해상도 필요 등)

원칙:
  - 미판정(빈칸)은 확정하지 않는다. blocked 로 뺀다.
  - 확정된 것만 verified=true. 나머지는 학습에 쓰되 verified=false 로 표시한다.
  - 차트/이미지 블록은 GT 대상에서 제외한다 (실행마다 출력이 달라 정답이 하나가 아님).
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canon import canon

ROOT = Path(__file__).resolve().parent.parent
CONS = ROOT / "eval_dataset/gt_consensus/consensus.jsonl"
SHEET = ROOT / "eval_dataset/gt_consensus/review_sheet.csv"
OUT = ROOT / "eval_dataset/gt_final"

# 실행마다 출력이 달라 GT 를 만들 수 없는 블록 (오늘 실측: chart 740자 vs 575자)
EXCLUDE_LABELS = ("chart", "image")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {r["id"]: r for r in
            (json.loads(l) for l in open(CONS, encoding="utf-8"))}

    verdicts = {}
    if SHEET.exists():
        with open(SHEET, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                v = (r.get("verdict") or "").strip().lower()
                if v:
                    verdicts[r["id"]] = (v, (r.get("correct") or "").strip())

    gt, blocked, stats = [], [], {"verified": 0, "consensus": 0,
                                  "blocked": 0, "excluded": 0}
    for rid, r in rows.items():
        label_type = rid.split("_")[-1] if "_" in rid else ""
        if any(k in r["image"] for k in EXCLUDE_LABELS):
            stats["excluded"] += 1
            continue

        need_human = r["grade"] == "low" or r["parser_changed"]
        v = verdicts.get(rid)

        if need_human and not v:
            blocked.append({**r, "reason": "사람 판정 대기"})
            stats["blocked"] += 1
            continue

        if v:
            kind, corrected = v
            if kind == "parser":
                label = r["sources"]["parser"]
            elif kind == "model":
                label = r["label"]
            elif kind == "other":
                if not corrected:
                    blocked.append({**r, "reason": "other 선택했으나 correct 비어 있음"})
                    stats["blocked"] += 1
                    continue
                label = corrected
            else:
                blocked.append({**r, "reason": f"알 수 없는 verdict: {kind}"})
                stats["blocked"] += 1
                continue
            verified = True
            stats["verified"] += 1
        else:
            label = r["label"]           # high/mid 는 합의 결과 그대로
            verified = False
            stats["consensus"] += 1

        gt.append({"id": rid, "image": r["image"], "label": canon(label),
                   "verified": verified, "grade": r["grade"],
                   "source": "human" if verified else "consensus"})

    for name, data in (("gt", gt), ("blocked", blocked)):
        p = OUT / f"{name}.jsonl"
        p.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in data),
                     encoding="utf-8")
        print(f"{name}: {len(data)}건 -> {p}")

    print(f"\n{'구분':<16}{'건수':>6}")
    print("-" * 24)
    print(f"{'사람 검증':<16}{stats['verified']:>6}")
    print(f"{'합의 (미검증)':<16}{stats['consensus']:>6}")
    print(f"{'판정 대기':<16}{stats['blocked']:>6}")
    print(f"{'GT 제외 (차트)':<16}{stats['excluded']:>6}")
    print("-" * 24)
    print(f"{'GT 총계':<16}{len(gt):>6}")
    if stats["blocked"]:
        print(f"\n⚠ {stats['blocked']}건이 판정 대기 상태입니다. "
              f"review_sheet.csv 의 verdict 열을 채우세요.")


if __name__ == "__main__":
    main()
