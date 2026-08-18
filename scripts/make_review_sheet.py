"""
사람 검토 시트 생성.

합의 파이프라인이 215블록 -> 21건으로 줄여준 것을, 사람이 실제로 판정할 수 있는
형태로 뽑는다. 판정 결과를 그대로 학습셋에 반영할 수 있도록 verdict 칸을 비워 둔다.

사용법:
  1. python3 scripts/make_review_sheet.py
  2. review_sheet.md 를 보면서 각 항목의 이미지를 확인
  3. review_sheet.csv 의 verdict 열에 parser / model / other 중 하나를 기입
     (other 면 correct 열에 정답을 직접 입력)
  4. python3 scripts/apply_review.py 로 학습셋에 반영
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONS = ROOT / "eval_dataset/gt_consensus/consensus.jsonl"
OUT = ROOT / "eval_dataset/gt_consensus"


def main():
    rows = [json.loads(l) for l in open(CONS, encoding="utf-8")]
    need = [r for r in rows if r["grade"] == "low" or r["parser_changed"]]
    need.sort(key=lambda r: (r["grade"] != "low", r["id"]))

    md = ["# GT 사람 검토 시트",
          "",
          f"전체 {len(rows)}블록 중 **{len(need)}건**만 확인하면 됩니다 "
          f"({len(need)/len(rows):.1%}).",
          "",
          "각 항목의 이미지를 열어 원본과 대조하고, 아래 셋 중 무엇이 맞는지 판정하세요.",
          "",
          "- **parser** — 파서 출력이 맞음",
          "- **model** — 모델 출력이 맞음",
          "- **other** — 둘 다 틀림 (정답을 직접 기입)",
          "", "---", ""]

    csv_rows = []
    for i, r in enumerate(need, 1):
        srcs = r["sources"]
        md += [f"## {i}. `{r['id']}`",
               "",
               f"등급 **{r['grade']}** · {r['note']}",
               "",
               f"이미지: `{r['image']}`",
               ""]
        for k, v in srcs.items():
            v = str(v).replace("\n", " ⏎ ")
            md += [f"**{k}**", "```", v[:400], "```", ""]
        md += ["판정: `[ ] parser   [ ] model   [ ] other`", "", "---", ""]

        csv_rows.append({
            "id": r["id"], "grade": r["grade"], "image": r["image"],
            "parser": srcs.get("parser", ""),
            **{k: v for k, v in srcs.items() if k != "parser"},
            "verdict": "", "correct": "",
        })

    (OUT / "review_sheet.md").write_text("\n".join(md), encoding="utf-8")

    keys = list(csv_rows[0].keys()) if csv_rows else []
    with open(OUT / "review_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(csv_rows)

    print(f"검토 대상 {len(need)}건 / 전체 {len(rows)}블록 ({len(need)/len(rows):.1%})")
    print(f"  low(전부 불일치)   {sum(1 for r in need if r['grade']=='low')}")
    print(f"  파서 교정 발생      {sum(1 for r in need if r['parser_changed'])}")
    print(f"\n마크다운 -> {OUT}/review_sheet.md")
    print(f"CSV     -> {OUT}/review_sheet.csv   (verdict 열에 기입)")


if __name__ == "__main__":
    main()
