"""정확도가 낮은 페이지를 분류하고 원인을 집계한다.

eval_kie.py 가 이미 per-doc 점수와 필드별 mismatch 를 runs/<ts>_eval_kie.json 에
전부 쓴다. 이 스크립트는 그 위에 얹는 것만 한다 — 다시 채점하지 않는다.

김동영 파트장 지시(2026-09-03)의 "추가 업무" 자리다.

    정확도가 크게 떨어지는 페이지 분류 후 원인 분석
    개선을 위한 데이터 유형 또는 패턴 정의 후 데이터 수집

내는 것 세 가지.

  ① 하위 N건        어느 페이지가 나쁜가
  ② 단계 분리        파싱에서 죽었나, 표 구조에서 죽었나, 필드 값에서 죽었나
  ③ 필드별 실패율    어느 필드가 반복해서 깨지나 (= 수집할 데이터의 단서)

단계 분리가 핵심이다. 같은 60% 라도 파싱이 백지를 뱉은 것과 표는 멀쩡한데 상호명만
틀리는 것은 수집해야 할 데이터가 전혀 다르다. 앞의 것은 해당 레이아웃의 페이지가
더 필요하고, 뒤의 것은 그 필드가 찍힌 크롭이 더 필요하다.

usage:
    python3 scripts/analyze_low_acc.py --parsing-dir vlm_output/best_exp013_ck7500
    python3 scripts/analyze_low_acc.py --report runs/20260907T...Z_eval_kie.json --bottom 30
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 파싱이 사실상 아무것도 못 뱉은 것으로 보는 길이. 영수증 한 장이 이보다 짧게 나오면
# 마크다운이 아니라 거절문·빈 표만 남은 경우다.
EMPTY_MD_CHARS = 120


def under_root(p: Path) -> Path:
    """상대 경로는 저장소 기준으로 읽는다 — 어느 디렉터리에서 불러도 같게 동작한다."""
    return p if p.is_absolute() else ROOT / p


def show(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def latest_report() -> Path:
    cands = sorted(glob.glob(str(ROOT / "runs" / "*_eval_kie.json")))
    if not cands:
        raise SystemExit("runs/ 에 eval_kie 리포트가 없다 — 먼저 scripts/eval_kie.py 를 돌려라")
    return Path(cands[-1])


def strip_row_index(field: str) -> str:
    """'row12.품명' -> 'Table.품명'. 행 번호는 집계에 의미가 없다."""
    m = re.match(r"^row\d+\.(.+)$", field)
    return f"Table.{m.group(1)}" if m else f"General.{field}"


def parsing_len(parsing_dir: Path | None, doc_type: str, name: str) -> int | None:
    """파싱 단계가 뱉은 마크다운 길이. 디렉터리를 안 주면 판정하지 않는다."""
    if parsing_dir is None:
        return None
    p = parsing_dir / doc_type / f"{name}.md"
    if not p.exists():
        return 0
    return len(p.read_text(encoding="utf-8", errors="replace").strip())


def classify(r: dict, md_chars: int | None) -> str:
    """이 문서가 어느 단계에서 무너졌는지 한 가지로 고른다. 앞선 단계가 우선한다."""
    if r["status"] != "ok":
        return "KIE 산출물 없음"
    if md_chars is not None and md_chars <= EMPTY_MD_CHARS:
        return "파싱 실패(빈 출력)"

    gt_rows = r.get("table_row_count_gt") or 0
    pred_rows = r.get("table_row_count_pred") or 0
    if gt_rows and pred_rows == 0:
        return "표 통째 누락"
    if gt_rows != pred_rows:
        return "표 행수 불일치"

    gen = r.get("general_accuracy")
    tab = r.get("table_accuracy")
    if gen is not None and tab is not None and gen < tab - 0.15:
        return "머리말 필드 오류"
    if tab is not None and gen is not None and tab < gen - 0.15:
        return "표 값 오류"
    return "전반 저조"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, help="eval_kie 리포트. 생략하면 runs/ 의 최신본")
    ap.add_argument("--parsing-dir", type=Path,
                    help="파싱 .md 디렉터리(vlm_output/<TAG>). 주면 파싱 단계 실패를 갈라낸다")
    ap.add_argument("--bottom", type=int, default=20, help="하위 몇 건을 볼 것인가")
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="이 total_accuracy 미만을 '크게 떨어지는' 것으로 본다")
    ap.add_argument("--out", type=Path, help="상세를 쓸 CSV 경로")
    a = ap.parse_args()

    report_path = under_root(a.report) if a.report else latest_report()
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    results = rep["results"]

    parsing_dir = under_root(a.parsing_dir) if a.parsing_dir else None
    if parsing_dir and not parsing_dir.exists():
        raise SystemExit(f"파싱 디렉터리가 없다: {parsing_dir}")

    rows = []
    for r in results:
        md = parsing_len(parsing_dir, r["doc_type"], r["name"])
        rows.append({
            "doc_type": r["doc_type"],
            "name": r["name"],
            "total": r.get("total_accuracy"),
            "general": r.get("general_accuracy"),
            "table": r.get("table_accuracy"),
            "rows_pred": r.get("table_row_count_pred"),
            "rows_gt": r.get("table_row_count_gt"),
            "md_chars": md,
            "cause": classify(r, md),
            "_raw": r,
        })

    print(f"리포트   : {show(report_path)}")
    print(f"파싱 출력: {show(parsing_dir) if parsing_dir else '(미지정 — 파싱 단계 판정 없음)'}")
    print(f"문서     : {len(rows)}건\n")

    # ── ① 크게 떨어지는 페이지
    scored = [x for x in rows if x["total"] is not None]
    low = sorted(scored, key=lambda x: x["total"])
    below = [x for x in low if x["total"] < a.threshold]
    missing = [x for x in rows if x["total"] is None]

    print(f"── ① total_accuracy < {a.threshold:.0%} : {len(below)}건"
          f" (전체 {len(rows)}건 중 {len(below)/max(len(rows),1):.1%})"
          + (f" · 산출물 없음 {len(missing)}건" if missing else ""))
    print()
    print("   %-10s %-14s %7s %8s %7s %9s %7s  %s"
          % ("유형", "문서", "total", "general", "table", "행수(예측/정답)", "md길이", "원인"))
    for x in low[:a.bottom]:
        rc = f"{x['rows_pred']}/{x['rows_gt']}"
        mc = "-" if x["md_chars"] is None else str(x["md_chars"])
        print("   %-10s %-14s %6.1f%% %7s %6s %9s %7s  %s" % (
            x["doc_type"], x["name"], x["total"] * 100,
            f"{x['general']*100:.1f}%" if x["general"] is not None else "-",
            f"{x['table']*100:.1f}%" if x["table"] is not None else "-",
            rc, mc, x["cause"]))

    # ── ② 단계별 원인 분포
    print(f"\n── ② 원인 분포 (하위 {len(below)}건 기준)")
    by_cause: dict[str, list] = defaultdict(list)
    for x in below + missing:
        by_cause[x["cause"]].append(x)
    for cause, xs in sorted(by_cause.items(), key=lambda kv: -len(kv[1])):
        types = ", ".join(f"{t}:{sum(1 for y in xs if y['doc_type']==t)}"
                          for t in sorted({y["doc_type"] for y in xs}))
        print(f"   {len(xs):>4}건  {cause:<16} ({types})")

    # ── ③ 필드별 실패율
    print("\n── ③ 필드별 실패 (전체 문서 기준 · 실패=score<1.0)")
    fail = defaultdict(int)
    worst_example: dict[str, dict] = {}
    for r in results:
        if r["status"] != "ok":
            continue
        for key in ("general_mismatches", "table_mismatches"):
            for m in r.get(key) or []:
                f = strip_row_index(m["field"])
                fail[f] += 1
                if f not in worst_example or m["score"] < worst_example[f]["score"]:
                    worst_example[f] = {**m, "doc": f"{r['doc_type']}/{r['name']}"}

    if not fail:
        print("   (필드별 mismatch 기록이 없다 — 이전 형식의 리포트이거나 전부 정답)")
    else:
        print("   %-28s %6s   %s" % ("필드", "실패건", "가장 나쁜 예 (예측 → 정답)"))
        for f, n in sorted(fail.items(), key=lambda kv: -kv[1])[:25]:
            ex = worst_example[f]
            pred = str(ex.get("pred"))[:26].replace("\n", " ")
            gt = str(ex.get("gt"))[:26].replace("\n", " ")
            print("   %-28s %6d   %-28s → %s" % (f, n, pred, gt))

    # ── CSV
    if a.out:
        out = under_root(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["doc_type", "name", "total_acc", "general_acc", "table_acc",
                        "rows_pred", "rows_gt", "md_chars", "cause"])
            for x in low + missing:
                w.writerow([x["doc_type"], x["name"], x["total"], x["general"], x["table"],
                            x["rows_pred"], x["rows_gt"], x["md_chars"], x["cause"]])
        print(f"\n상세 CSV: {show(out)}")


if __name__ == "__main__":
    main()
