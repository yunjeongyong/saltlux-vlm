"""
Compare test_output/kie/<type>/<name>.json (model predictions, from
kie_from_parsing.py or kie_label.py) against kie_dataset/<type>/<name>.json
(ground truth), field by field.

For each doc: General fields are compared value-by-value — numbers use
numeric tolerance, strings are scored by normalized edit-distance similarity
(1.0 = identical, 0.0 = completely different) so near-miss OCR/formatting
differences aren't scored as full misses. null/missing counts as a value
like any other. Table rows are matched order-independently by best field
overlap; extra/missing rows count as full mismatches for their fields.

Reports per-doc, per-type and overall field-accuracy, plus table row-count
mismatches, and writes a full report to ./runs/<ts>_eval_kie.json.

Usage:
    uv run python scripts/eval_kie.py
    uv run python scripts/eval_kie.py --types bill invoice
    uv run python scripts/eval_kie.py --pred-dir test_output/kie
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from kie_label import DOC_TYPES

ROOT = Path(__file__).resolve().parent.parent
GT_DIR = ROOT / "kie_dataset"
RUNS_DIR = ROOT / "runs"


def normalize(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    s = re.sub(r"\s+", " ", str(v).strip().lower())
    return s or None


NUMERIC_TOLERANCE = 1e-4


def as_number(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            )
        prev = cur
    return prev[-1]


def string_similarity(a: str | None, b: str | None) -> float:
    a, b = a or "", b or ""
    if a == b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / max_len


def value_score(pred, gt) -> float:
    gt_num = as_number(gt)
    if gt_num is not None:
        pred_num = as_number(pred)
        return 1.0 if pred_num is not None and abs(pred_num - gt_num) <= NUMERIC_TOLERANCE else 0.0
    return string_similarity(normalize(str(pred)), normalize(str(gt)))


@dataclass
class FieldTally:
    score_sum: float = 0.0
    total: int = 0
    mismatches: list[dict] = field(default_factory=list)

    def add(self, key: str, pred, gt) -> None:
        self.total += 1
        score = value_score(pred, gt)
        self.score_sum += score
        if score < 1.0:
            self.mismatches.append({"field": key, "pred": pred, "gt": gt, "score": round(score, 3)})

    def accuracy(self) -> float | None:
        return None if self.total == 0 else self.score_sum / self.total


def compare_general(pred: dict, gt: dict) -> FieldTally:
    tally = FieldTally()
    for key in gt.get("General", {}):
        tally.add(key, (pred.get("General") or {}).get(key), gt["General"][key])
    return tally


def row_score(pred_row: dict, gt_row: dict) -> float:
    return sum(value_score(pred_row.get(key), gt_row[key]) for key in gt_row)


def match_rows(pred_rows: list[dict], gt_rows: list[dict]) -> dict[int, int]:
    """Greedy best-match pairing of gt row index -> pred row index, order-independent."""
    pairs = [
        (row_score(pred_rows[pi], gt_rows[gi]), gi, pi)
        for gi in range(len(gt_rows))
        for pi in range(len(pred_rows))
    ]
    pairs.sort(reverse=True)
    used_gt, used_pred = set(), set()
    matches: dict[int, int] = {}
    for score, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches[gi] = pi
    return matches


def compare_table(pred: dict, gt: dict) -> tuple[FieldTally, int, int]:
    pred_rows = pred.get("Table") or []
    gt_rows = gt.get("Table") or []
    tally = FieldTally()
    matches = match_rows(pred_rows, gt_rows)
    for gi, gt_row in enumerate(gt_rows):
        pred_row = pred_rows[matches[gi]] if gi in matches else {}
        for key in gt_row:
            tally.add(f"row{gi}.{key}", pred_row.get(key), gt_row[key])
    return tally, len(pred_rows), len(gt_rows)


def find_gt_docs(doc_type: str) -> list[Path]:
    return sorted((GT_DIR / doc_type).glob("*.json"))


def eval_doc(doc_type: str, name: str, pred_dir: Path) -> dict:
    gt_path = GT_DIR / doc_type / f"{name}.json"
    pred_path = pred_dir / doc_type / f"{name}.json"

    gt = json.loads(gt_path.read_text(encoding="utf-8"))

    if not pred_path.exists():
        return {
            "doc_type": doc_type,
            "name": name,
            "status": "missing_prediction",
            "general_accuracy": None,
            "table_accuracy": None,
            "total_accuracy": None,
            "table_row_count_pred": None,
            "table_row_count_gt": len(gt.get("Table") or []),
        }

    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    general_tally = compare_general(pred, gt)
    table_tally, n_pred_rows, n_gt_rows = compare_table(pred, gt)

    total_fields = general_tally.total + table_tally.total
    total_accuracy = (
        None
        if total_fields == 0
        else (general_tally.score_sum + table_tally.score_sum) / total_fields
    )

    return {
        "doc_type": doc_type,
        "name": name,
        "status": "ok",
        "general_accuracy": general_tally.accuracy(),
        "general_score_sum": general_tally.score_sum,
        "general_total": general_tally.total,
        "general_mismatches": general_tally.mismatches,
        "table_accuracy": table_tally.accuracy(),
        "table_score_sum": table_tally.score_sum,
        "table_total": table_tally.total,
        "total_accuracy": total_accuracy,
        "table_row_count_pred": n_pred_rows,
        "table_row_count_gt": n_gt_rows,
        "table_mismatches": table_tally.mismatches,
    }


def weighted_accuracy(score_sums: list[float], totals: list[int]) -> float | None:
    total = sum(totals)
    return None if total == 0 else sum(score_sums) / total


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if r["status"] == "ok"]
    general_score_sums = [r["general_score_sum"] for r in ok]
    general_totals = [r["general_total"] for r in ok]
    table_score_sums = [r["table_score_sum"] for r in ok]
    table_totals = [r["table_total"] for r in ok]
    return {
        "num_docs": len(results),
        "num_missing_predictions": len(results) - len(ok),
        "general_accuracy": weighted_accuracy(general_score_sums, general_totals),
        "table_accuracy": weighted_accuracy(table_score_sums, table_totals),
        "total_accuracy": weighted_accuracy(
            general_score_sums + table_score_sums, general_totals + table_totals
        ),
        "table_row_count_mismatches": sum(
            1 for r in ok if r["table_row_count_pred"] != r["table_row_count_gt"]
        ),
    }


def run(doc_types: list[str], pred_dir: Path) -> None:
    all_results = []
    for dt in doc_types:
        for gt_path in find_gt_docs(dt):
            all_results.append(eval_doc(dt, gt_path.stem, pred_dir))

    print(f"Evaluated {len(all_results)} documents against {pred_dir}\n")
    for dt in doc_types:
        dt_results = [r for r in all_results if r["doc_type"] == dt]
        if not dt_results:
            continue
        s = summarize(dt_results)
        ga = f"{s['general_accuracy']:.1%}" if s["general_accuracy"] is not None else "n/a"
        ta = f"{s['table_accuracy']:.1%}" if s["table_accuracy"] is not None else "n/a"
        tta = f"{s['total_accuracy']:.1%}" if s["total_accuracy"] is not None else "n/a"
        print(
            f"[{dt}] docs={s['num_docs']} missing={s['num_missing_predictions']} "
            f"general_acc={ga} table_acc={ta} total_acc={tta} "
            f"row_count_mismatches={s['table_row_count_mismatches']}"
        )

    overall = summarize(all_results)
    ga = f"{overall['general_accuracy']:.1%}" if overall["general_accuracy"] is not None else "n/a"
    ta = f"{overall['table_accuracy']:.1%}" if overall["table_accuracy"] is not None else "n/a"
    tta = f"{overall['total_accuracy']:.1%}" if overall["total_accuracy"] is not None else "n/a"
    print(
        f"\n[overall] docs={overall['num_docs']} missing={overall['num_missing_predictions']} "
        f"general_acc={ga} table_acc={ta} total_acc={tta} "
        f"row_count_mismatches={overall['table_row_count_mismatches']}"
    )

    RUNS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "task": "eval_kie",
        "timestamp_utc": ts,
        "doc_types": doc_types,
        "pred_dir": str(pred_dir),
        "overall": overall,
        "by_type": {
            dt: summarize([r for r in all_results if r["doc_type"] == dt]) for dt in doc_types
        },
        "results": all_results,
    }
    report_path = RUNS_DIR / f"{ts}_eval_kie.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull report written to {report_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", nargs="+", choices=DOC_TYPES, default=DOC_TYPES)
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=ROOT / "test_output" / "kie",
        help="directory of predictions to evaluate, shaped <pred-dir>/<type>/<name>.json",
    )
    args = parser.parse_args()

    run(doc_types=args.types, pred_dir=args.pred_dir)


if __name__ == "__main__":
    main()
