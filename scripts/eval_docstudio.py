"""
Document Studio 출력 평가 하네스 (v1)

두 가지를 한다.
  A. Layout 분석 — GT 없이도 동작. threshold sweep으로 "이 값을 쓰면 무엇이 사라지는지"를 산출.
  B. 표 대조    — GT가 있을 때만. 행 recall / 숫자 정확도 / 계정명 CER / 회계 항등식.

설계 근거 (실측으로 확인된 것):
  - layout score 최저값은 본문이 아니라 작고 짧은 구조 메타데이터에 몰린다.
    (Table Sample: "(단위 : 원)" 0.5616 / Text Sample: "사업개요" 0.5911)
    표·본문은 0.93+ 라 threshold 위험이 없다. 그래서 sweep은 "무엇이" 빠지는지를 보여줘야 한다.
  - 계정명에 중복이 있다(보증금, 당기손익-공정가치금융자산이 유동/비유동에 각 1회).
    따라서 이름 key 매칭은 깨진다. 반드시 시퀀스 정렬을 쓴다.
  - GT는 아직 사람 검증 전이다. verified=False면 모든 지표에 라벨을 강제로 붙인다.

usage:
    python3 scripts/eval_docstudio.py "Table Sample.json" --gt eval_dataset/annotations/doc001_saltlux_bs_page000.json
    python3 scripts/eval_docstudio.py "Text Sample.json"
"""
import argparse
import csv
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
NUM_RE = re.compile(r"^-?[\d,]+$")


# ---------------------------------------------------------------- 로딩

def load_doc(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    boxes = d.get("layout_det_res", {}).get("boxes", [])
    blocks = d.get("parsing_res_list", [])
    # box <-> block 매칭 (bbox 근접). score를 본문에 붙이기 위해 필요.
    for b in boxes:
        b["content"] = ""
        for p in blocks:
            bb = p.get("block_bbox") or []
            if len(bb) == 4 and max(abs(x - y) for x, y in zip(b["coordinate"], bb)) < 6:
                b["content"] = str(p.get("block_content", ""))
                b["block_label"] = p.get("block_label")
                break
    return d, boxes, blocks


def table_html(blocks):
    for p in blocks:
        if p.get("block_label") == "table":
            return str(p.get("block_content", ""))
    return ""


def parse_table(html):
    """<table> HTML -> [[cell,...], ...]

    rowspan/colspan을 실제 격자로 전개한다. 이걸 안 하면 병합셀이 있는 표에서
    열이 밀려 "파서가 틀렸다"는 오탐이 난다. (연구개발계획6에서 실제로 겪음)
    """
    grid, pending = [], {}   # pending: {(row, col): 남은 값}
    for r, tr in enumerate(re.findall(r"<tr>(.*?)</tr>", html, re.S)):
        row, col = [], 0
        for attrs, inner in re.findall(r"<t[dh]([^>]*)>(.*?)</t[dh]>", tr, re.S):
            while (r, col) in pending:          # 위 행 rowspan이 덮은 자리
                row.append(pending.pop((r, col)))
                col += 1
            text = re.sub(r"<[^>]+>", "", inner).strip()
            rs = int((re.search(r'rowspan="(\d+)"', attrs) or [0, 1])[1])
            cs = int((re.search(r'colspan="(\d+)"', attrs) or [0, 1])[1])
            for c in range(cs):
                row.append(text)
                for extra in range(1, rs):
                    pending[(r + extra, col + c)] = text
            col += cs
        while (r, col) in pending:              # 행 끝에 걸린 rowspan
            row.append(pending.pop((r, col)))
            col += 1
        if row:
            grid.append(row)
    return grid


# ---------------------------------------------------------------- A. layout

def analyze_layout(boxes):
    rows = []
    for b in sorted(boxes, key=lambda x: x["score"]):
        x0, y0, x1, y1 = b["coordinate"]
        rows.append({
            "score": round(b["score"], 4),
            "label": b.get("label"),
            "w": round(x1 - x0), "h": round(y1 - y0),
            "content": b["content"][:60].replace("\n", " "),
        })

    sweep = []
    for t in THRESHOLDS:
        dropped = [r for r in rows if r["score"] < t]
        sweep.append({
            "threshold": t,
            "kept": len(rows) - len(dropped),
            "dropped": len(dropped),
            "dropped_content": [f'{r["label"]}:{r["content"]}' for r in dropped],
        })
    return rows, sweep


# ---------------------------------------------------------------- B. 표 대조

def align(gt_names, pr_names):
    """시퀀스 정렬. 이름 중복·행 누락이 있어도 깨지지 않는다."""
    sm = SequenceMatcher(None, gt_names, pr_names, autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pairs += [(i, j) for i, j in zip(range(i1, i2), range(j1, j2))]
        elif tag == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                gi = i1 + k if i1 + k < i2 else None
                pj = j1 + k if j1 + k < j2 else None
                pairs.append((gi, pj))
        elif tag == "delete":
            pairs += [(i, None) for i in range(i1, i2)]
        elif tag == "insert":
            pairs += [(None, j) for j in range(j1, j2)]
    return pairs


def cer(ref, hyp):
    if not ref:
        return 0.0 if not hyp else 1.0
    sm = SequenceMatcher(None, ref, hyp, autojunk=False)
    dist = sum(max(i2 - i1, j2 - j1)
               for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    return dist / len(ref)


def compare_table(gt, html):
    gt_rows = gt["ground_truth_table"]["rows"]
    cols = [c for c in gt["ground_truth_table"]["columns"] if c != "계정과목"]
    pr_rows = [r for r in parse_table(html) if len(r) >= 2]
    # 헤더행 제거 (첫 열이 비었거나 기수 표기)
    pr_rows = [r for r in pr_rows if r[0].strip() and not re.match(r"^제\s*\d+\s*기", r[0])]

    gt_names = [r["계정과목"] for r in gt_rows]
    pr_names = [r[0] for r in pr_rows]
    pairs = align(gt_names, pr_names)

    matched = [(i, j) for i, j in pairs if i is not None and j is not None]
    missing = [gt_names[i] for i, j in pairs if j is None]
    spurious = [pr_names[j] for i, j in pairs if i is None]

    name_errs, ref_chars, err_chars = [], 0, 0
    num_total = num_ok = 0
    num_errs = []
    for i, j in matched:
        ref, hyp = gt_names[i], pr_names[j]
        ref_chars += len(ref)
        if ref != hyp:
            e = cer(ref, hyp) * len(ref)
            err_chars += e
            name_errs.append({"gt": ref, "parser": hyp})
        for ci, col in enumerate(cols, start=1):
            g = gt_rows[i].get(col, "").strip()
            p = pr_rows[j][ci].strip() if ci < len(pr_rows[j]) else ""
            if not g and not p:
                continue
            num_total += 1
            if g == p:
                num_ok += 1
            else:
                num_errs.append({"row": ref, "col": col, "gt": g, "parser": p})

    return {
        "gt_rows": len(gt_rows), "parser_rows": len(pr_rows),
        "row_recall": round(len(matched) / len(gt_rows), 4) if gt_rows else None,
        "missing_rows": missing, "spurious_rows": spurious,
        "name_cer": round(err_chars / ref_chars, 4) if ref_chars else None,
        "name_error_count": len(name_errs), "name_errors": name_errs,
        "numeric_cells": num_total,
        "numeric_accuracy": round(num_ok / num_total, 4) if num_total else None,
        "numeric_errors": num_errs,
    }


def check_reading_order(blocks):
    """배열 순서가 시각적 읽기 순서(위->아래, 겹치면 왼쪽->오른쪽)와 맞는지.

    block_order 키는 레거시 더미라 신뢰할 수 없다. 배열 순서가 유일한 근거이므로
    그것이 기하학적으로 타당한지를 직접 검증한다.
    """
    seq = []
    for i, p in enumerate(blocks):
        bb = p.get("block_bbox") or []
        if len(bb) == 4:
            seq.append((i, bb[1], bb[0], p.get("block_label"), bb))
    def contains(o, i):
        return o[0] <= i[0] + 8 and o[1] <= i[1] + 8 and o[2] >= i[2] - 8 and o[3] >= i[3] - 8

    inversions = []
    for a in range(len(seq)):
        for b in range(a + 1, len(seq)):
            _, ya, xa, la, ba = seq[a]
            _, yb, xb, lb, bb = seq[b]
            # 한쪽이 다른 쪽을 감싸면 순서 판단 대상이 아니다.
            # (PPT 전면 배경 그래픽이 본문 뒤에 오는 건 정상)
            if contains(ba, bb) or contains(bb, ba):
                continue
            if yb + 12 < ya:
                inversions.append({"earlier": la, "later": lb,
                                   "y_earlier": round(ya), "y_later": round(yb)})
    return {"blocks_with_bbox": len(seq), "inversions": len(inversions),
            "ok": not inversions, "detail": inversions[:5]}


def check_identity(html):
    """자산총계 == 부채와 자본총계. GT 없이도 숫자 오류를 잡는 유일한 장치."""
    rows = parse_table(html)
    def find(*keys):
        for r in rows:
            n = r[0].replace(" ", "")
            if any(k in n for k in keys):
                return r[1:]
        return None
    a, b = find("자산총계"), find("부채와자본총계", "부채및자본총계")
    if not a or not b:
        return {"checked": False, "reason": "총계 행 미발견"}
    res = [{"period_idx": i, "자산총계": x, "부채와자본총계": y, "match": x == y}
           for i, (x, y) in enumerate(zip(a, b))]
    return {"checked": True, "all_match": all(r["match"] for r in res), "detail": res}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_json")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--out", default="eval_dataset/reports")
    args = ap.parse_args()

    d, boxes, blocks = load_doc(args.doc_json)
    name = Path(args.doc_json).stem
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    layout_rows, sweep = analyze_layout(boxes)
    report = {
        "doc": name,
        "size": [d.get("width"), d.get("height")],
        "model_settings": d.get("model_settings"),
        "block_count": len(blocks),
        # block_order 는 레거시 더미 키(파트장 확인, 2026-08-05). 배열 순서가 읽기 순서다.
        # 따라서 확인해야 할 것은 "배열 순서가 시각적 읽기 순서와 맞는가" 이다.
        "reading_order": check_reading_order(blocks),
        "layout": layout_rows,
        "threshold_sweep": sweep,
    }

    gt_verified = None
    html = table_html(blocks)
    if html:
        report["accounting_identity"] = check_identity(html)
    if args.gt:
        gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
        gt_verified = bool(gt.get("meta", {}).get("gt_verified", False))
        report["gt_verified"] = gt_verified
        if not gt_verified:
            report["WARNING"] = "GT 미검증 초안 기준. 보고용 수치로 사용 금지."
        if html:
            report["table"] = compare_table(gt, html)

    (out / f"{name}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(out / f"{name}_layout.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["score", "label", "w", "h", "content"])
        w.writeheader(); w.writerows(layout_rows)

    # ---- 콘솔 리포트
    print(f"\n{'='*72}\n{name}   {d.get('width')}x{d.get('height')}   블록 {len(blocks)}개\n{'='*72}")
    ro = report["reading_order"]
    if ro["ok"]:
        print("읽기순서 : 정상 (배열 순서 = 위->아래)")
    else:
        print(f"읽기순서 : ⚠ 역전 {ro['inversions']}건")
        for d in ro["detail"]:
            print(f"    {d['earlier']}(y={d['y_earlier']}) 뒤에 {d['later']}(y={d['y_later']})")

    print("\n[ layout score 하위 5 ]")
    for r in layout_rows[:5]:
        print(f"  {r['score']:.4f}  {r['label']:<16} {r['content'][:44]}")

    print("\n[ threshold sweep — 이 값을 쓰면 무엇이 사라지나 ]")
    for s in sweep:
        tag = "" if s["dropped"] == 0 else "  <= " + " | ".join(s["dropped_content"])[:70]
        print(f"  {s['threshold']:.2f} : 유지 {s['kept']:2d} / 손실 {s['dropped']}{tag}")

    if "accounting_identity" in report:
        ai = report["accounting_identity"]
        print(f"\n[ 회계 항등식 ] {'전 기수 일치' if ai.get('all_match') else ai}")

    if "table" in report:
        t = report["table"]
        label = "" if gt_verified else "   ⚠ 미검증 GT 기준"
        print(f"\n[ 표 대조 ]{label}")
        print(f"  행 recall      : {t['row_recall']:.1%}  (GT {t['gt_rows']} / 파서 {t['parser_rows']})")
        print(f"  숫자 정확도    : {t['numeric_accuracy']:.2%}  ({t['numeric_cells']}셀)")
        print(f"  계정명 CER     : {t['name_cer']:.2%}  (오류 {t['name_error_count']}건)")
        if t["missing_rows"]:
            print(f"  누락 행        : {t['missing_rows']}")
        if t["numeric_errors"]:
            print(f"  숫자 오류      : {len(t['numeric_errors'])}건")
            for e in t["numeric_errors"][:5]:
                print(f"      {e['row']} / {e['col']}: GT={e['gt']} vs {e['parser']}")
    print(f"\n-> {out}/{name}_report.json")


if __name__ == "__main__":
    sys.exit(main())
