"""
문서 유형별 숫자 정합성 검증.

GT 없이도 숫자 오류를 잡아내는 장치. 재무제표에서 회계 항등식으로
161셀을 검증했던 것과 같은 원리를 영수증·세금계산서로 확장한다.

OCR 오류가 나면 합계가 안 맞으므로, 합계 불일치 = 숫자 오류 신호다.
반대로 합계가 맞으면 그 숫자들은 서로를 검증한 셈이 된다.

지원 유형:
  balance_sheet  자산총계 == 부채와 자본총계
  tax_invoice    공급가액 + 세액 == 합계금액
  receipt        품목 합 == 소계,  소계 + 부가세 == 총액

usage:
    python3 scripts/validate_numeric.py <문서>.json [--type auto|tax_invoice|receipt]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canon import canon

# 유형 판별에 쓰는 키워드
HINTS = {
    "tax_invoice": ("세금계산서", "공급가액", "공급받는자", "사업자등록번호", "세액"),
    "receipt": ("영수증", "합계", "부가세", "받을금액", "결제금액", "품목", "단가"),
    "balance_sheet": ("자산총계", "부채와자본총계", "부채및자본총계", "재무상태표"),
}

NUM = re.compile(r"-?\d[\d,]*")


def to_num(s):
    s = str(s).strip().replace(",", "").replace(" ", "")
    m = re.fullmatch(r"-?\d+", s)
    return int(m.group()) if m else None


def all_text(doc):
    return "\n".join(str(b.get("block_content", "")) for b in doc["parsing_res_list"])


def detect_type(text):
    score = {k: sum(1 for w in ws if w in text) for k, ws in HINTS.items()}
    best = max(score, key=score.get)
    return best if score[best] >= 2 else "unknown"


def cells(doc):
    """모든 표 셀을 (행텍스트, 값) 형태로 편다."""
    out = []
    for b in doc["parsing_res_list"]:
        if b.get("block_label") != "table":
            continue
        html = canon(b.get("block_content", ""))
        for tr in re.findall(r"<tr>(.*?)</tr>", html, re.S):
            tds = [re.sub(r"<[^>]+>", "", c).strip()
                   for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            if tds:
                out.append(tds)
    return out


def find_labeled(rows, text, *keys):
    """라벨 키워드가 있는 행/줄에서 첫 숫자를 뽑는다."""
    for r in rows:
        joined = "".join(r).replace(" ", "")
        if any(k in joined for k in keys):
            for c in r[1:] or r:
                v = to_num(c)
                if v is not None:
                    return v
    for line in text.splitlines():
        s = line.replace(" ", "")
        if any(k in s for k in keys):
            m = NUM.findall(line)
            if m:
                return to_num(m[-1])
    return None


def check_tax_invoice(doc, text, rows):
    supply = find_labeled(rows, text, "공급가액")
    tax = find_labeled(rows, text, "세액", "부가가치세")
    total = find_labeled(rows, text, "합계금액", "총액", "합계")
    checks = []
    if supply is not None and tax is not None and total is not None:
        ok = supply + tax == total
        checks.append({"rule": "공급가액 + 세액 == 합계금액",
                       "values": {"공급가액": supply, "세액": tax, "합계": total},
                       "expected": supply + tax, "actual": total, "match": ok})
    if supply is not None and tax is not None:
        # 부가세율 10% 관행 확인 (반올림 오차 1원 허용)
        ok = abs(round(supply * 0.1) - tax) <= 1
        checks.append({"rule": "세액 == 공급가액 x 10%",
                       "expected": round(supply * 0.1), "actual": tax, "match": ok})
    return checks


def check_receipt(doc, text, rows):
    subtotal = find_labeled(rows, text, "소계", "과세물품가액", "공급가액")
    vat = find_labeled(rows, text, "부가세", "세액")
    total = find_labeled(rows, text, "합계", "총액", "받을금액", "결제금액")
    checks = []
    if subtotal is not None and vat is not None and total is not None:
        ok = subtotal + vat == total
        checks.append({"rule": "소계 + 부가세 == 총액",
                       "values": {"소계": subtotal, "부가세": vat, "총액": total},
                       "expected": subtotal + vat, "actual": total, "match": ok})
    # 품목 단가 x 수량 == 금액 (행 내부 검증)
    line_bad = 0, 0
    ok_n = bad_n = 0
    for r in rows:
        nums = [to_num(c) for c in r]
        nums = [n for n in nums if n is not None]
        if len(nums) >= 3:
            qty, price, amt = nums[-3], nums[-2], nums[-1]
            if qty and price and 0 < qty < 1000:
                (ok_n := ok_n + 1) if qty * price == amt else (bad_n := bad_n + 1)
    if ok_n or bad_n:
        checks.append({"rule": "수량 x 단가 == 금액 (품목 행)",
                       "ok": ok_n, "mismatch": bad_n, "match": bad_n == 0})
    return checks


def check_balance_sheet(doc, text, rows):
    def find(*keys):
        for r in rows:
            if any(k in r[0].replace(" ", "") for k in keys):
                return r[1:]
        return None
    a, b = find("자산총계"), find("부채와자본총계", "부채및자본총계")
    if not a or not b:
        return [{"rule": "자산총계 == 부채와 자본총계", "match": None,
                 "note": "총계 행 미발견"}]
    return [{"rule": "자산총계 == 부채와 자본총계",
             "periods": [{"자산총계": x, "부채와자본총계": y, "match": x == y}
                         for x, y in zip(a, b)],
             "match": all(x == y for x, y in zip(a, b))}]


CHECKERS = {"tax_invoice": check_tax_invoice, "receipt": check_receipt,
            "balance_sheet": check_balance_sheet}


def validate(doc, doc_type="auto"):
    text = all_text(doc)
    rows = cells(doc)
    t = detect_type(text) if doc_type == "auto" else doc_type
    fn = CHECKERS.get(t)
    checks = fn(doc, text, rows) if fn else []
    return {"doc_type": t, "checks": checks,
            "all_match": all(c.get("match") for c in checks) if checks else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_json")
    ap.add_argument("--type", default="auto",
                    choices=["auto", "tax_invoice", "receipt", "balance_sheet"])
    args = ap.parse_args()

    doc = json.loads(Path(args.doc_json).read_text(encoding="utf-8"))
    r = validate(doc, args.type)

    print(f"문서 유형: {r['doc_type']}")
    if not r["checks"]:
        print("  적용 가능한 검증 규칙 없음 (필요한 항목을 못 찾음)")
        return
    for c in r["checks"]:
        mark = "통과" if c.get("match") else ("실패" if c.get("match") is False else "판정불가")
        print(f"\n  [{mark}] {c['rule']}")
        for k, v in c.items():
            if k not in ("rule", "match"):
                print(f"      {k}: {v}")
    print(f"\n종합: {'전체 통과' if r['all_match'] else '불일치 있음 — 숫자 오류 의심'}")


if __name__ == "__main__":
    main()
