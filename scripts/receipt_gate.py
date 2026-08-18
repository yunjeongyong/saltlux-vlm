"""
영수증 마크다운 출력의 산술 검증 게이트.

실전에는 GT 가 없으므로 CER 로 옳고 그름을 못 잰다. 대신 영수증 자체가
가진 항등식을 쓴다 — 이건 정답을 몰라도 성립 여부를 확인할 수 있다:

    품목 금액의 합 == 합계
    공급가액 + 부가세 == 합계

test GT 300건 전부가 두 식을 만족하므로 규칙으로 삼을 수 있다.
게이트에 걸리면 그 출력은 숫자가 틀린 것이 확정이다(역은 성립하지 않는다 —
통과했다고 맞는 건 아니다. 한글 오독은 산술로 안 잡힌다).

usage:
    python3 scripts/receipt_gate.py ml/infer_test_contract.log
"""
import re
import sys
from pathlib import Path

ROW = re.compile(r"^\s*\|\s*([^|]+?)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*$")
SUM = re.compile(r"공급가액:?\s*([\d,]+)\s+부가세:?\s*([\d,]+)\s+\*{0,2}합계:?\s*([\d,]+)\s*원")


def num(s):
    return int(s.replace(",", ""))


def check(text):
    """(통과여부, 사유목록) — 파싱 실패도 불통과로 본다."""
    items = [num(m.group(3)) for l in text.splitlines()
             if (m := ROW.match(l)) and not set(m.group(1)) <= set(":- ")]
    m = SUM.search(text)
    why = []
    if not items:
        why.append("품목 행 없음")
    if not m:
        why.append("합계 줄 없음/형식 불일치")
    if why:
        return False, why
    sup, vat, tot = num(m.group(1)), num(m.group(2)), num(m.group(3))
    if sum(items) != tot:
        why.append(f"품목합 {sum(items):,} != 합계 {tot:,}")
    if sup + vat != tot:
        why.append(f"공급가액+부가세 {sup+vat:,} != 합계 {tot:,}")
    return not why, why


def main():
    log = Path(sys.argv[1] if len(sys.argv) > 1 else "ml/infer_test_contract.log")
    txt = log.read_text(errors="replace").replace("\r", "")
    blocks = re.split(r"^={60,}$", txt, flags=re.M)[1:]

    # 게이트 판정 x 실제 정오 를 교차시켜 게이트가 쓸모 있는지 본다
    tp = fp = tn = fn = 0
    caught, missed = [], []
    for b in blocks:
        img = re.search(r"^\[\d+/\d+\] (\S+)", b.strip())
        out = re.search(r"── 베이스.*?─+\n(.*?)(?=\n\s+\[)", b, re.S)
        flag = re.search(r"\[(OK|숫자일치|불일치)\]", b)
        if not (img and out and flag):
            continue
        body = "\n".join(l[4:] for l in out.group(1).splitlines())
        passed, why = check(body)
        # '숫자가 맞았는가' 기준 — OK/숫자일치 는 숫자가 GT 와 같음
        num_ok = flag.group(1) in ("OK", "숫자일치")
        if num_ok and passed:
            tn += 1
        elif num_ok and not passed:
            fp += 1
            missed.append((Path(img.group(1)).name, "오탐", why))
        elif not num_ok and not passed:
            tp += 1
            caught.append((Path(img.group(1)).name, why))
        else:
            fn += 1
            missed.append((Path(img.group(1)).name, "놓침", []))

    n = tp + fp + tn + fn
    print(f"\n{log.name} — 영수증 {n}건\n")
    print(f"  숫자 틀림 & 게이트 걸림   {tp:>3}   ← 잡아냄")
    print(f"  숫자 틀림 & 게이트 통과   {fn:>3}   ← 놓침")
    print(f"  숫자 맞음 & 게이트 걸림   {fp:>3}   ← 오탐 (헛되이 검수)")
    print(f"  숫자 맞음 & 게이트 통과   {tn:>3}   ← 정상")
    bad = tp + fn
    if bad:
        print(f"\n  검출률  {tp}/{bad} = {tp/bad*100:.1f}%")
    if tp + fp:
        print(f"  정밀도  {tp}/{tp+fp} = {tp/(tp+fp)*100:.1f}%")
    print(f"  검수 대상  {tp+fp}/{n} = {(tp+fp)/n*100:.1f}%  (사람이 봐야 할 비율)")

    if caught:
        print("\n잡아낸 것:")
        for name, why in caught:
            print(f"   {name}  —  {' / '.join(why)}")
    if missed:
        print("\n놓치거나 헛짚은 것:")
        for name, kind, why in missed[:8]:
            print(f"   {name}  [{kind}]  {' / '.join(why) if why else ''}")


if __name__ == "__main__":
    main()
