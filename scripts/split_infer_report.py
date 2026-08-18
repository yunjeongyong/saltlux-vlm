"""
infer_receipt.py 로그를 과제별로 갈라 다시 집계한다.

통합셋은 성격이 다른 두 과제가 섞여 있다 — 필드 크롭 전사(korie)와
전체 영수증 마크다운(receipts3000). 평균 하나로 내면 둘이 뭉개져서
"어느 쪽이 문제인지"가 안 보인다. 원본 디렉터리로 갈라서 따로 낸다.

usage:
    python3 scripts/split_infer_report.py ml/infer_n60.log
"""
import re
import sys
from pathlib import Path

HDR   = re.compile(r"^\[\d+/\d+\] (\S+)")
TAG   = re.compile(r"^\s+── (베이스.*?|어댑터 적용|베이스) ─+\s*$")
SCORE = re.compile(r"\[(OK|숫자일치|불일치)\]\s+문자정확도\s+([\d.]+)%\s+CER\s+([\d.]+)"
                   r"(?:\s+길이배율\s+([\d.]+)x\s+숫자재현\s+([\d.]+)%\s+여분숫자\s+(\d+))?")


def parse(path):
    rows, img, tag = [], None, None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.replace("\r", "")
        m = HDR.match(line)
        if m:
            img, tag = m.group(1), None
            continue
        m = TAG.match(line)
        if m:
            tag = "base" if m.group(1).startswith("베이스") else "adapter"
            continue
        m = SCORE.search(line)
        if m and img and tag:
            rows.append({"img": img, "arm": tag, "flag": m.group(1),
                         "acc": float(m.group(2)), "cer": float(m.group(3)),
                         "lr": float(m.group(4)) if m.group(4) else None,
                         "rec": float(m.group(5)) if m.group(5) else None,
                         "extra": int(m.group(6)) if m.group(6) else None})
            tag = None
    return rows


def task_of(img):
    return "전체 영수증 → 마크다운" if "receipts3000" in img else "필드 크롭 → 전사"


def agg(rows):
    n = len(rows)
    if not n:
        return None
    def avg(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return sum(v) / len(v) if v else None
    return {"n": n,
            "acc": sum(r["acc"] for r in rows) / n,
            "cer": sum(r["cer"] for r in rows) / n,
            "exact": sum(r["flag"] == "OK" for r in rows),
            "digit": sum(r["flag"] in ("OK", "숫자일치") for r in rows),
            "lr": avg("lr"), "rec": avg("rec"), "extra": avg("extra")}


def line(label, a):
    if not a:
        return f"  {label:<14} (없음)"
    out = (f"  {label:<14} n={a['n']:<4} 정확도 {a['acc']:5.1f}%  "
           f"CER {a['cer']:6.3f}  완전일치 {a['exact']:>3}/{a['n']:<4}"
           f"숫자일치 {a['digit']:>3}/{a['n']}")
    if a["lr"] is not None:
        out += (f"  |  길이배율 {a['lr']:5.2f}x  숫자재현 {a['rec']:5.1f}%  "
                f"여분 {a['extra']:5.1f}개")
    return out


def main():
    rows = parse(sys.argv[1] if len(sys.argv) > 1 else "ml/infer_n60.log")
    if not rows:
        sys.exit("채점 줄을 못 찾음 — 실행이 아직 안 끝났거나 GT 없는 입력")
    arms = sorted({r["arm"] for r in rows})
    print(f"\n채점된 출력 {len(rows)}개\n")
    for task in ("필드 크롭 → 전사", "전체 영수증 → 마크다운"):
        sub = [r for r in rows if task_of(r["img"]) == task]
        if not sub:
            continue
        print(f"{task}")
        for arm in arms:
            print(line(arm, agg([r for r in sub if r["arm"] == arm])))
        print()
    print("전체(두 과제 합산 — 참고용)")
    for arm in arms:
        print(line(arm, agg([r for r in rows if r["arm"] == arm])))

    if len(arms) == 2:
        print("\n샘플별 변화 (어댑터 − 베이스, 문자정확도 %p)")
        by = {}
        for r in rows:
            by.setdefault(r["img"], {})[r["arm"]] = r["acc"]
        d = [(v.get("adapter", 0) - v.get("base", 0), k) for k, v in by.items()
             if len(v) == 2]
        d.sort()
        up = sum(1 for x, _ in d if x > 0.5)
        dn = sum(1 for x, _ in d if x < -0.5)
        print(f"  개선 {up}개 / 악화 {dn}개 / 변화없음 {len(d)-up-dn}개")
        for x, k in d[:3]:
            print(f"    {x:+6.1f}p  {Path(k).name}")
        print("    ...")
        for x, k in d[-3:]:
            print(f"    {x:+6.1f}p  {Path(k).name}")


if __name__ == "__main__":
    main()
