"""
infer_receipt.py 로그를 이미지 품질 등급별로 쪼개 집계한다.

등급을 섞으면 판단이 흐려진다 — clean/light/screenshot 은 이미 완전일치 100%
라 올릴 데가 없고, 실패는 heavy 에 몰려 있다. 평균만 보면 학습이 어디에
효과가 있었는지(또는 없었는지)가 안 보인다.

등급은 receipts3000/manifest.jsonl 의 quality 필드를 따른다.

usage:
    python3 scripts/eval_by_quality.py ml/eval_clean_v1.log
    python3 scripts/eval_by_quality.py 어댑터.log 베이스.log      # 두 로그 비교
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/workspace")
ORDER = ["clean", "light", "sharp_photo", "screenshot", "heavy", "extreme", "?"]
HDR = re.compile(r"^\[(\d+)/(\d+)\] (\S+)")
TAG = re.compile(r"^\s+── (베이스.*?|어댑터 적용) ─+\s*$")
SCORE = re.compile(r"\[(OK|숫자일치|불일치)\]\s+문자정확도\s+([\d.]+)%\s+CER\s+([\d.]+)"
                   r"(?:\s+길이배율\s+([\d.]+)x)?")


def tiers():
    t = {}
    p = ROOT / "receipt_data/receipts3000/manifest.jsonl"
    for l in open(p, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            t[r["image_path"].split("/")[-1]] = r["quality"]
    return t


def parse(path, tmap):
    rows, img, arm = [], None, None
    for line in Path(path).read_text(errors="replace").replace("\r", "").splitlines():
        m = HDR.match(line.strip())
        if m:
            img, arm = m.group(3), None
            continue
        m = TAG.match(line)
        if m:
            arm = "base" if m.group(1).startswith("베이스") else "adapter"
            continue
        m = SCORE.search(line)
        if m and img and arm:
            rows.append({"img": Path(img).name, "arm": arm,
                         "tier": tmap.get(Path(img).name, "?"),
                         "flag": m.group(1), "acc": float(m.group(2)),
                         "cer": float(m.group(3)),
                         "lr": float(m.group(4)) if m.group(4) else None})
            arm = None
    return rows


def block(rows, label):
    arms = sorted({r["arm"] for r in rows})
    present = [t for t in ORDER if any(r["tier"] == t for r in rows)]
    print(f"\n{label}")
    print(f"  {'등급':<14}" + "".join(f"{a:>16}" for a in arms) + f"{'차이':>10}")
    for t in present:
        line = f"  {t:<14}"
        vals = []
        for a in arms:
            sub = [r for r in rows if r["tier"] == t and r["arm"] == a]
            if sub:
                e = sum(r["flag"] == "OK" for r in sub)
                vals.append(e / len(sub))
                line += f"{e}/{len(sub)}".rjust(16)
            else:
                vals.append(None)
                line += "".rjust(16)
        if len(vals) == 2 and None not in vals:
            d = (vals[1] - vals[0]) * 100
            line += f"{d:+9.1f}p" if abs(d) > 0.05 else f"{'—':>10}"
        print(line)
    print("  " + "─" * (14 + 16 * len(arms) + 10))
    for a in arms:
        sub = [r for r in rows if r["arm"] == a]
        n = len(sub)
        print(f"  {a:<14}n={n:<4} 완전일치 {sum(r['flag']=='OK' for r in sub):>3}/{n:<4}"
              f" 숫자일치 {sum(r['flag'] in ('OK','숫자일치') for r in sub):>3}/{n:<4}"
              f" 정확도 {sum(r['acc'] for r in sub)/n:5.1f}%"
              f" CER {sum(r['cer'] for r in sub)/n:.4f}")


def main():
    tmap = tiers()
    logs = sys.argv[1:] or ["ml/eval_clean_v1.log"]
    for lg in logs:
        rows = parse(lg, tmap)
        if not rows:
            print(f"{lg}: 채점 줄 없음 — 실행이 아직 안 끝났을 수 있음")
            continue
        block(rows, f"■ {Path(lg).name}")

        # 등급별 개별 실패 목록
        bad = [r for r in rows if r["flag"] != "OK"]
        if bad:
            by = defaultdict(list)
            for r in bad:
                by[r["tier"]].append(r)
            print("\n  실패 내역")
            for t in ORDER:
                if t in by:
                    names = ", ".join(sorted({r["img"].replace(".jpg", "") for r in by[t]}))
                    print(f"    {t:<12}{len(by[t]):>3}건  {names[:70]}")


if __name__ == "__main__":
    main()
