#!/usr/bin/env python3
"""택소노미에서 파싱 오류가 명백한 건을 제거한다.

'표 빈칸' 은 파서가 칸을 못 읽은 것이라 라벨링으로 고칠 대상이므로 남긴다.
정답 문자열 자체가 깨진 것(반복·대체문자·길이 이상)만 걷어낸다.

원본은 <파일명>.bak 으로 남기고, 제거된 이미지 심볼릭 링크도 함께 지운다.

사용:
  python3 filter_vlm_taxonomy.py --file ... --images ... --dry
  python3 filter_vlm_taxonomy.py --file ... --images ...
"""
import argparse
import collections
import json
import re
import shutil
import sys
from pathlib import Path

# 제거 사유. 표 빈칸은 일부러 넣지 않는다.
RULES = {
    "대체문자": lambda a: "�" in a,
    "같은 줄 반복": None,          # 아래에서 따로 계산
    "정답 너무 짧음(<30자)": lambda a: len(a) < 30,
    "정답 너무 긺(>8000자)": lambda a: len(a) > 8000,
    "HTML 잔여": lambda a: "<td" in a or "<table" in a,
}


def repeated(ans):
    lines = [l.strip() for l in ans.split("\n") if l.strip()]
    if not lines:
        return False
    top, c = collections.Counter(lines).most_common(1)[0]
    return c >= 4 and len(top) > 5


def reasons(ans):
    out = []
    for name, fn in RULES.items():
        if name == "같은 줄 반복":
            if repeated(ans):
                out.append(name)
        elif fn(ans):
            out.append(name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--images")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--drop", help="이미지 파일명 목록 (쉼표 구분), 또는 목록이 담긴 텍스트 파일 경로")
    ap.add_argument("--only-drop", action="store_true",
                    help="--drop 목록만 제거하고 자동 규칙은 적용하지 않는다")
    a = ap.parse_args()

    p = Path(a.file)
    rows = json.loads(p.read_text(encoding="utf-8"))

    manual = set()
    if a.drop:
        f = Path(a.drop)
        raw = f.read_text(encoding="utf-8") if f.is_file() else a.drop
        manual = {x.strip() for x in re.split(r"[,\s]+", raw) if x.strip()}
        print(f"수동 제거 지정 {len(manual)}건")

    keep, drop = [], []
    for r in rows:
        rs = [] if a.only_drop else reasons(r["completion"]["chosen"])
        if r["prompt"]["image"][0] in manual:
            rs = rs + ["수동 지정"]
        (drop if rs else keep).append((r, rs))

    cnt = collections.Counter()
    for _, rs in drop:
        for x in rs:
            cnt[x] += 1

    print(f"전체 {len(rows)}건")
    print(f"  제거 {len(drop)}건 → 남는 데이터 {len(keep)}건\n")
    print("사유별 (중복 포함):")
    for k, v in cnt.most_common():
        print(f"   {k:24s} {v:4d}")
    print("\n제거 목록 (앞 15건):")
    for r, rs in drop[:15]:
        print(f"   {r['prompt']['image'][0]:24s} {len(r['completion']['chosen']):6d}자  {', '.join(rs)}")
    if len(drop) > 15:
        print(f"   … 외 {len(drop)-15}건")

    if a.dry:
        print("\n(--dry: 아무것도 바꾸지 않았습니다)")
        return 0

    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"\n원본 백업 → {bak.name}")

    p.write_text(json.dumps([r for r, _ in keep], ensure_ascii=False, indent=4),
                 encoding="utf-8")
    print(f"택소노미 갱신 → {p}  ({len(keep)}건)")

    if a.images:
        imgdir = Path(a.images)
        n = 0
        for r, _ in drop:
            link = imgdir / r["prompt"]["image"][0]
            if link.is_symlink() or link.exists():
                link.unlink()
                n += 1
        print(f"이미지 링크 제거 {n}개  (남은 링크 {len(list(imgdir.iterdir()))}개)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
