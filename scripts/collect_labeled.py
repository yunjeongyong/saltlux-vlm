#!/usr/bin/env python3
"""라벨링 툴에서 검수한 파일만 골라 작업 폴더로 복사한다.

툴은 저장할 때 train/json/<id>.json 을 덮어쓴다. 업로드 시각 이후로
수정된 파일이 곧 '사람이 손댄 것'이다. 그 기준으로 골라낸다.

이미지는 툴 폴더가 읽기 전용이라 심볼릭 링크가 아니라 실체를 복사한다.

사용:
  python3 collect_labeled.py --dry
  python3 collect_labeled.py                      # 기본 폴더로 복사
  python3 collect_labeled.py --after "2026-08-10 03:00"
"""
import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

SRC = Path("/data/workspace/VLM/labling-tool/luxia-labeling-tool/data/train")
DST = Path("/data/workspace/yjyong/receipt_data/labeled/train")
# 업로드가 끝난 시각. 이후 mtime 이면 툴에서 저장한 것.
DEFAULT_AFTER = "2026-08-10 03:00"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", default=DEFAULT_AFTER,
                    help='이 시각 이후 수정된 json 만 대상 ("YYYY-MM-DD HH:MM")')
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--dst", default=str(DST))
    ap.add_argument("--upto", action="append", default=[],
                    help="이 파일ID 이하는 수정이 없어도 포함 (접두사별로 여러 번 지정 가능). "
                         "예: --upto gcse_00725 --upto kid_IMG00779")
    ap.add_argument("--include", help="추가로 포함할 파일ID 목록 (쉼표 구분 또는 파일 경로)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    cutoff = dt.datetime.strptime(a.after, "%Y-%m-%d %H:%M").timestamp()

    # 고칠 게 없어 저장을 누르지 않은 파일은 mtime 이 안 바뀐다.
    # 훑어본 범위를 통째로 가져오려면 --upto 로 경계를 준다.
    extra = set()
    if a.include:
        p = Path(a.include)
        raw = p.read_text(encoding="utf-8") if p.is_file() else a.include
        extra |= {x.strip() for x in raw.replace(",", " ").split() if x.strip()}

    imgs = {p.stem: p for p in (src / "images").iterdir()}
    picked, missing = [], []
    for j in sorted((src / "json").glob("*.json")):
        in_range = any(j.stem.startswith(u.split("_")[0] + "_") and j.stem <= u
                       for u in a.upto)
        if j.stat().st_mtime <= cutoff and not in_range and j.stem not in extra:
            continue
        img = imgs.get(j.stem)
        if img is None:
            missing.append(j.stem)
            continue
        picked.append((j, img))

    print(f"검수 완료로 판단된 파일: {len(picked)}건  (기준 {a.after} 이후 수정)")
    if missing:
        print(f"  ⚠ 이미지 없음 {len(missing)}건: {missing[:5]}")
    for j, img in picked[-5:]:
        t = dt.datetime.fromtimestamp(j.stat().st_mtime).strftime("%H:%M")
        print(f"    {j.stem:18s} {t} 저장")
    if len(picked) > 5:
        print(f"    … 외 {len(picked)-5}건")

    if a.dry:
        print(f"\n(--dry: 복사하지 않았습니다. 대상 {dst})")
        return 0

    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "json").mkdir(parents=True, exist_ok=True)

    n_new = n_upd = 0
    for j, img in picked:
        dj, di = dst / "json" / j.name, dst / "images" / img.name
        is_new = not dj.exists()
        # 툴 폴더가 읽기 전용이라 링크가 아니라 실체를 복사한다
        shutil.copy2(j, dj)
        shutil.copy2(img, di)
        n_new += is_new
        n_upd += not is_new

    print(f"\n복사 완료 → {dst}")
    print(f"  신규 {n_new}건 · 갱신 {n_upd}건")
    print(f"  images {len(list((dst/'images').iterdir()))}장"
          f"   json {len(list((dst/'json').iterdir()))}개")

    # 이어서 택소노미를 만들 수 있게 안내
    print("\n이 폴더로 학습셋을 만들려면:")
    print("  python3 scripts/make_vlm_taxonomy.py \\")
    print(f"      --src {dst} \\")
    print("      --out yjyong/vlm-dataset-maker/data --link-images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
