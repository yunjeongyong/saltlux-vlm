#!/usr/bin/env python3
"""검수에서 고른 제거 목록을 exclude_list.txt 에 누적하고 택소노미에 적용한다.

검수 HTML 의 [선택 목록 보기] 에서 복사한 내용을 그대로 넘기면 된다.
이미 등록된 파일명은 건너뛰므로, 전체 목록을 매번 붙여넣어도 안전하다.

사용:
  python3 add_exclude.py --list drop.txt --note "육안 검수 4차"
  cat drop.txt | python3 add_exclude.py --note "육안 검수 4차"
  python3 add_exclude.py --list drop.txt --dry
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDE = ROOT / "receipt_data" / "exclude_list.txt"
TAXO = (ROOT / "yjyong/vlm-dataset-maker/data/instruct"
        "/luxia3_instruct_v0.0.0/private/luxia3_receipt_v0.0.0.json")
IMAGES = ROOT / "yjyong/vlm-dataset-maker/data/trainset/image/instruct"


def read_names(text):
    return [x for x in re.split(r"[,\s]+", text) if x.strip()]


def current():
    if not EXCLUDE.exists():
        return []
    out = []
    for line in EXCLUDE.read_text(encoding="utf-8").splitlines():
        n = line.split("#")[0].strip()
        if n:
            out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", help="파일명 목록 파일 (없으면 표준입력)")
    ap.add_argument("--note", default="육안 검수", help="exclude_list.txt 에 남길 메모")
    ap.add_argument("--date", default="", help="메모에 붙일 날짜 (예: 2026-08-10)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    raw = Path(a.list).read_text(encoding="utf-8") if a.list else sys.stdin.read()
    got = read_names(raw)
    if not got:
        sys.exit("목록이 비어 있습니다.")

    cur = current()
    known = set(cur)
    add, dup = [], []
    for n in got:
        (dup if n in known or n in add else add).append(n)

    print(f"받은 목록 {len(got)}건  |  이미 등록 {len(dup)}건  |  새로 추가 {len(add)}건")
    if add:
        print("  추가:", ", ".join(add[:12]) + (" …" if len(add) > 12 else ""))
    if a.dry:
        print("\n(--dry: 아무것도 바꾸지 않았습니다)")
        return 0

    if add:
        header = f"\n# {a.date + '  ' if a.date else ''}{a.note}\n"
        with EXCLUDE.open("a", encoding="utf-8") as f:
            f.write(header + "\n".join(add) + "\n")
        print(f"exclude_list.txt 에 {len(add)}건 추가 (누적 {len(cur)+len(add)}건)")

    tmp = Path("/tmp/exclude_clean.txt")
    tmp.write_text("\n".join(cur + add), encoding="utf-8")

    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/filter_vlm_taxonomy.py"),
         "--file", str(TAXO), "--images", str(IMAGES),
         "--drop", str(tmp), "--only-drop"],
        capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "택소노미 갱신" in line or "이미지 링크" in line:
            print(" ", line.strip())
    if r.returncode:
        print(r.stderr[-500:])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
