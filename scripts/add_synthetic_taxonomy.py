#!/usr/bin/env python3
"""합성 영수증(receipts3000)을 택소노미로 변환한다.

이 데이터는 생성 시점의 값이 그대로 GT 라 정답이 100% 정확하다.
파싱을 거치지 않으므로 환각·오독이 없고, 사람 검수가 필요 없다.

실물 검수분과 별도 파일로 두어 vlm-dataset-maker 가 섞어 쓰게 한다.

사용:
  python3 add_synthetic_taxonomy.py --dry
  python3 add_synthetic_taxonomy.py --limit 1000 --link-images
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path("/data/workspace/yjyong")
SRC = ROOT / "receipt_data/receipts3000"
OUT = ROOT / "yjyong/vlm-dataset-maker/data"

TASK_C1, TASK_C2, DOMAIN = "OCR", "Document Understanding", "Other"
QUESTION = "이 영수증의 내용을 마크다운으로 정리해줘."


def detect_language(text):
    kor = len(re.findall(r"[가-힣]", text))
    eng = len(re.findall(r"[A-Za-z]", text))
    if kor and eng:
        r = kor / (kor + eng)
        return "kor" if r > 0.8 else ("eng" if r < 0.2 else "mix")
    return "kor" if kor else ("eng" if eng else "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="앞에서 N건만")
    ap.add_argument("--dataset-name", default="luxia3_receipt_syn")
    ap.add_argument("--source-dataset", default="luxia3_instruct_v0.0.0")
    ap.add_argument("--link-images", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    gts = sorted((SRC / "gt").glob("*.json"))
    if a.limit:
        gts = gts[: a.limit]

    imgs = {p.stem: p for p in (SRC / "images").iterdir()}
    rows, skipped = [], []

    for g in gts:
        img = imgs.get(g.stem)
        if img is None:
            skipped.append((g.stem, "이미지 없음"))
            continue
        d = json.loads(g.read_text(encoding="utf-8"))
        md = (d.get("markdown") or "").strip()
        if not md:
            skipped.append((g.stem, "markdown 없음"))
            continue
        rows.append({
            "dataset_name": a.dataset_name,
            "version": "0.0.0",
            "task": {"category_1": TASK_C1, "category_2": TASK_C2},
            "domain": {"category_1": DOMAIN},
            "metadata": {"image_cnt": 1, "source_file": g.name},
            "prompt": {"question": QUESTION, "input": [],
                       "image": [img.name], "conversation": []},
            "completion": {"chosen": md},
            "feature": {"completion": {"chosen": {"language": detect_language(md)}}},
        })

    print(f"변환 {len(rows)}건 / 건너뜀 {len(skipped)}건")
    lens = sorted(len(r["completion"]["chosen"]) for r in rows)
    if lens:
        print(f"정답 길이  중앙값 {lens[len(lens)//2]}자  최소 {lens[0]}  최대 {lens[-1]}")

    if a.dry:
        if rows:
            s = dict(rows[0])
            s["completion"] = {"chosen": s["completion"]["chosen"][:200] + " …"}
            print("\n=== 샘플 ===")
            print(json.dumps(s, ensure_ascii=False, indent=2))
        print("\n(--dry: 파일을 쓰지 않았습니다)")
        return 0

    tgt = (OUT / "instruct" / a.source_dataset / "private"
           / f"{a.dataset_name}_v0.0.0.json")
    tgt.parent.mkdir(parents=True, exist_ok=True)
    tgt.write_text(json.dumps(rows, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"\n택소노미 저장 → {tgt}")

    if a.link_images:
        imgdir = OUT / "trainset" / "image" / "instruct"
        imgdir.mkdir(parents=True, exist_ok=True)
        n = 0
        for r in rows:
            name = r["prompt"]["image"][0]
            link = imgdir / name
            if not link.exists():
                link.symlink_to(SRC / "images" / name)
                n += 1
        print(f"이미지 링크 {n}개 추가 → {imgdir}  (전체 {len(list(imgdir.iterdir()))}개)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
