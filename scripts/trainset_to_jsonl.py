#!/usr/bin/env python3
"""vlm-dataset-maker 산출물을 train_vlm.py 가 먹는 JSONL 로 바꾼다.

두 가지가 어긋나 있어 그대로는 못 쓴다.
  ① JSON 배열  →  JSONL (한 줄 = 한 샘플)
  ② 이미지 경로가 도커 기준 /app/data/... → train_vlm.py 는 ROOT 기준 상대경로

train_vlm.py 는 Image.open(ROOT / rel) 로 여는데 ROOT 는 /data/workspace/yjyong 이다.

사용:
  python3 trainset_to_jsonl.py \
      --file yjyong/vlm-dataset-maker/data/trainset/text/receipt_v1.json \
      --images yjyong/vlm-dataset-maker/data/trainset/image/instruct \
      --out receipt_data/unified_receipt --val-ratio 0.1
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path("/data/workspace/yjyong")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--images", required=True,
                    help="실제 이미지 폴더 (ROOT 기준 상대경로로 변환된다)")
    ap.add_argument("--out", required=True, help="출력 폴더")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rows = json.loads(Path(a.file).read_text(encoding="utf-8"))
    imgdir = Path(a.images)
    rel_dir = imgdir.resolve().relative_to(ROOT)

    out_rows, missing = [], []
    for r in rows:
        name = Path(r["images"][0]).name          # /app/... 에서 파일명만 취한다
        p = imgdir / name
        if not p.exists():
            missing.append(name)
            continue
        out_rows.append({
            "images": [str(rel_dir / name)],
            "messages": r["messages"],
        })

    if missing:
        print(f"⚠ 이미지 없음 {len(missing)}건: {missing[:5]}")

    random.seed(a.seed)
    random.shuffle(out_rows)
    n_val = max(1, int(len(out_rows) * a.val_ratio))
    val, train = out_rows[:n_val], out_rows[n_val:]

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val)):
        fp = outdir / f"{name}.jsonl"
        with fp.open("w", encoding="utf-8") as f:
            for x in part:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"  {name}.jsonl  {len(part)}건  → {fp}")

    # 학습 코드가 실제로 열 수 있는지 표본 검증
    from PIL import Image
    ok = 0
    for x in train[:5]:
        try:
            Image.open(ROOT / x["images"][0]).convert("RGB")
            ok += 1
        except Exception as e:
            print(f"  ✗ 이미지 열기 실패 {x['images'][0]}: {e}")
    print(f"\n경로 검증: 표본 5건 중 {ok}건 정상 로드")
    print(f"이미지 경로 형식: {train[0]['images'][0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
