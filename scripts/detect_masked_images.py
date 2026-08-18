#!/usr/bin/env python3
"""모자이크·블러로 크게 가려진 영수증 이미지를 찾는다.

가림 처리된 영역은 주변과 달리 '넓고 평평한 중간 톤 덩어리' 로 나타난다.
종이 여백(거의 흰색)이나 글자(대비 큼)와 구분하기 위해 세 조건을 모두 본다.

  ① 블록 내부 표준편차가 낮다        → 질감이 없다
  ② 밝기가 흰색도 검정도 아니다       → 여백이 아니다
  ③ 그런 블록이 넓게 이어져 있다      → 점 노이즈가 아니다

사용:
  python3 detect_masked_images.py --images <폴더> --top 40
  python3 detect_masked_images.py --images <폴더> --threshold 0.15 --out drop.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BLOCK = 12          # 블록 한 변(px). 축소본 기준
FLAT_STD = 6.0      # 이 값 미만이면 '평평한' 블록
LO, HI = 45, 225    # 이 밝기 범위 밖이면 여백/그림자로 보고 제외
LONGEST = 700       # 검사용 축소 크기


def masked_ratio(path):
    """가려진 것으로 보이는 면적 비율과 가장 큰 덩어리 비율을 돌려준다."""
    im = Image.open(path).convert("L")
    if max(im.size) > LONGEST:
        s = LONGEST / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
    a = np.asarray(im, dtype=np.float32)
    h, w = a.shape
    bh, bw = h // BLOCK, w // BLOCK
    if bh < 3 or bw < 3:
        return 0.0, 0.0
    a = a[:bh * BLOCK, :bw * BLOCK].reshape(bh, BLOCK, bw, BLOCK)
    mean = a.mean(axis=(1, 3))
    std = a.std(axis=(1, 3))

    flat = (std < FLAT_STD) & (mean > LO) & (mean < HI)
    ratio = float(flat.mean())

    # 가장 큰 연결 덩어리 (4방향, 반복 확장)
    lab = np.zeros_like(flat, dtype=np.int32)
    cur, best = 0, 0
    for i in range(bh):
        for j in range(bw):
            if flat[i, j] and lab[i, j] == 0:
                cur += 1
                stack, size = [(i, j)], 0
                lab[i, j] = cur
                while stack:
                    y, x = stack.pop()
                    size += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < bh and 0 <= nx < bw and flat[ny, nx] and lab[ny, nx] == 0:
                            lab[ny, nx] = cur
                            stack.append((ny, nx))
                best = max(best, size)
    return ratio, best / (bh * bw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--threshold", type=float,
                    help="가장 큰 덩어리 비율이 이 값 이상이면 목록에 넣는다 (예: 0.12)")
    ap.add_argument("--out", help="제거 후보 파일명을 저장할 텍스트 경로")
    ap.add_argument("--dump-scores", help="전 파일 점수를 JSON 으로 저장 (검수 HTML 에서 사용)")
    a = ap.parse_args()

    files = sorted(p for p in Path(a.images).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"검사 {len(files)}장\n")

    scored = []
    for i, f in enumerate(files, 1):
        try:
            r, big = masked_ratio(f)
        except Exception as e:
            print(f"  ✗ {f.name}: {e}")
            continue
        scored.append((big, r, f.name))
        if i % 200 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    if a.dump_scores:
        import json
        Path(a.dump_scores).write_text(json.dumps(
            {n: round(big, 4) for big, r, n in scored}, indent=0), encoding="utf-8")
        print(f"점수 덤프 → {a.dump_scores}")

    scored.sort(reverse=True)
    print(f"\n{'파일':24s} {'최대덩어리':>8s} {'평평면적':>8s}")
    for big, r, n in scored[:a.top]:
        print(f"{n:24s} {big*100:7.1f}% {r*100:7.1f}%")

    if a.threshold is not None:
        hit = [n for big, r, n in scored if big >= a.threshold]
        print(f"\n임계값 {a.threshold*100:.0f}% 이상: {len(hit)}건")
        if a.out:
            Path(a.out).write_text("\n".join(hit), encoding="utf-8")
            print(f"  → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
