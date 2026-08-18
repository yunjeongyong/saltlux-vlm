"""
네 소스의 파서 BBOX 결과를 한 장의 그리드로 모아 본다.

소스마다 성격이 달라서(합성/실사크롤/스캔/촬영) 같은 파서를 태워도 결과가
크게 다르다. 나란히 놓고 봐야 어디를 검수해야 할지 판단이 선다.

아직 파싱이 안 끝난 소스는 조용히 건너뛴다. 진행 중에도 계속 돌릴 수 있게.

usage:
    python3 scripts/grid_all_sources.py
    python3 scripts/grid_all_sources.py --per-source 5 --seed 7
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

R = Path("/workspace/receipt_data")

SOURCES = [
    ("clean_v1 (합성+korie)", R / "clean_v1/bbox"),
    ("크롤링 (구글)",          R / "crawl_google/bbox"),
    ("SROIE 2019",            R / "public/sroie/bbox"),
    ("CORD-v2",               R / "public/cord/bbox"),
]

COLORS = {
    "table": (0, 170, 60), "text": (0, 110, 255), "figure_title": (240, 120, 0),
    "doc_title": (210, 0, 160), "paragraph_title": (140, 0, 220),
    "header": (0, 175, 175), "image": (130, 130, 130), "seal": (210, 0, 0),
    "footer": (150, 110, 0), "vision_footnote": (180, 140, 0),
    "header_image": (130, 130, 130), "footer_image": (130, 130, 130),
    "number": (90, 90, 90), "algorithm": (0, 90, 90), "aside_text": (170, 90, 0),
}


def font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except OSError:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--tile-w", type=int, default=300)
    ap.add_argument("--tile-h", type=int, default=430)
    ap.add_argument("--out", default=str(R / "bbox_grid_all_sources.png"))
    args = ap.parse_args()

    random.seed(args.seed)
    F, FB = font(13), font(19)
    TW, TH = args.tile_w, args.tile_h
    HDR = 34

    rows = []
    for name, d in SOURCES:
        if not d.exists():
            print(f"  건너뜀: {name} (아직 없음)")
            continue
        files = [f for f in sorted(d.glob("*.json"))
                 if json.loads(f.read_text(encoding="utf-8"))["n_boxes"] > 0]
        if not files:
            print(f"  건너뜀: {name} (박스 있는 문서 없음)")
            continue
        picks = random.sample(files, min(args.per_source, len(files)))
        tot_box = sum(json.loads(f.read_text(encoding="utf-8"))["n_boxes"] for f in files)

        row = Image.new("RGB", (args.per_source * TW, TH), "white")
        for i, f in enumerate(picks):
            b = json.loads(f.read_text(encoding="utf-8"))
            try:
                im = Image.open(R.parent / b["image"] if not Path(b["image"]).is_absolute()
                                else b["image"]).convert("RGB")
            except FileNotFoundError:
                continue
            dr = ImageDraw.Draw(im)
            w = max(2, int(min(im.size) / 220))
            for bx in b["boxes"]:
                dr.rectangle(bx["bbox"], outline=COLORS.get(bx["label"], (255, 0, 0)), width=w)
            im.thumbnail((TW - 8, TH - HDR - 6))
            t = Image.new("RGB", (TW, TH), "white")
            t.paste(im, ((TW - im.width) // 2, HDR))
            dd = ImageDraw.Draw(t)
            dd.text((4, 3), b["doc_id"][:30], fill=(20, 20, 20), font=F)
            dd.text((4, 18), f"박스 {b['n_boxes']}개  {b['width']}x{b['height']}  "
                             f"{b['scale_from_parse'][0]}x", fill=(120, 120, 120), font=F)
            dd.rectangle([0, 0, TW - 1, TH - 1], outline=(215, 215, 215))
            row.paste(t, (i * TW, 0))
        rows.append((f"{name}  —  {len(files)}장 / 박스 {tot_box:,}개", row))

    if not rows:
        raise SystemExit("그릴 소스가 없다.")

    LAB = 34
    W = args.per_source * TW
    canvas = Image.new("RGB", (W, len(rows) * (TH + LAB)), "white")
    dc = ImageDraw.Draw(canvas)
    y = 0
    for label, row in rows:
        dc.rectangle([0, y, W, y + LAB], fill=(38, 38, 42))
        dc.text((10, y + 8), label, fill="white", font=FB)
        canvas.paste(row, (0, y + LAB))
        y += TH + LAB
    canvas.save(args.out)
    print(f"\n저장: {args.out}  {canvas.size}  ({len(rows)}개 소스)")


if __name__ == "__main__":
    main()
