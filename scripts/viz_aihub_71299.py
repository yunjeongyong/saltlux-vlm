"""
AIHub 71299(대규모 OCR 데이터, 공공)의 이미지와 라벨을 겹쳐 본다.

71845 와 달리 이건 진짜 OCR 데이터다. 스캔한 공공문서에 **글자 단위 폴리곤 +
텍스트**가 붙어 있다. 다만 학습 원천(이미지)이 없어 실제로 쓸 수 있는 것은
검증셋 64,228장뿐이다.

라벨 형태:
    Bbox[] = { data: "기안용지", x:[x1,x1,x2,x2], y:[y1,y2,y1,y2],
               type, typeface, id }
    x/y 가 각각 4개인데 축별로 나열된 형태라 (min,max) 를 취하면 사각형이 된다.

usage:
    python3 scripts/viz_aihub_71299.py
    python3 scripts/viz_aihub_71299.py -n 3 --seed 2
"""
import argparse
import glob
import json
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/data/workspace/yjyong/aihub/71299/extracted")
FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"
BOX = "#2a78d6"
BOX2 = "#eb6834"
INK, MUTED, RULE = "#14171c", "#5d6674", "#dce0e6"


def pairs(n, seed):
    """VS(이미지) 와 VL(라벨) 은 폴더가 갈려 있다. stem 으로 맞춘다."""
    imgs = glob.glob(str(ROOT / "VS_*" / "*.jpg"))
    rng = random.Random(seed)
    rng.shuffle(imgs)
    lab = {Path(p).stem: p for p in glob.glob(str(ROOT / "VL_*" / "*.json"))}
    out = []
    for ip in imgs:
        lp = lab.get(Path(ip).stem)
        if lp:
            out.append((ip, lp))
        if len(out) >= n:
            break
    return out


def rect(b):
    """x:[x1,x1,x2,x2], y:[y1,y2,y1,y2] → (x1,y1,x2,y2)"""
    return min(b["x"]), min(b["y"]), max(b["x"]), max(b["y"])


def render(ip, lp, target_h=1000):
    d = json.loads(Path(lp).read_text(encoding="utf-8"))
    im = Image.open(ip).convert("RGB")
    W, H = im.size
    s = target_h / H
    im = im.resize((max(1, int(W * s)), target_h))
    dr = ImageDraw.Draw(im)
    boxes = d.get("Bbox") or []
    for b in boxes:
        try:
            x1, y1, x2, y2 = rect(b)
        except Exception:
            continue
        col = BOX if b.get("typeface") == 1 else BOX2
        dr.rectangle([x1 * s, y1 * s, x2 * s, y2 * s], outline=col, width=2)
    return im, d, boxes


def panel(d, boxes, w, h):
    im = Image.new("RGB", (w, h), "#fbfbfb")
    dr = ImageDraw.Draw(im)
    f15 = ImageFont.truetype(FONT, 15)
    f12 = ImageFont.truetype(FONT, 12)
    f11 = ImageFont.truetype(FONT, 11)
    I = d.get("Images", {})
    y = 8
    dr.text((10, y), "라벨 원문 — Bbox[]", fill="#c0392b", font=f15)
    y += 22
    dr.text((10, y), f"{I.get('width')}x{I.get('height')}px · {I.get('dpi')}dpi · "
                     f"박스 {len(boxes)}개", fill=MUTED, font=f12)
    y += 20
    dr.line([8, y, w - 8, y], fill=RULE)
    y += 8
    for b in boxes:
        if y > h - 20:
            dr.text((12, y), "…", fill="#888", font=f11)
            break
        try:
            x1, y1, x2, y2 = rect(b)
        except Exception:
            continue
        col = BOX if b.get("typeface") == 1 else BOX2
        dr.text((10, y), f"[{b.get('id')}]", fill=col, font=f11)
        txt = str(b.get("data", ""))
        dr.text((46, y), txt[:22], fill=INK, font=f12)
        dr.text((w - 190, y), f"({x1},{y1},{x2},{y2})", fill=MUTED, font=f11)
        y += 16
    dr.rectangle([0, 0, w - 1, h - 1], outline="#bbb", width=2)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--out", default="/data/workspace/yjyong/receipt_data/review/aihub_71299.png")
    args = ap.parse_args()

    cells = []
    for ip, lp in pairs(args.n, args.seed):
        img, d, boxes = render(ip, lp, args.height)
        pn = panel(d, boxes, 380, img.height)
        bar = 78
        cell = Image.new("RGB", (img.width + pn.width + 10, bar + img.height), "white")
        cell.paste(img, (0, bar))
        cell.paste(pn, (img.width + 10, bar))
        dr = ImageDraw.Draw(cell)
        I = d.get("Images", {})
        dr.text((4, 4), Path(ip).stem, fill=INK, font=ImageFont.truetype(FONT, 19))
        dr.text((4, 28), f"{d.get('Dataset',{}).get('name','')} · "
                         f"{I.get('width')}x{I.get('height')}px · {I.get('dpi')}dpi · 글자박스 {len(boxes)}개",
                fill=MUTED, font=ImageFont.truetype(FONT, 14))
        dr.text((4, 50), "파랑 = typeface 1(인쇄체) · 주황 = 그 외(필기·도장 등)",
                fill=MUTED, font=ImageFont.truetype(FONT, 13))
        cells.append(cell)

    pad = 18
    W = sum(c.width for c in cells) + pad * (len(cells) - 1)
    H = max(c.height for c in cells)
    out = Image.new("RGB", (W, H), "white")
    x = 0
    for c in cells:
        out.paste(c, (x, 0))
        x += c.width + pad
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"저장: {args.out}  {out.size}")


if __name__ == "__main__":
    main()
