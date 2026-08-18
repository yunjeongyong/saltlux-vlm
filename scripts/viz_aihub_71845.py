"""
AIHub 71845(논문 요약) 데이터가 실제로 무엇을 담고 있는지 눈으로 확인한다.

이름만 보면 문서 데이터라 파싱 학습에 쓸 수 있을 것 같지만, 열어보면 목적이 다르다.
그 차이를 말로 설명하는 대신 이미지와 GT 를 나란히 붙여 드러낸다.

  - 페이지 이미지가 없다. PPTX 에서 뽑아낸 **그림·표 조각 PNG** 만 있다.
  - 그래서 레이아웃 bbox 를 얹을 판이 없다. location 은 PPTX 슬라이드 좌표(EMU)다.
  - 표(TA) 이미지의 GT 는 표 구조가 아니라 **서술형 캡션**이다.

usage:
    python3 scripts/viz_aihub_71845.py
    python3 scripts/viz_aihub_71845.py --per-cat 3 --split "TL_과학기술(ST)"
"""
import argparse
import glob
import json
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/data/workspace/yjyong/aihub/71845/extracted")
FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"

CAT_NAME = {"TA": "표 (Table)", "PI": "그림 (Picture)", "CH": "차트 (Chart)"}
CAT_COLOR = {"TA": "#d62728", "PI": "#1f77b4", "CH": "#2ca02c"}


def collect(split, per_cat, seed):
    """카테고리별로 (이미지경로, 메타, 논문제목) 을 모은다."""
    d = ROOT / split
    pool = {"TA": [], "PI": [], "CH": []}
    for f in sorted(glob.glob(str(d / "*.json"))):
        j = json.loads(Path(f).read_text(encoding="utf-8"))
        title = j["raw_data_meta_info"].get("doc_title", "")
        for i in (j["training_data_info"].get("image_info") or []):
            cat = i.get("image_category")
            if cat not in pool:
                continue
            p = d / Path(i["image_file_name"]).name
            if p.exists():
                pool[cat].append((p, i, title))
    rng = random.Random(seed)
    return {c: rng.sample(v, min(per_cat, len(v))) for c, v in pool.items() if v}


def cell(img_path, meta, title, cat, w=560, img_h=330):
    im = Image.new("RGB", (w, img_h + 250), "white")
    dr = ImageDraw.Draw(im)
    fh = ImageFont.truetype(FONT, 18)
    fb = ImageFont.truetype(FONT, 15)
    fs = ImageFont.truetype(FONT, 13)

    col = CAT_COLOR.get(cat, "#333")
    dr.rectangle([0, 0, w - 1, 26], fill=col)
    dr.text((7, 4), f"{CAT_NAME.get(cat, cat)}   {meta.get('image_name','')}", fill="white", font=fh)

    # 이미지: 비율 유지해 가운데
    pic = Image.open(img_path).convert("RGB")
    pw, ph = pic.size
    s = min((w - 16) / pw, img_h / ph)
    pic = pic.resize((max(1, int(pw * s)), max(1, int(ph * s))))
    im.paste(pic, ((w - pic.width) // 2, 32 + (img_h - pic.height) // 2))
    dr.rectangle([8, 32, w - 9, 32 + img_h], outline="#cccccc")
    dr.text((10, 34 + img_h), f"원본 {pw}x{ph}px", fill="#888", font=fs)

    y = 32 + img_h + 24
    dr.text((8, y), "GT = image_caption", fill=col, font=fb)
    y += 20
    cap = (meta.get("image_caption") or "").strip()
    for line in textwrap.wrap(cap, width=44)[:8]:
        dr.text((10, y), line, fill="#222", font=fs)
        y += 17
    if len(textwrap.wrap(cap, width=44)) > 8:
        dr.text((10, y), "…", fill="#888", font=fs)
        y += 17
    y += 6
    dr.text((10, y), f"논문: {title[:36]}", fill="#777", font=fs)
    y += 17
    dr.text((10, y), f"location(EMU): {meta.get('image_location','')}", fill="#777", font=fs)
    dr.rectangle([0, 0, w - 1, im.height - 1], outline="#bbbbbb", width=2)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="TL_과학기술(ST)")
    ap.add_argument("--per-cat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="/data/workspace/yjyong/receipt_data/review/aihub_71845.png")
    args = ap.parse_args()

    picks = collect(args.split, args.per_cat, args.seed)
    rows = []
    for cat in ["TA", "PI", "CH"]:
        if cat not in picks:
            continue
        cells = [cell(p, m, t, cat) for p, m, t in picks[cat]]
        pad = 14
        W = sum(c.width for c in cells) + pad * (len(cells) - 1)
        H = max(c.height for c in cells)
        row = Image.new("RGB", (W, H), "white")
        x = 0
        for c in cells:
            row.paste(c, (x, 0))
            x += c.width + pad
        rows.append(row)

    W = max(r.width for r in rows)
    bar = 54
    H = bar + sum(r.height for r in rows) + 14 * (len(rows) - 1)
    out = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(out)
    dr.text((8, 6), f"AIHub 71845  {args.split}", fill="black",
            font=ImageFont.truetype(FONT, 24))
    dr.text((8, 32), "페이지 이미지 없음 · PPTX 에서 뽑은 조각 PNG + 서술형 캡션이 전부",
            fill="#c00000", font=ImageFont.truetype(FONT, 17))
    y = bar
    for r in rows:
        out.paste(r, (0, y))
        y += r.height + 14
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"저장: {args.out}  {out.size}")


if __name__ == "__main__":
    main()
