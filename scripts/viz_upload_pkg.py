"""
업로드 대기 중인 두 패키지가 각각 무엇인지 한 장으로 보인다.

    train/human_annotated/receipt      영수증 84건   — 사람 검수
    train/public/aihub-71299-ocr       문서 36,598건 — AIHub 라벨

성격이 완전히 다르다. 하나는 사람이 검수한 소량 영수증이고, 다른 하나는
공공문서 대량 전사다. 정답 형식도 마크다운 표 vs 평문으로 갈린다.
그 차이를 이미지와 정답을 나란히 놓아 드러낸다.

usage:
    python3 scripts/viz_upload_pkg.py
"""
import argparse
import json
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path("/data/workspace/yjyong/vlm_dataset_upload/train")
FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"
C_RCP, C_DOC = "#2a78d6", "#eb6834"
INK, MUTED, RULE = "#14171c", "#5d6674", "#dce0e6"

PKGS = [
    ("영수증 · 사람 검수", "human_annotated/receipt", C_RCP, "receipt_markdown",
     "마크다운 표"),
    ("공공문서 · AIHub 라벨", "public/aihub-71299-ocr", C_DOC, "page_ocr",
     "평문 (표 제외)"),
]


def draw_boxes(pic, meta):
    """라벨의 좌표를 이미지에 얹는다. 두 패키지의 좌표계가 다르다.

      영수증  block_bbox 는 파서가 처리한 해상도 기준이라 원본 크기로 환산해야 한다.
      71299   Bbox 의 x/y 는 이미 원본 픽셀이라 그대로 쓴다.
    """
    W, H = pic.size
    dr = ImageDraw.Draw(pic)
    n = 0
    if "parsing_res_list" in meta:                      # 영수증 (블록 단위)
        pw = meta.get("width") or W
        ph = meta.get("height") or H
        sx, sy = W / pw, H / ph
        for b in meta["parsing_res_list"]:
            bb = b.get("block_bbox")
            if not bb or len(bb) < 4:
                continue
            col = "#d62728" if b.get("block_label") == "table" else "#2a78d6"
            dr.rectangle([bb[0]*sx, bb[1]*sy, bb[2]*sx, bb[3]*sy], outline=col, width=3)
            n += 1
    else:                                               # 71299 (어절 단위)
        for b in meta.get("Bbox", []):
            try:
                x1, y1, x2, y2 = min(b["x"]), min(b["y"]), max(b["x"]), max(b["y"])
            except (KeyError, ValueError, TypeError):
                continue
            col = "#eb6834" if b.get("typeface") != 1 else "#2a78d6"
            dr.rectangle([x1, y1, x2, y2], outline=col, width=4)
            n += 1
    return n


def sample_cell(pkg_dir, row, color, w=420, ih=520):
    did = row["doc_id"]
    img = next((p for p in (pkg_dir / "data").glob(f"{did}.*") if p.suffix != ".json"), None)
    meta = json.loads((pkg_dir / "data" / f"{did}.json").read_text(encoding="utf-8"))
    im = Image.new("RGB", (w, ih + 250), "white")
    dr = ImageDraw.Draw(im)
    n_box = 0
    if img:
        pic = Image.open(img).convert("RGB")
        n_box = draw_boxes(pic, meta)                   # 원본 해상도에서 그려야 선이 안 뭉갠다
        pw, ph = pic.size
        s = min((w - 16) / pw, ih / ph)
        pic = pic.resize((max(1, int(pw * s)), max(1, int(ph * s))))
        im.paste(pic, ((w - pic.width) // 2, 4))
        dr.rectangle([8, 4, w - 9, 4 + ih], outline=RULE)
    y = ih + 14
    dr.text((10, y), f"{did[:30]}   박스 {n_box}개", fill=MUTED, font=ImageFont.truetype(FONT, 13))
    y += 20
    dr.text((10, y), "정답 (assistant)", fill=color, font=ImageFont.truetype(FONT, 14))
    y += 20
    f12 = ImageFont.truetype(FONT, 12)
    txt = row["messages"][1]["content"]
    for para in txt.split("\n"):
        if y > im.height - 20:
            dr.text((12, y), "…", fill="#888", font=f12)
            break
        if not para.strip():
            y += 6
            continue
        for ln in textwrap.wrap(para, width=46) or [""]:
            if y > im.height - 20:
                break
            dr.text((12, y), ln, fill=INK, font=f12)
            y += 15
    dr.rectangle([0, 0, w - 1, im.height - 1], outline="#ccc")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--out", default="/data/workspace/yjyong/receipt_data/review/upload_packages.png")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    blocks = []
    for title, rel, color, task, fmt in PKGS:
        p = BASE / rel
        rows = [json.loads(l) for l in (p / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        picks = rng.sample(rows, min(args.per, len(rows)))
        cells = [sample_cell(p, r, color) for r in picks]
        pad = 12
        W = sum(c.width for c in cells) + pad * (len(cells) - 1)
        H = max(c.height for c in cells)
        row_im = Image.new("RGB", (W, H), "white")
        x = 0
        for c in cells:
            row_im.paste(c, (x, 0))
            x += c.width + pad
        head = 74
        blk = Image.new("RGB", (max(W, 900), head + H), "white")
        dr = ImageDraw.Draw(blk)
        dr.rectangle([0, 0, 8, head - 12], fill=color)
        dr.text((18, 2), title, fill=INK, font=ImageFont.truetype(FONT, 21))
        n_files = len(list((p / "data").glob("*")))
        size = sum(f.stat().st_size for f in (p / "data").glob("*")) / 1e9
        dr.text((18, 30), f"{len(rows):,}건  ·  data/ {n_files:,}개 파일(이미지+라벨)  ·  "
                          f"{size:.1f}GB  ·  task={task}", fill=MUTED,
                font=ImageFont.truetype(FONT, 15))
        legend = ("파랑=text 블록 · 빨강=table 블록"
                  if "receipt" in rel else "파랑=인쇄체 · 주황=손글씨")
        dr.text((18, 50), f"정답 형식: {fmt}   ·   train/{rel}/   ·   {legend}", fill=color,
                font=ImageFont.truetype(FONT, 14))
        blk.paste(row_im, (0, head))
        blocks.append(blk)

    pad = 20
    W = max(b.width for b in blocks)
    head = 92
    H = head + sum(b.height for b in blocks) + pad * (len(blocks) - 1) + 20
    out = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(out)
    dr.text((14, 12), "업로드 대기 데이터셋 2종", fill=INK, font=ImageFont.truetype(FONT, 26))
    dr.text((14, 48), "→ /data/workspace/VLM/dataset/train/   ·   합계 36,682건 / 8.9GB",
            fill=MUTED, font=ImageFont.truetype(FONT, 16))
    dr.line([14, head - 10, W - 14, head - 10], fill=RULE, width=2)
    y = head
    for b in blocks:
        out.paste(b, (14, y))
        y += b.height + pad
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"저장: {args.out}  {out.size}")


if __name__ == "__main__":
    main()
