"""
born-digital PDF 에서 OCR 없이 텍스트와 좌표를 그대로 꺼내 보인다.

AIHub 71845 의 TS/VS 폴더에는 논문 PDF 9,000편이 들어 있다. 스캔본이 아니라
편집기에서 바로 만든 PDF 라 텍스트 레이어가 살아 있다. 그래서 파싱을 돌릴
이유가 없다 — 글자와 좌표가 파일 안에 이미 정답으로 들어 있기 때문이다.

  파서 경로   이미지 렌더 → 레이아웃 검출 → VLM 인식   (오차가 두 번 낀다)
  추출 경로   PDF 텍스트 레이어 읽기                   (오차 0)

왼쪽에 페이지를 렌더해 텍스트 블록 좌표를 얹고, 오른쪽에 뽑아낸 글자를 붙인다.
한국어가 깨지지 않는지도 같이 본다.

usage:
    python3 scripts/viz_pdf_textlayer.py
    python3 scripts/viz_pdf_textlayer.py --ids ST_0008_0001055 --dpi 150
"""
import argparse
import glob
import random
import re
import textwrap
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/data/workspace/yjyong/aihub/71845/extracted")
FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"
BOX = "#2a78d6"
INK, MUTED, RULE = "#14171c", "#5d6674", "#dce0e6"

HANGUL = re.compile(r"[가-힣]")


def pick(n, seed, min_kor=0.2):
    """한글 비중이 있는 페이지를 고른다. 영문 초록만 있는 첫 장은 피한다."""
    out = []
    pdfs = sorted(glob.glob(str(ROOT / "TS_*" / "*.pdf")))
    rng = random.Random(seed)
    for p in rng.sample(pdfs, min(400, len(pdfs))):
        try:
            d = pymupdf.open(p)
        except Exception:
            continue
        for pno in range(min(4, d.page_count)):
            t = d[pno].get_text()
            if len(t) < 500:
                continue
            if len(HANGUL.findall(t)) / max(1, len(t)) >= min_kor:
                out.append((p, pno))
                break
        d.close()
        if len(out) >= n:
            break
    return out


def render(pdf_path, pno, dpi, max_h):
    d = pymupdf.open(pdf_path)
    pg = d[pno]
    pm = pg.get_pixmap(dpi=dpi)
    im = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
    s_pdf = dpi / 72.0                      # PDF 포인트 → 렌더 픽셀
    s = min(1.0, max_h / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
    dr = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 13)
    blocks = [b for b in pg.get_text("blocks") if b[6] == 0 and b[4].strip()]
    for b in blocks:
        x0, y0, x1, y1 = [v * s_pdf * s for v in b[:4]]
        dr.rectangle([x0, y0, x1, y1], outline=BOX, width=2)
        tag = str(b[5])
        tw = dr.textlength(tag, font=f)
        dr.rectangle([x0, max(0, y0 - 15), x0 + tw + 5, max(0, y0 - 15) + 15], fill=BOX)
        dr.text((x0 + 2, max(0, y0 - 15)), tag, fill="white", font=f)
    meta = dict(page=pno + 1, npage=d.page_count,
                pt=(pg.rect.width, pg.rect.height),
                nblock=len(blocks),
                nchar=len(pg.get_text()),
                kor=len(HANGUL.findall(pg.get_text())))
    blocks_txt = [(b[5], b[:4], b[4].strip()) for b in blocks]
    d.close()
    return im, meta, blocks_txt


def text_panel(blocks, w, h):
    im = Image.new("RGB", (w, h), "#fbfbfb")
    dr = ImageDraw.Draw(im)
    f15 = ImageFont.truetype(FONT, 15)
    f12 = ImageFont.truetype(FONT, 12)
    f11 = ImageFont.truetype(FONT, 11)
    y = 8
    dr.text((10, y), "PDF 텍스트 레이어 — 그대로 꺼낸 값", fill="#c0392b", font=f15)
    y += 24
    for bno, bb, txt in blocks:
        if y > h - 30:
            dr.text((12, y), f"… 외 {len(blocks) - blocks.index((bno, bb, txt))}개 블록", fill="#888", font=f11)
            break
        head = f"[{bno}] ({bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f})"
        dr.text((10, y), head, fill=BOX, font=f11)
        y += 15
        for ln in textwrap.wrap(" ".join(txt.split()), width=46)[:3]:
            dr.text((14, y), ln, fill=INK, font=f12)
            y += 15
        y += 4
    dr.rectangle([0, 0, w - 1, h - 1], outline="#bbb", width=2)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=2)
    ap.add_argument("--ids", default="")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--height", type=int, default=980)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="/data/workspace/yjyong/receipt_data/review/pdf_textlayer.png")
    args = ap.parse_args()

    if args.ids:
        targets = []
        for i in args.ids.split(","):
            hits = glob.glob(str(ROOT / "TS_*" / f"{i.strip()}.pdf"))
            if hits:
                targets.append((hits[0], 0))
    else:
        targets = pick(args.n, args.seed)
    if not targets:
        raise SystemExit("대상 PDF 를 못 찾았다")

    cells = []
    for p, pno in targets:
        img, meta, blocks = render(p, pno, args.dpi, args.height)
        pn = text_panel(blocks, 470, img.height)
        bar = 78
        cell = Image.new("RGB", (img.width + pn.width + 10, bar + img.height), "white")
        cell.paste(img, (0, bar))
        cell.paste(pn, (img.width + 10, bar))
        dr = ImageDraw.Draw(cell)
        dr.text((4, 5), Path(p).stem, fill=INK, font=ImageFont.truetype(FONT, 20))
        dr.text((4, 30), f"{meta['page']}/{meta['npage']}쪽 · {meta['pt'][0]:.0f}x{meta['pt'][1]:.0f}pt · "
                         f"텍스트 블록 {meta['nblock']}개 · {meta['nchar']:,}자 (한글 {meta['kor']:,})",
                fill=MUTED, font=ImageFont.truetype(FONT, 15))
        dr.text((4, 52), "OCR 0회 · VLM 0회 — 좌표와 글자를 파일에서 그대로 읽었다",
                fill="#0f7a4d", font=ImageFont.truetype(FONT, 15))
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
