"""
receipt 패키지에서 사람 검수분만 골라 이미지·박스·정답을 나란히 본다.

패키지에는 사람 검수 84건과 합성 92건이 섞여 있다(train.jsonl 의 source 로 구분).
합성본은 파서 박스에 생성 GT 를 얹은 것이라 성격이 다르니, 검수분만 따로 본다.

    왼쪽   이미지 + block_bbox (번호 + block_label)
    오른쪽 train.jsonl 의 assistant 정답 그대로

블록 좌표는 파서가 처리한 해상도 기준이라 원본 크기로 환산해서 얹는다.

usage:
    python3 scripts/viz_receipt_human.py -n 2
    python3 scripts/viz_receipt_human.py --ids gcse_00001,kid_IMG00005
"""
import argparse
import json
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PKG = Path("/data/workspace/yjyong/vlm_dataset_upload/train/human_annotated/receipt")
FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"
INK, MUTED, RULE = "#14171c", "#5d6674", "#dce0e6"
COLOR = {
    "table": "#d62728", "text": "#1f77b4", "doc_title": "#9467bd",
    "paragraph_title": "#8c564b", "figure_title": "#e377c2", "header": "#ff7f0e",
    "image": "#7f7f7f", "vision_footnote": "#2ca02c",
    "header_image": "#bcbd22", "footer_image": "#17becf", "aside_text": "#555555",
}


def page_of(d):
    """labeled/ 는 API 응답 규격이라 블록이 result.elements[0].json 안에 있다."""
    els = (d.get("result") or {}).get("elements") or []
    return (els[0].get("json") or {}) if els else d


def render(did, target_h=960):
    img = next((p for p in (PKG / "data").glob(f"{did}.*") if p.suffix != ".json"), None)
    page = page_of(json.loads((PKG / "data" / f"{did}.json").read_text(encoding="utf-8")))
    blocks = sorted(page.get("parsing_res_list") or [],
                    key=lambda b: b.get("block_order") if b.get("block_order") is not None else 0)
    im = Image.open(img).convert("RGB")
    W, H = im.size
    # 파서 처리 해상도 -> 원본 크기 환산
    sx = W / (page.get("width") or W)
    sy = H / (page.get("height") or H)
    s = target_h / H
    im = im.resize((max(1, int(W * s)), target_h))
    dr = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 17)
    for b in blocks:
        bb = b.get("block_bbox")
        if not bb or len(bb) < 4:
            continue
        lab = b.get("block_label", "?")
        c = COLOR.get(lab, "#000000")
        x1, y1 = bb[0] * sx * s, bb[1] * sy * s
        x2, y2 = bb[2] * sx * s, bb[3] * sy * s
        dr.rectangle([x1, y1, x2, y2], outline=c, width=3)
        tag = f"{b.get('block_order')} {lab}"
        tw = dr.textlength(tag, font=f)
        ty = max(0, y1 - 20)
        dr.rectangle([x1, ty, x1 + tw + 8, ty + 20], fill=c)
        dr.text((x1 + 4, ty + 1), tag, fill="white", font=f)
    return im, page, len(blocks)


def text_panel(row, w, h):
    im = Image.new("RGB", (w, h), "#fbfbfb")
    dr = ImageDraw.Draw(im)
    y = 8
    dr.text((10, y), "train.jsonl 정답 (assistant)", fill="#c0392b",
            font=ImageFont.truetype(FONT, 16))
    y += 24
    dr.line([8, y, w - 8, y], fill=RULE)
    y += 8
    f13 = ImageFont.truetype(FONT, 13)
    for para in row["messages"][1]["content"].split("\n"):
        if y > h - 22:
            dr.text((12, y), "…", fill="#888", font=f13)
            break
        if not para.strip():
            y += 7
            continue
        for ln in textwrap.wrap(para, width=44) or [""]:
            if y > h - 22:
                break
            dr.text((12, y), ln, fill=INK, font=f13)
            y += 16
    dr.rectangle([0, 0, w - 1, h - 1], outline="#bbb", width=2)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=2)
    ap.add_argument("--ids", default="")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--out", default="/data/workspace/yjyong/receipt_data/review/receipt_human.png")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (PKG / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    human = [r for r in rows if r.get("source") == "human_annotated"]
    if args.ids:
        want = [x.strip() for x in args.ids.split(",")]
        picks = [r for r in human if r["doc_id"] in want]
    else:
        picks = random.Random(args.seed).sample(human, min(args.n, len(human)))

    cells = []
    for r in picks:
        img, page, nb = render(r["doc_id"], args.height)
        pn = text_panel(r, 430, img.height)
        bar = 76
        cell = Image.new("RGB", (img.width + pn.width + 10, bar + img.height), "white")
        cell.paste(img, (0, bar))
        cell.paste(pn, (img.width + 10, bar))
        dr = ImageDraw.Draw(cell)
        dr.text((4, 4), r["doc_id"], fill=INK, font=ImageFont.truetype(FONT, 20))
        dr.text((4, 29), f"블록 {nb}개 · 파서 해상도 {page.get('width')}x{page.get('height')}",
                fill=MUTED, font=ImageFont.truetype(FONT, 14))
        dr.text((4, 50), "사람 검수 완료 (라벨링 툴)", fill="#0f7a4d",
                font=ImageFont.truetype(FONT, 14))
        cells.append(cell)

    pad = 18
    W = sum(c.width for c in cells) + pad * (len(cells) - 1)
    head = 76
    out = Image.new("RGB", (W, head + max(c.height for c in cells)), "white")
    dr = ImageDraw.Draw(out)
    dr.text((14, 10), "human_annotated/receipt — 사람 검수분 84건 중", fill=INK,
            font=ImageFont.truetype(FONT, 24))
    dr.text((14, 44), "박스 색 = block_label (파랑 text · 빨강 table · 보라 doc_title · 주황 header)",
            fill=MUTED, font=ImageFont.truetype(FONT, 15))
    x = 0
    for c in cells:
        out.paste(c, (x, head))
        x += c.width + pad
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"저장: {args.out}  {out.size}  ({', '.join(r['doc_id'] for r in picks)})")


if __name__ == "__main__":
    main()
