"""
포장이 끝난 aihub-71299-ocr 패키지가 실제로 어떻게 생겼는지 확인한다.

손글씨를 걸러낸 뒤(손글씨 박스 10% 초과 장 제외) 남은 장이 정말 인쇄체인지,
그리고 어절 박스를 줄로 묶어 만든 정답이 읽을 만한지를 눈으로 본다.

왼쪽  이미지 + 박스 (파랑=인쇄체 / 주황=남아있는 손글씨)
오른쪽 train.jsonl 의 assistant 정답 그대로

usage:
    python3 scripts/viz_71299_pkg.py
    python3 scripts/viz_71299_pkg.py -n 3 --seed 5
"""
import argparse
import json
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PKG = Path("/data/workspace/yjyong/aihub/71299/pkg/aihub-71299-ocr")
FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"
PRINTED, HAND = "#2a78d6", "#eb6834"
INK, MUTED, RULE = "#14171c", "#5d6674", "#dce0e6"


def load_rows(n, seed, prefer_zero=True):
    rows = [json.loads(l) for l in (PKG / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(seed)
    if prefer_zero:
        # 손글씨 0% 인 장을 우선 보여주되, 남아있는 소량 손글씨 장도 하나 섞는다
        zero = [r for r in rows if r.get("handwriting_ratio", 0) == 0]
        some = [r for r in rows if 0 < r.get("handwriting_ratio", 0) <= 0.10]
        pick = rng.sample(zero, min(n - 1, len(zero)))
        if some and n > 1:
            pick.append(rng.choice(some))
        return pick
    return rng.sample(rows, n)


def render(row, target_h=980):
    did = row["doc_id"]
    ip = PKG / "data" / f"{did}.jpg"
    lp = PKG / "data" / f"{did}.json"
    d = json.loads(lp.read_text(encoding="utf-8"))
    boxes = d.get("Bbox") or []
    im = Image.open(ip).convert("RGB")
    W, H = im.size
    s = target_h / H
    im = im.resize((max(1, int(W * s)), target_h))
    dr = ImageDraw.Draw(im)
    n_hand = 0
    for b in boxes:
        try:
            x1, y1 = min(b["x"]) * s, min(b["y"]) * s
            x2, y2 = max(b["x"]) * s, max(b["y"]) * s
        except (KeyError, ValueError, TypeError):
            continue
        printed = b.get("typeface") == 1
        n_hand += not printed
        dr.rectangle([x1, y1, x2, y2], outline=PRINTED if printed else HAND, width=2)
    return im, d, len(boxes), n_hand


def text_panel(row, w, h):
    im = Image.new("RGB", (w, h), "#fbfbfb")
    dr = ImageDraw.Draw(im)
    f15 = ImageFont.truetype(FONT, 15)
    f13 = ImageFont.truetype(FONT, 13)
    y = 8
    dr.text((10, y), "train.jsonl 정답 (assistant)", fill="#c0392b", font=f15)
    y += 24
    dr.line([8, y, w - 8, y], fill=RULE)
    y += 8
    txt = row["messages"][1]["content"]
    for para in txt.split("\n"):
        if y > h - 24:
            dr.text((12, y), "…", fill="#888", font=f13)
            break
        if not para.strip():
            y += 8
            continue
        for ln in textwrap.wrap(para, width=42) or [""]:
            if y > h - 24:
                break
            dr.text((12, y), ln, fill=INK, font=f13)
            y += 17
    dr.rectangle([0, 0, w - 1, h - 1], outline="#bbb", width=2)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--height", type=int, default=980)
    ap.add_argument("--out", default="/data/workspace/yjyong/receipt_data/review/aihub_71299_pkg.png")
    args = ap.parse_args()

    cells = []
    for row in load_rows(args.n, args.seed):
        img, d, nb, nh = render(row, args.height)
        pn = text_panel(row, 420, img.height)
        bar = 82
        cell = Image.new("RGB", (img.width + pn.width + 10, bar + img.height), "white")
        cell.paste(img, (0, bar))
        cell.paste(pn, (img.width + 10, bar))
        dr = ImageDraw.Draw(cell)
        I = d.get("Images", {})
        dr.text((4, 4), row["doc_id"], fill=INK, font=ImageFont.truetype(FONT, 18))
        dr.text((4, 28), f"{I.get('width')}x{I.get('height')}px · {I.get('dpi')}dpi · 박스 {nb}개",
                fill=MUTED, font=ImageFont.truetype(FONT, 14))
        hr = row.get("handwriting_ratio", 0)
        dr.text((4, 50), f"손글씨 {nh}개 ({100*hr:.1f}%)"
                         + ("  — 완전 인쇄체" if nh == 0 else "  — 임계 10% 이내로 통과"),
                fill="#0f7a4d" if nh == 0 else "#9a6600",
                font=ImageFont.truetype(FONT, 14))
        cells.append(cell)

    pad = 16
    W = sum(c.width for c in cells) + pad * (len(cells) - 1)
    head = 74
    out = Image.new("RGB", (W, head + max(c.height for c in cells)), "white")
    dr = ImageDraw.Draw(out)
    dr.text((14, 10), "aihub-71299-ocr — 손글씨 제외 후 패키지 (45,098장)", fill=INK,
            font=ImageFont.truetype(FONT, 24))
    dr.text((14, 44), "박스 색: 파랑=인쇄체(typeface 1) · 주황=손글씨(2·3·4)   "
                      "제외 19,124장 = 손글씨 박스 10% 초과",
            fill=MUTED, font=ImageFont.truetype(FONT, 15))
    x = 0
    for c in cells:
        out.paste(c, (x, head))
        x += c.width + pad
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"저장: {args.out}  {out.size}")


if __name__ == "__main__":
    main()
