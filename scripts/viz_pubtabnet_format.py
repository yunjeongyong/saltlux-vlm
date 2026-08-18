"""
PubTabNet-HTML 의 포맷이 실제로 잘 나뉘어 있는지 눈으로 확인한다.

이 데이터는 표를 세 갈래로 쪼개 저장한다.

    structure.tokens   표의 뼈대 (HTML 태그 시퀀스). 셀 내용 없음
    cells[].tokens     셀 내용 (문자 단위). 좌표 없음
    cells[].bbox       셀 좌표. 빈 셀은 이 키가 아예 없다

셋이 따로 놀면 쓸모가 없으니, 이미지 위에 bbox 를 얹고 그 옆에 구조 토큰과
셀 내용을 나란히 붙여 **같은 표를 가리키고 있는지** 확인한다.

usage:
    python3 scripts/viz_pubtabnet_format.py --ids 000000,366212
"""
import argparse
import ast
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DS = Path("/data/workspace/VLM/dataset/train/public/pubtabnet-html")
FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"

# 헤더 셀은 구조 토큰에서 <th> 로 나온다. 색을 갈라 두면 thead/tbody 경계가 보인다.
C_HEAD = "#d62728"
C_BODY = "#1f77b4"
C_EMPTY = "#bbbbbb"


def load(did):
    d = json.loads((DS / "data" / f"{did}.json").read_text(encoding="utf-8"))
    h = ast.literal_eval(d["html"])          # 작은따옴표라 json.loads 는 실패한다
    return d, h


def cell_kinds(struct_tokens):
    """구조 토큰을 훑어 셀마다 (헤더인가, colspan/rowspan) 을 뽑는다.

    주의: 이 데이터에는 <th> 셀이 없다. 헤더 행도 셀 태그는 <td> 이고,
    헤더인지 아닌지는 <thead>...</thead> 구역에 들어 있느냐로만 구분된다.
    (500장 실측: <th> 0건 / <thead> 500건)
    """
    kinds = []
    in_head = False
    i = 0
    T = struct_tokens
    while i < len(T):
        t = T[i]
        if t == "<thead>":
            in_head = True
            i += 1
        elif t == "</thead>":
            in_head = False
            i += 1
        elif t in ("<td>", "<th>"):
            kinds.append((in_head, ""))
            i += 1
        elif t in ("<td", "<th"):
            span = ""
            j = i + 1
            while j < len(T) and T[j] != ">":
                span += T[j]
                j += 1
            kinds.append((in_head, span.strip()))
            i = j + 1
        else:
            i += 1
    return kinds


def draw_boxes(did, h, target_w=780):
    """표 이미지는 가로로 길고 세로가 짧다. 가로 기준으로 키워야 글자가 읽힌다."""
    im = Image.open(DS / "data" / f"{did}.jpg").convert("RGB")
    W, H = im.size
    s = target_w / W
    im = im.resize((target_w, max(1, int(H * s))))
    dr = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 13)
    kinds = cell_kinds(h["structure"]["tokens"])
    n_box = n_nobox = 0
    for idx, c in enumerate(h["cells"]):
        b = c.get("bbox")
        head = kinds[idx][0] if idx < len(kinds) else False
        span = kinds[idx][1] if idx < len(kinds) else ""
        if not b:
            n_nobox += 1
            continue
        n_box += 1
        col = C_HEAD if head else C_BODY
        x1, y1, x2, y2 = [v * s for v in b]
        dr.rectangle([x1, y1, x2, y2], outline=col, width=2)
        tag = f"{idx} {'thead' if head else 'td'}" + (f" {span}" if span else "")
        tw = dr.textlength(tag, font=f)
        ty = max(0, y1 - 15)
        dr.rectangle([x1, ty, x1 + tw + 5, ty + 15], fill=col)
        dr.text((x1 + 2, ty), tag, fill="white", font=f)
    return im, n_box, n_nobox, kinds


def panel(did, h, kinds, w, hgt, n_box, n_nobox):
    im = Image.new("RGB", (w, hgt), "#fbfbfb")
    dr = ImageDraw.Draw(im)
    fb = ImageFont.truetype(FONT, 17)
    fs = ImageFont.truetype(FONT, 14)
    fm = ImageFont.truetype(FONT, 13)
    y = 10
    dr.text((10, y), f"cells {len(h['cells'])}개  ·  bbox 있음 {n_box} / 없음 {n_nobox}",
            fill="#000", font=fb)
    y += 26
    st = h["structure"]["tokens"]
    dr.text((10, y), f"structure.tokens  ({len(st)}개) — 뼈대만", fill="#c00000", font=fs)
    y += 20
    line = ""
    for t in st:
        if dr.textlength(line + t, font=fm) > w - 24:
            dr.text((12, y), line, fill="#444", font=fm)
            y += 16
            line = ""
            if y > hgt * 0.42:
                dr.text((12, y), "…", fill="#888", font=fm)
                y += 16
                break
        line += t
    if line and y <= hgt * 0.42:
        dr.text((12, y), line, fill="#444", font=fm)
        y += 16
    y += 12
    dr.line([8, y, w - 8, y], fill="#ccc", width=1)
    y += 10
    dr.text((10, y), "cells[] — 내용 + 좌표", fill="#c00000", font=fs)
    y += 20
    for idx, c in enumerate(h["cells"]):
        if y > hgt - 22:
            dr.text((12, y), f"… 외 {len(h['cells'])-idx}개", fill="#888", font=fm)
            break
        head = kinds[idx][0] if idx < len(kinds) else False
        span = kinds[idx][1] if idx < len(kinds) else ""
        txt = "".join(c.get("tokens", []))
        b = c.get("bbox")
        col = C_HEAD if head else (C_EMPTY if not b else "#333")
        tag = f"[{idx}] <td{(' ' + span) if span else ''}>" + (" thead" if head else "")
        dr.text((12, y), tag, fill=col, font=fm)
        body = (txt[:34] + "…") if len(txt) > 34 else txt
        dr.text((132, y), body if body.strip() else "(빈 셀)", fill=col, font=fm)
        dr.text((w - 178, y), str(b) if b else "bbox 없음", fill=col, font=fm)
        y += 16
    dr.rectangle([0, 0, w - 1, hgt - 1], outline="#bbb", width=2)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="000000,366212")
    ap.add_argument("--width", type=int, default=780, help="표 한 장의 가로 픽셀")
    ap.add_argument("--panel", action="store_true", help="구조 토큰·셀 목록 패널을 아래 붙인다")
    ap.add_argument("--out", default="/data/workspace/yjyong/receipt_data/review/pubtabnet_format.png")
    args = ap.parse_args()

    cells_img = []
    for did in [x.strip() for x in args.ids.split(",") if x.strip()]:
        d, h = load(did)
        img, n_box, n_nobox, kinds = draw_boxes(did, h, args.width)
        bar = 60
        parts = [img]
        if args.panel:
            ph = 46 + 16 * min(len(h["cells"]) + 6, 40)
            parts.append(panel(did, h, kinds, img.width, ph, n_box, n_nobox))
        hh = bar + sum(p.height for p in parts) + 8 * (len(parts) - 1)
        cell = Image.new("RGB", (img.width, hh), "white")
        y = bar
        for p in parts:
            cell.paste(p, (0, y))
            y += p.height + 8
        dr = ImageDraw.Draw(cell)
        dr.text((6, 6), did, fill="black", font=ImageFont.truetype(FONT, 21))
        dr.text((6, 33), f"셀 {len(h['cells'])}개 · bbox 있음 {n_box} / 없음 {n_nobox}",
                fill="#c00000", font=ImageFont.truetype(FONT, 17))
        cells_img.append(cell)

    pad = 18
    W = sum(c.width for c in cells_img) + pad * (len(cells_img) - 1)
    H = max(c.height for c in cells_img)
    out = Image.new("RGB", (W, H), "white")
    x = 0
    for c in cells_img:
        out.paste(c, (x, 0))
        x += c.width + pad
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"저장: {args.out}  {out.size}")


if __name__ == "__main__":
    main()
