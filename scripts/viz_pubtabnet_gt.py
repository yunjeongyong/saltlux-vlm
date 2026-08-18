"""
PubTabNet-HTML 의 이미지와 GT 표를 나란히 놓고 대조한다.

이 데이터의 GT 는 완성된 표가 아니라 두 조각으로 흩어져 있다.

    structure.tokens   <thead><tr><td colspan="2">... 같은 뼈대 토큰 열
    cells[].tokens     ['K','i','n','e','t','i','c',...] 문자 단위 셀 내용

둘을 합쳐야 비로소 표가 된다. "합쳐지긴 하는가"를 눈으로 확인하려고,
토큰을 실제 격자로 복원해 이미지 옆에 그린다. 왼쪽 사진과 오른쪽 격자가
같은 표로 보이면 GT 가 제대로 맞물린 것이다.

colspan/rowspan 을 반영하고, <thead> 구역은 색을 달리한다.
(이 데이터에는 <th> 셀이 없다. 헤더는 <thead> 구역으로만 구분된다.)

usage:
    python3 scripts/viz_pubtabnet_gt.py --ids 000000,000008
"""
import argparse
import ast
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DS = Path("/data/workspace/VLM/dataset/train/public/pubtabnet-html")
FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"

HEAD_BG = "#fde8e8"
HEAD_FG = "#a01d1d"
BODY_FG = "#1a1a1a"
EMPTY_BG = "#f2f2f2"
GRID = "#999999"


def load(did):
    d = json.loads((DS / "data" / f"{did}.json").read_text(encoding="utf-8"))
    return d, ast.literal_eval(d["html"])


def cell_text(tokens):
    """셀 토큰(문자 단위 + 인라인 태그)을 표시용 문자열로. 태그는 걷어낸다."""
    s = "".join(tokens or [])
    s = re.sub(r"</?(b|i|sup|sub|em|strong|u|small)>", "", s)
    return re.sub(r"\s+", " ", s).strip()


def build_grid(struct_tokens, cells):
    """구조 토큰을 격자로 복원한다. 반환: rows[r][c] = (text, is_head, is_origin)

    colspan/rowspan 이 덮는 칸은 is_origin=False 로 두고 내용을 비운다.
    (마크다운 표로 내릴 때 빈 칸이 되는 자리와 같다)
    """
    # 1) 행별로 (colspan, rowspan, is_head, cell_index) 뽑기
    rows_spec = []
    cur = None
    in_head = False
    ci = 0
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
        elif t == "<tr>":
            cur = []
            i += 1
        elif t == "</tr>":
            rows_spec.append(cur or [])
            cur = None
            i += 1
        elif t in ("<td>", "<th>"):
            (cur if cur is not None else []).append((1, 1, in_head, ci))
            ci += 1
            i += 1
        elif t in ("<td", "<th"):
            attr = ""
            j = i + 1
            while j < len(T) and T[j] != ">":
                attr += T[j]
                j += 1
            cs = int((re.search(r'colspan="(\d+)"', attr) or [0, 1])[1]) if "colspan" in attr else 1
            rs = int((re.search(r'rowspan="(\d+)"', attr) or [0, 1])[1]) if "rowspan" in attr else 1
            (cur if cur is not None else []).append((cs, rs, in_head, ci))
            ci += 1
            i = j + 1
        else:
            i += 1

    # 2) 열 수 = 첫 행들의 colspan 합 최대값
    n_col = max((sum(c[0] for c in r) for r in rows_spec), default=1)
    n_row = len(rows_spec)
    grid = [[None] * n_col for _ in range(n_row)]

    for r, spec in enumerate(rows_spec):
        c = 0
        for cs, rs, head, idx in spec:
            while c < n_col and grid[r][c] is not None:
                c += 1
            if c >= n_col:
                break
            txt = cell_text(cells[idx].get("tokens")) if idx < len(cells) else ""
            grid[r][c] = (txt, head, True)
            for dr_ in range(rs):
                for dc in range(cs):
                    if dr_ == 0 and dc == 0:
                        continue
                    rr, cc = r + dr_, c + dc
                    if rr < n_row and cc < n_col:
                        grid[rr][cc] = ("", head, False)
            c += cs
    for r in range(n_row):
        for c in range(n_col):
            if grid[r][c] is None:
                grid[r][c] = ("", False, True)
    return grid


def wrap(dr, text, font, w):
    if not text:
        return [""]
    out, line = [], ""
    for ch in text:
        if dr.textlength(line + ch, font=font) > w - 8:
            out.append(line)
            line = ch
        else:
            line += ch
    out.append(line)
    return out[:4]


def draw_grid(grid, width, title):
    f = ImageFont.truetype(FONT, 13)
    ft = ImageFont.truetype(FONT, 17)
    n_row, n_col = len(grid), len(grid[0])
    cw = width // n_col
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    heights = []
    for r in range(n_row):
        mx = 1
        for c in range(n_col):
            mx = max(mx, len(wrap(probe, grid[r][c][0], f, cw)))
        heights.append(mx * 17 + 8)
    top = 28
    im = Image.new("RGB", (cw * n_col + 1, top + sum(heights) + 1), "white")
    dr = ImageDraw.Draw(im)
    dr.text((2, 4), title, fill="#c00000", font=ft)
    y = top
    for r in range(n_row):
        x = 0
        for c in range(n_col):
            txt, head, origin = grid[r][c]
            bg = HEAD_BG if head else (EMPTY_BG if not origin else "white")
            dr.rectangle([x, y, x + cw, y + heights[r]], fill=bg, outline=GRID)
            ty = y + 4
            for ln in wrap(dr, txt, f, cw):
                dr.text((x + 4, ty), ln, fill=HEAD_FG if head else BODY_FG, font=f)
                ty += 17
            x += cw
        y += heights[r]
    return im


def draw_image(did, h, width):
    im = Image.open(DS / "data" / f"{did}.jpg").convert("RGB")
    W, H = im.size
    s = width / W
    im = im.resize((width, max(1, int(H * s))))
    dr = ImageDraw.Draw(im)
    for c in h["cells"]:
        b = c.get("bbox")
        if not b:
            continue
        dr.rectangle([v * s for v in b], outline="#1f77b4", width=1)
    out = Image.new("RGB", (width, im.height + 28), "white")
    out.paste(im, (0, 28))
    ImageDraw.Draw(out).text((2, 4), "이미지 + cells[].bbox", fill="#1f77b4",
                             font=ImageFont.truetype(FONT, 17))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="000000,000008")
    ap.add_argument("--width", type=int, default=880)
    ap.add_argument("--out", default="/data/workspace/yjyong/receipt_data/review/pubtabnet_gt.png")
    args = ap.parse_args()

    blocks = []
    for did in [x.strip() for x in args.ids.split(",") if x.strip()]:
        d, h = load(did)
        grid = build_grid(h["structure"]["tokens"], h["cells"])
        nb = sum(1 for c in h["cells"] if c.get("bbox"))
        img = draw_image(did, h, args.width)
        gd = draw_grid(grid, args.width,
                       f"GT 표 복원 — {len(grid)}행 x {len(grid[0])}열 "
                       f"(분홍=thead / 회색=병합에 덮인 칸)")
        bar = 56
        cell = Image.new("RGB", (args.width, bar + img.height + 10 + gd.height), "white")
        cell.paste(img, (0, bar))
        cell.paste(gd, (0, bar + img.height + 10))
        dr = ImageDraw.Draw(cell)
        dr.text((4, 5), did, fill="black", font=ImageFont.truetype(FONT, 21))
        dr.text((4, 31), f"셀 {len(h['cells'])}개 · bbox 있음 {nb} / 없음 {len(h['cells'])-nb}",
                fill="#555", font=ImageFont.truetype(FONT, 16))
        blocks.append(cell)

    pad = 20
    W = sum(b.width for b in blocks) + pad * (len(blocks) - 1)
    H = max(b.height for b in blocks)
    out = Image.new("RGB", (W, H), "white")
    x = 0
    for b in blocks:
        out.paste(b, (x, 0))
        x += b.width + pad
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"저장: {args.out}  {out.size}")


if __name__ == "__main__":
    main()
