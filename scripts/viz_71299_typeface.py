"""
71299 의 typeface 값이 무엇을 뜻하는지 크롭을 잘라 눈으로 확인한다.

라벨에 typeface 가 1~4 로 들어있는데 그 뜻이 배포 문서에 없다. 손글씨를 빼려면
어느 값이 손글씨인지부터 알아야 하므로, 값별로 실제 글자 크롭을 뽑아 나열한다.

writing_style(Images 안의 값) 도 같이 본다 — 장 단위 속성이라 이쪽이 더 굵은
필터가 될 수 있다.

usage:
    python3 scripts/viz_71299_typeface.py
    python3 scripts/viz_71299_typeface.py --per 14
"""
import argparse
import collections
import glob
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/data/workspace/yjyong/aihub/71299/extracted")
FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"
CAT = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
INK, MUTED, RULE = "#14171c", "#5d6674", "#dce0e6"


def collect(per, seed, pad=4, min_w=40):
    """typeface 값별로 크롭 이미지를 모은다."""
    lab = {Path(p).stem: p for p in glob.glob(str(ROOT / "VL_*" / "*.json"))}
    imgs = glob.glob(str(ROOT / "VS_*" / "*.jpg"))
    rng = random.Random(seed)
    rng.shuffle(imgs)
    bucket = collections.defaultdict(list)
    style = collections.Counter()
    for ip in imgs:
        if all(len(bucket[k]) >= per for k in (1, 2, 3, 4)):
            break
        lp = lab.get(Path(ip).stem)
        if not lp:
            continue
        d = json.loads(Path(lp).read_text(encoding="utf-8"))
        boxes = d.get("Bbox") or []
        ws = d.get("Images", {}).get("writing_style")
        want = [b for b in boxes if len(bucket[b.get("typeface")]) < per]
        if not want:
            continue
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        for b in want:
            tf = b.get("typeface")
            try:
                x1, y1 = max(0, min(b["x"]) - pad), max(0, min(b["y"]) - pad)
                x2, y2 = min(W, max(b["x"]) + pad), min(H, max(b["y"]) + pad)
            except Exception:
                continue
            if x2 - x1 < min_w or y2 - y1 < 12:
                continue
            bucket[tf].append((im.crop((x1, y1, x2, y2)), str(b.get("data", "")),
                               Path(ip).stem, ws))
            style[(tf, ws)] += 1
    return bucket, style


def row(crops, tf, w, ch=52):
    """한 typeface 의 크롭들을 가로로 이어붙인다."""
    cells = []
    f = ImageFont.truetype(FONT, 12)
    for im, txt, stem, ws in crops:
        s = ch / im.height
        im2 = im.resize((max(1, int(im.width * s)), ch))
        cw = min(im2.width, 260)
        cell = Image.new("RGB", (cw + 8, ch + 22), "white")
        cell.paste(im2.crop((0, 0, cw, ch)), (4, 0))
        d = ImageDraw.Draw(cell)
        d.rectangle([4, 0, 4 + cw - 1, ch - 1], outline=CAT.get(tf, "#999"))
        d.text((5, ch + 2), (txt[:14] or "·"), fill=MUTED, font=f)
        cells.append(cell)
    if not cells:
        return None
    # 폭에 맞춰 여러 줄로
    lines, cur, cw = [], [], 0
    for c in cells:
        if cw + c.width > w - 130:
            lines.append(cur)
            cur, cw = [], 0
        cur.append(c)
        cw += c.width + 6
    if cur:
        lines.append(cur)
    H = sum(max(c.height for c in ln) + 6 for ln in lines)
    im = Image.new("RGB", (w, H), "white")
    y = 0
    for ln in lines:
        x = 130
        for c in ln:
            im.paste(c, (x, y))
            x += c.width + 6
        y += max(c.height for c in ln) + 6
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=12)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--out", default="/data/workspace/yjyong/receipt_data/review/aihub_71299_typeface.png")
    args = ap.parse_args()

    bucket, style = collect(args.per, args.seed)
    rows = []
    for tf in (1, 2, 3, 4):
        r = row(bucket.get(tf, []), tf, args.width)
        if r is not None:
            rows.append((tf, r, len(bucket[tf])))

    head = 86
    H = head + sum(r.height + 34 for _, r, _ in rows) + 110
    out = Image.new("RGB", (args.width, H), "white")
    dr = ImageDraw.Draw(out)
    dr.text((14, 12), "71299 typeface 값별 실제 크롭 — 무엇이 손글씨인가", fill=INK,
            font=ImageFont.truetype(FONT, 24))
    dr.text((14, 48), "배포 문서에 값의 뜻이 없어 직접 잘라 확인한다. 아래 글자는 라벨 텍스트.",
            fill=MUTED, font=ImageFont.truetype(FONT, 15))
    dr.line([14, head - 12, args.width - 14, head - 12], fill=RULE, width=2)

    y = head
    share = {1: 80, 3: 11, 2: 9, 4: 0.4}
    for tf, r, n in rows:
        dr.rectangle([14, y, 120, y + 26], fill=CAT.get(tf, "#999"))
        dr.text((22, y + 3), f"typeface {tf}", fill="white",
                font=ImageFont.truetype(FONT, 17))
        dr.text((14, y + 30), f"전체의 {share.get(tf, 0)}%", fill=MUTED,
                font=ImageFont.truetype(FONT, 13))
        out.paste(r, (0, y))
        y += r.height + 34

    dr.line([14, y, args.width - 14, y], fill=RULE, width=2)
    y += 12
    dr.text((14, y), "장 단위 속성 writing_style 분포 (표본 1,500장)", fill=INK,
            font=ImageFont.truetype(FONT, 16))
    y += 24
    dr.text((16, y), "1: 4장 · 2: 1장 · 3: 1,495장(99.7%) — 거의 한 값이라 필터로 못 쓴다",
            fill=MUTED, font=ImageFont.truetype(FONT, 14))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"저장: {args.out}  {out.size}")
    print("typeface x writing_style:", dict(style))


if __name__ == "__main__":
    main()
