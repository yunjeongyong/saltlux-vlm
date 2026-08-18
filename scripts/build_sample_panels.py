"""라벨링·정답·추론을 한 장에 담은 패널을 만든다.

세 가지를 따로 보면 대조가 안 된다. 왼쪽에 이미지(크롭이면 원본에서의 위치까지),
오른쪽에 정답과 모델 출력을 위아래로 놓아 한눈에 비교되게 한다.

usage: python3 scripts/build_sample_panels.py <images폴더>
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "receipt_data"
IMG = Path(sys.argv[1])
# PaddleOCR 동봉 폰트는 필기체라 본문에 쓰면 읽기 어렵다. 나눔바른고딕으로 간다.
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf"

PAPER = (246, 246, 243)
CARD = (255, 255, 255)
INK = (25, 27, 30)
DIM = (99, 104, 110)
RULE = (223, 223, 217)
TX = (27, 175, 122)
TB = (235, 104, 52)
GT_C = (44, 86, 120)
PR_C = (31, 107, 74)

W = 1500
PAD = 26
IMGW = 620            # 왼쪽 이미지 열 폭
GAP = 24

F_TITLE = ImageFont.truetype(FONT_BOLD, 26)
F_META = ImageFont.truetype(FONT_PATH, 15)
F_LABEL = ImageFont.truetype(FONT_BOLD, 17)
F_BODY = ImageFont.truetype(FONT_PATH, 16)


def wrap(draw, text, font, width):
    """한글은 단어 경계가 없어 글자 단위로 접는다. 줄바꿈은 그대로 살린다."""
    lines = []
    for raw in text.split("\n"):
        if not raw:
            lines.append("")
            continue
        cur = ""
        for ch in raw:
            if draw.textlength(cur + ch, font=font) > width and cur:
                lines.append(cur)
                cur = ch
            else:
                cur += ch
        lines.append(cur)
    return lines


def clean(s, limit=1400, is_html=False):
    """정답과 추론을 같은 규칙으로 다듬는다.

    HTML 표는 한쪽만 태그마다 줄바꿈이 들어 있어 그대로 두면 길이가 3배 차이 나
    비교가 안 된다. 태그 사이 공백을 걷어내 양쪽을 같은 모양으로 만든다.
    """
    s = "".join("□" if c == "�" or (ord(c) < 32 and c != "\n") else c
                for c in s)
    if is_html:
        s = re.sub(r">\s+<", "><", s)
        s = re.sub(r"\s+", " ", s).strip()
    else:
        s = re.sub(r"\n{3,}", "\n\n", s)
    return s if len(s) <= limit else s[:limit] + " …(이하 생략)"


def page_of(doc_id):
    for split in ("train", "val", "test"):
        for ext in (".jpg", ".png", ".jpeg"):
            p = RD / f"labeled/{split}/images/{doc_id}{ext}"
            if p.exists():
                return p, RD / f"labeled/{split}/json/{doc_id}.json"
    return None, None


def crop_manifest():
    m = {}
    p = RD / "train_crops/manifest.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                r = json.loads(line)
                m[r["crop_id"]] = r
    return m


MAN = crop_manifest()


def left_column(src_path, crop_id, color):
    """이미지 열을 만든다. 크롭이면 원본(bbox 표시) 위, 크롭 아래로 쌓는다."""
    im = Image.open(src_path).convert("RGB")
    meta = MAN.get(crop_id) if crop_id else None
    parts = []
    if meta:
        doc = crop_id.replace("train_", "").rsplit("_", 1)[0]
        page, jp = page_of(doc)
        if page and jp.exists():
            pg = Image.open(page).convert("RGB")
            PW, PH = pg.size
            pj = json.loads(jp.read_text(encoding="utf-8"))["result"]["elements"][0]["json"]
            if isinstance(pj, str):
                pj = json.loads(pj)
            sx, sy = PW / pj["width"], PH / pj["height"]
            bb = [meta["bbox"][0] * sx, meta["bbox"][1] * sy,
                  meta["bbox"][2] * sx, meta["bbox"][3] * sy]
            ImageDraw.Draw(pg).rectangle(bb, outline=color,
                                         width=max(4, int(PW / 180)))
            parts.append(("원본에서의 위치 (라벨링 bbox)", pg))
    parts.append(("학습에 들어간 이미지", im))

    tiles = []
    for cap, pic in parts:
        w = IMGW
        h = max(1, int(pic.height * w / pic.width))
        h = min(h, 560)
        w2 = int(pic.width * h / pic.height)
        w2 = min(w2, IMGW)
        h = max(1, int(pic.height * w2 / pic.width))
        tiles.append((cap, pic.resize((w2, h), Image.LANCZOS)))
    return tiles


def build(rec, idx):
    task = rec["task"]
    color = TB if "table" in task else TX
    src = ROOT / rec.get("src", "")
    if not src.exists():
        src = IMG / rec["file"]
    crop_id = None
    if task.startswith("receipt_crop"):
        crop_id = Path(rec.get("src", "")).stem
    tiles = left_column(src, crop_id, color)

    scratch = Image.new("RGB", (10, 10))
    d0 = ImageDraw.Draw(scratch)
    tw = W - IMGW - GAP - PAD * 2
    is_html = "<table" in rec["gt"].lower()
    gt_lines = wrap(d0, clean(rec["gt"], is_html=is_html), F_BODY, tw - 24)
    pr_lines = wrap(d0, clean(rec["pred"], is_html=is_html), F_BODY, tw - 24)

    lh = 23
    head_h = 96
    img_h = sum(t[1].height + 26 for t in tiles)
    txt_h = (34 + len(gt_lines) * lh + 22) + (34 + len(pr_lines) * lh + 22) + 18
    H = PAD * 2 + head_h + max(img_h, txt_h) + 16

    cv = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(cv)

    # 머리
    d.text((PAD, PAD), f"{idx:02d}  {task}", font=F_TITLE, fill=INK)
    d.text((PAD, PAD + 38),
           f"{rec['file']}   ·   원본 {rec['size'][0]}×{rec['size'][1]}"
           f"   ·   추론 {rec['sec']}초", font=F_META, fill=DIM)
    d.line([(PAD, PAD + 68), (W - PAD, PAD + 68)], fill=INK, width=2)

    # 왼쪽 이미지
    y = PAD + head_h
    for cap, pic in tiles:
        d.text((PAD, y - 20), cap, font=F_META, fill=DIM)
        cv.paste(pic, (PAD, y))
        d.rectangle([PAD - 1, y - 1, PAD + pic.width, y + pic.height],
                    outline=RULE, width=1)
        y += pic.height + 26

    # 오른쪽 텍스트 두 칸
    x = PAD + IMGW + GAP
    ty = PAD + head_h
    for label, lines, c in (("정답 (GT)", gt_lines, GT_C),
                            ("모델 추론 (exp_003)", pr_lines, PR_C)):
        bh = 34 + len(lines) * lh + 22
        d.rectangle([x, ty, W - PAD, ty + bh], fill=CARD, outline=RULE, width=1)
        d.rectangle([x, ty, x + 4, ty + bh], fill=c)
        d.text((x + 16, ty + 9), label, font=F_LABEL, fill=c)
        yy = ty + 38
        for ln in lines:
            d.text((x + 16, yy), ln, font=F_BODY, fill=INK)
            yy += lh
        ty += bh + 18

    out = IMG / f"panel_{rec['file']}"
    cv.save(out, optimize=True)
    return out


def main():
    recs = json.loads((IMG / "_pred.json").read_text(encoding="utf-8"))
    for i, r in enumerate(recs, 1):
        p = build(r, i)
        print(f"  {p.name:44} {Image.open(p).size}")
    print(f"\n{len(recs)}장 생성 → {IMG}")


if __name__ == "__main__":
    main()
