"""미학습 샘플 패널을 그린다.

build_sample_panels.py 와 골격은 같되 두 가지가 다르다.
  · 캡션이 "학습에 들어간 이미지" 가 아니라 미학습 근거를 그대로 찍는다
  · 크롭 원본 위치는 train_crops 가 아니라 평가셋(eval_crops) 좌표로 그린다

usage: python3 scripts/build_unseen_panels.py <폴더>
"""
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "receipt_data"
IMG = Path(sys.argv[1])
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
BADGE = (157, 58, 39)

W = 1500
PAD = 26
IMGW = 620
GAP = 24

F_TITLE = ImageFont.truetype(FONT_BOLD, 26)
F_META = ImageFont.truetype(FONT_PATH, 15)
F_BADGE = ImageFont.truetype(FONT_BOLD, 15)
F_LABEL = ImageFont.truetype(FONT_BOLD, 17)
F_BODY = ImageFont.truetype(FONT_PATH, 16)


def wrap(draw, text, font, width):
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
    s = "".join("□" if c == "�" or (ord(c) < 32 and c != "\n") else c
                for c in s)
    if is_html:
        s = re.sub(r">\s+<", "><", s)
        s = re.sub(r"\s+", " ", s).strip()
    else:
        s = re.sub(r"\n{3,}", "\n\n", s)
    return s if len(s) <= limit else s[:limit] + " …(이하 생략)"


def is_transposed(W_, H_, pw, ph, tol=0.12):
    ar_i, ar_j = W_ / H_, pw / ph
    if abs(ar_i - ar_j) < 0.3:
        return False
    return abs(ar_i - 1 / ar_j) < tol * max(1.0, ar_i)


def page_of(doc_id, split):
    for ext in (".jpg", ".png", ".jpeg"):
        p = RD / f"labeled/{split}/images/{doc_id}{ext}"
        if p.exists():
            return p, RD / f"labeled/{split}/json/{doc_id}.json"
    return None, None


def page_with_bbox(crop, color):
    """평가셋 크롭이 원본 어디서 나왔는지 표시한다."""
    page, jp = page_of(crop["doc_id"], crop["split"])
    if not page or not jp.exists():
        return None
    pg = Image.open(page).convert("RGB")
    pj = json.loads(jp.read_text(encoding="utf-8"))["result"]["elements"][0]["json"]
    if isinstance(pj, str):
        pj = json.loads(pj)
    PW, PH = pg.size
    pw, ph = pj["width"], pj["height"]
    if is_transposed(PW, PH, pw, ph):        # 좌표계가 90도 돌아간 장
        pg = pg.transpose(Image.ROTATE_270)
        PW, PH = pg.size
    sx, sy = PW / pw, PH / ph
    x1, y1, x2, y2 = crop["bbox"]
    ImageDraw.Draw(pg).rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                                 outline=color, width=max(4, int(PW / 180)))
    return pg


def left_column(rec, color):
    parts = []
    if rec.get("crop"):
        pg = page_with_bbox(rec["crop"], color)
        if pg:
            parts.append(("원본에서의 위치 (평가셋 라벨링 bbox)", pg))
        parts.append(("평가에 쓴 크롭 — 학습 미사용",
                      Image.open(ROOT / rec["src"]).convert("RGB")))
    else:
        parts.append(("학습에 쓰지 않은 이미지",
                      Image.open(ROOT / rec["src"]).convert("RGB")))

    tiles = []
    for cap, pic in parts:
        h = min(max(1, int(pic.height * IMGW / pic.width)), 560)
        w2 = min(int(pic.width * h / pic.height), IMGW)
        h = max(1, int(pic.height * w2 / pic.width))
        tiles.append((cap, pic.resize((w2, h), Image.LANCZOS)))
    return tiles


def build(rec, idx):
    task = rec["task"]
    color = TB if "table" in task else TX
    tiles = left_column(rec, color)

    scratch = Image.new("RGB", (10, 10))
    d0 = ImageDraw.Draw(scratch)
    tw = W - IMGW - GAP - PAD * 2
    is_html = "<table" in rec["gt"].lower()
    gt_lines = wrap(d0, clean(rec["gt"], is_html=is_html), F_BODY, tw - 24)
    pr_lines = wrap(d0, clean(rec["pred"], is_html=is_html), F_BODY, tw - 24)

    lh = 23
    head_h = 140                       # 머리글 3줄 + 구분선 + 캡션 자리
    img_h = sum(t[1].height + 26 for t in tiles)
    txt_h = (34 + len(gt_lines) * lh + 22) + (34 + len(pr_lines) * lh + 22) + 18
    H = PAD * 2 + head_h + max(img_h, txt_h) + 16

    cv = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(cv)

    d.text((PAD, PAD), f"{idx:02d}  {task}", font=F_TITLE, fill=INK)
    sc = (f"   ·   {rec['metric']} {rec['score']:.4f}"
          if rec.get("metric") else "")
    d.text((PAD, PAD + 38),
           f"{rec['file']}   ·   원본 {rec['size'][0]}×{rec['size'][1]}"
           f"   ·   추론 {rec['sec']}초{sc}", font=F_META, fill=DIM)
    # 미학습 근거와 정답 출처는 패널 안에 남긴다 — 떼어놓으면 다시 오해가 생긴다
    d.text((PAD, PAD + 60), f"▪ {rec['note']}", font=F_BADGE, fill=BADGE)
    d.text((PAD, PAD + 80), f"▪ 정답 출처: {rec['gt_src']}", font=F_META, fill=DIM)
    d.line([(PAD, PAD + 104), (W - PAD, PAD + 104)], fill=INK, width=2)

    y = PAD + head_h
    for cap, pic in tiles:
        d.text((PAD, y - 20), cap, font=F_META, fill=DIM)
        cv.paste(pic, (PAD, y))
        d.rectangle([PAD - 1, y - 1, PAD + pic.width, y + pic.height],
                    outline=RULE, width=1)
        y += pic.height + 26

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
    src = IMG / "_pred_best.json"
    if not src.exists():
        src = IMG / "_pred.json"
    recs = json.loads(src.read_text(encoding="utf-8"))
    for i, r in enumerate(recs, 1):
        p = build(r, i)
        print(f"  {p.name:40} {Image.open(p).size}")
    print(f"\n{len(recs)}장 생성 → {IMG}")


if __name__ == "__main__":
    main()
