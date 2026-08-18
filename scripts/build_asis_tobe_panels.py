"""AS-IS(베이스) / TO-BE(학습 모델) 짝비교 패널.

같은 크롭에 대해 베이스가 무엇을 틀렸고 학습 모델이 그걸 어떻게 고쳤는지를
한 장에 담는다. 이미지는 전부 평가셋(978건 / 90장) — 학습에 쓰지 않았다.

  AS-IS  luxia-document-parsing-high (사내 서빙 베이스)
  TO-BE  upload-mix-qwen36/checkpoint-1000 (재채점 결과 가장 좋은 학습 모델)

usage: python3 scripts/build_asis_tobe_panels.py <출력폴더>
"""
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "receipt_data"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

SCORED = RD / "review/asis_tobe.json"
A = "베이스(luxia)"
B = "학습 ckpt1000"

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
BAD = (157, 58, 39)
GOOD = (31, 107, 74)

W = 1560
PAD = 26
IMGW = 560
GAP = 24

F_TITLE = ImageFont.truetype(FONT_BOLD, 26)
F_META = ImageFont.truetype(FONT_PATH, 15)
F_BADGE = ImageFont.truetype(FONT_BOLD, 16)
F_LABEL = ImageFont.truetype(FONT_BOLD, 17)
F_BODY = ImageFont.truetype(FONT_PATH, 16)

# 유형마다 두 건씩. crop_id 를 직접 지정한다 — 조건식으로 뽑으면 다음에 데이터가
# 바뀔 때 조용히 다른 게 들어와 보고서와 어긋난다.
PICKS = [
    ("이웃 줄 혼입", "크롭 밖 이웃 줄까지 덧붙여 읽는다 — 긴 오답(CER>1)의 주된 형태",
     ["val_receipt56_9", "test_receipt05_6"],
     "※ 크롭 패딩(pad_ratio 0.02)으로 이웃 줄이 화면에 걸쳐 있다. "
     "학습 모델은 '크롭 하나 = 한 덩어리'를 지켜 정답과 맞고, "
     "순수 인식력 향상과는 구분해서 읽어야 한다."),
    ("서식 기호 삽입", "숫자를 수식 기호로 감싸 원문에 없는 문자가 섞인다",
     ["test_receipt25_37", "val_receipt49_11"], ""),
    # 글자 오독 유형은 넣지 않았다. 후보 5건뿐이고 전부 한두 글자 차이인 데다
    # 크롭이 흐려 사람 눈으로도 판정이 안 된다 — 근거로 쓸 수 없다.
    # 베이스의 실패는 글자를 잘못 읽는 쪽이 아니라 범위·구조 쪽에 몰려 있다.
    ("표 출력 절단", "표를 끝까지 내지 못하고 중간에서 잘린다 — 닫는 태그가 없어 TEDS 가 0",
     ["test_receipt65_11"],
     "※ AS-IS 출력 1,708자에 </table> 가 없다(행 중간에서 끊김). "
     "TO-BE 는 1,803자로 표를 닫는다 — 정답 1,823자."),
    ("표 행 누락", "표는 닫지만 품목 절반을 빠뜨린다",
     ["test_receipt20_10"],
     "※ 양쪽 다 </table> 까지 낸다. 베이스는 11행 중 6행만 냈고, "
     "TO-BE 는 11행을 모두 낸다(정답과 동일)."),
]


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


def clean(s, limit=900, is_html=False):
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


def page_with_bbox(rec, color):
    page, jp = page_of(rec["doc_id"], rec["split"])
    if not page or not jp.exists():
        return None
    pg = Image.open(page).convert("RGB")
    pj = json.loads(jp.read_text(encoding="utf-8"))["result"]["elements"][0]["json"]
    if isinstance(pj, str):
        pj = json.loads(pj)
    PW, PH = pg.size
    pw, ph = pj["width"], pj["height"]
    if is_transposed(PW, PH, pw, ph):
        pg = pg.transpose(Image.ROTATE_270)
        PW, PH = pg.size
    sx, sy = PW / pw, PH / ph
    x1, y1, x2, y2 = rec["bbox"]
    ImageDraw.Draw(pg).rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                                 outline=color, width=max(4, int(PW / 180)))
    return pg


def tiles_of(rec, color):
    parts = []
    pg = page_with_bbox(rec, color)
    if pg:
        parts.append(("원본에서의 위치 (평가셋 bbox)", pg))
    parts.append(("평가에 쓴 크롭 — 학습 미사용",
                  Image.open(RD / "eval_crops" / rec["image"]).convert("RGB")))
    out = []
    for cap, pic in parts:
        h = min(max(1, int(pic.height * IMGW / pic.width)), 520)
        w2 = min(int(pic.width * h / pic.height), IMGW)
        h = max(1, int(pic.height * w2 / pic.width))
        out.append((cap, pic.resize((w2, h), Image.LANCZOS)))
    return out


def metric(rec, key):
    if rec["task"] == "table":
        return "TEDS", rec[key]["teds"], False
    return "CER", rec[key]["cer"], True


def build(rec, idx, kind, why, caveat=""):
    color = TB if rec["task"] == "table" else TX
    tiles = tiles_of(rec, color)
    m, sa, lower = metric(rec, A)
    _, sb, _ = metric(rec, B)
    delta = (sa - sb) if lower else (sb - sa)

    scratch = Image.new("RGB", (10, 10))
    d0 = ImageDraw.Draw(scratch)
    tw = W - IMGW - GAP - PAD * 2
    html = rec["task"] == "table"
    cards = [
        ("정답 (GT) — 검수 정정본", clean(rec["gt"], is_html=html), GT_C, ""),
        (f"AS-IS  베이스 모델 (luxia)", clean(rec[A]["pred"], is_html=html),
         BAD, f"{m} {sa:.4f}"),
        (f"TO-BE  학습 모델 (ckpt-1000)", clean(rec[B]["pred"], is_html=html),
         GOOD, f"{m} {sb:.4f}"),
    ]
    wrapped = [(t, wrap(d0, body, F_BODY, tw - 24), c, s) for t, body, c, s in cards]

    scratch2 = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    cav = wrap(scratch2, caveat, F_META, W - PAD * 2) if caveat else []

    lh = 23
    head_h = 142 + len(cav) * 20
    img_h = sum(t[1].height + 26 for t in tiles)
    txt_h = sum(34 + len(w) * lh + 22 + 16 for _, w, _, _ in wrapped)
    H = PAD * 2 + head_h + max(img_h, txt_h) + 16

    cv = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(cv)

    d.text((PAD, PAD), f"{idx:02d}  {kind}", font=F_TITLE, fill=INK)
    d.text((PAD, PAD + 38), why, font=F_META, fill=DIM)
    d.text((PAD, PAD + 62),
           f"▪ 평가셋 {rec['split']} · {rec['crop_id']} · 학습 미사용   "
           f"·   {rec['label']}", font=F_BADGE, fill=BAD)
    arrow = "↓" if lower else "↑"
    d.text((PAD, PAD + 86),
           f"▪ {m} {sa:.4f} → {sb:.4f}  ({arrow} {abs(delta):.4f} 개선)",
           font=F_BADGE, fill=GOOD)
    yy = PAD + 110
    for ln_ in cav:
        d.text((PAD, yy), ln_, font=F_META, fill=DIM)
        yy += 20
    d.line([(PAD, yy + 2), (W - PAD, yy + 2)], fill=INK, width=2)

    y = PAD + head_h
    for cap, pic in tiles:
        d.text((PAD, y - 20), cap, font=F_META, fill=DIM)
        cv.paste(pic, (PAD, y))
        d.rectangle([PAD - 1, y - 1, PAD + pic.width, y + pic.height],
                    outline=RULE, width=1)
        y += pic.height + 26

    x = PAD + IMGW + GAP
    ty = PAD + head_h
    for label, lines, c, score in wrapped:
        bh = 34 + len(lines) * lh + 22
        d.rectangle([x, ty, W - PAD, ty + bh], fill=CARD, outline=RULE, width=1)
        d.rectangle([x, ty, x + 4, ty + bh], fill=c)
        d.text((x + 16, ty + 9), label, font=F_LABEL, fill=c)
        if score:
            d.text((W - PAD - 16 - d0.textlength(score, font=F_LABEL), ty + 9),
                   score, font=F_LABEL, fill=c)
        yy = ty + 38
        for ln in lines:
            d.text((x + 16, yy), ln, font=F_BODY, fill=INK)
            yy += lh
        ty += bh + 16

    # 마크다운 링크에 넣으므로 공백을 쓰지 않는다
    p = OUT / f"panel_{idx:02d}_{kind.replace(' ', '')}_{rec['crop_id']}.png"
    cv.save(p, optimize=True)
    return p, m, sa, sb


def main():
    rows = {r["crop_id"]: r for r in json.loads(
        SCORED.read_text(encoding="utf-8"))["rows"]}
    idx = 0
    made = []
    for kind, why, ids, caveat in PICKS:
        for cid in ids:
            rec = rows.get(cid)
            if not rec or A not in rec or B not in rec:
                print(f"  ✗ {cid} 없음")
                continue
            idx += 1
            p, m, sa, sb = build(rec, idx, kind, why, caveat)
            made.append((kind, cid, m, sa, sb))
            print(f"  {p.name:52} {m} {sa:.4f} → {sb:.4f}")
    print(f"\n{len(made)}장 생성 → {OUT}")


if __name__ == "__main__":
    main()
