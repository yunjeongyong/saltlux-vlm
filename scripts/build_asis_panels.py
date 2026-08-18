"""AS-IS 문제 정의 패널 — 베이스 모델(luxia)이 무엇을 틀리는가.

images_asis_tobe 는 "TO-BE 가 고친 것"만 골라서 개선 확인용이다. 이쪽은 반대로
개선 여부와 무관하게 베이스의 실패를 유형별로 정의한다. 고쳐지지 않은 건도 넣는다
— 문제 정의에서 골라내면 그건 정의가 아니라 홍보다.

이미지는 전부 평가셋(978건 / 90장)이며 학습에 쓰지 않았다.

usage: python3 scripts/build_asis_panels.py <출력폴더>
"""
import difflib
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

W = 1500
PAD = 26
IMGW = 560
GAP = 24

F_TITLE = ImageFont.truetype(FONT_BOLD, 26)
F_META = ImageFont.truetype(FONT_PATH, 15)
F_BADGE = ImageFont.truetype(FONT_BOLD, 16)
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


# ── 문제 유형 정의 ──────────────────────────────────────────────────────
def nrows(s):
    return len(re.findall(r"<tr[ >]", s))


def has_fmt(pred, gt):
    return bool(re.search(r"\$[^$]+\$|\*\*?[0-9][^*]*\*", pred)) \
        and not re.search(r"[$*]", gt)


def squash(s):
    return re.sub(r"[\s\W]+", "", str(s))


def doc_coverage(r, bydoc):
    """베이스 출력이 그 영수증의 정답 어딘가에 있는 비율.

    높으면 크롭에 이웃 블록이 걸쳐 들어온 것이고(평가셋 범위 문제),
    낮으면 문서 어디에도 없는 내용을 지어낸 것이다(모델 결함).
    """
    doc = squash(" ".join(x["gt"] for x in bydoc[r["doc_id"]]))
    pred = squash(r[A]["pred"])
    if not pred:
        return 1.0
    sm = difflib.SequenceMatcher(None, pred, doc, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(pred)


def defect_halluc(r):
    p = re.sub(r"\s+", " ", r[A]["pred"])
    return f"문서에 없는 내용을 지어냄 — 정답 {len(r['gt'])}자 → 출력 {len(p)}자"


def defect_fmt(r):
    m = re.findall(r"\$[^$]{1,30}\$|\*\*?[0-9][^*]{0,30}\*", r[A]["pred"])
    ex = " · ".join(m[:2]) if m else "수식/강조 기호"
    return f"원문에 없는 기호를 붙임 — {ex}"


def defect_cut(r):
    return (f"</table> 없음 — {len(r[A]['pred']):,}자에서 끊김 "
            f"(정답 {len(r['gt']):,}자)")


def defect_rows(r):
    return f"정답 {nrows(r['gt'])}행 → 베이스 출력 {nrows(r[A]['pred'])}행"


CATS = [
    ("수식 환각", "읽지 못한 영역을 LaTeX 수식으로 지어낸다 — 긴 오답(CER>1)의 실체",
     "text", lambda r: r[A]["cer"] > 1 and r["_cov"] < 0.9
     and re.search(r"\\\\[a-z]+|\$", r[A]["pred"]),
     lambda r: r["_cov"], defect_halluc, 2),
    ("서식 기호 삽입", "숫자를 수식·강조 기호로 감싸 원문에 없는 문자가 섞인다",
     "text", lambda r: has_fmt(r[A]["pred"], r["gt"]) and len(r["gt"]) >= 18,
     lambda r: -r[A]["cer"], defect_fmt, 2),
    ("표 출력 절단", "표를 끝까지 내지 못하고 잘린다 — 닫는 태그가 없어 TEDS 가 0",
     "table", lambda r: "</table>" not in r[A]["pred"],
     lambda r: -len(r["gt"]), defect_cut, 2),
    ("표 행 누락", "표는 닫지만 품목을 빠뜨린다",
     "table", lambda r: "</table>" in r[A]["pred"]
     and nrows(r[A]["pred"]) < nrows(r["gt"]),
     lambda r: nrows(r[A]["pred"]) - nrows(r["gt"]), defect_rows, 2),
]


def build(rec, idx, kind, why, defect, fixed_note):
    color = TB if rec["task"] == "table" else TX
    tiles = tiles_of(rec, color)
    m = "TEDS" if rec["task"] == "table" else "CER"
    sa = rec[A]["teds"] if m == "TEDS" else rec[A]["cer"]

    scratch = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    tw = W - IMGW - GAP - PAD * 2
    html = rec["task"] == "table"
    cards = [
        ("정답 (GT) — 검수 정정본", clean(rec["gt"], is_html=html), GT_C, ""),
        ("AS-IS  베이스 모델 (luxia)", clean(rec[A]["pred"], is_html=html),
         BAD, f"{m} {sa:.4f}"),
    ]
    wrapped = [(t, wrap(scratch, b, F_BODY, tw - 24), c, s) for t, b, c, s in cards]

    lh = 23
    head_h = 142
    img_h = sum(t[1].height + 26 for t in tiles)
    txt_h = sum(34 + len(w) * lh + 22 + 16 for _, w, _, _ in wrapped)
    H = PAD * 2 + head_h + max(img_h, txt_h) + 16

    cv = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(cv)
    d.text((PAD, PAD), f"{idx:02d}  AS-IS · {kind}", font=F_TITLE, fill=INK)
    d.text((PAD, PAD + 38), why, font=F_META, fill=DIM)
    d.text((PAD, PAD + 62), f"▪ {defect}", font=F_BADGE, fill=BAD)
    d.text((PAD, PAD + 86),
           f"▪ 평가셋 {rec['split']} · {rec['crop_id']} · 학습 미사용   ·   "
           f"{m} {sa:.4f}   ·   {fixed_note}", font=F_META, fill=DIM)
    d.line([(PAD, PAD + 112), (W - PAD, PAD + 112)], fill=INK, width=2)

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
            d.text((W - PAD - 16 - scratch.textlength(score, font=F_LABEL),
                    ty + 9), score, font=F_LABEL, fill=c)
        yy = ty + 38
        for ln in lines:
            d.text((x + 16, yy), ln, font=F_BODY, fill=INK)
            yy += lh
        ty += bh + 16

    p = OUT / f"asis_{idx:02d}_{kind.replace(' ', '')}_{rec['crop_id']}.png"
    cv.save(p, optimize=True)
    return p


def main():
    rows = [r for r in json.loads(SCORED.read_text(encoding="utf-8"))["rows"]
            if A in r and B in r]
    bydoc = {}
    for r in rows:
        bydoc.setdefault(r["doc_id"], []).append(r)
    for r in rows:
        r["_cov"] = doc_coverage(r, bydoc) if r["task"] == "text" else 1.0

    # 긴 오답 중 이웃 블록 정답으로 설명되는 건은 모델 문제가 아니다 — 따로 센다
    long_ = [r for r in rows if r["task"] == "text" and r[A]["cer"] > 1]
    scope = sum(1 for r in long_ if r["_cov"] >= 0.9)
    print(f"긴 오답 {len(long_)}건 = 평가셋 범위 문제 {scope}건 + "
          f"모델 결함 {len(long_) - scope}건\n")

    idx = 0
    for kind, why, task, cond, key, defect, n in CATS:
        pool = [r for r in rows if r["task"] == task and cond(r)]
        print(f"[{kind}] 해당 {len(pool)}건")
        used = set()
        picked = 0
        for r in sorted(pool, key=key):
            if r["doc_id"] in used:          # 한 영수증에 몰리지 않게
                continue
            used.add(r["doc_id"])
            idx += 1
            picked += 1
            # 이 사례가 TO-BE 에서 어떻게 됐는지도 적는다 (고른 기준은 아니다)
            if task == "text":
                ok = r[B]["cer"] <= 0.1
                note = f"TO-BE CER {r[B]['cer']:.4f}" + (" (해결)" if ok else "")
            else:
                ok = r[B]["teds"] >= 0.9
                note = f"TO-BE TEDS {r[B]['teds']:.4f}" + (" (해결)" if ok else "")
            p = build(r, idx, kind, why, defect(r), note)
            print(f"  {p.name:54} {note}")
            if picked == n:
                break
    print(f"\n{idx}장 생성 → {OUT}")


if __name__ == "__main__":
    main()
