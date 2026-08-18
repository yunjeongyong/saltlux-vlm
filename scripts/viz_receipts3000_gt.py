"""
합성 영수증 3000장의 GT 를 이미지 옆에 붙여 눈으로 확인한다.

이 데이터의 GT 는 bbox 가 아니다. 원본을 렌더링할 때 쓴 필드값·품목표를
그대로 저장한 것이라 열화(회전/그림자/저해상도)를 아무리 걸어도 값은 정확하다.
그래서 그릴 것은 '박스가 맞나'가 아니라 '이 이미지에서 이 값이 실제로 읽히나',
즉 열화 강도별로 GT 가 여전히 사람 눈에 검증 가능한지다.

quality 6단계(clean/light/sharp_photo/heavy/screenshot/extreme)에서 골고루 뽑아
왼쪽 이미지, 오른쪽 GT 패널로 나란히 그린다. --parser 를 주면 파서 박스도 겹친다.

usage:
    python3 scripts/viz_receipts3000_gt.py
    python3 scripts/viz_receipts3000_gt.py --per-quality 2 --parser
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/data/workspace/yjyong/receipt_data")
DS = ROOT / "receipts3000"
PRED_DIR = ROOT / "clean_v1" / "bbox"
FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"

QUALITIES = ["clean", "light", "sharp_photo", "heavy", "screenshot", "extreme"]
PRED_COLOR = {"table": "#ff2d2d", "text": "#00a0ff", "header": "#ffb000",
              "figure": "#00c853", "title": "#aa00ff"}


def gt_lines(gt):
    """GT json 을 사람이 훑기 좋은 줄 목록으로 편다."""
    L = []
    for k in ["상호명", "업종", "사업자번호", "주소", "전화번호", "거래일시"]:
        if gt.get(k):
            L.append((f"{k}", str(gt[k]), "#1a1a1a"))
    L.append(("", "", None))
    L.append(("품목", f"{len(gt.get('품목', []))}건", "#000080"))
    for it in gt.get("품목", []):
        L.append(("  " + str(it.get("명", "")),
                  f"{it.get('수량','')} x {it.get('단가',''):,} = {it.get('금액',0):,}"
                  if isinstance(it.get("단가"), int) else str(it),
                  "#333333"))
    L.append(("", "", None))
    for k in ["공급가액", "부가세", "합계", "결제수단"]:
        v = gt.get(k)
        if v is None:
            continue
        L.append((k, f"{v:,}" if isinstance(v, int) else str(v),
                  "#c00000" if k == "합계" else "#1a1a1a"))
    return L


def gt_panel(gt, meta, w, h):
    im = Image.new("RGB", (w, h), "#fbfbfb")
    d = ImageDraw.Draw(im)
    fb = ImageFont.truetype(FONT, 21)
    fk = ImageFont.truetype(FONT, 19)
    fs = ImageFont.truetype(FONT, 17)
    y = 12
    d.text((14, y), f"GT  ({meta['quality']} / {meta['subdomain']})", fill="#000", font=fb)
    y += 34
    d.line([10, y, w-10, y], fill="#cccccc", width=2)
    y += 10
    for k, v, c in gt_lines(gt):
        if c is None:
            y += 10
            continue
        if y > h - 24:
            d.text((14, y), "... (이하 생략)", fill="#888", font=fs)
            break
        f = fk if not k.startswith("  ") else fs
        d.text((14, y), k[:26], fill=c, font=f)
        tw = d.textlength(v, font=f)
        d.text((max(14, w - 14 - tw), y), v, fill=c, font=f)
        y += 25 if f is fk else 22
    d.rectangle([0, 0, w-1, h-1], outline="#bbbbbb", width=2)
    return im


def render_image(path, target_h, boxes=None, title=""):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    s = target_h / H
    im = im.resize((max(1, int(W*s)), target_h))
    if boxes:
        d = ImageDraw.Draw(im)
        f = ImageFont.truetype(FONT, 15)
        for label, bb, sc in boxes:
            c = PRED_COLOR.get(label, "#ff2d2d")
            d.rectangle([v*s for v in bb], outline=c, width=2)
            d.text((bb[0]*s+2, max(0, bb[1]*s-16)), f"{label} {sc:.2f}", fill=c, font=f)
    bar = 30
    out = Image.new("RGB", (im.width, target_h + bar), "white")
    out.paste(im, (0, bar))
    ImageDraw.Draw(out).text((5, 5), title, fill="black",
                             font=ImageFont.truetype(FONT, 19))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-quality", type=int, default=2)
    ap.add_argument("--parser", action="store_true", help="파서 박스도 겹쳐 그린다")
    ap.add_argument("--height", type=int, default=760)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "review" / "receipts3000_gt.png"))
    args = ap.parse_args()

    man = [json.loads(l) for l in (DS / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_q = {q: [m for m in man if m["quality"] == q] for q in QUALITIES}
    rng = random.Random(args.seed)

    rows = []
    for q in QUALITIES:
        picks = rng.sample(by_q[q], min(args.per_quality, len(by_q[q])))
        cells = []
        for m in picks:
            gt = json.loads((DS / m["gt_path"]).read_text(encoding="utf-8"))
            img = DS / m["image_path"]
            doc_id = "syn_" + Path(m["image_path"]).stem
            boxes = None
            if args.parser:
                pf = PRED_DIR / f"{doc_id}.json"
                if pf.exists():
                    p = json.loads(pf.read_text(encoding="utf-8"))
                    boxes = [(b["label"], b["bbox"], b.get("score", 0)) for b in p["boxes"]]
            left = render_image(img, args.height, boxes,
                                f"{Path(m['image_path']).stem}  {q}"
                                + (f"  파서 {len(boxes)}박스" if boxes else ""))
            panel = gt_panel(gt, m, 430, left.height)
            cell = Image.new("RGB", (left.width + panel.width + 8, left.height), "white")
            cell.paste(left, (0, 0))
            cell.paste(panel, (left.width + 8, 0))
            cells.append(cell)
        h = max(c.height for c in cells)
        w = sum(c.width for c in cells) + 24*(len(cells)-1)
        row = Image.new("RGB", (w, h), "white")
        x = 0
        for c in cells:
            row.paste(c, (x, 0))
            x += c.width + 24
        rows.append(row)

    W = max(r.width for r in rows)
    H = sum(r.height for r in rows) + 16*(len(rows)-1)
    out = Image.new("RGB", (W, H), "#dddddd")
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height + 16
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.save(p)
    print(f"저장: {p}  {out.size}  (quality {len(QUALITIES)}종 x {args.per_quality}장)")


if __name__ == "__main__":
    main()
