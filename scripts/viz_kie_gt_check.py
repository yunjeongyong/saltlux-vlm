"""
KIE 필드 라벨이 실제로 맞는지, 그 필드의 bbox 를 잘라 값과 나란히 붙여 본다.

compare_kie_vs_parser.py 가 뽑은 불일치 목록은 "파서가 틀렸다"와
"KorIE 라벨이 틀렸다"가 섞여 있다. 구분하려면 결국 그 자리를 봐야 한다.
KID 라벨에 필드별 bbox 가 있으니, 그 영역만 잘라서 GT 문자열과 붙이면
한 줄만 보고도 어느 쪽이 틀렸는지 판정할 수 있다.

usage:
    python3 scripts/viz_kie_gt_check.py                 # 불일치 24건
    python3 scripts/viz_kie_gt_check.py --mode match    # 일치 건(대조군)
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/data/workspace/yjyong/receipt_data")
GT_DIR = ROOT / "kid_gt"
MISS = ROOT / "review" / "kie_vs_parser_mismatch.jsonl"
FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"

FIELD_CLS = {"MerchantName": 5, "Total": 6, "Subtotal": 7, "TotalTax": 8,
             "TransactionDate": 9, "TransactionTime": 10,
             "MerchantPhoneNumber": 12, "ReceiptNumber": 13,
             "MerchantAddress": 14}


def crop_field(image_id, field, pad=8, crop_h=64):
    g = json.loads((GT_DIR / f"{image_id}.json").read_text(encoding="utf-8"))
    cls = FIELD_CLS.get(field)
    boxes = [b for b in g.get("layout", []) if b["cls_id"] == cls]
    if not boxes:
        return None
    path = ROOT.parent / g["image"]
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    W, H = im.size
    x1, y1, x2, y2 = boxes[0]["bbox_norm"]
    box = (max(0, x1*W-pad), max(0, y1*H-pad), min(W, x2*W+pad), min(H, y2*H+pad))
    c = im.crop([int(v) for v in box])
    if c.height < 4:
        return None
    s = crop_h / c.height
    return c.resize((max(1, int(c.width*s)), crop_h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=24)
    ap.add_argument("--mode", choices=["mismatch", "match"], default="mismatch")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in MISS.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["field"] in FIELD_CLS]
    if args.mode == "match":
        # 대조군: 불일치 목록에 없는 (문서,필드) 조합에서 뽑는다
        bad = {(r["image_id"], r["field"]) for r in rows}
        rows = []
        for gf in sorted(GT_DIR.glob("*.json")):
            if gf.stem.endswith("-result"):
                continue
            g = json.loads(gf.read_text(encoding="utf-8"))
            for k, v in (g.get("fields") or {}).items():
                if k in FIELD_CLS and (gf.stem, k) not in bad:
                    rows.append({"image_id": gf.stem, "field": k, "gt": v})
    picks = random.Random(args.seed).sample(rows, min(args.n, len(rows)))

    cells = []
    for r in picks:
        c = crop_field(r["image_id"], r["field"])
        if c is None:
            continue
        cells.append((r, c))
    if not cells:
        raise SystemExit("자를 수 있는 건이 없다")

    label_w = 620
    crop_w = max(c.width for _, c in cells)
    rowh = 76
    W = label_w + min(crop_w, 1000) + 30
    H = rowh*len(cells) + 60
    out = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(out)
    fh = ImageFont.truetype(FONT, 24)
    f = ImageFont.truetype(FONT, 20)
    title = ("파서 텍스트와 불일치한 KIE 필드 — 잘라낸 실제 영역"
             if args.mode == "mismatch" else "일치 건 (대조군)")
    d.text((14, 14), title, fill="black", font=fh)
    y = 54
    for r, c in cells:
        d.text((14, y+8), f"{r['image_id']}", fill="#666", font=f)
        d.text((130, y+8), f"{r['field']}", fill="#0050c0", font=f)
        d.text((130, y+34), f"GT: {r['gt']}"[:46], fill="#c00000", font=f)
        if c.width > 1000:
            c = c.crop((0, 0, 1000, c.height))
        out.paste(c, (label_w, y))
        d.line([10, y+rowh-4, W-10, y+rowh-4], fill="#e0e0e0")
        y += rowh
    p = Path(args.out or ROOT / "review" / f"kie_gt_check_{args.mode}.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    out.save(p)
    print(f"저장: {p}  {out.size}  ({len(cells)}건)")


if __name__ == "__main__":
    main()
