"""
라벨링 포맷이 갈린 실제 사례를 그림으로 뽑는다.

검수는 끝났는데 같은 종류의 내용이 문서마다 다른 라벨로 붙어 있다.
말로 하면 "기준이 모호합니다"지만, 두 장을 나란히 놓으면 한눈에 보인다.
포맷 결정을 요청할 때 붙일 근거 그림이다.

usage:
    python3 scripts/viz_format_cases.py --ids gcse_00001,gcse_00003
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/data/workspace/yjyong/receipt_data/labeled")
FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"

COLOR = {
    "table": "#d62728", "text": "#1f77b4", "doc_title": "#9467bd",
    "paragraph_title": "#8c564b", "figure_title": "#e377c2",
    "header": "#ff7f0e", "image": "#7f7f7f", "vision_footnote": "#2ca02c",
    "header_image": "#bcbd22", "footer_image": "#17becf", "aside_text": "#555555",
}


def find(doc_id):
    for s in ["train", "val", "test"]:
        j = ROOT / s / "json" / f"{doc_id}.json"
        if j.exists():
            imgs = list((ROOT / s / "images").glob(f"{doc_id}.*"))
            if imgs:
                return j, imgs[0]
    return None, None


def render(doc_id, target_h=1050, caption=""):
    jf, imf = find(doc_id)
    if not jf:
        return None
    d = json.loads(jf.read_text(encoding="utf-8"))
    im = Image.open(imf).convert("RGB")
    W, H = im.size
    # 저장된 width/height 는 파서 처리 해상도. 원본과 다르면 환산한다.
    pw, ph = d.get("width") or W, d.get("height") or H
    sx, sy = W / pw, H / ph
    s = target_h / H
    im = im.resize((max(1, int(W * s)), target_h))
    dr = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 17)

    blocks = sorted(d.get("parsing_res_list", []),
                    key=lambda b: b.get("block_order") if b.get("block_order") is not None else 0)
    for b in blocks:
        bb = b.get("block_bbox")
        if not bb or len(bb) < 4:
            continue
        lab = b.get("block_label", "?")
        c = COLOR.get(lab, "#000000")
        x1, y1, x2, y2 = bb[0]*sx*s, bb[1]*sy*s, bb[2]*sx*s, bb[3]*sy*s
        dr.rectangle([x1, y1, x2, y2], outline=c, width=3)
        tag = f"{b.get('block_order')} {lab}"
        tw = dr.textlength(tag, font=f)
        ty = max(0, y1 - 20)
        dr.rectangle([x1, ty, x1 + tw + 8, ty + 20], fill=c)
        dr.text((x1 + 4, ty + 1), tag, fill="white", font=f)

    bar = 62
    out = Image.new("RGB", (im.width, target_h + bar), "white")
    out.paste(im, (0, bar))
    d2 = ImageDraw.Draw(out)
    d2.text((6, 6), doc_id, fill="black", font=ImageFont.truetype(FONT, 22))
    d2.text((6, 34), caption, fill="#c00000", font=ImageFont.truetype(FONT, 19))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="쉼표 구분. 'id:설명' 형태 가능")
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=1050)
    args = ap.parse_args()

    cells = []
    for item in args.ids.split(","):
        did, _, cap = item.partition(":")
        img = render(did.strip(), args.height, cap.strip())
        if img is None:
            print(f"없음: {did}")
            continue
        cells.append(img)
    if not cells:
        raise SystemExit("그릴 게 없다")
    pad = 18
    W = sum(c.width for c in cells) + pad * (len(cells) - 1)
    H = max(c.height for c in cells)
    out = Image.new("RGB", (W, H), "white")
    x = 0
    for c in cells:
        out.paste(c, (x, 0))
        x += c.width + pad
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"저장: {args.out}  {out.size}")


if __name__ == "__main__":
    main()
