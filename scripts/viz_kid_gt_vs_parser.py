"""
KorIE KID 사람라벨(GT) 과 document-parse 오토라벨을 같은 이미지 위에 나란히 그린다.

KID 는 이미 라벨이 붙은 데이터셋이다. 그래서 "파서를 왜 또 돌려 비교하냐"는
질문이 나오는데, 두 라벨의 성격이 다르기 때문이다.

  KID GT   : 영수증 전용 필드 17종 (상호명/합계/품목명/수량/단가 ...) — 줄·셀 단위
  파서 출력 : 범용 문서 레이아웃 (table/text/header ...)      — 블록 단위

즉 이름 체계가 겹치지 않는다. 여기서 재는 것은 '이름을 맞췄나'가 아니라
'파서가 GT 필드 영역을 잡아내기는 했나(덮었나)'다.

  bbox 정답이 있는 실사진 데이터는 KID 가 유일하다. clean_v1/크롤링은
  bbox GT 가 없어 오토라벨 품질을 숫자로 말할 수 없다.

usage:
    python3 scripts/viz_kid_gt_vs_parser.py                # 샘플 8장 + 지표
    python3 scripts/viz_kid_gt_vs_parser.py -n 12 --ids IMG00001,IMG00529
"""
import argparse
import json
import random
import statistics as st
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/data/workspace/yjyong/receipt_data")
GT_DIR = ROOT / "kid_gt"
PRED_DIR = ROOT / "clean_v1" / "bbox"
FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"

# 배포본에 클래스명이 없다. OCR 크롭(IMG_필드명.jpg) 크기를 라벨 박스 크기와
# 맞춰 역추정했다. 15/16 은 같은 자리(품목 둘째 줄 바코드)에 겹쳐 붙어 있다.
CLS = {
    0: "Item_Name", 1: "Item_Qty", 2: "Item_TotalPrice", 3: "Item_UnitPrice",
    4: "Item_Row", 5: "MerchantName", 6: "Total", 7: "Subtotal", 8: "TotalTax",
    9: "TransactionDate", 10: "TransactionTime", 11: "Tip",
    12: "MerchantPhone", 13: "ReceiptNumber", 14: "MerchantAddress",
    15: "Item_barcode", 16: "Item_barcode2",
}
COLOR = {
    0: "#e6194b", 1: "#3cb44b", 2: "#4363d8", 3: "#f58231", 4: "#911eb4",
    5: "#42d4f4", 6: "#f032e6", 7: "#bfef45", 8: "#fabed4", 9: "#469990",
    10: "#dcbeff", 11: "#9a6324", 12: "#800000", 13: "#aaffc3", 14: "#808000",
    15: "#ffd8b1", 16: "#000075",
}
PRED_COLOR = {"table": "#ff2d2d", "text": "#00a0ff", "header": "#ffb000",
              "figure": "#00c853", "title": "#aa00ff"}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def covered(gt, pred):
    ix = max(0.0, min(gt[2], pred[2]) - max(gt[0], pred[0]))
    iy = max(0.0, min(gt[3], pred[3]) - max(gt[1], pred[1]))
    area = (gt[2]-gt[0]) * (gt[3]-gt[1])
    return (ix * iy) / area if area > 0 else 0.0


def load_pair(img_id):
    gf = GT_DIR / f"{img_id}.json"
    pf = PRED_DIR / f"kid_{img_id}.json"
    if not (gf.exists() and pf.exists()):
        return None
    g = json.loads(gf.read_text(encoding="utf-8"))
    p = json.loads(pf.read_text(encoding="utf-8"))
    if not g.get("layout"):
        return None
    W, H = p["width"], p["height"]
    gt = [(b["cls_id"], [b["bbox_norm"][0]*W, b["bbox_norm"][1]*H,
                         b["bbox_norm"][2]*W, b["bbox_norm"][3]*H])
          for b in g["layout"]]
    pred = [(b["label"], b["bbox"], b.get("score", 0)) for b in p["boxes"]]
    return g, p, gt, pred, W, H


def render(img_path, boxes, title, target_h=1100, font_sz=18):
    """boxes: [(color, [x1,y1,x2,y2], text)]"""
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    s = target_h / H
    im = im.resize((max(1, int(W*s)), target_h))
    d = ImageDraw.Draw(im, "RGBA")
    f = ImageFont.truetype(FONT, font_sz)
    for color, bb, txt in boxes:
        x1, y1, x2, y2 = [v*s for v in bb]
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if txt:
            tw = d.textlength(txt, font=f)
            ty = max(0, y1 - font_sz - 3)
            d.rectangle([x1, ty, x1+tw+6, ty+font_sz+3], fill=color)
            d.text((x1+3, ty+1), txt, fill="white", font=f)
    # 제목 띠
    bar = 34
    out = Image.new("RGB", (im.width, im.height + bar), "white")
    out.paste(im, (0, bar))
    d2 = ImageDraw.Draw(out)
    d2.text((6, 7), title, fill="black", font=ImageFont.truetype(FONT, 22))
    return out


def hstack(imgs, pad=10, bg="white"):
    h = max(i.height for i in imgs)
    w = sum(i.width for i in imgs) + pad*(len(imgs)-1)
    out = Image.new("RGB", (w, h), bg)
    x = 0
    for i in imgs:
        out.paste(i, (x, 0))
        x += i.width + pad
    return out


def vstack(imgs, pad=14, bg="#dddddd"):
    w = max(i.width for i in imgs)
    h = sum(i.height for i in imgs) + pad*(len(imgs)-1)
    out = Image.new("RGB", (w, h), bg)
    y = 0
    for i in imgs:
        out.paste(i, (0, y))
        y += i.height + pad
    return out


def metrics(ids, iou_thr=0.5):
    TP = FP = FN = 0
    covs, best_ious = [], []
    per_cls = Counter()
    per_cls_cov = Counter()
    per_cls_hit = Counter()
    n_gt_per_doc, n_pred_per_doc = [], []
    for i in ids:
        r = load_pair(i)
        if not r:
            continue
        _, _, gt, pred, W, H = r
        pb = [b for _, b, _ in pred]
        n_gt_per_doc.append(len(gt))
        n_pred_per_doc.append(len(pb))
        used = set()
        for cls, gb in gt:
            per_cls[cls] += 1
            best, bi, bcov = 0.0, -1, 0.0
            for k, p in enumerate(pb):
                v = iou(gb, p)
                if v > best:
                    best, bi = v, k
                bcov = max(bcov, covered(gb, p))
            covs.append(bcov)
            best_ious.append(best)
            if bcov >= 0.9:
                per_cls_cov[cls] += 1
            if best >= iou_thr and bi not in used:
                TP += 1
                used.add(bi)
                per_cls_hit[cls] += 1
            else:
                FN += 1
        FP += len(pb) - len(used)
    prec = TP/(TP+FP) if TP+FP else 0
    rec = TP/(TP+FN) if TP+FN else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    return dict(n_doc=len(n_gt_per_doc), TP=TP, FP=FP, FN=FN,
                precision=prec, recall=rec, f1=f1,
                gt_per_doc=st.mean(n_gt_per_doc), pred_per_doc=st.mean(n_pred_per_doc),
                cov_mean=st.mean(covs), cov90=sum(1 for c in covs if c >= 0.9)/len(covs),
                iou_mean=st.mean(best_ious),
                per_cls={CLS.get(c, c): (per_cls_cov[c], per_cls_hit[c], n)
                         for c, n in per_cls.most_common()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("--ids", default="")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "review" / "kid_gt_vs_parser.png"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    all_ids = sorted(p.stem for p in GT_DIR.glob("*.json")
                     if not p.stem.endswith("-result"))
    all_ids = [i for i in all_ids if (PRED_DIR / f"kid_{i}.json").exists()]

    m = metrics(all_ids, args.iou)
    print(f"KID GT vs 파서 오토라벨  (문서 {m['n_doc']}장)")
    print(f"  문서당 GT 필드 {m['gt_per_doc']:.1f}개 / 파서 박스 {m['pred_per_doc']:.1f}개")
    print(f"\n  [영역 매칭] IoU>={args.iou}")
    print(f"    precision {m['precision']*100:5.1f}%  recall {m['recall']*100:5.1f}%  F1 {m['f1']*100:5.1f}%")
    print(f"    TP {m['TP']} / FP {m['FP']} / FN {m['FN']}")
    print(f"\n  [포함 기준] GT 필드가 파서 박스 안에 들어간 비율")
    print(f"    평균 덮임 {m['cov_mean']*100:.1f}%   90% 이상 덮인 GT {m['cov90']*100:.1f}%")
    print(f"    최고 IoU 평균 {m['iou_mean']*100:.1f}%")
    print(f"\n  클래스별  (90%덮임 / IoU매칭 / 전체)")
    for name, (c90, hit, n) in m["per_cls"].items():
        print(f"    {str(name):18} {c90:5}/{hit:5}/{n:<5}  덮임 {100*c90/n:5.1f}%  매칭 {100*hit/n:5.1f}%")

    ids = [i.strip() for i in args.ids.split(",") if i.strip()] or \
        random.Random(args.seed).sample(all_ids, min(args.n, len(all_ids)))

    rows = []
    for i in ids:
        r = load_pair(i)
        if not r:
            print(f"skip {i}")
            continue
        g, p, gt, pred, W, H = r
        img = ROOT.parent / g["image"]
        if not img.exists():
            img = Path("/data/workspace") / g["image"]
        gt_boxes = [(COLOR.get(c, "#ff0000"), b, CLS.get(c, str(c))) for c, b in gt]
        pr_boxes = [(PRED_COLOR.get(l, "#ff2d2d"), b, f"{l} {s:.2f}")
                    for l, b, s in pred]
        left = render(img, gt_boxes, f"{i}  GT(사람라벨) {len(gt)}개")
        right = render(img, pr_boxes, f"{i}  파서 오토라벨 {len(pred)}개")
        plain = render(img, [], f"{i}  원본")
        rows.append(hstack([plain, left, right]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vstack(rows).save(out)
    print(f"\n저장: {out}  ({', '.join(ids)})")


if __name__ == "__main__":
    main()
