"""
파서 오토라벨을 사람 라벨과 대조한다 (CORD 전용).

clean_v1 과 크롤링에는 bbox 정답이 없어서 오토라벨 품질을 숫자로 말할 수 없다.
CORD 는 NAVER CLOVA 가 사람 손으로 라벨링한 valid_line 을 갖고 있으므로,
같은 이미지에 파서를 돌려 대조하면 오토라벨 품질을 처음으로 정량화할 수 있다.

대조 방식.
  - 사람 라벨(줄 단위)을 GT, 파서 박스를 예측으로 놓고 IoU 매칭.
  - 클래스는 비교하지 않는다. CORD 는 영수증 전용 카테고리(menu.nm 등),
    파서는 범용 문서 카테고리(text/table 등)라 라벨 체계가 아예 다르다.
    여기서 재는 것은 '영역을 잡았는가'지 '이름을 맞췄는가'가 아니다.
  - CORD 는 상단 헤더가 미라벨이다(평균 46%). 그 영역의 파서 박스를 FP 로
    세면 부당하게 나쁘게 나온다. --gt-region 으로 GT 가 존재하는 세로 구간만
    비교하는 것을 기본으로 한다.

usage:
    python3 scripts/compare_parser_vs_gt.py
    python3 scripts/compare_parser_vs_gt.py --iou 0.5 --no-gt-region
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace/receipt_data/public/cord")


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def covered(gt, pred):
    """GT 가 예측 박스에 얼마나 덮였는가. 파서가 여러 줄을 한 덩어리로 잡는
    경우 IoU 는 낮게 나오지만 실제로는 영역을 포함하고 있다. 그걸 따로 본다."""
    gx1, gy1, gx2, gy2 = gt
    px1, py1, px2, py2 = pred
    ix1, iy1 = max(gx1, px1), max(gy1, py1)
    ix2, iy2 = min(gx2, px2), min(gy2, py2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = (gx2 - gx1) * (gy2 - gy1)
    return inter / area if area > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default=str(ROOT / "gt"))
    ap.add_argument("--pred-dir", default=str(ROOT / "bbox"))
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--no-gt-region", action="store_true",
                    help="GT 가 없는 상단 영역도 FP 로 센다")
    ap.add_argument("--out", default=str(ROOT / "parser_vs_gt.json"))
    args = ap.parse_args()

    gt_dir, pred_dir = Path(args.gt_dir), Path(args.pred_dir)
    preds = {p.stem: p for p in pred_dir.glob("*.json")}
    if not preds:
        raise SystemExit(f"{pred_dir} 에 파서 결과가 없다. CORD 파싱이 끝났는지 확인해라.")

    TP = FP = FN = 0
    covs, ious = [], []
    per_cat = Counter()
    per_cat_hit = Counter()
    ratio_pred_per_gt = []
    n_doc = 0

    for gf in sorted(gt_dir.glob("*.json")):
        if gf.stem not in preds:
            continue
        g = json.loads(gf.read_text(encoding="utf-8"))
        p = json.loads(preds[gf.stem].read_text(encoding="utf-8"))
        gt_boxes = [(b["bbox"], b["category"]) for b in g["layout"]]
        if not gt_boxes:
            continue
        n_doc += 1

        pred_boxes = [b["bbox"] for b in p["boxes"]]
        if not args.no_gt_region:
            # GT 가 존재하는 세로 구간으로 예측을 제한한다.
            ys = [bb[1] for bb, _ in gt_boxes] + [bb[3] for bb, _ in gt_boxes]
            lo, hi = min(ys), max(ys)
            pred_boxes = [b for b in pred_boxes
                          if not (b[3] < lo - 5 or b[1] > hi + 5)]

        used = set()
        for gbox, cat in gt_boxes:
            per_cat[cat] += 1
            best, bi = 0.0, -1
            bestcov = 0.0
            for i, pb in enumerate(pred_boxes):
                v = iou(gbox, pb)
                if v > best:
                    best, bi = v, i
                bestcov = max(bestcov, covered(gbox, pb))
            covs.append(bestcov)
            ious.append(best)
            if best >= args.iou and bi not in used:
                TP += 1
                used.add(bi)
                per_cat_hit[cat] += 1
            else:
                FN += 1
        FP += len(pred_boxes) - len(used)
        ratio_pred_per_gt.append(len(pred_boxes) / len(gt_boxes))

    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    import statistics as st

    print(f"대조 문서 {n_doc}개 / GT 블록 {TP + FN}개 / 파서 박스 {TP + FP}개")
    print(f"  GT 1개당 파서 박스 {st.mean(ratio_pred_per_gt):.2f}개")
    print(f"\nIoU>={args.iou} 매칭")
    print(f"  precision {prec*100:5.1f}%   recall {rec*100:5.1f}%   F1 {f1*100:5.1f}%")
    print(f"  TP {TP} / FP {FP} / FN {FN}")
    print(f"\nGT 블록이 파서 박스에 덮인 비율(포함 기준)")
    print(f"  평균 {st.mean(covs)*100:.1f}%  중앙값 {st.median(covs)*100:.1f}%")
    print(f"  90% 이상 덮인 GT: {sum(1 for c in covs if c >= 0.9)}개 "
          f"({100*sum(1 for c in covs if c>=0.9)/len(covs):.1f}%)")
    print(f"  최고 IoU 평균 {st.mean(ious)*100:.1f}%")

    print(f"\n카테고리별 재현율 (IoU>={args.iou}):")
    for cat, n in per_cat.most_common(12):
        print(f"    {cat:26} {per_cat_hit[cat]:5}/{n:<5} {100*per_cat_hit[cat]/n:5.1f}%")

    Path(args.out).write_text(json.dumps({
        "n_doc": n_doc, "iou_thr": args.iou,
        "precision": prec, "recall": rec, "f1": f1,
        "TP": TP, "FP": FP, "FN": FN,
        "gt_covered_mean": st.mean(covs), "best_iou_mean": st.mean(ious),
        "per_category_recall": {c: [per_cat_hit[c], n] for c, n in per_cat.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
