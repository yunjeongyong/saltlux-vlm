#!/usr/bin/env python3
"""파싱 결과(parse/*.json)를 라벨링 툴 업로드 형식으로 변환한다.

라벨링 툴은 PP-StructureV3 산출물을 최상위에 기대하고, 저장 시
 - 모든 BBox 에 label 이 있어야 하고
 - order 가 중복이면 안 되고
 - BBox 가 이미지 영역 안에 있어야 한다
는 검증을 건다. parse/*.json 을 그대로 올리면 네 가지가 걸린다.

  1. parsing_res_list / layout_det_res 가 result.elements[0].json 안에 중첩
  2. parsing_res_list 와 layout_det_res.boxes 길이 불일치 (로드 실패)
  3. 좌표가 축소본 기준 (parse 1032x1960 vs 원본 1739x3303)
  4. 이미지와 JSON 의 basename 불일치

사용:
  python3 make_labeling_upload.py --source clean_v1 --prefix kid_
  python3 make_labeling_upload.py --source crawl_google --limit 20
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "receipt_data"

# 툴 Label 탭에 등록할 라벨과 cls_id. 파싱 엔진 카테고리를 그대로 쓴다.
FALLBACK_CLS = {
    "text": 0, "paragraph_title": 1, "table": 2, "figure_title": 3,
    "chart": 4, "number": 5, "header": 12, "footer": 13, "image": 14,
}


def extract(parse_json):
    """result.elements[0].json 을 꺼낸다. 페이지가 여러 개면 첫 장만."""
    els = parse_json.get("result", {}).get("elements") or []
    if not els:
        return None
    return els[0].get("json")


def convert(inner, orig_w, orig_h, input_path):
    """중첩 해제 + 좌표 환산 + boxes 재생성. 최상위 PP-Structure 형태로 반환."""
    pw, ph = inner.get("width"), inner.get("height")
    if not pw or not ph:
        return None, "parse 에 width/height 없음"

    sx, sy = orig_w / pw, orig_h / ph
    blocks = inner.get("parsing_res_list") or []
    if not blocks:
        return None, "parsing_res_list 비어 있음"

    # 기존 boxes 에서 label -> cls_id 를 주워온다 (없으면 fallback).
    cls_of = {}
    for b in ((inner.get("layout_det_res") or {}).get("boxes") or []):
        if b.get("label") and b.get("cls_id") is not None:
            cls_of.setdefault(b["label"], b["cls_id"])

    def scale_box(bb):
        x0, y0, x1, y1 = bb
        x0, x1 = sorted((x0 * sx, x1 * sx))
        y0, y1 = sorted((y0 * sy, y1 * sy))
        # 이미지 밖으로 튀어나온 좌표를 잘라낸다 (툴 저장 검증 대상).
        x0 = max(0.0, min(x0, orig_w - 1))
        y0 = max(0.0, min(y0, orig_h - 1))
        x1 = max(x0 + 1, min(x1, orig_w))
        y1 = max(y0 + 1, min(y1, orig_h))
        return [round(v, 1) for v in (x0, y0, x1, y1)]

    def scale_poly(pts, bb):
        if not pts:
            x0, y0, x1, y1 = bb
            return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        return [[round(px * sx, 1), round(py * sy, 1)] for px, py in pts]

    new_blocks, new_boxes = [], []
    for i, b in enumerate(blocks):
        label = b.get("block_label") or "text"
        bbox = scale_box(b["block_bbox"])
        poly = scale_poly(b.get("block_polygon_points"), bbox)
        # block_order 가 null 인 게 2,052개 있다. 인덱스로 채워 중복을 없앤다.
        new_blocks.append({
            "block_label": label,
            "block_content": b.get("block_content") or "",
            "block_bbox": bbox,
            "block_id": i,
            "block_order": i,
            "group_id": b.get("group_id", 0),
            "block_polygon_points": poly,
        })
        new_boxes.append({
            "cls_id": cls_of.get(label, FALLBACK_CLS.get(label, 0)),
            "label": label,
            "score": 1.0,
            "coordinate": bbox,
            "order": i,
            "polygon_points": poly,
        })

    out = {
        "input_path": input_path,
        "page_index": 0,
        "page_count": 1,
        "width": orig_w,
        "height": orig_h,
        "model_settings": inner.get("model_settings", {}),
        "parsing_res_list": new_blocks,
        "layout_det_res": {
            "input_path": input_path,
            "page_index": 0,
            "boxes": new_boxes,
        },
    }
    return out, None


def validate(doc):
    """툴이 저장 시 거는 검증을 미리 돌린다."""
    errs = []
    blocks, boxes = doc["parsing_res_list"], doc["layout_det_res"]["boxes"]
    if len(blocks) != len(boxes):
        errs.append(f"길이 불일치 {len(blocks)} vs {len(boxes)}")
    orders = [b["block_order"] for b in blocks]
    if len(orders) != len(set(orders)):
        errs.append("order 중복")
    W, H = doc["width"], doc["height"]
    for b in blocks:
        if not b["block_label"]:
            errs.append(f"block {b['block_id']} label 없음")
        x0, y0, x1, y1 = b["block_bbox"]
        if x1 <= x0 or y1 <= y0:
            errs.append(f"block {b['block_id']} 크기 0 이하")
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            errs.append(f"block {b['block_id']} 이미지 영역 밖")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["clean_v1", "crawl_google"])
    ap.add_argument("--prefix", default="", help="doc_id 접두어 필터 (예: kid_)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="심볼릭 링크 대신 이미지 복사")
    args = ap.parse_args()

    src = R / args.source
    out = Path(args.out) if args.out else R / "labeling_upload" / (
        args.source + ("_" + args.prefix.rstrip("_") if args.prefix else ""))
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(src / "manifest.jsonl")]
    if args.prefix:
        rows = [r for r in rows if r["doc_id"].startswith(args.prefix)]
    if args.limit:
        rows = rows[:args.limit]

    ok = 0
    skipped = []
    for r in rows:
        doc_id = r["doc_id"]
        pf = src / "parse" / f"{doc_id}.json"
        if not pf.exists():
            skipped.append((doc_id, "parse 없음"))
            continue

        inner = extract(json.load(open(pf)))
        if inner is None:
            skipped.append((doc_id, "elements 없음"))
            continue

        img_src = R.parent / r["image"] if not (R.parent / r["image"]).exists() \
            else R.parent / r["image"]
        if not img_src.exists():
            skipped.append((doc_id, f"이미지 없음: {r['image']}"))
            continue

        ext = img_src.suffix.lower()
        img_dst = out / f"{doc_id}{ext}"

        doc, err = convert(inner, r["width"], r["height"], f"{doc_id}{ext}")
        if err:
            skipped.append((doc_id, err))
            continue

        errs = validate(doc)
        if errs:
            skipped.append((doc_id, "; ".join(errs[:2])))
            continue

        if img_dst.exists() or img_dst.is_symlink():
            img_dst.unlink()
        if args.copy:
            shutil.copy2(img_src, img_dst)
        else:
            img_dst.symlink_to(os.path.relpath(img_src, out))

        with open(out / f"{doc_id}.json", "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        ok += 1

    print(f"변환 완료: {ok}건 -> {out}")
    if skipped:
        print(f"건너뜀: {len(skipped)}건")
        for d, why in skipped[:10]:
            print(f"  {d}: {why}")
        if len(skipped) > 10:
            print(f"  ... 외 {len(skipped) - 10}건")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
