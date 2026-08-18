"""
document-parse 산출물에서 Detection 학습용 바운딩박스 라벨을 뽑는다.

파이프라인 1단계(`1_parse.py`)가 만든 layout_det_res.boxes 가 곧 오토라벨이다.
다만 그대로 쓰면 안 되는 이유가 하나 있다.

  Document Studio 는 입력 이미지를 확대한 뒤 검출한다. 배율이 장마다 다르다
  (실측 2.2~3.1배). 그래서 좌표가 원본 이미지 기준이 아니다. 원본에 그대로
  올리면 박스가 전부 밖으로 나간다. 여기서 원본 좌표계로 되돌린다.

  sx = parse_width / source_width,  sy = parse_height / source_height
  (실측상 sx ≈ sy 라 종횡비는 유지된다. 어긋나면 경고를 낸다.)

출력:
  bbox/{doc_id}.json          원본 좌표계 박스 + 라벨 + score
  yolo/labels/{doc_id}.txt    --yolo 를 주면 YOLO 포맷도 같이 (정규화 cxcywh)
  yolo/classes.txt            클래스 순서

score 가 낮은 박스가 적지 않다(실측 34.5%가 0.5 미만). 오토라벨을 그대로
학습에 넣기 전에 --min-score 로 자르거나, 1차 검수 대상으로 따로 빼라.

usage:
    python3 scripts/parse_to_bbox.py
    python3 scripts/parse_to_bbox.py --min-score 0.5 --yolo
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace")


def extract_page(doc):
    """저장된 API 응답에서 페이지 JSON을 꺼낸다. 응답 껍데기가 여러 겹이다."""
    res = doc.get("result") or {}
    els = res.get("elements") or []
    if not els:
        return None
    return els[0].get("json")


def convert(doc, min_score, warn_ratio):
    page = extract_page(doc)
    if not page:
        return None, "no_page_json"

    sw, sh = doc.get("_source_width"), doc.get("_source_height")
    pw, ph = page.get("width"), page.get("height")
    if not all((sw, sh, pw, ph)):
        return None, "no_size"

    sx, sy = pw / sw, ph / sh
    # sx 와 sy 가 크게 다르면 종횡비가 깨진 것이다. 그 경우 축별로 따로 나누면
    # 되지만, 원인을 모르는 채 넘기면 안 되니 표시해 둔다.
    aspect_off = abs(sx - sy) / max(sx, sy) > warn_ratio

    boxes = []
    for b in page.get("layout_det_res", {}).get("boxes", []):
        score = b.get("score", 0.0)
        if score < min_score:
            continue
        x1, y1, x2, y2 = b["coordinate"]
        # 원본 좌표계로 되돌리고, 이미지 밖으로 나간 건 잘라낸다.
        x1, x2 = max(0.0, x1 / sx), min(float(sw), x2 / sx)
        y1, y2 = max(0.0, y1 / sy), min(float(sh), y2 / sy)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({
            "label": b.get("label"),
            "cls_id": b.get("cls_id"),
            "score": score,
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })

    out = {
        "doc_id": doc.get("_source_doc_id"),
        "image": doc.get("_source_image"),
        "width": sw,
        "height": sh,
        "scale_from_parse": [round(sx, 4), round(sy, 4)],
        "aspect_mismatch": aspect_off,
        "n_boxes": len(boxes),
        "boxes": boxes,
    }
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse-dir", default="receipt_data/clean_v1/parse")
    ap.add_argument("--out", default="receipt_data/clean_v1/bbox")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--warn-ratio", type=float, default=0.02,
                    help="sx/sy 불일치 경고 기준 (기본 2%)")
    ap.add_argument("--yolo", action="store_true", help="YOLO 포맷도 같이 생성")
    args = ap.parse_args()

    parse_dir = ROOT / args.parse_dir
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(parse_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"{parse_dir} 에 파싱 결과가 없다.")

    labels = Counter()
    errs = Counter()
    n_box = []
    aspect_bad = []
    converted = []

    for f in files:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errs["broken_json"] += 1
            continue
        out, err = convert(doc, args.min_score, args.warn_ratio)
        if err:
            errs[err] += 1
            continue
        (out_dir / f.name).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        converted.append(out)
        n_box.append(out["n_boxes"])
        for b in out["boxes"]:
            labels[b["label"]] += 1
        if out["aspect_mismatch"]:
            aspect_bad.append(out["doc_id"])

    print(f"변환 {len(converted)}장 / 입력 {len(files)}장")
    if errs:
        print("  건너뜀:", dict(errs))
    if n_box:
        print(f"  박스: 총 {sum(n_box)}개, 장당 평균 {sum(n_box)/len(n_box):.1f} "
              f"(최소 {min(n_box)} 최대 {max(n_box)})")
        print(f"  박스 0개인 장: {sum(1 for n in n_box if n == 0)}장")
    print(f"  라벨 분포: {dict(labels.most_common())}")
    if aspect_bad:
        print(f"  종횡비 불일치 {len(aspect_bad)}장 (예: {aspect_bad[:3]}) — 축별 배율이 다르다")

    if args.yolo:
        classes = [l for l, _ in labels.most_common()]
        cls_idx = {c: i for i, c in enumerate(classes)}
        ydir = out_dir.parent / "yolo"
        (ydir / "labels").mkdir(parents=True, exist_ok=True)
        for out in converted:
            w, h = out["width"], out["height"]
            lines = []
            for b in out["boxes"]:
                x1, y1, x2, y2 = b["bbox"]
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{cls_idx[b['label']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (ydir / "labels" / f"{out['doc_id']}.txt").write_text("\n".join(lines), encoding="utf-8")
        (ydir / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
        print(f"  YOLO: {ydir} (클래스 {len(classes)}개)")

    print(f"\n산출: {out_dir}")


if __name__ == "__main__":
    main()
