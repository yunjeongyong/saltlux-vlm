"""
공개 영수증 데이터셋을 받아 clean_v1 규칙으로 내보낸다.

clean_v1과 같은 규칙을 지킨다: 이미지 1장 ↔ JSON 1개, manifest.jsonl 한 줄이
문서 하나. 그래야 기존 스크립트(build_clean_dataset, parse_clean_v1 등)를
그대로 태울 수 있다.

  receipt_data/public/{name}/images/{doc_id}.jpg
  receipt_data/public/{name}/gt/{doc_id}.json
  receipt_data/public/{name}/manifest.jsonl

CORD-v2:
  gt_parse   -> fields / items   (영수증 필드와 품목)
  valid_line -> layout           (단어 단위 quad + category, layout 학습에 쓴다)
  원본 gt는 raw_gt 로 통째로 보존한다. 매핑에서 흘린 게 있어도 되살릴 수 있게.

usage:
    python3 scripts/fetch_public_receipts.py --dataset cord
    python3 scripts/fetch_public_receipts.py --dataset cord --limit 20   # 확인용
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path("/workspace/receipt_data/public")

# CORD 가격 표기는 "75,000" / "40,000." 처럼 지저분하다.
NUM_RE = re.compile(r"[^\d.-]")


def to_num(s):
    """'75,000' -> 75000, '1 x' -> 1. 실패하면 None (원문은 따로 보존된다)."""
    if s is None:
        return None
    t = NUM_RE.sub("", str(s)).rstrip(".")
    if not t or t in {"-", "."}:
        return None
    try:
        return int(float(t))
    except ValueError:
        return None


def quad_to_bbox(q):
    xs = [q["x1"], q["x2"], q["x3"], q["x4"]]
    ys = [q["y1"], q["y2"], q["y3"], q["y4"]]
    return [min(xs), min(ys), max(xs), max(ys)]


def cord_items(gt_parse):
    """menu -> items. CORD는 menu가 dict 하나로 올 때도 있다(품목 1개짜리)."""
    menu = gt_parse.get("menu", [])
    if isinstance(menu, dict):
        menu = [menu]
    out = []
    for m in menu:
        if not isinstance(m, dict):
            continue
        out.append({
            "name": m.get("nm"),
            "qty": to_num(m.get("cnt")),
            "qty_raw": m.get("cnt"),
            "price": to_num(m.get("price")),
            "price_raw": m.get("price"),
            "unit_price": to_num(m.get("unitprice")),
        })
    return out


def cord_fields(gt_parse):
    """total / sub_total -> 평평한 필드. 키 이름은 CORD 원본을 따른다."""
    fields = {}
    for section in ("sub_total", "total"):
        blk = gt_parse.get(section)
        if isinstance(blk, dict):
            for k, v in blk.items():
                if isinstance(v, (str, int, float)):
                    fields[f"{section}.{k}"] = v
                    n = to_num(v)
                    if n is not None:
                        fields[f"{section}.{k}__num"] = n
    return fields


def cord_layout(gt):
    """valid_line -> 블록 리스트. 단어 quad를 줄 단위로 합쳐 bbox 하나로 만든다."""
    out = []
    for i, line in enumerate(gt.get("valid_line", [])):
        words = line.get("words", [])
        boxes = [quad_to_bbox(w["quad"]) for w in words if "quad" in w]
        if not boxes:
            continue
        bbox = [
            min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes),
        ]
        out.append({
            "block_id": i,
            "category": line.get("category"),
            "group_id": line.get("group_id"),
            "text": " ".join(w.get("text", "") for w in words).strip(),
            "bbox": bbox,
            "words": [
                {"text": w.get("text"), "bbox": quad_to_bbox(w["quad"]), "is_key": w.get("is_key")}
                for w in words if "quad" in w
            ],
        })
    return out


def export_cord(out_root, limit):
    from datasets import load_dataset

    ds = load_dataset("naver-clova-ix/cord-v2")
    img_dir, gt_dir = out_root / "images", out_root / "gt"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    manifest, n = [], 0
    for split, rows in ds.items():
        for i, row in enumerate(rows):
            if limit and n >= limit:
                break
            gt = json.loads(row["ground_truth"])
            meta = gt.get("meta", {})
            size = meta.get("image_size", {})
            img = row["image"]
            w = size.get("width") or img.width
            h = size.get("height") or img.height

            doc_id = f"cord_{split}_{meta.get('image_id', i):04d}"
            rel_img = f"receipt_data/public/cord/images/{doc_id}.jpg"
            # 원본이 PNG인데 clean_v1 쪽이 전부 jpg라 맞춘다. RGB 변환은 필수(P/RGBA 섞여 있다).
            img.convert("RGB").save(img_dir / f"{doc_id}.jpg", quality=95)

            gp = gt.get("gt_parse", {})
            doc = {
                "doc_id": doc_id,
                "image": rel_img,
                "width": w,
                "height": h,
                "source": "cord-v2",
                "capture": "photo",          # CORD는 실사 촬영본이다 (합성 아님)
                "quality": None,             # 원본에 품질 라벨이 없다
                "subdomain": "음식점",        # CORD는 대부분 인도네시아 식당 영수증
                "split": split if split != "validation" else "val",
                "gt_status": "verified",     # 공개셋 배포본 기준
                "fields": cord_fields(gp),
                "items": cord_items(gp),
                "layout": cord_layout(gt),
                "raw_gt": gt,
            }
            (gt_dir / f"{doc_id}.json").write_text(
                json.dumps(doc, ensure_ascii=False), encoding="utf-8"
            )
            manifest.append({
                "doc_id": doc_id, "image": rel_img, "width": w, "height": h,
                "source": "cord-v2", "capture": "photo",
                "split": doc["split"], "gt_status": "verified",
                "n_fields": len(doc["fields"]), "n_items": len(doc["items"]),
                "n_layout": len(doc["layout"]),
            })
            n += 1
        if limit and n >= limit:
            break

    mf = out_root / "manifest.jsonl"
    with mf.open("w", encoding="utf-8") as fh:
        for m in manifest:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    return manifest


def export_sroie(out_root, limit):
    """SROIE (ICDAR 2019 Task 3). podbilabs/sroie-donut 은 CORD와 같은 donut 포맷이라
    gt_parse 를 그대로 쓴다. 단 SROIE는 품목(menu)이 없고 헤더 4필드만 있다:
    company / date / address / total. 그래서 items 는 항상 빈 리스트다.

    darentang/sroie 를 안 쓴 이유: image_path 가 HF 빌드 머신 경로라 이미지가 없다.
    """
    from datasets import load_dataset

    ds = load_dataset("podbilabs/sroie-donut")
    img_dir, gt_dir = out_root / "images", out_root / "gt"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    manifest, n = [], 0
    for split, rows in ds.items():
        for i, row in enumerate(rows):
            if limit and n >= limit:
                break
            try:
                gt = json.loads(row["ground_truth"])
            except (json.JSONDecodeError, TypeError):
                continue
            gp = gt.get("gt_parse", {}) or {}
            img = row["image"]
            w, h = img.width, img.height

            doc_id = f"sroie_{split if split != 'validation' else 'val'}_{i:04d}"
            rel_img = f"receipt_data/public/sroie/images/{doc_id}.jpg"
            img.convert("RGB").save(img_dir / f"{doc_id}.jpg", quality=95)

            fields = {}
            for k in ("company", "date", "address", "total"):
                if gp.get(k) is not None:
                    fields[k] = gp[k]
            if "total" in fields:
                tn = to_num(fields["total"])
                if tn is not None:
                    fields["total__num"] = tn

            doc = {
                "doc_id": doc_id,
                "image": rel_img,
                "width": w,
                "height": h,
                "source": "sroie-2019",
                "capture": "scan",           # SROIE는 스캔본이다 (CORD의 촬영본과 대비된다)
                "quality": None,
                "subdomain": "유통",
                "split": split if split != "validation" else "val",
                "gt_status": "verified",
                "fields": fields,
                "items": [],                 # SROIE에는 품목 라벨이 없다
                "layout": [],                # Task 3 배포본에는 bbox가 없다
                "raw_gt": gt,
            }
            (gt_dir / f"{doc_id}.json").write_text(
                json.dumps(doc, ensure_ascii=False), encoding="utf-8"
            )
            manifest.append({
                "doc_id": doc_id, "image": rel_img, "width": w, "height": h,
                "source": "sroie-2019", "capture": "scan",
                "split": doc["split"], "gt_status": "verified",
                "n_fields": len(fields), "n_items": 0, "n_layout": 0,
            })
            n += 1
        if limit and n >= limit:
            break

    mf = out_root / "manifest.jsonl"
    with mf.open("w", encoding="utf-8") as fh:
        for m in manifest:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    return manifest


EXPORTERS = {"cord": export_cord, "sroie": export_sroie}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cord", choices=sorted(EXPORTERS) + ["all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT))
    args = ap.parse_args()

    names = sorted(EXPORTERS) if args.dataset == "all" else [args.dataset]
    for name in names:
        out_root = Path(args.out) / name
        manifest = EXPORTERS[name](out_root, args.limit)
        _report(name, out_root, manifest)


def _report(name, out_root, manifest):
    from collections import Counter
    c = Counter(m["split"] for m in manifest)
    print(f"\n{name}: {len(manifest)}장  splits={dict(c)}")
    print(f"  이미지 {out_root/'images'}")
    print(f"  라벨   {out_root/'gt'}")
    print(f"  매니페스트 {out_root/'manifest.jsonl'}")
    if manifest:
        avg_items = sum(m["n_items"] for m in manifest) / len(manifest)
        avg_layout = sum(m["n_layout"] for m in manifest) / len(manifest)
        print(f"  평균 품목 {avg_items:.1f}개 / 평균 layout 블록 {avg_layout:.1f}개")


if __name__ == "__main__":
    main()
