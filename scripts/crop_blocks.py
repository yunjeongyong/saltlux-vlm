"""
bbox 에서 블록 크롭 이미지를 잘라낸다 (파이프라인 2단계 `2_crop`).

크롭 1장마다 세 가지가 따라붙는다. 셋 다 이미 있으므로 새로 만들 게 없다.
  label         파서가 붙인 블록 라벨 (bbox/*.json)
  text          파서가 읽은 블록 텍스트 (parse/*.json -> parsing_res_list.block_content)
  bbox          원본 좌표계 좌표

text 가 붙는다는 게 중요하다. crop_ocr 학습 데이터를 GT 추출 API 없이
바로 만들 수 있다는 뜻이다. 다만 파서 OCR 이 정답이므로 noisy supervision 이고,
사전학습 단계용이지 최종 파인튜닝용은 아니다.

bbox 와 parsing_res_list 는 좌표로 맞춘다. 두 배열의 순서가 같다는 보장이
없어서 인덱스로 짝지으면 라벨과 텍스트가 어긋난다.

패딩: 박스에 딱 맞춰 자르면 글자 끝이 잘려 전사 품질이 떨어진다.
박스 크기에 비례해 여백을 준다.

usage:
    python3 scripts/crop_blocks.py --source clean_v1 --limit 20
    python3 scripts/crop_blocks.py --source all
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

R = Path("/workspace/receipt_data")

SOURCES = {
    "clean_v1":     R / "clean_v1",
    "crawl_google": R / "crawl_google",
    "sroie":        R / "public/sroie",
    "cord":         R / "public/cord",
}

# 글자가 없는 블록은 전사 대상이 아니다. 크롭은 뜨되 기본적으로 건너뛴다.
NON_TEXT_LABELS = {"image", "header_image", "footer_image", "seal"}


def page_json(doc):
    """저장된 API 응답에서 페이지 JSON을 꺼낸다."""
    els = (doc.get("result") or {}).get("elements") or []
    return els[0].get("json") if els else None


def text_by_bbox(page, sw, sh):
    """parsing_res_list 를 원본 좌표계 bbox -> (label, text) 로 만든다.

    parse 쪽 좌표는 리사이즈된 값이므로 bbox/ 와 같은 방식으로 되돌린다.
    """
    if not page:
        return {}
    pw, ph = page.get("width"), page.get("height")
    if not (pw and ph and sw and sh):
        return {}
    sx, sy = pw / sw, ph / sh
    out = {}
    for b in page.get("parsing_res_list", []):
        bb = b.get("block_bbox")
        if not bb or len(bb) != 4:
            continue
        key = (round(bb[0] / sx, 1), round(bb[1] / sy, 1),
               round(bb[2] / sx, 1), round(bb[3] / sy, 1))
        out[key] = (b.get("block_label"), b.get("block_content"))
    return out


def match_text(bbox, table, tol=3.0):
    """좌표로 텍스트를 찾는다. 반올림 오차가 있어 근사 매칭한다."""
    x1, y1, x2, y2 = bbox
    best, bestd = None, tol
    for (kx1, ky1, kx2, ky2), v in table.items():
        d = max(abs(kx1 - x1), abs(ky1 - y1), abs(kx2 - x2), abs(ky2 - y2))
        if d <= bestd:
            best, bestd = v, d
    return best


def enclosing_block(bbox, table, slack=4.0):
    """이 박스를 감싸는 파스 블록을 찾는다.

    파서는 merge_layout_blocks 로 인접 블록을 합친다. 그래서 검출 박스
    (layout_det_res.boxes) 는 파스 블록보다 잘게 쪼개져 있고, 1:1 대응이
    없는 박스가 생긴다. 그 경우 '어느 병합 블록에 속했는지'라도 남겨두면
    나중에 크롭을 되짚을 수 있다. 텍스트는 붙이지 않는다 — 병합 블록의
    텍스트는 이 작은 박스의 텍스트가 아니다.
    """
    x1, y1, x2, y2 = bbox
    best, area = None, None
    for (kx1, ky1, kx2, ky2), v in table.items():
        if kx1 - slack <= x1 and ky1 - slack <= y1 and kx2 + slack >= x2 and ky2 + slack >= y2:
            a = (kx2 - kx1) * (ky2 - ky1)
            if area is None or a < area:   # 가장 작게 감싸는 것
                best, area = v, a
    return best


def crop_one(im, bbox, pad_ratio, min_pad):
    """패딩을 주고 자른다. 이미지 밖으로 나가지 않게 자른다."""
    W, H = im.size
    x1, y1, x2, y2 = bbox
    px = max(min_pad, (x2 - x1) * pad_ratio)
    py = max(min_pad, (y2 - y1) * pad_ratio)
    return im.crop((max(0, int(x1 - px)), max(0, int(y1 - py)),
                    min(W, int(x2 + px)), min(H, int(y2 + py))))


def run_source(name, root, args):
    bbox_dir = root / "bbox"
    parse_dir = root / "parse"
    if not bbox_dir.exists():
        print(f"  건너뜀: {name} (bbox 없음)")
        return None

    out_dir = root / "crops"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(bbox_dir.glob("*.json"))
    if args.limit:
        files = files[:args.limit]

    labels, stats = Counter(), Counter()
    rows = []
    for f in files:
        b = json.loads(f.read_text(encoding="utf-8"))
        img_path = R.parent / b["image"] if not Path(b["image"]).is_absolute() else Path(b["image"])
        try:
            im = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            stats["image_missing"] += 1
            continue

        table = {}
        pf = parse_dir / f.name
        if pf.exists():
            try:
                table = text_by_bbox(page_json(json.loads(pf.read_text(encoding="utf-8"))),
                                     b["width"], b["height"])
            except (json.JSONDecodeError, KeyError):
                stats["parse_unreadable"] += 1

        for i, bx in enumerate(b["boxes"]):
            label = bx["label"]
            if label in NON_TEXT_LABELS and not args.keep_nontext:
                stats["skip_nontext"] += 1
                continue
            x1, y1, x2, y2 = bx["bbox"]
            if (x2 - x1) < args.min_side or (y2 - y1) < args.min_side:
                stats["too_small"] += 1
                continue

            crop = crop_one(im, bx["bbox"], args.pad, args.min_pad)
            cid = f"{b['doc_id']}_b{i:03d}"
            rel = f"receipt_data/{root.relative_to(R)}/crops/images/{cid}.jpg"
            crop.save(img_dir / f"{cid}.jpg", quality=95)

            hit = match_text(bx["bbox"], table)
            text = hit[1] if hit else None
            merged_into = None
            if hit is None:
                enc = enclosing_block(bx["bbox"], table)
                merged_into = enc[0] if enc else None
                stats["merged_no_text" if merged_into else "no_text_match"] += 1

            labels[label] += 1
            stats["kept"] += 1
            rows.append({
                "crop_id": cid,
                "doc_id": b["doc_id"],
                "image": rel,
                "source_image": b["image"],
                "label": label,
                "score": bx["score"],
                "bbox": bx["bbox"],
                "width": crop.width,
                "height": crop.height,
                # 파서가 읽은 텍스트. GT 추출 없이 crop_ocr 학습에 쓸 수 있다.
                # 다만 파서 OCR 이므로 noisy 다.
                "parser_text": text,
                "text_source": "document-parse" if text is not None else None,
                # 텍스트가 없는 이유. 병합 블록에 흡수된 경우 그 블록의 라벨이 들어간다.
                "merged_into": merged_into,
            })

    mf = out_dir / "manifest.jsonl"
    with mf.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_text = sum(1 for r in rows if r["parser_text"])
    print(f"  {name:13} 문서 {len(files):>5} -> 크롭 {stats['kept']:>6}장  "
          f"텍스트 있음 {n_text:>6} ({100 * n_text / max(1, stats['kept']):.0f}%)")
    if stats:
        skipped = {k: v for k, v in stats.items() if k != "kept"}
        if skipped:
            print(f"  {'':13} 제외/경고: {skipped}")
    print(f"  {'':13} 라벨: {dict(labels.most_common(6))}")
    return {"name": name, "docs": len(files), "crops": stats["kept"],
            "with_text": n_text, "dir": str(img_dir), "manifest": str(mf)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="all",
                    choices=sorted(SOURCES) + ["all"])
    ap.add_argument("--limit", type=int, default=0, help="문서 N개만 (확인용)")
    ap.add_argument("--pad", type=float, default=0.04,
                    help="박스 크기 대비 여백 비율")
    ap.add_argument("--min-pad", type=float, default=3.0, help="최소 여백(px)")
    ap.add_argument("--min-side", type=float, default=8.0,
                    help="이보다 얇은 박스는 버린다")
    ap.add_argument("--keep-nontext", action="store_true",
                    help="image/seal 등 글자 없는 블록도 크롭")
    args = ap.parse_args()

    names = sorted(SOURCES) if args.source == "all" else [args.source]
    print(f"크롭 생성 — 소스 {len(names)}개\n")
    results = [r for r in (run_source(n, SOURCES[n], args) for n in names) if r]

    if results:
        tc = sum(r["crops"] for r in results)
        tt = sum(r["with_text"] for r in results)
        print(f"\n합계: 크롭 {tc:,}장 / 텍스트 있음 {tt:,}장 ({100 * tt / max(1, tc):.0f}%)")
        for r in results:
            print(f"  {r['manifest']}")


if __name__ == "__main__":
    main()
