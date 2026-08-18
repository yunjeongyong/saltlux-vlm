"""
데이터셋 인벤토리 — 엑셀 시트용 통계를 실측해서 뽑는다.

수량·이미지 규격은 문서에 적힌 값을 옮기지 않고 파일을 직접 세고 연다.
규격은 표본을 열어 중간값과 범위를 낸다(PIL 은 헤더만 읽어 크기를 알아내므로
수천 장이어도 빠르다).

usage:
    python3 scripts/inventory_datasets.py
    python3 scripts/inventory_datasets.py --sample 500 --out /tmp/inv.json
"""
import argparse
import json
import statistics as st
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imgs(d, pattern="**/*"):
    d = ROOT / d
    if not d.exists():
        return []
    return [p for p in d.glob(pattern)
            if p.is_file() and p.suffix.lower() in IMG_EXT]


def dims(paths, sample):
    """표본을 고르게 뽑아 (w, h) 를 잰다. 파일 순서가 곧 수집 순서라 등간격으로 뽑는다."""
    if not paths:
        return None
    step = max(1, len(paths) // sample)
    sel = paths[::step][:sample]
    out = []
    for p in sel:
        try:
            with Image.open(p) as im:
                out.append(im.size)
        except Exception:
            continue
    if not out:
        return None
    w = sorted(x[0] for x in out)
    h = sorted(x[1] for x in out)
    mp = sorted(x[0] * x[1] / 1e6 for x in out)
    return {
        "n_sampled": len(out),
        "w_med": w[len(w) // 2], "h_med": h[len(h) // 2],
        "w_min": w[0], "w_max": w[-1], "h_min": h[0], "h_max": h[-1],
        "mp_med": round(mp[len(mp) // 2], 2),
        "ext": sorted({p.suffix.lower().lstrip(".") for p in sel}),
    }


def lines(p):
    p = ROOT / p
    if not p.exists():
        return 0
    return sum(1 for l in p.open(encoding="utf-8") if l.strip())


def count(d, pattern="**/*.json"):
    d = ROOT / d
    return len(list(d.glob(pattern))) if d.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "receipt_data/review/inventory.json"))
    args = ap.parse_args()

    # (키, 이미지 디렉토리, glob) — 이미지 규격을 잴 대상
    IMG_SETS = {
        "aihub71299_val_img": ("aihub/71299/val_image", "**/*"),
        "aihub71299_pkg": ("vlm_dataset_upload/train/public/aihub-71299-ocr/data", "*"),
        "aihub71845": ("aihub/71845/extracted", "**/*"),
        "korie_ocr": ("receipt_data/korie/ocr", "**/*"),
        "korie_kid": ("receipt_data/korie/kid", "**/*"),
        "sroie": ("receipt_data/public/sroie", "**/*"),
        "cord": ("receipt_data/public/cord", "**/*"),
        "receipts3000": ("receipt_data/receipts3000/images", "*"),
        "crawl_google": ("receipt_data/crawl_google/images", "*"),
        "crawl_google_up": ("receipt_data/crawl_google/images_up", "*"),
        "labeled_train": ("receipt_data/labeled/train/images", "*"),
        "labeled_val": ("receipt_data/labeled/val/images", "*"),
        "labeled_test": ("receipt_data/labeled/test/images", "*"),
        "receipt_pkg": ("vlm_dataset_upload/train/human_annotated/receipt/data", "*"),
        "eval_crops": ("receipt_data/eval_crops/images", "*"),
    }

    inv = {}
    for k, (d, pat) in IMG_SETS.items():
        ps = sorted(imgs(d, pat))
        inv[k] = {"n_images": len(ps), "dims": dims(ps, args.sample)}
        m = inv[k]["dims"]
        print(f"{k:<22} 이미지 {len(ps):>7,}장"
              + (f" | 중간값 {m['w_med']}x{m['h_med']} ({m['mp_med']}MP)"
                 f" | 범위 {m['w_min']}~{m['w_max']} x {m['h_min']}~{m['h_max']}"
                 f" | {'/'.join(m['ext'])}" if m else ""))

    # jsonl 행 수
    JSONL = {
        "trainset_v3_train": "receipt_data/trainset_v3/train.jsonl",
        "trainset_v3_val": "receipt_data/trainset_v3/val.jsonl",
        "trainset_v3_test": "receipt_data/trainset_v3/test.jsonl",
        "upload_mix_train": "receipt_data/upload_mix/train.jsonl",
        "upload_mix_val": "receipt_data/upload_mix/val.jsonl",
        "upload_mix_test": "receipt_data/upload_mix/test.jsonl",
        "unified_train": "receipt_data/unified/train.jsonl",
        "unified_val": "receipt_data/unified/val.jsonl",
        "unified_test": "receipt_data/unified/test.jsonl",
        "pkg_receipt": "vlm_dataset_upload/train/human_annotated/receipt/train.jsonl",
        "pkg_aihub": "vlm_dataset_upload/train/public/aihub-71299-ocr/train.jsonl",
        "eval_crops": "receipt_data/eval_crops/manifest.jsonl",
    }
    print()
    for k, p in JSONL.items():
        inv[k] = {"n_rows": lines(p)}
        print(f"{k:<22} {inv[k]['n_rows']:>7,}행")

    # 라벨 파일 수
    print()
    for k, d, pat in [("labeled_train_json", "receipt_data/labeled/train/json", "*.json"),
                      ("labeled_val_json", "receipt_data/labeled/val/json", "*.json"),
                      ("labeled_test_json", "receipt_data/labeled/test/json", "*.json"),
                      ("receipts3000_gt", "receipt_data/receipts3000/gt", "*.json"),
                      ("clean_v1_parse", "receipt_data/clean_v1/parse", "*.json")]:
        inv[k] = {"n_files": count(d, pat)}
        print(f"{k:<22} {inv[k]['n_files']:>7,}개")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inv, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
