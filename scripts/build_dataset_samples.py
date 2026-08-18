"""학습 데이터셋 8종의 샘플 이미지를 뽑아 주간보고에 첨부할 그림을 만든다.

크롭 태스크(receipt_crop_*)는 크롭만 보면 어디서 잘렸는지 알 수 없으므로
원본 페이지에 bbox 를 그려 위치를 함께 보여준다. 나머지는 이미지 자체가 학습
단위라 프레임만 두른다.

usage: python3 scripts/build_dataset_samples.py <출력폴더>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "receipt_data"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

TX, TB, ACC = (27, 175, 122), (235, 104, 52), (44, 86, 120)
MAXW = 900

# (task, 파일이름, 몇 장, 크롭이면 원본에 bbox 를 그릴지)
PICK = [
    ("html_table", "01_표크롭_pubtabnet", 2, False),
    ("receipt_crop_table", "02_표크롭_영수증", 2, True),
    ("receipt_crop_text", "03_텍스트크롭_영수증", 2, True),
    ("receipt_markdown", "04_영수증_페이지", 1, False),
    ("page_ocr", "05_페이지OCR_aihub", 1, False),
    ("parsing/전체텍스트", "06_문서전사_multimodal", 1, False),
    ("doc_parsing/government", "07_공공문서", 1, False),
    ("doc_parsing/paper", "08_논문", 1, False),
]


def crop_bbox(crop_id):
    """크롭이 원본 어디서 나왔는지 찾는다. 학습 크롭은 train_crops 매니페스트에 있다."""
    man = RD / "train_crops/manifest.jsonl"
    if not man.exists():
        return None
    for line in man.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["crop_id"] == crop_id:
            return r
    return None


def page_of(doc_id):
    for split in ("train", "val", "test"):
        for ext in (".jpg", ".png", ".jpeg"):
            p = RD / f"labeled/{split}/images/{doc_id}{ext}"
            if p.exists():
                return p, RD / f"labeled/{split}/json/{doc_id}.json"
    return None, None


def scale(im, w=MAXW):
    if im.width <= w:
        return im
    return im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)


def save_plain(im, name, color):
    """이미지 자체가 학습 단위인 경우 — 테두리만 둘러 경계를 보이게 한다."""
    im = scale(im.convert("RGB"))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width - 1, im.height - 1], outline=color, width=3)
    p = OUT / f"{name}.png"
    im.save(p, optimize=True)
    return p


def save_with_source(crop_im, row, name, color):
    """크롭 + 원본에서의 위치를 나란히 붙인다."""
    doc = row["crop_id"].replace("train_", "").rsplit("_", 1)[0]
    src, jpath = page_of(doc)
    crop_im = scale(crop_im.convert("RGB"), 460)
    if src is None or not jpath.exists():
        return save_plain(crop_im, name, color)
    page = Image.open(src).convert("RGB")
    W, H = page.size
    pj = json.loads(jpath.read_text(encoding="utf-8"))["result"]["elements"][0]["json"]
    if isinstance(pj, str):
        pj = json.loads(pj)
    sx, sy = W / pj["width"], H / pj["height"]
    bb = [row["bbox"][0] * sx, row["bbox"][1] * sy,
          row["bbox"][2] * sx, row["bbox"][3] * sy]
    d = ImageDraw.Draw(page)
    d.rectangle(bb, outline=color, width=max(4, int(W / 200)))
    page = scale(page, 420)
    # 좌: 원본에서의 위치 · 우: 잘라낸 크롭
    gap, pad = 24, 14
    cw = max(page.width + crop_im.width + gap + pad * 2, 300)
    ch = max(page.height, crop_im.height) + pad * 2
    canvas = Image.new("RGB", (cw, ch), (246, 246, 243))
    canvas.paste(page, (pad, pad))
    canvas.paste(crop_im, (pad + page.width + gap,
                           pad + (ch - pad * 2 - crop_im.height) // 2))
    dd = ImageDraw.Draw(canvas)
    dd.rectangle([pad + page.width + gap - 1,
                  pad + (ch - pad * 2 - crop_im.height) // 2 - 1,
                  pad + page.width + gap + crop_im.width,
                  pad + (ch - pad * 2 - crop_im.height) // 2 + crop_im.height],
                 outline=color, width=3)
    p = OUT / f"{name}.png"
    canvas.save(p, optimize=True)
    return p


def main():
    rows = defaultdict(list)
    for line in (RD / "exp004c_260813/train.jsonl").read_text(
            encoding="utf-8").split("\n"):
        if line.strip():
            r = json.loads(line)
            rows[r["task"]].append(r)

    made = []
    for task, name, n, with_src in PICK:
        pool = rows.get(task, [])
        if not pool:
            print(f"  ✗ {task} 없음")
            continue
        # 너무 작거나 큰 것 말고 중간 크기를 고른다
        pool = sorted(pool, key=lambda r: len(json.dumps(r["messages"],
                                                         ensure_ascii=False)))
        picks = [pool[len(pool) // 3], pool[len(pool) * 2 // 3]][:n]
        for i, r in enumerate(picks):
            ip = ROOT / r["images"][0]
            if not ip.exists():
                ip = RD / r["images"][0].replace("receipt_data/", "")
            if not ip.exists():
                print(f"  ✗ 이미지 없음 {r['images'][0]}")
                continue
            im = Image.open(ip)
            color = TB if "table" in task or task == "html_table" else TX
            fn = name if n == 1 else f"{name}_{i+1}"
            meta = crop_bbox(r["doc_id"]) if with_src else None
            p = (save_with_source(im, meta, fn, color) if meta
                 else save_plain(im, fn, color))
            gt = r["messages"][-1]["content"]
            made.append((task, p, im.size, gt))
            print(f"  {p.name:34} {im.size}  gt {len(gt):,}자")

    (OUT / "_gt.json").write_text(
        json.dumps([{"task": t, "file": p.name, "size": list(s), "gt": g}
                    for t, p, s, g in made], ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n{len(made)}장 저장 → {OUT}")


if __name__ == "__main__":
    main()
