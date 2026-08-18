"""
블록 단위 학습셋 빌더 + 시각화.

운영 파이프라인이 Layout Detection -> 블록 크롭 -> VLM 파싱 구조이므로,
학습도 같은 구조(블록 크롭 -> 블록 텍스트)로 맞춘다.
페이지 통째로 넣는 것보다 블록당 유효 해상도가 크게 올라간다.

주의: JSON 좌표계와 실제 이미지 해상도가 다른 문서가 있다.
      (연구개발계획: JSON 1386x1960 vs PNG 595x841)
      반드시 스케일 보정 후 크롭한다.

출력:
    eval_dataset/train/blocks/*.png      크롭 이미지
    eval_dataset/train/train_block.jsonl 학습셋
    eval_dataset/train/viz/*.png         블록 경계 시각화
"""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "eval_dataset"
OUT = DS / "train"
CROP_DIR = OUT / "blocks"
VIZ_DIR = OUT / "viz"

INSTRUCTION = {
    "table": "이 표 이미지를 HTML <table>로 변환하세요.",
    "_default": "이 이미지의 텍스트를 그대로 옮겨 적으세요.",
}
PAD = 6          # 크롭 여백(px). 글자가 경계에 붙어 잘리는 것 방지
MIN_SIDE = 12    # 이보다 작으면 글자 자체가 안 보인다
MIN_CHARS = 2
MIN_CROP_H = 48  # 저해상도 원본(연구개발계획: 0.43배 축소본) 대응.
                 # 업스케일은 정보를 늘리지 못하지만, 문서 간 글자 크기를 맞춰
                 # 모델이 일관된 스케일을 보게 한다. OCR 전처리의 표준 절차.
MAX_CROP_H = 1600

IMG_ALIAS = {
    "Table Sample": "doc001_saltlux_bs_page000",
    "연구개발계획14_highnocaption": "연구개발계획14",
    "연구개발계획15_hightnocaption": "연구개발계획15",
}
LABEL_COLOR = {
    "table": (220, 40, 40), "text": (40, 120, 220),
    "paragraph_title": (30, 170, 90), "doc_title": (30, 170, 90),
    "header": (200, 130, 0), "footer": (200, 130, 0),
    "image": (150, 60, 200), "chart": (150, 60, 200),
    "figure_title": (0, 170, 170), "vision_footnote": (120, 120, 120),
    "number": (120, 120, 120),
}


def find_image(stem):
    name = IMG_ALIAS.get(stem, stem)
    for ext in (".png", ".jpg", ".jpeg"):
        p = DS / "images" / (name + ext)
        if p.exists():
            return p
    return None


def gt_corrections(gt):
    return {e["parser_output"]: e["ground_truth"] for e in gt.get("errors", [])}


def main():
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    gt = json.loads((DS / "annotations/doc001_saltlux_bs_page000.json")
                    .read_text(encoding="utf-8"))
    fixes = gt_corrections(gt)

    samples, stats, scale_warn = [], {}, []
    for jf in sorted((DS / "json").glob("*.json")):
        stem = jf.stem
        imgp = find_image(stem)
        if not imgp:
            continue
        doc = json.loads(jf.read_text(encoding="utf-8"))
        im = Image.open(imgp).convert("RGB")

        # 좌표계 보정 — JSON 기준 해상도와 실제 이미지가 다를 수 있다
        sx = im.width / doc["width"]
        sy = im.height / doc["height"]
        if abs(sx - 1) > 0.02:
            scale_warn.append((stem, f"{doc['width']}x{doc['height']} -> "
                                     f"{im.width}x{im.height} (x{sx:.2f})"))

        scores = {tuple(round(v) for v in b["coordinate"]): b["score"]
                  for b in doc.get("layout_det_res", {}).get("boxes", [])}

        viz = im.copy()
        d = ImageDraw.Draw(viz)
        try:
            font = ImageFont.load_default(size=max(11, int(im.width / 70)))
        except Exception:
            font = ImageFont.load_default()

        for i, b in enumerate(doc["parsing_res_list"]):
            bb = b.get("block_bbox") or []
            label = b.get("block_label", "text")
            content = str(b.get("block_content", "")).strip()
            if len(bb) != 4:
                continue
            key = tuple(round(v) for v in bb)
            score = next((v for k, v in scores.items()
                          if max(abs(a - c) for a, c in zip(k, key)) < 6), None)

            x0, y0, x1, y1 = [bb[0] * sx, bb[1] * sy, bb[2] * sx, bb[3] * sy]
            col = LABEL_COLOR.get(label, (100, 100, 100))
            d.rectangle([x0, y0, x1, y1], outline=col, width=3)
            tag = f"{i}:{label}" + (f" {score:.2f}" if score is not None else "")
            d.text((x0 + 3, max(0, y0 - 15)), tag, fill=col, font=font)

            if not content or len(content) < MIN_CHARS:
                stats[f"{label}:빈내용"] = stats.get(f"{label}:빈내용", 0) + 1
                continue
            if (x1 - x0) < MIN_SIDE or (y1 - y0) < MIN_SIDE:
                stats[f"{label}:너무작음"] = stats.get(f"{label}:너무작음", 0) + 1
                continue

            crop = im.crop((max(0, x0 - PAD), max(0, y0 - PAD),
                            min(im.width, x1 + PAD), min(im.height, y1 + PAD)))
            if crop.height < MIN_CROP_H:            # 저해상도 원본 보정
                f = min(MIN_CROP_H / crop.height, MAX_CROP_H / max(crop.width, 1))
                if f > 1:
                    crop = crop.resize((max(1, int(crop.width * f)),
                                        max(1, int(crop.height * f))), Image.LANCZOS)
            if stem == "Table Sample":
                for w, r in fixes.items():
                    content = content.replace(f">{w}<", f">{r}<")

            cp = CROP_DIR / f"{stem}__{i:02d}_{label}.png"
            crop.save(cp)
            samples.append({
                "id": f"{stem}#{i}",
                "images": [str(cp.relative_to(ROOT))],
                "messages": [
                    {"role": "user",
                     "content": "<image>" + INSTRUCTION.get(label, INSTRUCTION["_default"])},
                    {"role": "assistant", "content": content},
                ],
                "_label": label, "_score": score, "_chars": len(content),
                "_size": f"{crop.width}x{crop.height}",
            })
            stats[label] = stats.get(label, 0) + 1

        viz.save(VIZ_DIR / f"{stem}.png")

    random.Random(0).shuffle(samples)
    n_val = max(2, len(samples) // 10)
    val, train = samples[:n_val], samples[n_val:]
    for name, rows in (("train_block", train), ("val_block", val)):
        p = OUT / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items()
                                    if not k.startswith("_")}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)}건 -> {p}")

    print(f"\n=== 라벨별 블록 수 ===")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k:<24}{v}")
    tot = sum(s["_chars"] for s in samples)
    print(f"\n총 {len(samples)}샘플 / 타겟 {tot:,}자 (평균 {tot // max(len(samples),1)}자)")
    print(f"크롭 이미지 -> {CROP_DIR}")
    print(f"시각화     -> {VIZ_DIR}")
    if scale_warn:
        print(f"\n=== 좌표계 스케일 보정 적용 {len(scale_warn)}건 ===")
        for s, w in scale_warn[:6]:
            print(f"  {s}: {w}")


if __name__ == "__main__":
    main()
