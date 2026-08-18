"""
VLM 학습셋 빌더.  (이미지, 정답 마크다운) 쌍을 만든다.

정답 소스:
  - 기본: Document Studio 파싱 결과 (parsing_res_list 배열 순서 = 읽기 순서)
  - 재무제표: 사람이 교정한 GT 계정명으로 치환 (파서 원본에 한글 오류 11건)

빈 내용 블록(도식이 image로 오분류된 건)은 학습 타겟에서 제외한다.
그 자리에 빈 값을 학습시키면 "도식은 무시하라"를 가르치게 되기 때문.

출력: JSONL. 한 줄 = 한 샘플.
    {"images": [...], "messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "eval_dataset"
IMG_DIR = DS / "images"
JSON_DIR = DS / "json"
GT = DS / "annotations/doc001_saltlux_bs_page000.json"
OUT_DIR = DS / "train"

INSTRUCTION = "이 문서 이미지를 마크다운으로 변환하세요. 표는 HTML <table>로 유지하세요."

# json 파일명 -> 이미지 파일명 (이름이 어긋나는 것들)
IMG_ALIAS = {
    "Table Sample": "doc001_saltlux_bs_page000",
    "연구개발계획14_highnocaption": "연구개발계획14",
    "연구개발계획15_hightnocaption": "연구개발계획15",
}


def find_image(stem):
    name = IMG_ALIAS.get(stem, stem)
    for ext in (".png", ".jpg", ".jpeg"):
        p = IMG_DIR / (name + ext)
        if p.exists():
            return p
    return None


def block_to_md(b):
    label = b.get("block_label")
    c = str(b.get("block_content", "")).strip()
    if not c:
        return None                      # 빈 도식 블록 — 타겟에서 제외
    if label == "table":
        return c                         # HTML 그대로
    if label in ("doc_title", "paragraph_title"):
        return f"## {c}"
    if label in ("figure_title", "vision_footnote"):
        return f"*{c}*"
    if label in ("chart", "image"):
        return None                      # 캡션은 비결정적이라 학습 타겟에 부적합
    return c


def build_target(doc):
    parts = []
    for b in doc["parsing_res_list"]:     # 배열 순서 = 읽기 순서 (파트장 확인)
        md = block_to_md(b)
        if md:
            parts.append(md)
    return "\n\n".join(parts)


def apply_gt_corrections(target, gt):
    """재무제표: 파서의 잘못된 계정명을 GT 정답으로 치환."""
    n = 0
    for e in gt.get("errors", []):
        wrong, right = e["parser_output"], e["ground_truth"]
        if wrong in target:
            target = target.replace(f">{wrong}<", f">{right}<")
            n += 1
    return target, n


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = json.loads(GT.read_text(encoding="utf-8"))

    samples, skipped = [], []
    for jf in sorted(JSON_DIR.glob("*.json")):
        stem = jf.stem
        img = find_image(stem)
        if not img:
            skipped.append((stem, "이미지 없음"))
            continue
        doc = json.loads(jf.read_text(encoding="utf-8"))
        target = build_target(doc)
        if len(target) < 40:
            skipped.append((stem, f"타겟 너무 짧음 ({len(target)}자)"))
            continue

        note = ""
        if stem == "Table Sample":
            target, n = apply_gt_corrections(target, gt)
            note = f"GT 교정 {n}건 적용"

        samples.append({
            "id": stem,
            "images": [str(img.relative_to(ROOT))],
            "messages": [
                {"role": "user", "content": f"<image>{INSTRUCTION}"},
                {"role": "assistant", "content": target},
            ],
            "_chars": len(target),
            "_note": note,
        })

    # 재무제표(유일한 검증 GT)는 반드시 학습에 넣고, 검증셋은 뒤에서 2건
    val = samples[-2:]
    train = samples[:-2]

    for name, rows in (("train", train), ("val", val)):
        p = OUT_DIR / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items()
                                    if not k.startswith("_")}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)}건 -> {p}")

    print(f"\n{'문서':<28}{'타겟 글자수':>10}  비고")
    print("-" * 60)
    for s in samples:
        print(f"{s['id'][:27]:<28}{s['_chars']:>10}  {s['_note']}")
    if skipped:
        print(f"\n제외 {len(skipped)}건:")
        for s, why in skipped:
            print(f"  {s} — {why}")
    tot = sum(s["_chars"] for s in samples)
    print(f"\n총 {len(samples)}건 / 타겟 {tot:,}자 (평균 {tot//max(len(samples),1):,}자)")


if __name__ == "__main__":
    main()
