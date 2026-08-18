"""
라벨링된 영수증에서 테이블/텍스트 블록 크롭을 잘라 평가셋을 만든다.

평가 방식이 "크롭 이미지를 VLM 에 직접 호출"이라 페이지 단위 정답으로는 못 쓴다.
블록 하나 = 평가 1건이 되도록 bbox 로 자르고, 그 블록의 정답을 같이 붙인다.

  table  블록 -> 정답은 HTML <table>  -> TEDS 로 채점
  text   블록 -> 정답은 평문           -> CER 로 채점

좌표계 주의. parsing_res_list 의 bbox 는 파서가 리사이즈한 페이지 좌표라
원본 이미지 크기와 다르다(이 데이터셋은 배율이 0.37~5.4 로 제각각이고,
가로/세로 배율이 다른 장도 있다). width/height 비로 축을 따로 되돌린다.

usage:
    python3 scripts/build_eval_crops.py                    # test+val 전량
    python3 scripts/build_eval_crops.py --splits test      # test 만
"""
import argparse
import ast
import json
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "receipt_data/labeled"

# 표는 HTML 로, 나머지 글자 블록은 평문으로 채점한다.
TABLE_LABELS = {"table"}
TEXT_LABELS = {"text", "doc_title", "paragraph_title", "footer", "header",
               "vision_footnote", "display_formula", "figure_title"}
# 글자가 없는 블록. 크롭해봐야 전사할 대상이 아니다.
SKIP_LABELS = {"image", "seal", "header_image", "footer_image"}


def page_json(doc):
    els = (doc.get("result") or {}).get("elements") or []
    return els[0].get("json") if els else None


def as_bbox(v):
    """block_bbox 는 리스트인 장도 있고 '[1, 2, 3, 4]' 문자열인 장도 있다."""
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        return [float(x) for x in v]
    except (TypeError, ValueError):
        return None


def is_transposed(W, H, pw, ph, tol=0.12):
    """이미지와 json 의 가로세로 비가 서로 역수면 좌표계가 90도 돌아간 것이다.

    4000x3000(1.333) 인데 json 이 1470x1960(0.750) 인 식. 배율 보정으로는 못
    고치고, 회전을 시켜야 한다. 회전 후 배율이 등방(sx==sy)이 되는 것으로
    확인된다.
    """
    ar_i, ar_j = W / H, pw / ph
    if abs(ar_i - ar_j) < 0.3:            # 애초에 방향이 같으면 회전 아님
        return False
    return abs(ar_i - 1 / ar_j) < tol * max(1.0, ar_i)


def crop(im, bbox, pad_ratio, min_pad):
    """박스에 딱 맞춰 자르면 글자 끝이 잘려 전사가 나빠진다. 크기 비례로 여백을 준다."""
    W, H = im.size
    x1, y1, x2, y2 = bbox
    px = max(min_pad, (x2 - x1) * pad_ratio)
    py = max(min_pad, (y2 - y1) * pad_ratio)
    box = (max(0, int(x1 - px)), max(0, int(y1 - py)),
           min(W, int(x2 + px)), min(H, int(y2 + py)))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return None
    return im.crop(box)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "val"])
    ap.add_argument("--out", default=str(ROOT / "receipt_data/eval_crops"))
    ap.add_argument("--pad-ratio", type=float, default=0.02)
    ap.add_argument("--min-pad", type=int, default=4)
    ap.add_argument("--min-chars", type=int, default=1,
                    help="정답 글자 수가 이보다 적은 블록은 버린다")
    # 학습/검증셋(build_labeled_valtest.py)과 같은 문서를 빼야 숫자가 맞는다.
    # receipt19 는 receipt21 과 같은 이미지를 두 번 라벨링한 것이고, 원본 대조
    # 결과 receipt21 이 맞아 19 를 뺀다.
    ap.add_argument("--drop-ids", nargs="*", default=["receipt19"],
                    help="제외할 doc_id")
    args = ap.parse_args()

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows, stats, labels = [], Counter(), Counter()
    for split in args.splits:
        jdir, idir = SRC / split / "json", SRC / split / "images"
        if not jdir.exists():
            raise SystemExit(f"없는 split: {jdir}")
        for jf in sorted(jdir.glob("*.json")):
            stem = jf.stem
            if stem in set(args.drop_ids or []):
                stats["제외 지정"] += 1
                continue
            cand = [p for p in idir.glob(stem + ".*")]
            if not cand:
                stats["이미지없음"] += 1
                continue
            im = Image.open(cand[0]).convert("RGB")
            W, H = im.size

            pj = page_json(json.loads(jf.read_text(encoding="utf-8")))
            if not pj:
                stats["페이지JSON없음"] += 1
                continue
            pw, ph = pj.get("width"), pj.get("height")
            if not (pw and ph):
                stats["크기정보없음"] += 1
                continue

            # 가로세로가 뒤집힌 장이 있다. 파서가 세운 이미지에 좌표를 매겼는데
            # 원본은 눕혀져 있는 경우다(이미지 AR ≈ 1/json AR). 이때는 배율로
            # 못 고친다 — 축별로 나눠봐야 좌표계가 90도 돌아간 채라 엉뚱한 데를
            # 자른다. 원본을 시계방향 90도 돌려 좌표계를 맞춘 뒤 자른다.
            # 방향은 receipt34('LOTTERIA')·receipt67('매출전표')로 실측해 정했다.
            if is_transposed(W, H, pw, ph):
                im = im.transpose(Image.ROTATE_270)      # 시계방향 90도
                W, H = im.size
                stats["회전보정"] += 1

            # 파서 좌표 -> 원본 이미지 좌표. 가로/세로 배율이 다른 장이 있어 따로 나눈다.
            sx, sy = W / pw, H / ph

            for b in pj.get("parsing_res_list", []):
                label = b.get("block_label")
                labels[label] += 1
                if label in SKIP_LABELS:
                    stats["스킵(비텍스트)"] += 1
                    continue
                if label in TABLE_LABELS:
                    task = "table"
                elif label in TEXT_LABELS:
                    task = "text"
                else:
                    stats[f"미분류라벨({label})"] += 1
                    continue

                bb = as_bbox(b.get("block_bbox"))
                if not bb:
                    stats["bbox없음"] += 1
                    continue
                gt = (b.get("block_content") or "").strip()
                if len(gt) < args.min_chars:
                    stats["정답없음"] += 1
                    continue
                # 표인데 HTML 이 아니면 TEDS 를 못 매긴다. 채점 불가라 뺀다.
                if task == "table" and "<table" not in gt.lower():
                    stats["표인데HTML아님"] += 1
                    continue

                c = crop(im, [bb[0] * sx, bb[1] * sy, bb[2] * sx, bb[3] * sy],
                         args.pad_ratio, args.min_pad)
                if c is None:
                    stats["크롭너무작음"] += 1
                    continue

                cid = f"{split}_{stem}_{b.get('block_id', len(rows))}"
                rel = f"images/{cid}.png"
                c.save(out / rel)
                rows.append({"crop_id": cid, "doc_id": stem, "split": split,
                             "task": task, "label": label,
                             "bbox": [round(v, 1) for v in bb],
                             "size": list(c.size), "image": rel, "gt": gt})
                stats[f"생성/{task}"] += 1

    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")

    docs = len({r["doc_id"] for r in rows})
    print(f"영수증 {docs}장 -> 크롭 {len(rows):,}건")
    for k, v in sorted(stats.items()):
        print(f"  {k:<22}{v:>6,}")
    print(f"\n라벨 원본 분포: {dict(labels.most_common())}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
