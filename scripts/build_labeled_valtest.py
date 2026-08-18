"""
라벨링 검수본의 val/test split 을 학습용 val/test jsonl 로 만든다.

학습셋(vlm_dataset_upload)의 영수증 정답은 make_vlm_taxonomy 의 조립 규칙으로
만들어졌다. 검증셋을 result.md 같은 다른 경로로 만들면 정답 형식이 미세하게
달라져, 모델이 잘하는데도 eval_loss 가 안 내려가는 착시가 생긴다. 그래서
여기서도 같은 build_markdown 을 쓴다.

이미지는 복사하지 않고 labeled/ 아래 원본을 그대로 가리킨다 (8.9GB 중복 방지).
경로는 train_vlm.py 와 같은 ROOT 기준 상대경로다.

usage:
    python3 scripts/build_labeled_valtest.py
    python3 scripts/build_labeled_valtest.py --splits val test --out ...
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_vlm_taxonomy import build_markdown  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "receipt_data/labeled"
PROMPT = "이 영수증의 내용을 마크다운으로 정리해줘."


def page_json(doc):
    els = (doc.get("result") or {}).get("elements") or []
    return els[0].get("json") if els else None


def build(split, min_chars, html_tables, jdir=None, idir=None):
    """split 이름으로 찾거나(labeled/{split}/{json,images}), 경로를 직접 받는다.

    평가셋이 팀 공용 위치(VLM/dataset/eval/...)로 옮겨가면 json 과 images 가
    한 폴더에 평평하게 놓이거나 서로 다른 곳에 있을 수 있어서, 경로를 직접
    받는 길을 열어 둔다.
    """
    jdir = Path(jdir) if jdir else SRC / split / "json"
    idir = Path(idir) if idir else SRC / split / "images"
    if not jdir.exists():
        raise SystemExit(f"json 디렉토리 없음: {jdir}")
    if not idir.exists():
        raise SystemExit(f"images 디렉토리 없음: {idir}")

    rows, drop = [], Counter()
    for jf in sorted(jdir.glob("*.json")):
        img = next((p for p in idir.glob(jf.stem + ".*")), None)
        if img is None:
            drop["이미지 없음"] += 1
            continue
        pj = page_json(json.loads(jf.read_text(encoding="utf-8")))
        if not pj:
            drop["페이지 JSON 없음"] += 1
            continue
        dropped = []
        md = build_markdown(pj.get("parsing_res_list", []), dropped,
                            keep_html_tables=html_tables).strip()
        if len(md) < min_chars:
            drop["정답 너무 짧음"] += 1
            continue
        if dropped:
            drop["블록 일부 제외된 문서"] += 1
        rows.append({
            "doc_id": jf.stem,
            "task": "receipt_markdown",
            "source": "human_annotated",
            "split": split,
            "images": [str(img.relative_to(ROOT))],
            "messages": [{"role": "user", "content": "<image>" + PROMPT},
                         {"role": "assistant", "content": md}],
        })
    return rows, drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out", default=str(ROOT / "receipt_data/upload_mix"))
    ap.add_argument("--min-chars", type=int, default=10)
    # 학습셋(build_upload_mix.py)과 같은 기본값이어야 한다. 한쪽만 HTML 이면
    # 모델이 배운 형식과 검증 정답이 달라 eval_loss 가 엉뚱하게 높게 나온다.
    ap.add_argument("--html-tables", dest="html_tables",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="표를 HTML 로 유지 (--no-html-tables 로 끔)")
    # val/test 를 나누면 각각 50건 아래로 내려간다. 그 크기에서는 한두 장
    # 차이로 지표가 흔들려 두 숫자를 비교하는 의미가 없다. 합쳐서 같은
    # 99건을 쓰면 적어도 하나의 숫자는 안정적으로 읽힌다.
    # 대신 test 가 더는 독립된 홀드아웃이 아니다 — val 로 모델을 고르는 순간
    # test 점수도 같이 낙관적으로 편향된다. 보고할 때 그 전제를 밝혀야 한다.
    ap.add_argument("--merge", action="store_true",
                    help="val/test 를 합쳐 동일한 셋으로 만든다")
    # 같은 영수증이 다른 doc_id 로 두 번 라벨링된 경우가 있다(receipt19/receipt21).
    # 자동 규칙으로 고르면 '먼저 온 것'처럼 임의 기준이 되므로, 어느 쪽이 맞는지
    # 원본 이미지로 확인한 뒤 틀린 쪽을 여기 명시한다.
    #   receipt19: "고객 콜센터" — 원본은 "콜센타". receipt21 이 맞아 19 를 뺀다.
    ap.add_argument("--drop-ids", nargs="*", default=["receipt19"],
                    help="제외할 doc_id (중복 라벨 등). 빈 목록이면 전부 유지")
    # 평가셋을 labeled/{split} 이 아닌 다른 위치에서 가져올 때 쓴다.
    # 이 두 개를 주면 --splits 는 무시되고 한 벌만 만들어 val/test 양쪽에 쓴다.
    ap.add_argument("--json-dir", help="페이지 JSON 폴더 (직접 지정)")
    ap.add_argument("--image-dir", help="이미지 폴더 (직접 지정)")
    args = ap.parse_args()

    if bool(args.json_dir) != bool(args.image_dir):
        raise SystemExit("--json-dir 과 --image-dir 은 같이 줘야 한다")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    built = {}
    skip = set(args.drop_ids or [])
    splits = ["eval"] if args.json_dir else args.splits
    for split in splits:
        rows, drop = build(split, args.min_chars, args.html_tables,
                           args.json_dir, args.image_dir)
        n0 = len(rows)
        rows = [r for r in rows if r["doc_id"] not in skip]
        if len(rows) != n0:
            print(f"  {split} 제외 지정: {n0 - len(rows)}건 "
                  f"{sorted(skip & {r['doc_id'] for r in built.get(split, [])} or skip)}")
        built[split] = rows
        for k, v in drop.items():
            print(f"  {split} {k}: {v}")

    # 경로를 직접 준 경우는 한 벌뿐이므로 val/test 양쪽에 같은 것을 쓴다.
    if args.merge or args.json_dir:
        merged = [r for split in splits for r in built[split]]
        written = {"val.jsonl": merged, "test.jsonl": merged}
    else:
        written = {f"{s}.jsonl": r for s, r in built.items()}

    for name, rows in written.items():
        f = out / name
        f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
                     + "\n", encoding="utf-8")
        chars = sorted(len(r["messages"][1]["content"]) for r in rows)
        srcs = {r["split"] for r in rows}
        print(f"{name:<12} {len(rows):>4}건  정답 길이 중간값 "
              f"{chars[len(chars)//2] if chars else 0}자  출처 {sorted(srcs)}  -> {f}")

    # 이미지가 실제로 열리는지 전수 확인 — 학습 중간에 죽는 것을 막는다
    miss = 0
    for split in args.splits:
        for line in (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip() and not (ROOT / json.loads(line)["images"][0]).exists():
                miss += 1
    print(f"이미지 경로 확인: 누락 {miss}건" + (" ✅" if not miss else " ❌"))


if __name__ == "__main__":
    main()
