"""
라벨링 툴로 검수한 실물 영수증을 팀 공용 데이터셋 규격으로 포장한다.

규격은 `dataset/train/public/pubtabnet-html` 와 같다.

    receipt/
      data/{id}.jpg      원본 이미지
      data/{id}.json     검수 완료 페이지 JSON (파서 포맷 그대로)
      manifest.json      info + schema + data[]
      train.jsonl        학습셋 (messages)

정답 마크다운은 `make_vlm_taxonomy.py` 의 조립 규칙을 그대로 쓴다. 같은 규칙을
두 벌 두면 나중에 한쪽만 고쳐져 학습셋이 갈라진다.

usage:
    python3 scripts/build_labeled_receipt_pkg.py --dry
    python3 scripts/build_labeled_receipt_pkg.py
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_vlm_taxonomy import build_markdown, is_hallucinated  # noqa: E402

SRC = Path("/data/workspace/yjyong/receipt_data/labeled")
OUT = Path("/data/workspace/yjyong/receipt_data/pkg/receipt")
DEST = "/data/workspace/VLM/dataset/train/human_annotated/receipt"
PROMPT = "이 영수증의 내용을 마크다운으로 정리해줘."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", help="labeled/ 아래 어느 split 을 담을지")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dest-root", default=DEST,
                    help="manifest 에 기록할 최종 배치 경로")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    jdir = SRC / args.split / "json"
    idir = SRC / args.split / "images"
    ids = sorted(p.stem for p in jdir.glob("*.json"))
    print(f"{args.split} 검수본 {len(ids)}건")

    out = Path(args.out)
    if not args.dry:
        (out / "data").mkdir(parents=True, exist_ok=True)

    rows, index, drop = [], [], Counter()
    src_cnt, block_cnt, dropped_blocks = Counter(), Counter(), Counter()
    kept = 0

    for i in ids:
        img = next((p for p in idir.glob(f"{i}.*")), None)
        if img is None:
            drop["이미지 없음"] += 1
            continue
        d = json.loads((jdir / f"{i}.json").read_text(encoding="utf-8"))
        # labeled/ 는 API 응답 규격으로 통일돼 있다 (unify_labeled_format.py).
        # 블록은 result.elements[0].json 안에 들어 있다.
        els = (d.get("result") or {}).get("elements") or []
        page = (els[0].get("json") or {}) if els else d
        blocks = page.get("parsing_res_list") or []
        if not blocks:
            drop["블록 없음"] += 1
            continue

        dl = []
        md = build_markdown(blocks, dl)
        if not md.strip():
            drop["정답 비어있음"] += 1
            continue
        for lab, _, why in dl:
            dropped_blocks[why] += 1

        kept += 1
        src_cnt[i.split("_")[0]] += 1
        for b in blocks:
            block_cnt[b.get("block_label")] += 1

        ext = img.suffix
        rel_img, rel_lab = f"data/{i}{ext}", f"data/{i}.json"
        if not args.dry:
            di, dj = out / rel_img, out / rel_lab
            if not di.exists():
                try:
                    os.link(img, di)
                except OSError:
                    di.write_bytes(img.read_bytes())
            if not dj.exists():
                dj.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

        rows.append({
            "doc_id": i,
            "task": "receipt_markdown",
            "images": [f"receipt/{rel_img}"],
            "messages": [
                {"role": "user", "content": f"<image>{PROMPT}"},
                {"role": "assistant", "content": md},
            ],
        })
        index.append({
            "id": i,
            "image_path": f"{args.dest_root}/{rel_img}",
            "annotation_path": f"{args.dest_root}/{rel_lab}",
        })

    print(f"\n채택 {kept}건" + (f" / 제외 {sum(drop.values())}건 {dict(drop)}" if drop else ""))
    print(f"  출처: {dict(src_cnt.most_common())}")
    print(f"  블록 라벨: {dict(block_cnt.most_common(8))}")
    if dropped_blocks:
        print(f"  ⚠ 환각으로 걸러진 블록: {dict(dropped_blocks)}")
    if args.dry:
        if rows:
            r = rows[0]
            print("\n샘플 1행:")
            print(json.dumps({**r, "messages": [r["messages"][0],
                  {"role": "assistant",
                   "content": r["messages"][1]["content"][:260] + " …"}]},
                  ensure_ascii=False, indent=1))
        print("\n(--dry: 저장하지 않았다)")
        return

    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "info": {
            "description": "사내 수집 실물 영수증 — 라벨링 툴로 사람이 검수한 문서 파싱 데이터.",
            "version": "1.0",
            "source": "internal (luxia labeling tool)",
            "date_created": "2026-08-11",
            "note": (
                "document-parse 로 초벌 파싱한 뒤 라벨링 툴에서 사람이 검수했다. "
                "정답 마크다운은 scripts/make_vlm_taxonomy.py 의 조립 규칙으로 만들었다 — "
                "block_order 순서, doc_title=H1, paragraph_title=H2, HTML 표는 마크다운 표로 변환, "
                "SKIP_LABELS(image/seal/header_image/footer_image/figure_title)는 제외."
            ),
        },
        "schema": {
            "_shape": "API 응답 규격 — result.elements[0].json 안에 페이지 JSON이 들어 있다 (labeled/val·test와 동일)",
            "parsing_res_list": "list — {block_label, block_content, block_bbox, block_id, block_order}",
            "block_content": "text 계열은 평문 / table 은 HTML <table> 문자열",
            "block_bbox": "[x1,y1,x2,y2] — 파서 처리 해상도 기준 (원본 크기와 다를 수 있음)",
            "block_order": "읽기 순서. 마크다운 조립 기준",
        },
        "data": index,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n저장: {out}")
    print(f"  data/       {kept*2}개 (이미지+json)")
    print(f"  train.jsonl {len(rows)}행")


if __name__ == "__main__":
    main()
