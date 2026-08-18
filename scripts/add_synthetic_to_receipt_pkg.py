"""
receipt 패키지에 합성 영수증(receipts3000)을 섞어 목표 장수를 채운다.

주의 — 이 패키지는 `human_annotated/` 아래에 있다. 합성 데이터는 사람이 라벨링한
것이 아니므로, 레코드와 manifest 에 `source` 를 남겨 나중에 출처를 가려낼 수 있게 한다.
(train.jsonl 의 source 로 걸러내면 사람 검수분만 다시 뽑을 수 있다)

JSON 규격은 labeled/val·test 와 같은 API 응답 형태로 맞춘다. 재료는 두 곳에서 온다.

    껍데기·블록·좌표   clean_v1/parse/syn_receipt_NNNN.json  (파서 응답)
    markdown 정답      receipts3000/gt/receipt_NNNN.json     (생성값이라 100% 정확)

파서 markdown 을 쓰지 않는 이유: 합성본은 원본 렌더링 값이 그대로 정답이라
파서가 읽어낸 것보다 정확하다. 좌표는 파서 것 말고 대안이 없어 그대로 쓴다.

usage:
    python3 scripts/add_synthetic_to_receipt_pkg.py --target 176 --dry
    python3 scripts/add_synthetic_to_receipt_pkg.py --target 176
"""
import argparse
import json
import os
import random
from pathlib import Path

ROOT = Path("/data/workspace/yjyong")
PKG = ROOT / "vlm_dataset_upload/train/human_annotated/receipt"
SYN_IMG = ROOT / "receipt_data/receipts3000/images"
SYN_GT = ROOT / "receipt_data/receipts3000/gt"
SYN_PARSE = ROOT / "receipt_data/clean_v1/parse"
DEST = "/data/workspace/VLM/dataset/train/human_annotated/receipt"
PROMPT = "이 영수증의 내용을 마크다운으로 정리해줘."


def build_env(parse_env, gt_md):
    """파서 응답 껍데기에 합성 GT markdown 을 얹는다. 규격은 val·test 와 동일."""
    env = json.loads(json.dumps(parse_env))          # 원본 훼손 방지
    for k in ("_source_doc_id", "_source_image", "_source_width", "_source_height"):
        env.pop(k, None)
    els = (env.get("result") or {}).get("elements") or []
    if not els:
        return None
    els[0]["markdown"] = gt_md
    env["result"]["md"] = gt_md
    env.setdefault("status", 200)
    env.setdefault("previewFile", None)
    env.setdefault("previewFilename", None)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=176, help="패키지 최종 장수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (PKG / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    have = {r["doc_id"] for r in rows}
    need = args.target - len(rows)
    print(f"현재 {len(rows)}건 → 목표 {args.target}건 · 합성 {need}건 추가")
    if need <= 0:
        print("이미 목표를 채웠다.")
        return

    # 파서 응답이 있는 합성본만 후보로 둔다
    cand = []
    for gt in sorted(SYN_GT.glob("receipt_*.json")):
        stem = gt.stem
        if (SYN_PARSE / f"syn_{stem}.json").exists() and (SYN_IMG / f"{stem}.jpg").exists():
            cand.append(stem)
    print(f"  합성 후보 {len(cand):,}건")
    picks = random.Random(args.seed).sample(cand, need)

    # 기존 레코드에도 출처를 남긴다 (사람 검수분)
    for r in rows:
        r.setdefault("source", "human_annotated")

    added, index_add = [], []
    for stem in picks:
        gt = json.loads((SYN_GT / f"{stem}.json").read_text(encoding="utf-8"))
        md = (gt.get("markdown") or "").strip()
        if not md:
            continue
        env = build_env(json.loads((SYN_PARSE / f"syn_{stem}.json").read_text(encoding="utf-8")), md)
        if env is None:
            continue
        did = f"syn_{stem}"
        if did in have:
            continue
        rel_img, rel_lab = f"data/{did}.jpg", f"data/{did}.json"
        if not args.dry:
            di = PKG / rel_img
            if not di.exists():
                try:
                    os.link(SYN_IMG / f"{stem}.jpg", di)
                except OSError:
                    di.write_bytes((SYN_IMG / f"{stem}.jpg").read_bytes())
            (PKG / rel_lab).write_text(json.dumps(env, ensure_ascii=False), encoding="utf-8")
        added.append({
            "doc_id": did, "task": "receipt_markdown", "source": "synthetic",
            "images": [f"receipt/{rel_img}"],
            "messages": [{"role": "user", "content": f"<image>{PROMPT}"},
                         {"role": "assistant", "content": md}],
        })
        index_add.append({"id": did,
                          "image_path": f"{DEST}/{rel_img}",
                          "annotation_path": f"{DEST}/{rel_lab}"})

    allrows = rows + added
    random.Random(args.seed).shuffle(allrows)
    print(f"  추가 {len(added)}건 → 합계 {len(allrows)}건 "
          f"(사람검수 {sum(1 for r in allrows if r['source']=='human_annotated')} / "
          f"합성 {sum(1 for r in allrows if r['source']=='synthetic')})")

    if args.dry:
        print("\n샘플:", json.dumps({**added[0], "messages": [
            added[0]["messages"][0],
            {"role": "assistant", "content": added[0]["messages"][1]["content"][:160] + " …"}]},
            ensure_ascii=False, indent=1))
        print("\n(--dry: 저장하지 않았다)")
        return

    (PKG / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in allrows) + "\n", encoding="utf-8")

    man = json.loads((PKG / "manifest.json").read_text(encoding="utf-8"))
    man["data"] = man["data"] + index_add
    man["info"]["note"] += (
        f" 목표 장수를 채우려고 합성 영수증(receipts3000) {len(added)}건을 섞었다 — "
        "합성분의 markdown 은 렌더링 생성값이라 정확하고, 좌표·블록은 파서 결과다. "
        "출처는 train.jsonl 의 source 필드(human_annotated / synthetic)로 구분한다."
    )
    man["info"]["composition"] = {
        "human_annotated": sum(1 for r in allrows if r["source"] == "human_annotated"),
        "synthetic": sum(1 for r in allrows if r["source"] == "synthetic"),
    }
    (PKG / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {PKG}")
    print(f"  data/       {len(list((PKG/'data').glob('*')))}개")
    print(f"  train.jsonl {len(allrows)}행 · manifest.data {len(man['data'])}건")


if __name__ == "__main__":
    main()
