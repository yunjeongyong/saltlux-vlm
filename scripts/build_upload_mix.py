"""
업로드 패키지 두 개를 train_vlm.py 가 먹을 수 있는 학습셋으로 합친다.

패키지의 images 경로는 배치 기준 상대경로(`receipt/data/...`)라 그대로는 못 쓴다.
train_vlm.py 는 `ROOT / rel` 로 여니 그 기준으로 다시 쓴다. ROOT 는 train_vlm.py
가 잡는 값(스크립트의 부모 = 리포 루트)과 같아야 한다. 컨테이너마다 마운트
위치가 달라서(`/workspace` vs `/data/workspace/yjyong`) --root 로 받는다.

섞는 비율에 손을 댄다. 영수증 84건과 공공문서 36,598건을 그대로 합치면
영수증이 0.2% 라 사실상 안 배운다. 영수증을 여러 번 넣어(oversample) 비중을
올린다 — 데이터가 늘어나는 건 아니고 같은 장을 반복해서 보는 것이다.

usage:
    python3 scripts/build_upload_mix.py --doc 3000 --rcp-repeat 8
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_vlm_taxonomy import build_markdown  # noqa: E402

# train_vlm.py 와 같은 규칙으로 리포 루트를 잡는다 (scripts/ 의 부모).
ROOT = Path(__file__).resolve().parent.parent
PKG_REL = {
    "receipt": "vlm_dataset_upload/train/human_annotated/receipt",
    "doc": "vlm_dataset_upload/train/public/aihub-71299-ocr",
}


def html_answer(pkg, doc_id):
    """패키지에 남아 있는 원본 페이지 JSON 으로 정답을 다시 조립한다 (표는 HTML 유지).

    패키지의 train.jsonl 정답은 표가 마크다운으로 변환된 상태다. 변환 과정에서
    colspan 이 빈 칸으로 펴져 되돌릴 수 없으므로, 문자열을 고치지 않고 원본
    블록에서 새로 만든다. data/{id}.json 이 없으면 None 을 돌려준다.
    """
    f = pkg / "data" / f"{doc_id}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    els = (d.get("result") or {}).get("elements") or []
    page = (els[0].get("json") or {}) if els else d
    blocks = page.get("parsing_res_list") or []
    if not blocks:
        return None
    md = build_markdown(blocks, keep_html_tables=True).strip()
    return md or None


def load(p, prefix, root, html_tables=False):
    rows, regen, failed = [], 0, []
    for line in (p / "train.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rel = r["images"][0]
        # "receipt/data/x.jpg" -> "vlm_dataset_upload/train/.../receipt/data/x.jpg"
        r["images"] = [str((p.parent / rel).relative_to(root))]
        r["_src"] = prefix
        if html_tables:
            md = html_answer(p, r["doc_id"])
            if md:
                r["messages"][1]["content"] = md
                regen += 1
            else:
                failed.append(r["doc_id"])
        rows.append(r)
    if html_tables:
        print(f"  {prefix}: 정답 재생성 {regen}건"
              + (f" / 실패 {len(failed)}건 {failed[:3]}" if failed else ""))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", type=int, default=3000, help="공공문서에서 뽑을 장수 (0=전량)")
    ap.add_argument("--rcp-repeat", type=int, default=8, help="영수증 반복 횟수")
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=str(ROOT),
                    help="train_vlm.py 가 이미지 상대경로를 여는 기준 디렉토리")
    ap.add_argument("--out", default=None)
    # 표를 마크다운으로 바꾸면 병합 셀이 사라지고 TEDS 로는 0 점이 된다.
    # 채점을 TEDS 로 하고 서비스 프롬프트도 HTML 을 요구하므로 기본값을 켠다.
    ap.add_argument("--html-tables", dest="html_tables",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="영수증 정답의 표를 HTML 로 유지 (--no-html-tables 로 끔)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = args.out or str(root / "receipt_data/upload_mix")
    pkgs = {k: root / v for k, v in PKG_REL.items()}
    for k, p in pkgs.items():
        if not (p / "train.jsonl").exists():
            raise SystemExit(f"패키지 없음: {p}/train.jsonl (--root 확인)")

    rng = random.Random(args.seed)
    # 공공문서는 평문 전사라 표 변환과 무관하다. 영수증에만 적용한다.
    rcp = load(pkgs["receipt"], "receipt", root, args.html_tables)
    doc = load(pkgs["doc"], "doc", root)
    rng.shuffle(rcp)
    rng.shuffle(doc)

    # 검증은 각 출처에서 떼어낸다. 한쪽만 보면 다른 쪽이 망가져도 모른다.
    # --val 0 은 "떼지 마라"는 뜻이다 — 검증셋을 별도 split(labeled/val)에서
    # 가져올 때 쓴다. 이때 학습에서 영수증을 빼면 187장이 179장으로 줄어든다.
    if args.val <= 0:
        v_rcp, v_doc = [], []
    else:
        v_rcp = rcp[:max(8, args.val // 8)]
        v_doc = doc[:args.val - len(v_rcp)]
    t_rcp = rcp[len(v_rcp):]
    t_doc = doc[len(v_doc):]
    if args.doc:
        t_doc = t_doc[:args.doc]

    train = t_rcp * args.rcp_repeat + t_doc
    rng.shuffle(train)
    val = v_rcp + v_doc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # val 을 안 뗐으면 val.jsonl 을 건드리지 않는다. 빈 파일로 덮어쓰면
    # 별도로 만들어 둔 검증셋(labeled/val)이 조용히 날아간다.
    files = [("train.jsonl", train)] + ([("val.jsonl", val)] if val else [])
    for name, rows in files:
        (out / name).write_text(
            "\n".join(json.dumps({k: v for k, v in r.items() if k != "_src"},
                                 ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8")

    def share(rows):
        n = len(rows)
        if not n:
            return "0건"
        c = sum(1 for r in rows if r["_src"] == "receipt")
        return f"{n:,}건 (영수증 {c:,} = {100*c/n:.1f}% / 문서 {n-c:,})"

    print(f"train  {share(train)}")
    print(f"val    {share(val)}" + ("" if val else "  (미생성 — 별도 split 사용)"))
    print(f"\n원본: 영수증 {len(rcp)}건 x {args.rcp_repeat}회 · 문서 {len(t_doc):,}건")
    print(f"저장: {out}")

    # 경로가 실제로 열리는지 전수 확인 — 학습 도중에 죽는 것을 막는다
    miss = [r["images"][0] for r in train + val if not (root / r["images"][0]).exists()]
    print(f"이미지 경로 확인: 누락 {len(miss)}건" + (f"  예: {miss[:2]}" if miss else " ✅"))


if __name__ == "__main__":
    main()
