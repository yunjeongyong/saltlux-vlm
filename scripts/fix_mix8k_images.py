"""
mix8k 학습셋의 이미지 경로를 실제 위치에 맞춰 보정하고 전량 존재 검증.

kdy_260813 에서 온 8,198건은 taxonomy 의 상대경로를 그대로 들고 있어서
(pubtabnet-html/... , data/only_text/... , kogovdoc-bench/...) 학습 스크립트가
ROOT 기준으로 열면 전부 없다. 이미지 패키지가 들어온 자리를 앞에 붙여야 한다.

train_vlm.py:44 가 `ROOT / rel` 로 여니까 최종 경로는 /workspace 기준 상대경로여야
한다. --images-root 도 /workspace 기준으로 준다 (절대경로를 주면 상대로 되돌린다).

루트별로 따로 지정하는 이유 — 세 종이 한 디렉토리에 나란히 있으면 --images-root
하나로 끝나지만, 흩어져 있으면 개별 지정이 필요하다. AI Hub 멀티모달 건은 최상위가
그냥 `data` 라 이름만으로 구분이 안 되므로 특히 그렇다.

usage:
    # 세 종이 vlm_dataset_src/ 아래 나란히 있는 경우
    python3 scripts/fix_mix8k_images.py --images-root vlm_dataset_src

    # 흩어져 있는 경우 — 루트별 지정 (앞의 --images-root 를 덮어쓴다)
    python3 scripts/fix_mix8k_images.py \
        --map pubtabnet-html=vlm_dataset_src \
        --map kogovdoc-bench=vlm_dataset_src \
        --map data=vlm_dataset_src/multimodal-retrieval

    # 검증만 (파일 안 씀)
    python3 scripts/fix_mix8k_images.py --images-root vlm_dataset_src --dry-run
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 이미 /workspace 기준으로 맞는 루트. 건드리지 않는다.
KEEP = {"vlm_dataset_upload"}


def rel_to_root(p):
    """절대경로로 줘도 /workspace 기준 상대경로로 되돌린다."""
    p = Path(p)
    if p.is_absolute():
        try:
            return p.relative_to(ROOT)
        except ValueError:
            raise SystemExit(f"--images-root 가 {ROOT} 밖이다: {p}\n"
                             "학습 스크립트가 ROOT 기준으로 열기 때문에 안 된다.")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "receipt_data/mix8k/train.jsonl"))
    ap.add_argument("--out", default=None, help="기본: --src 를 제자리에 덮어씀")
    ap.add_argument("--images-root", default=None,
                    help="보정 대상 전체에 붙일 프리픽스 (/workspace 기준)")
    ap.add_argument("--map", action="append", default=[], metavar="ROOT=PREFIX",
                    help="루트별 프리픽스. --images-root 보다 우선한다")
    ap.add_argument("--dry-run", action="store_true", help="검증만, 파일 안 씀")
    args = ap.parse_args()

    per_root = {}
    for m in args.map:
        if "=" not in m:
            raise SystemExit(f"--map 형식은 ROOT=PREFIX 다: {m}")
        k, v = m.split("=", 1)
        per_root[k] = rel_to_root(v)
    default_prefix = rel_to_root(args.images_root) if args.images_root else None

    # read_text().splitlines() 를 쓰면 안 된다. splitlines 는 \n 말고 U+2028/U+2029/
    # \x85 에서도 쪼개는데, json.dumps(ensure_ascii=False) 는 이것들을 이스케이프하지
    # 않는다 — GT 본문에 섞여 있으면 한 줄이 두 동강 나 JSONDecodeError 가 난다.
    # 파일 객체 순회는 \n 에서만 끊는다.
    with open(args.src, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    roots = Counter(r["images"][0].split("/")[0] for r in rows)
    todo = [k for k in roots if k not in KEEP]
    unmapped = [k for k in todo if k not in per_root and default_prefix is None]
    if unmapped:
        raise SystemExit(f"프리픽스가 안 정해진 루트: {unmapped}\n"
                         "--images-root 또는 --map 으로 지정할 것.")

    missing = defaultdict(list)
    fixed = Counter()
    for r in rows:
        new = []
        for rel in r["images"]:
            root = rel.split("/")[0]
            if root not in KEEP:
                prefix = per_root.get(root, default_prefix)
                rel = str(prefix / rel)
                fixed[root] += 1
            if not (ROOT / rel).exists():
                missing[root].append(rel)
            new.append(rel)
        r["images"] = new

    total = sum(len(r["images"]) for r in rows)
    n_missing = sum(len(v) for v in missing.values())
    print(f"이미지 참조 {total:,}개 / 경로 보정 {sum(fixed.values()):,}개")
    for k, n in sorted(roots.items()):
        mark = "(그대로)" if k in KEEP else f"-> {per_root.get(k, default_prefix)}/"
        print(f"  {k:<20} {n:>6,}건  {mark}")

    if n_missing:
        print(f"\n없는 이미지 {n_missing:,}개:")
        for k, v in sorted(missing.items()):
            print(f"  {k:<20} {len(v):>6,}개   예: {v[0]}")
        print("\n학습 안 띄운다 — 첫 배치에서 FileNotFoundError 로 죽는다.")
        raise SystemExit(1)

    print("\n전량 확인됨.")
    if args.dry_run:
        print("(--dry-run 이라 파일은 안 씀)")
        return

    out = Path(args.out or args.src)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")
    print(f"기록: {out} ({len(rows):,}행)")


if __name__ == "__main__":
    main()
