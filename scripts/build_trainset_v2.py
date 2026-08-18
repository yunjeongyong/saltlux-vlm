"""
합의 라벨 기반 학습셋 v2.

gen_gt.py 의 합의 결과를 학습셋에 반영한다. 등급별로 다르게 취급:

  high (3자 일치)     -> 신뢰. 정상 샘플 풀
  mid  (2자 일치)     -> 채택하되 표시
  low  (전부 불일치)   -> 학습에서 제외. 사람 검토 대기
  파서 교정 발생       -> hard example 로 분리. 사람 검토 대기

hard example 만으로 학습하면 모델이 '무조건 뭔가 고쳐야 한다'를 배워
정상 텍스트까지 건드린다(실측: LLM 후처리에서 오교정 2건 발생).
그래서 정상 샘플과 섞고, 평가도 두 집합을 분리해서 본다.

저화질 증강: 현재 이미지를 더 축소한 쌍을 추가한다. 해상도 저하 강건성 확보.
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "eval_dataset"
CONS = DS / "gt_consensus/consensus.jsonl"
OUT = DS / "train"
LOWRES_DIR = OUT / "blocks_lowres"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-ratio", type=float, default=0.35,
                    help="hard example 비율. 너무 높으면 과교정을 학습한다")
    ap.add_argument("--lowres-ratio", type=float, default=0.25,
                    help="저화질 증강 샘플 비율")
    ap.add_argument("--lowres-scale", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(CONS, encoding="utf-8")]
    rnd = random.Random(args.seed)

    clean, hard, dropped = [], [], []
    for r in rows:
        sample = {
            "id": r["id"], "images": [r["image"]],
            "messages": [
                {"role": "user", "content": "<image>" + (
                    "이 표 이미지를 HTML <table>로 변환하세요."
                    if "<table" in r["label"] else
                    "이 이미지의 텍스트를 그대로 옮겨 적으세요.")},
                {"role": "assistant", "content": r["label"]},
            ],
            "_grade": r["grade"], "_changed": r["parser_changed"],
        }
        if r["grade"] == "low":
            dropped.append(sample)          # 정답 불명 — 학습에서 뺀다
        elif r["parser_changed"]:
            hard.append(sample)             # 파서가 틀렸던 자리
        else:
            clean.append(sample)

    # hard 비율 맞추기 — hard 는 적으므로 clean 을 그에 맞춰 샘플링
    n_hard = len(hard)
    n_clean_want = int(n_hard * (1 - args.hard_ratio) / args.hard_ratio) if n_hard else len(clean)
    n_clean_want = min(max(n_clean_want, len(clean) // 2), len(clean))
    rnd.shuffle(clean)
    picked_clean = clean[:n_clean_want]

    samples = hard + picked_clean
    rnd.shuffle(samples)

    # 저화질 증강
    LOWRES_DIR.mkdir(parents=True, exist_ok=True)
    n_aug = int(len(samples) * args.lowres_ratio)
    aug = []
    for s in rnd.sample(samples, min(n_aug, len(samples))):
        src = ROOT / s["images"][0]
        im = Image.open(src).convert("RGB")
        w, h = int(im.width * args.lowres_scale), int(im.height * args.lowres_scale)
        if w < 8 or h < 8:
            continue
        dst = LOWRES_DIR / (Path(src).stem + "_lowres.png")
        im.resize((w, h), Image.LANCZOS).save(dst)
        a = json.loads(json.dumps(s))
        a["id"] = s["id"] + "@lowres"
        a["images"] = [str(dst.relative_to(ROOT))]
        a["_aug"] = True
        aug.append(a)
    samples += aug
    rnd.shuffle(samples)

    n_val = max(4, len(samples) // 10)
    val, train = samples[:n_val], samples[n_val:]
    for name, rs in (("train_v2", train), ("val_v2", val)):
        p = OUT / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({k: v for k, v in r.items()
                                    if not k.startswith("_")}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rs)}건 -> {p}")

    # 평가용으로 hard / clean 분리 저장 — 과교정 감지에 쓴다
    for name, rs in (("eval_hard", hard), ("eval_clean", picked_clean[:30])):
        p = OUT / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({k: v for k, v in r.items()
                                    if not k.startswith("_")}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rs)}건 -> {p}")

    print(f"\n=== 구성 ===")
    print(f"  hard (파서 교정된 자리)  {len(hard)}")
    print(f"  clean (합의 일치)        {len(picked_clean)} / 전체 clean {len(clean)}")
    print(f"  저화질 증강              {len(aug)}  (x{args.lowres_scale})")
    print(f"  학습 제외 (low, 정답불명) {len(dropped)}")
    print(f"  합계                     {len(samples)}")
    if hard:
        print(f"\n=== hard example ({len(hard)}건) ===")
        for h in hard:
            print(f"  {h['id']:<34} {h['messages'][1]['content'][:56]}")


if __name__ == "__main__":
    main()
