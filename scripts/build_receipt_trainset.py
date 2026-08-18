"""
영수증 데이터셋 4종을 VLM 학습용 단일 형식으로 통합.

문제: 확보한 데이터가 전부 다른 형식이다.
    receipts3000   gt/*.json (구조화 + markdown) + manifest.jsonl
    KorIE ocr      *.txt (필드 크롭 텍스트)
    KorIE kid      *.txt (YOLO bbox)
    KorIE ie       *.csv (품목 표)

출력: train_vlm.py 가 그대로 먹는 JSONL 한 줄 = 한 샘플
    {"images": ["경로"],
     "messages": [{"role":"user","content":"<image>지시문"},
                  {"role":"assistant","content":"정답 텍스트"}]}

태스크 3종을 섞는다. 섞는 이유는 단일 태스크만 학습하면 모델이
그 형식만 뱉게 되어 범용성을 잃기 때문이다.

  full_md    영수증 전체 이미지 -> 마크다운      (receipts3000)
  field      필드 크롭        -> 텍스트         (KorIE ocr)
  full_json  영수증 전체 이미지 -> 구조화 JSON   (receipts3000)

usage:
    python3 scripts/build_receipt_trainset.py --dry
    python3 scripts/build_receipt_trainset.py
    python3 scripts/build_receipt_trainset.py --tasks full_md,field --max-field 3000
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace")
RD = ROOT / "receipt_data"
OUT = RD / "unified"

PROMPT = {
    "full_md":   "이 영수증을 마크다운 표로 정리해줘.",
    "full_json": "이 영수증을 JSON으로 파싱해줘. "
                 "상호명·사업자번호·거래일시·품목(명/수량/단가/금액)·"
                 "공급가액·부가세·합계·결제수단을 포함해줘.",
    "field":     "이 이미지에 적힌 텍스트를 그대로 옮겨 적어줘.",
}

# GT 에서 학습 타겟으로 쓸 필드 (markdown 은 별도 태스크)
JSON_KEYS = ["상호명", "업종", "사업자번호", "주소", "전화번호", "거래일시",
             "품목", "공급가액", "부가세", "합계", "결제수단"]


def sample(img: Path, task: str, target: str, meta=None):
    return {
        "images": [str(img.relative_to(ROOT))],
        "messages": [
            {"role": "user", "content": "<image>" + PROMPT[task]},
            {"role": "assistant", "content": target},
        ],
        "_task": task, **({"_meta": meta} if meta else {}),
    }


def load_receipts3000(tasks):
    """합성 3,000장. quality 를 층화 분할에 쓴다."""
    base = RD / "receipts3000"
    mani = {}
    mf = base / "manifest.jsonl"
    if mf.exists():
        for line in mf.open(encoding="utf-8"):
            r = json.loads(line)
            mani[Path(r["image_path"]).stem] = r

    out = []
    for gt_path in sorted((base / "gt").glob("*.json")):
        stem = gt_path.stem
        img = base / "images" / f"{stem}.jpg"
        if not img.exists():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        q = mani.get(stem, {}).get("quality", "unknown")

        if "full_md" in tasks and gt.get("markdown"):
            out.append(sample(img, "full_md", gt["markdown"].strip(),
                              {"quality": q, "src": "receipts3000"}))
        if "full_json" in tasks:
            slim = {k: gt[k] for k in JSON_KEYS if k in gt}
            out.append(sample(img, "full_json",
                              json.dumps(slim, ensure_ascii=False, indent=2),
                              {"quality": q, "src": "receipts3000"}))
    return out


def load_korie_ocr(tasks, max_eval=600, seed=0):
    """필드 크롭 8,927쌍.

    KorIE 원본 split 은 60:20:20 이라 평가셋이 과하다. 평가셋 크기는
    '통계적으로 필요한 만큼'이면 충분하다 — CER 3%대를 ±0.5%p 로 재려면
    약 4,500자, 필드 평균 10.3자 기준 440건이면 된다.
    초과분은 버리지 않고 train 으로 돌린다.
    """
    if "field" not in tasks:
        return {}
    base = RD / "korie" / "ocr"
    rnd = random.Random(seed)
    loaded = {}
    for split in ("train", "val", "test"):
        d = base / split / split
        if not d.exists():
            d = base / split
        rows = []
        for t in sorted(d.glob("*.txt")):
            img = t.with_suffix(".jpg")
            if not img.exists():
                continue
            text = t.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            field = t.stem.split("_", 1)[1] if "_" in t.stem else "unknown"
            rows.append(sample(img, "field", text,
                               {"field": field, "src": "korie_ocr"}))
        loaded[split] = rows

    # ⚠ KorIE 배포 split 은 영수증 단위로 나뉘어 있지 않다.
    #   train 674종 / test 638종인데 train∩test 가 638건 — test 영수증 전부가
    #   train 에도 있다. 같은 영수증의 다른 필드가 양쪽에 흩어져 있어서,
    #   그대로 쓰면 모델이 학습 때 본 매장·폰트·날짜를 평가에서 다시 만난다.
    #   따라서 배포 split 을 버리고 영수증 ID 기준으로 다시 나눈다.
    allrows = loaded["train"] + loaded["val"] + loaded["test"]
    by_receipt = {}
    for r in allrows:
        m = re.search(r"(IMG\d+)", r["images"][0])
        by_receipt.setdefault(m.group(1) if m else r["images"][0], []).append(r)

    rids = sorted(by_receipt)
    rnd.shuffle(rids)
    n = len(rids)
    n_test = max(1, int(n * 0.10))
    n_val = max(1, int(n * 0.10))
    grp = {"test": rids[:n_test],
           "val": rids[n_test:n_test + n_val],
           "train": rids[n_test + n_val:]}

    res = {}
    for split, ids in grp.items():
        rows = [r for i in ids for r in by_receipt[i]]
        if split in ("val", "test"):
            # 필드 유형이 고르게 남도록 유형별 균등 추출
            rnd.shuffle(rows)
            by = {}
            for r in rows:
                by.setdefault(r["_meta"]["field"], []).append(r)
            per = max(1, max_eval // max(len(by), 1))
            rows = [r for _, g in sorted(by.items()) for r in g[:per]]
        res[split] = rows
    return res


def stratified_split(rows, ratios=(0.8, 0.1, 0.1), seed=0):
    """quality 별로 고르게 나눈다. 한쪽에 heavy 가 몰리면 평가가 왜곡된다."""
    rnd = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault((r.get("_meta") or {}).get("quality", "?"), []).append(r)
    tr, va, te = [], [], []
    for _, group in sorted(by.items()):
        rnd.shuffle(group)
        n = len(group)
        a = int(n * ratios[0])
        b = a + int(n * ratios[1])
        tr += group[:a]; va += group[a:b]; te += group[b:]
    return tr, va, te


def write(path, rows, keep_meta=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            o = {k: v for k, v in r.items() if not k.startswith("_")}
            if keep_meta and "_meta" in r:
                o["meta"] = r["_meta"]
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="full_md,field",
                    help="full_md,full_json,field 중 쉼표 구분")
    ap.add_argument("--max-eval-field", type=int, default=600,
                    help="필드 평가셋 크기. 통계상 440건이면 CER ±0.5%p 확보")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    tasks = set(a.tasks.split(","))

    r3k = load_receipts3000(tasks)
    tr3, va3, te3 = stratified_split(r3k, seed=a.seed)
    korie = load_korie_ocr(tasks, a.max_eval_field, a.seed)

    train = tr3 + korie.get("train", [])
    val = va3 + korie.get("val", [])
    test = te3 + korie.get("test", [])
    random.Random(a.seed).shuffle(train)

    print(f"{'split':<8}{'합계':>8}  구성")
    print("-" * 62)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        c = Counter(r["_task"] for r in rows)
        print(f"{name:<8}{len(rows):>8,}  {dict(c)}")

    q = Counter((r.get('_meta') or {}).get('quality')
                for r in test if (r.get('_meta') or {}).get('src') == 'receipts3000')
    if q:
        print(f"\ntest 셋 quality 분포: {dict(q)}")
    fld = Counter((r.get('_meta') or {}).get('field')
                  for r in test if (r.get('_meta') or {}).get('src') == 'korie_ocr')
    if fld:
        print(f"test 셋 필드 상위: {dict(fld.most_common(5))}")

    if a.dry:
        print("\n(--dry: 파일 생성 안 함)")
        return

    for name, rows in (("train", train), ("val", val), ("test", test)):
        p = OUT / f"{name}.jsonl"
        write(p, rows, keep_meta=(name != "train"))
        print(f"\n{p}  {len(rows):,}건")

    # 평가용으로 quality/필드별 분할본도 남긴다 — 평균만 보면 진단이 안 된다
    evd = OUT / "eval_by_quality"
    byq = {}
    for r in test:
        m = r.get("_meta") or {}
        key = m.get("quality") if m.get("src") == "receipts3000" else "korie_field"
        byq.setdefault(key, []).append(r)
    for k, rows in byq.items():
        write(evd / f"{k}.jsonl", rows, keep_meta=True)
    print(f"\n평가 분할 -> {evd}/  ({', '.join(f'{k}:{len(v)}' for k, v in sorted(byq.items()))})")


if __name__ == "__main__":
    main()
