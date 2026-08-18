"""
학습 jsonl 생성기 — 교사 모델 없이, 기존 GT 만으로 7과제를 전개한다.

핵심 규칙은 하나다: **GT 에 실제로 있는 필드만 질문한다.**
필드가 없는 문서에 그 질문을 만들면 정답이 빈 값이 되고, 모델은
'모르면 빈칸'을 학습한다. clean_v1 의 필드 보유율이 79~97% 로
균일하지 않으므로(사업자번호·결제수단·items 가 79%), 문서 단위로
보유 여부를 확인하고 라우팅한다.

소스마다 가진 GT 가 다르므로 만들 수 있는 과제도 다르다.
  clean_v1  fields 9종 + items + markdown  -> 7과제 전부
  CORD-v2   items + 금액 3종               -> items_only, amount_qa
  SROIE     헤더 4필드                     -> field_qa 3종, amount_qa 합계
  크롤링     값 GT 없음                     -> crop_ocr 만

crop_ocr 는 정답이 파서 OCR 이다(사람 검증 아님). LLaVAR 방식대로
사전학습 단계용으로 따로 뽑고, 본 학습 파일에는 섞지 않는다.

말투(variant)는 (doc_id, task, field) 해시로 고른다. 랜덤이면 재생성할
때마다 프롬프트가 달라져 실험 비교가 불가능하다.

출력 형식은 train_vlm.py 가 읽는 구조를 따른다.
  {"doc_id","task","images":[상대경로],"messages":[{user},{assistant}]}
  user content 는 "<image>" + 프롬프트.

usage:
    python3 scripts/build_trainset_v3.py --out receipt_data/trainset_v3
    python3 scripts/build_trainset_v3.py --limit 50 --dry-run
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import prompt_templates as PT

R = Path("/workspace/receipt_data")
IMG_TOKEN = "<image>"

# 소스별 GT 키 매핑. 같은 의미의 필드가 소스마다 다른 이름으로 들어있다.
FIELD_MAP = {
    "clean_v1": {
        "상호명": ["merchant_name"],
        "거래일시": ["transaction_datetime", "transaction_date"],
        "사업자번호": ["business_no"],
        "전화번호": ["merchant_phone"],
        "주소": ["merchant_address"],
        "결제수단": ["payment_method"],
    },
    "sroie": {
        "상호명": ["company"],
        "거래일시": ["date"],
        "주소": ["address"],
    },
    "cord": {},
}
AMOUNT_MAP = {
    "clean_v1": {"합계": ["total"], "공급가액": ["subtotal"], "부가세": ["tax"]},
    "cord": {"합계": ["total.total_price"],
             "공급가액": ["sub_total.subtotal_price"],
             "부가세": ["sub_total.tax_price"]},
    "sroie": {"합계": ["total"]},
}

SOURCES = {
    "clean_v1": R / "clean_v1",
    "cord":     R / "public/cord",
    "sroie":    R / "public/sroie",
}


def pick_variant(doc_id, task, field, n):
    """(문서, 과제, 필드) 로 말투를 결정한다. 재생성해도 같은 결과가 나온다."""
    h = hashlib.md5(f"{doc_id}|{task}|{field or ''}".encode()).hexdigest()
    return int(h[:8], 16) % n


def get_field(gt, keys):
    f = gt.get("fields") or {}
    for k in keys:
        v = f.get(k)
        if v not in (None, ""):
            return v
    return None


def to_int(v):
    """'84,600원' -> 84600. 실패하면 None."""
    if v is None:
        return None
    if isinstance(v, int):
        return v
    t = "".join(ch for ch in str(v) if ch.isdigit() or ch == "-")
    try:
        return int(t)
    except ValueError:
        return None


def sample(doc_id, task, images, prompt, answer):
    return {
        "doc_id": doc_id,
        "task": task,
        "images": images,
        "messages": [
            {"role": "user", "content": IMG_TOKEN + prompt},
            {"role": "assistant", "content": answer},
        ],
    }


# ---------------------------------------------------------------- 과제별 정답

def ans_items_only(gt):
    """'상품명 수량 금액' 한 줄씩."""
    lines = []
    for it in gt.get("items") or []:
        name = it.get("name")
        if not name:
            continue
        qty = it.get("qty")
        amt = it.get("amount", it.get("price"))
        lines.append(f"{name} {qty if qty is not None else ''} "
                     f"{amt if amt is not None else ''}".strip())
    return "\n".join(lines) if lines else None


def ans_json_kie(gt, src):
    """채택 스키마 8키. 프롬프트에 못박은 키와 정확히 일치해야 한다."""
    fm, am = FIELD_MAP.get(src, {}), AMOUNT_MAP.get(src, {})
    items = []
    for it in gt.get("items") or []:
        items.append({
            "name": it.get("name"),
            "unit_price": to_int(it.get("unit_price")),
            "qty": to_int(it.get("qty")),
            "amount": to_int(it.get("amount", it.get("price"))),
        })
    obj = {
        "merchant_name": get_field(gt, fm.get("상호명", [])),
        "business_no": get_field(gt, fm.get("사업자번호", [])),
        "transaction_datetime": get_field(gt, fm.get("거래일시", [])),
        "items": items,
        "subtotal": to_int(get_field(gt, am.get("공급가액", []))),
        "tax": to_int(get_field(gt, am.get("부가세", []))),
        "total": to_int(get_field(gt, am.get("합계", []))),
        "payment_method": get_field(gt, fm.get("결제수단", [])),
    }
    return json.dumps(obj, ensure_ascii=False)


def ans_merchant_info(gt, src):
    fm = FIELD_MAP.get(src, {})
    obj = {
        "merchant_name": get_field(gt, fm.get("상호명", [])),
        "business_no": get_field(gt, fm.get("사업자번호", [])),
        "transaction_datetime": get_field(gt, fm.get("거래일시", [])),
    }
    if obj["merchant_name"] is None:
        return None
    return json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------- 문서 전개

def expand_doc(gt, src, stats):
    """문서 하나 -> 생성 가능한 샘플 전부. GT 에 있는 것만 만든다."""
    doc_id = gt["doc_id"]
    imgs = [gt["image"]]
    fm, am = FIELD_MAP.get(src, {}), AMOUNT_MAP.get(src, {})
    out = []

    def add(task, field, answer):
        if answer is None or answer == "":
            stats[f"skip:{task}" + (f"/{field}" if field else "")] += 1
            return
        n = len(PT.TASKS[task]["variants"])
        p = PT.build_prompt(task, field=field,
                            variant=pick_variant(doc_id, task, field, n))
        out.append(sample(doc_id, task if not field else f"{task}/{field}", imgs, p, answer))

    # 1. md_table — GT 에 markdown 이 보존된 소스만
    if gt.get("markdown"):
        add("md_table", None, gt["markdown"])

    # 2/3. json_kie, items_only — items 가 있어야 의미가 있다
    if gt.get("items"):
        add("json_kie", None, ans_json_kie(gt, src))
        add("items_only", None, ans_items_only(gt))

    # 4. field_qa — 문서가 실제로 가진 필드만
    for field, keys in fm.items():
        v = get_field(gt, keys)
        if v is not None:
            add("field_qa", field, str(v))

    # 5. amount_qa — 숫자만. 정수 변환에 실패하면 만들지 않는다.
    for field, keys in am.items():
        n = to_int(get_field(gt, keys))
        if n is not None:
            add("amount_qa", field, str(n))

    # 7. merchant_info
    if get_field(gt, fm.get("상호명", [])):
        add("merchant_info", None, ans_merchant_info(gt, src))

    return out


def expand_crops(manifest, stats, max_chars):
    """crop_ocr. 정답은 파서 OCR 이라 noisy 하다 — 본 학습과 분리해서 낸다."""
    out = []
    for line in open(manifest, encoding="utf-8"):
        r = json.loads(line)
        text = (r.get("parser_text") or "").strip()
        if not text:
            stats["skip:crop_ocr/no_text"] += 1
            continue
        if len(text) > max_chars:
            # 표 하나가 통째로 들어온 경우. 한 줄 전사 과제의 성격에서 벗어난다.
            stats["skip:crop_ocr/too_long"] += 1
            continue
        n = len(PT.TASKS["crop_ocr"]["variants"])
        p = PT.build_prompt("crop_ocr",
                            variant=pick_variant(r["crop_id"], "crop_ocr", None, n))
        out.append(sample(r["crop_id"], "crop_ocr", [r["image"]], p, text))
    return out


# ---------------------------------------------------------------- split

def doc_split(gt, doc_id, ratios, seed):
    """GT 에 split 이 있으면 따르고, 없으면 doc_id 해시로 고정 배분한다.
    같은 문서의 샘플이 train/val 로 흩어지면 누수가 생기므로 문서 단위로 나눈다."""
    s = gt.get("split")
    if s in ("train", "val", "test"):
        return s
    h = int(hashlib.md5(f"{seed}|{doc_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    tr, va = ratios
    return "train" if h < tr else ("val" if h < tr + va else "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "trainset_v3"))
    ap.add_argument("--limit", type=int, default=0, help="소스당 문서 N개만")
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crop-max-chars", type=int, default=200)
    ap.add_argument("--no-crops", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않는다")
    args = ap.parse_args()

    random.seed(args.seed)
    stats = Counter()
    by_split = defaultdict(list)
    task_count = Counter()
    src_count = Counter()

    print("문서 전개 (GT 에 있는 필드만)\n")
    for src, root in SOURCES.items():
        gtdir = root / "gt"
        if not gtdir.exists():
            print(f"  {src:10} 건너뜀 (gt 없음)")
            continue
        files = sorted(gtdir.glob("*.json"))
        if args.limit:
            files = files[:args.limit]
        n_doc = n_s = 0
        for f in files:
            gt = json.loads(f.read_text(encoding="utf-8"))
            samples = expand_doc(gt, src, stats)
            if not samples:
                stats[f"skip:{src}/empty_doc"] += 1
                continue
            sp = doc_split(gt, gt["doc_id"], (args.train_ratio, args.val_ratio), args.seed)
            for s in samples:
                by_split[sp].append(s)
                task_count[s["task"].split("/")[0]] += 1
            src_count[src] += len(samples)
            n_doc += 1
            n_s += len(samples)
        print(f"  {src:10} 문서 {n_doc:>5,} -> 샘플 {n_s:>7,}  (문서당 {n_s/max(1,n_doc):.1f})")

    # crop_ocr 은 별도 파일. 파서 OCR 이 정답이라 본 학습과 성격이 다르다.
    crops = []
    if not args.no_crops:
        for m in [R / "clean_v1/crops/manifest.jsonl",
                  R / "crawl_google/crops/manifest.jsonl"]:
            if m.exists():
                got = expand_crops(m, stats, args.crop_max_chars)
                crops += got
                print(f"  {'crop_ocr':10} {m.parent.parent.name:12} -> 샘플 {len(got):>7,}")

    out = Path(args.out)
    print(f"\n본 학습 (검증된 GT)")
    tot = 0
    for sp in ("train", "val", "test"):
        n = len(by_split[sp])
        tot += n
        print(f"  {sp:6} {n:>8,}")
    print(f"  {'합계':6} {tot:>8,}")
    print(f"\n과제별: {dict(task_count.most_common())}")
    print(f"소스별: {dict(src_count.most_common())}")

    if crops:
        random.shuffle(crops)
        n_tr = int(len(crops) * args.train_ratio)
        n_va = int(len(crops) * args.val_ratio)
        print(f"\n사전학습 (crop_ocr, 파서 OCR — noisy)")
        print(f"  train {n_tr:,} / val {n_va:,} / test {len(crops)-n_tr-n_va:,}")

    skips = {k: v for k, v in stats.items() if k.startswith("skip:")}
    if skips:
        print(f"\n생성 안 함 (GT 없음/부적합): {dict(Counter(skips).most_common(8))}")

    if args.dry_run:
        print("\n--dry-run — 파일을 쓰지 않았다.")
        return

    out.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        with (out / f"{sp}.jsonl").open("w", encoding="utf-8") as fh:
            for s in by_split[sp]:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    if crops:
        pre = out / "pretrain_crop_ocr"
        pre.mkdir(exist_ok=True)
        n_tr = int(len(crops) * args.train_ratio)
        n_va = int(len(crops) * args.val_ratio)
        for sp, chunk in (("train", crops[:n_tr]),
                          ("val", crops[n_tr:n_tr + n_va]),
                          ("test", crops[n_tr + n_va:])):
            with (pre / f"{sp}.jsonl").open("w", encoding="utf-8") as fh:
                for s in chunk:
                    fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"\n사전학습 파일: {pre}")
    print(f"본 학습 파일: {out}")


if __name__ == "__main__":
    main()
