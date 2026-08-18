"""
검증 통과분만 모아 공통 스키마 데이터셋을 만든다.

지금 두 소스가 스키마가 다르다 — receipts3000 은 한국어 키(상호명/공급가액),
korie/kid 는 영어 키(MerchantName/Subtotal). 합칠 때마다 변환 코드를 쓰게 되므로
공통 필드명으로 한 번 정규화해 둔다.

검증 규칙은 소스마다 다르다:
    receipts3000  산술 항등식 — 품목합==합계, 공급가액+부가세==합계
                  (GT 300건 전수가 만족함을 확인했으므로 규칙으로 삼을 수 있다)
    korie/kid     필드 형식 — 전화번호 형식, 이상 문자, 금액에 한글 등
                  (배포본 라벨에 오독이 섞여 있어 형식 검사로 걸러낸다)

usage:
    python3 scripts/build_clean_dataset.py
"""
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path("/workspace")
RD = ROOT / "receipt_data"
OUT = RD / "clean_v1"

HAN = re.compile(r"[가-힣]")
DIGIT = re.compile(r"\d")
PHONE = re.compile(r"\d{2,4}[-\s)]?\s?\d{3,4}[-\s]?\d{4}")
ODD = re.compile(r"[^\w\s가-힣.,\-:()/#*&'\"~+@%]")
BIZNO = re.compile(r"^\d{3}-\d{2}-\d{5}$")
ROW = re.compile(r"^\s*\|\s*([^|]+?)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*$")

# 공통 필드명 — 두 소스의 서로 다른 키를 여기로 모은다
KO2COMMON = {"상호명": "merchant_name", "주소": "merchant_address",
             "전화번호": "merchant_phone", "사업자번호": "business_no",
             "거래일시": "transaction_datetime", "공급가액": "subtotal",
             "부가세": "tax", "합계": "total", "결제수단": "payment_method",
             "업종": "business_type"}
EN2COMMON = {"MerchantName": "merchant_name", "MerchantAddress": "merchant_address",
             "MerchantPhoneNumber": "merchant_phone", "TransactionDate": "transaction_date",
             "TransactionTime": "transaction_time", "Subtotal": "subtotal",
             "TotalTax": "tax", "Total": "total", "ReceiptNumber": "receipt_no"}

QUESTION = {
    "merchant_name":        "이 영수증의 상호명이 뭐야?",
    "merchant_address":     "이 영수증의 주소가 뭐야?",
    "merchant_phone":       "이 영수증의 전화번호가 뭐야?",
    "transaction_date":     "이 영수증의 거래 날짜가 뭐야?",
    "transaction_time":     "이 영수증의 거래 시각이 뭐야?",
    "transaction_datetime": "이 영수증의 거래일시가 뭐야?",
    "subtotal":             "이 영수증의 공급가액이 얼마야?",
    "tax":                  "이 영수증의 부가세가 얼마야?",
    "total":                "이 영수증의 합계 금액이 얼마야?",
    "receipt_no":           "이 영수증의 영수증 번호가 뭐야?",
    "business_no":          "이 영수증의 사업자번호가 뭐야?",
    "payment_method":       "이 영수증의 결제수단이 뭐야?",
}
MONEY = {"subtotal", "tax", "total"}


def field_flags(name, text):
    text = str(text).strip()
    f = []
    if not text:
        return ["빈 값"]
    if len(text) <= 1:
        f.append("한 글자")
    if ODD.search(text):
        f.append("이상 문자")
    if name == "merchant_phone":
        if BIZNO.match(text):
            f.append("사업자번호가 전화번호 칸에")
        elif not PHONE.search(text):
            f.append("전화번호 형식 아님")
    if name in MONEY:
        if not DIGIT.search(text):
            f.append("금액에 숫자 없음")
        if HAN.search(text):
            f.append("금액에 한글 섞임")
    if name.startswith("transaction_") and not DIGIT.search(text):
        f.append("날짜/시각에 숫자 없음")
    return f


def num(s):
    return int(re.sub(r"[^\d]", "", str(s)) or 0)


# ── receipts3000 (합성) ────────────────────────────────────────────
def load_synth():
    mf = {}
    for l in open(RD / "receipts3000/manifest.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            mf[r["image_path"].split("/")[-1]] = r
    split_of = {}
    for s in ("train", "val", "test"):
        for l in open(RD / f"unified/{s}.jsonl", encoding="utf-8"):
            if l.strip():
                p = json.loads(l)["images"][0]
                if "receipts3000" in p:
                    split_of[p.split("/")[-1]] = s

    docs = []
    for fn, meta in sorted(mf.items()):
        gt = json.loads((RD / "receipts3000" / meta["gt_path"]).read_text(encoding="utf-8"))
        fields, flags = {}, []
        for ko, com in KO2COMMON.items():
            if ko in gt and str(gt[ko]).strip():
                v = str(gt[ko]).strip()
                fl = field_flags(com, v)
                if fl:
                    flags += [f"{com}: {x}" for x in fl]
                else:
                    fields[com] = v
        items = [{"name": it["명"], "qty": it["수량"],
                  "unit_price": it["단가"], "amount": it["금액"]}
                 for it in gt.get("품목", [])]
        # 산술 항등식
        tot = num(gt.get("합계", 0))
        if items and sum(i["amount"] for i in items) != tot:
            flags.append("품목합 != 합계")
        if num(gt.get("공급가액", 0)) + num(gt.get("부가세", 0)) != tot:
            flags.append("공급가액+부가세 != 합계")

        img = RD / "receipts3000/images" / fn
        w, h = Image.open(img).size
        docs.append({
            "doc_id": "syn_" + fn.replace(".jpg", ""),
            "image": str(img)[len(str(ROOT)) + 1:],
            "width": w, "height": h,
            "source": "receipts3000", "capture": "synthetic",
            "quality": meta.get("quality"), "subdomain": meta.get("subdomain"),
            "split": split_of.get(fn, "train"),
            "gt_status": "verified" if not flags else "flagged",
            "fields": fields, "items": items,
            "markdown": gt.get("markdown"),
            "layout": [],
            "checks": {"passed": not flags, "flags": flags},
        })
    return docs


# ── korie/kid (실촬영) ─────────────────────────────────────────────
def load_kid():
    R = RD / "korie"
    SP = (("train", "train/train"), ("val", "val/val"), ("test", "test/test"))
    # 크롭이 세 split 에 흩어져 있으므로 전부에서 모은다
    texts = defaultdict(dict)
    for _, d in SP:
        for p in (R / "ocr" / d).glob("*.txt"):
            m = re.match(r"(IMG\d+)_(.+)", p.stem)
            if m and m.group(2) in EN2COMMON:
                texts[m.group(1)][EN2COMMON[m.group(2)]] = p.read_text(encoding="utf-8").strip()

    docs = []
    for split, d in SP:
        # images/ 에 루시아 추론 결과(*-result.json) 등이 섞여 있으므로 확장자로 거른다
        for img in sorted((R / "kid" / d / "images").glob("*")):
            if not img.is_file() or img.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            iid = img.stem
            fields, flags = {}, []
            for com, v in sorted(texts.get(iid, {}).items()):
                fl = field_flags(com, v)
                if fl:
                    flags += [f"{com}: {x}" for x in fl]
                else:
                    fields[com] = v
            layout = []
            lp = R / "kid" / d / "labels" / f"{iid}.txt"
            if lp.exists():
                for line in lp.read_text().splitlines():
                    t = line.split()
                    if len(t) == 5:
                        c, cx, cy, w_, h_ = int(t[0]), *map(float, t[1:])
                        layout.append({"cls_id": c, "bbox_norm": [
                            round(cx - w_ / 2, 6), round(cy - h_ / 2, 6),
                            round(cx + w_ / 2, 6), round(cy + h_ / 2, 6)]})
            if not fields:
                flags.append("정상 필드 0개")
            w, h = Image.open(img).size
            docs.append({
                "doc_id": "kid_" + iid,
                "image": str(img)[len(str(ROOT)) + 1:],
                "width": w, "height": h,
                "source": "korie/kid", "capture": "photo",
                "quality": None, "subdomain": None,
                "split": split,
                "gt_status": "unverified",
                "fields": fields, "items": [],
                "markdown": None,
                "layout": layout,
                "checks": {"passed": not flags, "flags": flags},
            })
    return docs


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "gt").mkdir(parents=True)

    docs = load_synth() + load_kid()
    for d in docs:
        (OUT / "gt" / f"{d['doc_id']}.json").write_text(
            json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({k: d[k] for k in
                                ("doc_id", "image", "width", "height", "source",
                                 "capture", "quality", "split", "gt_status")} |
                               {"n_fields": len(d["fields"]), "n_items": len(d["items"]),
                                "n_layout": len(d["layout"]),
                                "passed": d["checks"]["passed"]},
                               ensure_ascii=False) + "\n")

    # 학습용 jsonl — 검증 통과 문서의 통과 필드만
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from infer_receipt import RECEIPT_CONTRACT
    counts = defaultdict(Counter)
    for split in ("train", "val", "test"):
        rows = []
        for d in docs:
            if d["split"] != split or not d["checks"]["passed"]:
                continue
            if d["markdown"]:
                rows.append({"doc_id": d["doc_id"], "task": "markdown",
                             "images": [d["image"]],
                             "messages": [{"role": "user", "content": "<image>" + RECEIPT_CONTRACT},
                                          {"role": "assistant", "content": d["markdown"]}]})
                counts[split]["markdown"] += 1
            for name, v in d["fields"].items():
                if name in QUESTION:
                    rows.append({"doc_id": d["doc_id"], "task": "vqa", "field": name,
                                 "images": [d["image"]],
                                 "messages": [{"role": "user", "content": "<image>" + QUESTION[name]},
                                              {"role": "assistant", "content": str(v)}]})
                    counts[split]["vqa"] += 1
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[split]["_합계"] = len(rows)
        counts[split]["_이미지"] = len({r["images"][0] for r in rows})

    src = Counter(d["source"] for d in docs)
    ok = Counter(d["source"] for d in docs if d["checks"]["passed"])
    print(f"\n문서 {len(docs)}개 → {OUT}")
    print(f"{'소스':<16}{'문서':>7}{'검증통과':>9}{'비율':>8}")
    for s in src:
        print(f"{s:<16}{src[s]:>7}{ok[s]:>9}{ok[s]/src[s]*100:>7.1f}%")
    print(f"\n{'split':<10}{'markdown':>10}{'vqa':>8}{'합계':>8}{'이미지':>8}")
    for s in ("train", "val", "test"):
        c = counts[s]
        print(f"{s:<10}{c['markdown']:>10}{c['vqa']:>8}{c['_합계']:>8}{c['_이미지']:>8}")
    fl = Counter(x.split(":")[-1].strip() for d in docs for x in d["checks"]["flags"])
    print("\n제외 사유:")
    for k, v in fl.most_common(8):
        print(f"  {k:<26}{v:>5}")


if __name__ == "__main__":
    main()
