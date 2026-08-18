"""
korie/kid 실촬영 영수증의 이미지별 GT JSON 생성 + 검증.

이미지 1장 ↔ JSON 1개로 만든다. 세 소스를 ID 로 조인한다:
    kid/images/IMG*.png        전체 영수증 실촬영 사진
    kid/labels/IMG*.txt        YOLO bbox (클래스명 매핑은 배포본에 없음)
    ocr/*/IMG*_{필드}.txt      필드별 텍스트

ie/*.csv 는 넣지 않는다 — 상품명이 30% 손상돼 있고 검수 여부가 불명이다.

GT 는 unverified 로 표기한다. 루시아 추론이 'MerchantName: 포시애플-청주본점'
이라고 낸 이미지의 KorIE GT 가 '포시애를-청주' 인 사례가 있어, 이 GT 자체에
오독이 섞여 있음이 확인됐다. verified 로 적으면 뒤에 쓰는 사람이 정답으로 믿는다.

usage:
    python3 scripts/build_kid_gt.py            # 생성 + 검증 리포트
    python3 scripts/build_kid_gt.py --emit     # 학습용 jsonl 까지 (의심 건 제외)
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/workspace")
R = ROOT / "receipt_data/korie"
OUT = ROOT / "receipt_data/kid_gt"

SPLITS = (("train", "train/train"), ("val", "val/val"), ("test", "test/test"))
HAN = re.compile(r"[가-힣]")
DIGIT = re.compile(r"\d")
PHONE = re.compile(r"\d{2,4}[-\s)]?\s?\d{3,4}[-\s]?\d{4}")
ODD = re.compile(r"[^\w\s가-힣.,\-:()/#*&'\"~+@%]")
BIZNO = re.compile(r"^\d{3}-\d{2}-\d{5}$")

# 질문 문안 — 필드마다 하나. 학습 시 지시문 다양성이 이 만큼 확보된다.
QUESTION = {
    "MerchantName":        "이 영수증의 상호명이 뭐야?",
    "MerchantAddress":     "이 영수증의 주소가 뭐야?",
    "MerchantPhoneNumber": "이 영수증의 전화번호가 뭐야?",
    "TransactionDate":     "이 영수증의 거래 날짜가 뭐야?",
    "TransactionTime":     "이 영수증의 거래 시각이 뭐야?",
    "Total":               "이 영수증의 합계 금액이 얼마야?",
    "TotalTax":            "이 영수증의 부가세가 얼마야?",
    "Subtotal":            "이 영수증의 공급가액이 얼마야?",
    "ReceiptNumber":       "이 영수증의 영수증 번호가 뭐야?",
}
MONEY = ("Total", "TotalTax", "Subtotal")


def flags(field, text):
    """이 값이 GT 로 쓰기에 의심스러운 이유들. 비어 있으면 통과."""
    f = []
    if not text:
        return ["빈 값"]
    if len(text) <= 1:
        f.append("한 글자")
    if ODD.search(text):
        f.append("이상 문자")
    if field == "MerchantPhoneNumber":
        if BIZNO.match(text):
            f.append("사업자번호가 전화번호 칸에")
        elif not PHONE.search(text):
            f.append("전화번호 형식 아님")
    if field in MONEY:
        if not DIGIT.search(text):
            f.append("금액에 숫자 없음")
        if HAN.search(text):
            f.append("금액에 한글 섞임")
    if field in ("TransactionDate", "TransactionTime") and not DIGIT.search(text):
        f.append("날짜/시각에 숫자 없음")
    if field == "MerchantName" and not re.search(r"[가-힣A-Za-z]", text):
        f.append("상호명에 문자 없음")
    return f


def build():
    OUT.mkdir(exist_ok=True)
    docs, stat = defaultdict(list), Counter()

    # 한 영수증의 크롭이 ocr 의 세 split 에 흩어져 있다(IMG00001: train 6 / val 2 / test 2).
    # 그래서 필드는 세 폴더 전부에서 모으고, 문서의 split 은 사진이 있는 kid 기준으로 정한다.
    # 사진은 kid 의 한 split 에만 존재하므로 이렇게 해도 split 간 누수는 없다.
    texts = defaultdict(dict)
    for _, od in SPLITS:
        for p in (R / "ocr" / od).glob("*.txt"):
            m = re.match(r"(IMG\d+)_(.+)", p.stem)
            if m and m.group(2) in QUESTION:
                texts[m.group(1)][m.group(2)] = p.read_text(encoding="utf-8").strip()

    for split, d in SPLITS:
        imgs = {p.stem: p for p in (R / "kid" / d / "images").glob("*") if p.is_file()}
        lbl_dir = R / "kid" / d / "labels"

        for iid, ipath in sorted(imgs.items()):
            fields, bad = {}, {}
            for fname, val in sorted(texts.get(iid, {}).items()):
                fl = flags(fname, val)
                (bad if fl else fields)[fname] = {"value": val, "flags": fl} if fl else val
                stat[f"필드_{'의심' if fl else '정상'}"] += 1
                for x in fl:
                    stat[f"사유_{x}"] += 1

            layout = []
            lp = lbl_dir / f"{iid}.txt"
            if lp.exists():
                for line in lp.read_text().splitlines():
                    t = line.split()
                    if len(t) == 5:
                        c, cx, cy, w, h = int(t[0]), *map(float, t[1:])
                        layout.append({"cls_id": c,
                                       "bbox_norm": [round(cx - w / 2, 6), round(cy - h / 2, 6),
                                                     round(cx + w / 2, 6), round(cy + h / 2, 6)]})

            doc = {
                "image": str(ipath)[len(str(ROOT)) + 1:],
                "image_id": iid,
                "split": split,
                "source": "korie/kid",
                "gt_status": "unverified",
                "gt_note": "KorIE 배포본 라벨 그대로. 오독 사례 확인됨 — 정답으로 신뢰 금지.",
                "fields": fields,
                "suspect_fields": bad,
                "layout": layout,
                "layout_note": "YOLO 정규화 bbox. 클래스명 매핑은 배포본에 없음(cls_id 0~16).",
            }
            (OUT / f"{iid}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            docs[split].append(doc)
            stat[f"문서_{split}"] += 1
            if bad:
                stat["의심 필드가 있는 문서"] += 1
            if not fields:
                stat["정상 필드가 0개인 문서"] += 1
    return docs, stat


def emit(docs):
    """검증 통과 필드만으로 학습용 jsonl 생성."""
    D = ROOT / "receipt_data/kid_vqa"
    D.mkdir(exist_ok=True)
    n = {}
    for split, ds in docs.items():
        rows = []
        for d in ds:
            for fname, val in d["fields"].items():
                rows.append({"images": [d["image"]], "field": fname,
                             "messages": [{"role": "user",
                                           "content": "<image>" + QUESTION[fname]},
                                          {"role": "assistant", "content": val}]})
        p = D / f"{split}_clean.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n[split] = (len(rows), len({r["images"][0] for r in rows}))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    args = ap.parse_args()

    docs, stat = build()
    tot = sum(stat[f"문서_{s}"] for s, _ in SPLITS)
    ok, sus = stat["필드_정상"], stat["필드_의심"]
    print(f"\nGT JSON {tot}개 생성 → {OUT}")
    print(f"  {'split':<8}" + "".join(f"{s:>9}" for s, _ in SPLITS))
    print(f"  {'문서':<8}" + "".join(f"{stat[f'문서_{s}']:>9}" for s, _ in SPLITS))
    print(f"\n필드 {ok + sus}개 · 정상 {ok} ({ok/(ok+sus)*100:.1f}%) · 의심 {sus} ({sus/(ok+sus)*100:.1f}%)")
    print(f"의심 필드가 있는 문서 {stat['의심 필드가 있는 문서']} / {tot}")
    print(f"정상 필드가 0개인 문서 {stat['정상 필드가 0개인 문서']}")
    print("\n의심 사유:")
    for k, v in sorted(stat.items(), key=lambda x: -x[1]):
        if k.startswith("사유_"):
            print(f"  {k[3:]:<28}{v:>5}")

    if args.emit:
        n = emit(docs)
        print("\n학습용 jsonl (의심 필드 제외):")
        for s, (r, i) in n.items():
            print(f"  {s+'_clean.jsonl':<22}{r:>6}건 · 이미지 {i}장")


if __name__ == "__main__":
    main()
