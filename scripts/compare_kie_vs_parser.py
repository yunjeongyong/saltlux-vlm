"""
KorIE 필드 라벨(KIE) 과 파서가 뽑아낸 텍스트를 대조한다.

KID 는 박스, KIE 는 값이다. 값 쪽은 파서 박스와 IoU 로 비교할 수 없으니
"GT 필드값 문자열이 파서 텍스트(markdown) 안에 실제로 등장하는가"로 본다.

이 수치는 두 방향으로 읽힌다.
  - 등장하면: 파서 텍스트에 그 값이 살아 있다 → VLM 학습 입력으로 쓸 만하다.
  - 안 등장하면: 파서 오독이거나, KorIE 라벨 자체가 오독이다.
    KorIE 배포본 라벨은 gt_status=unverified 다. 불일치 건은 둘 중
    누가 틀렸는지 사람이 봐야 하는 목록이라서 따로 뽑아 저장한다.

usage:
    python3 scripts/compare_kie_vs_parser.py
"""
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/data/workspace/yjyong/receipt_data")
GT_DIR = ROOT / "kid_gt"
PARSE_DIR = ROOT / "clean_v1" / "parse"
OUT = ROOT / "review" / "kie_vs_parser_mismatch.jsonl"


def norm(s):
    """공백·구두점·전각을 털어낸 비교용 문자열. 자릿수 콤마와 하이픈은
    표기 차이일 뿐이라 지운다. 한글/영문/숫자만 남긴다."""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"[^0-9A-Za-z가-힣]", "", s).lower()


def parse_text(doc):
    els = (doc.get("result") or {}).get("elements") or []
    if els:
        return els[0].get("markdown") or ""
    return doc.get("md") or ""


def main():
    stat = defaultdict(lambda: [0, 0])     # field -> [hit, total]
    miss = []
    n_doc = 0
    for gf in sorted(GT_DIR.glob("*.json")):
        if gf.stem.endswith("-result"):
            continue
        pf = PARSE_DIR / f"kid_{gf.stem}.json"
        if not pf.exists():
            continue
        g = json.loads(gf.read_text(encoding="utf-8"))
        fields = g.get("fields") or {}
        if not fields:
            continue
        text = norm(parse_text(json.loads(pf.read_text(encoding="utf-8"))))
        if not text:
            continue
        n_doc += 1
        for k, v in fields.items():
            nv = norm(v)
            if not nv:
                continue
            stat[k][1] += 1
            if nv in text:
                stat[k][0] += 1
            else:
                miss.append({"image_id": gf.stem, "field": k, "gt": v})

    print(f"KIE 필드값이 파서 텍스트에 그대로 있는가  (문서 {n_doc}장)\n")
    tot_h = tot_n = 0
    for k, (h, n) in sorted(stat.items(), key=lambda x: -x[1][1]):
        tot_h += h
        tot_n += n
        print(f"  {k:22} {h:5}/{n:<5}  {100*h/n:5.1f}%")
    print(f"  {'합계':22} {tot_h:5}/{tot_n:<5}  {100*tot_h/tot_n:5.1f}%")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for m in miss:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"\n불일치 {len(miss)}건 저장: {OUT}")
    print("  필드별 불일치 상위:", Counter(m["field"] for m in miss).most_common(5))
    print("\n  샘플 10건 (GT 값이 파서 텍스트에 없음):")
    for m in miss[:10]:
        print(f"    {m['image_id']:10} {m['field']:20} {m['gt']!r}")


if __name__ == "__main__":
    main()
