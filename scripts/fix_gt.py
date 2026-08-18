"""정답 라벨의 오독을 고친다. 정답 필드만, 지정한 문서에서만 바꾼다.

전역 문자열 치환을 쓰면 안 된다. 같은 문자열이 모델 예측(pred)에도, 다른
영수증에도 들어 있어서 관측값을 덮어쓰거나 무관한 문서를 건드린다(실제로
'가지(전남, 강원,' 한 건이 40개 파일 · receipt27 · gcse_00061 · 과거 평가
스냅샷까지 번졌다). 그래서 doc_id 로 범위를 좁히고 정답 필드만 손댄다.

usage:
    python3 scripts/fix_gt.py receipt37 "풍듀핫웜" "퐁듀핫윙"
    python3 scripts/fix_gt.py receipt37 "풍듀핫웜" "퐁듀핫윙" --dry
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "receipt_data"

# 크롭 정답이 사는 곳: rows 의 gt 필드
CROP_MANIFESTS = ["eval_crops/manifest.jsonl", "eval_crops_v2/manifest.jsonl",
                  "eval_crops_v3/manifest.jsonl", "train_crops/manifest.jsonl"]
# 평가 결과: gt 만 고치고 pred 는 절대 건드리지 않는다(모델이 낸 관측값이다)
EVAL_JSONS = ["review/eval_crops_luxia.json", "review/eval_crops_base.json",
              "review/eval_crops_exp003.json"]
# 학습·검증 데이터셋: messages 의 정답(assistant) 쪽
DATASETS = [f"{d}/{s}.jsonl"
            for d in ("upload_mix", "merged_260813",
                      "exp004_260813", "exp004b_260813")
            for s in ("train", "val", "test", "eval_labeled")]


def backup(p, tag):
    b = p.with_suffix(p.suffix + f".bak_{tag}")
    if not b.exists():
        shutil.copy2(p, b)


def fix_jsonl(p, doc, old, new, field, tag, dry):
    """jsonl 을 줄 단위로 읽어 doc_id 가 맞는 행만 고친다."""
    if not p.exists():
        return 0
    lines, hits = [], 0
    # splitlines() 는 U+2028·U+000B 같은 문자에서도 줄을 나눈다. 영수증 정답에
    # 그런 문자가 섞여 있으면 한 줄을 두 조각으로 잘라 JSON 파싱이 깨진다.
    for line in p.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("doc_id") == doc:
            if field == "gt" and old in (r.get("gt") or ""):
                r["gt"] = r["gt"].replace(old, new)
                hits += 1
            elif field == "messages":
                for m in r.get("messages", []):
                    if isinstance(m.get("content"), str) and old in m["content"]:
                        m["content"] = m["content"].replace(old, new)
                        hits += 1
        lines.append(json.dumps(r, ensure_ascii=False))
    if hits and not dry:
        backup(p, tag)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hits


def fix_evaljson(p, doc, old, new, tag, dry):
    """평가 결과 json 의 gt 만. pred 는 손대지 않는다."""
    if not p.exists():
        return 0
    d = json.loads(p.read_text(encoding="utf-8"))
    hits = 0
    for r in d.get("rows", []):
        if r.get("doc_id") == doc and old in (r.get("gt") or ""):
            r["gt"] = r["gt"].replace(old, new)
            hits += 1
    if hits and not dry:
        backup(p, tag)
        p.write_text(json.dumps(d, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    return hits


def fix_label(doc, old, new, tag, dry):
    """원천 라벨 파일. 같은 오독이 markdown/html/md/parsing_res_list 에
    복제돼 있으므로 그 파일 안에서는 전부 바꾼다(파일 자체가 이 문서의 것이다)."""
    total = 0
    for split in ("train", "val", "test"):
        p = RD / f"labeled/{split}/json/{doc}.json"
        if not p.exists():
            continue
        s = p.read_text(encoding="utf-8")
        n = s.count(old)
        if n and not dry:
            backup(p, tag)
            p.write_text(s.replace(old, new), encoding="utf-8")
        if n:
            print(f"  {'[dry] ' if dry else ''}labeled/{split}/json/{doc}.json"
                  f"  {n}곳")
        total += n
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_id")
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--tag", default="gtfix")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.old == a.new:
        sys.exit("old 와 new 가 같다")

    print(f"{a.doc_id}: {a.old!r} → {a.new!r}{'  (dry run)' if a.dry else ''}")
    total = fix_label(a.doc_id, a.old, a.new, a.tag, a.dry)
    for rel in CROP_MANIFESTS:
        n = fix_jsonl(RD / rel, a.doc_id, a.old, a.new, "gt", a.tag, a.dry)
        if n:
            print(f"  {'[dry] ' if a.dry else ''}{rel}  {n}행")
            total += n
    for rel in EVAL_JSONS:
        n = fix_evaljson(RD / rel, a.doc_id, a.old, a.new, a.tag, a.dry)
        if n:
            print(f"  {'[dry] ' if a.dry else ''}{rel}  {n}행 (gt 만)")
            total += n
    for rel in DATASETS:
        n = fix_jsonl(RD / rel, a.doc_id, a.old, a.new, "messages", a.tag, a.dry)
        if n:
            print(f"  {'[dry] ' if a.dry else ''}{rel}  {n}곳")
            total += n
    print(f"합계 {total}곳{' (변경 없음 — dry run)' if a.dry else ' 수정'}")


if __name__ == "__main__":
    main()
