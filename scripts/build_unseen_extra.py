"""약하게 나온 슬롯만 후보를 더 뽑는다.

build_unseen_samples.py 가 슬롯당 2~3장만 담아서, 영수증 페이지와 논문처럼
난도가 높은 태스크는 표본이 모자란다. 이미 돌린 것은 빼고 추가분만 만든다.

usage: python3 scripts/build_unseen_extra.py <추가폴더> <기존폴더>
"""
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "receipt_data"
OUT = Path(sys.argv[1])
PREV = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

DROP_DOCS = {"receipt34", "receipt67"}


def jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n")
            if l.strip()]


def page_of(doc_id):
    for split in ("val", "test"):
        for ext in (".jpg", ".png", ".jpeg"):
            p = RD / f"labeled/{split}/images/{doc_id}{ext}"
            if p.exists():
                return p, RD / f"labeled/{split}/json/{doc_id}.json"
    return None, None


def page_markdown(doc_id):
    _, jp = page_of(doc_id)
    if not jp or not jp.exists():
        return None
    d = json.loads(jp.read_text(encoding="utf-8"))
    return d["result"]["elements"][0].get("markdown") or d["result"].get("md")


def main():
    done = {m["src"] for m in json.loads(
        (PREV / "_gt.json").read_text(encoding="utf-8"))}
    seen3 = set()
    for r in jsonl(RD / "upload_mix/train.jsonl"):
        for im in r.get("images", []):
            seen3.add(Path(im).stem)

    made = []

    def add(slot, task, name, src, gt, note, gt_src):
        rel = str(Path(src).relative_to(ROOT))
        if rel in done:
            return False
        im = Image.open(src)
        made.append({"slot": slot, "task": task, "file": f"{name}.png",
                     "size": list(im.size), "gt": gt, "src": rel,
                     "note": note, "gt_src": gt_src, "crop": None,
                     "known": None})
        return True

    # ── 영수증 페이지 후보 6장 ──────────────────────────────────────────
    ev = [r for r in jsonl(RD / "eval_crops/manifest.jsonl")
          if r["doc_id"] not in DROP_DOCS]
    rec = {r["crop_id"]: r["upload-mix-qwen36"] for r in json.loads(
        (RD / "review/eval_crops_exp003.json").read_text(encoding="utf-8"))["rows"]}
    by_doc = {}
    for r in ev:
        s = rec.get(r["crop_id"], {})
        v = (1 - min(s["cer"], 1.0)) if s.get("cer") is not None else s.get("teds")
        if v is not None:
            by_doc.setdefault(r["doc_id"], []).append(v)
    rank = sorted(by_doc, key=lambda d: -sum(by_doc[d]) / len(by_doc[d]))
    n = 0
    for doc in rank:
        md, page = page_markdown(doc), page_of(doc)[0]
        if md and page and 400 <= len(md) <= 2200:
            if add("영수증_페이지", "receipt_markdown", f"03_영수증_페이지_x{n+1}",
                   page, md, f"평가셋 90장 중 {doc} · 학습 미사용",
                   "라벨링 파서 출력 — 페이지 단위는 미검수"):
                n += 1
                if n == 6:
                    break

    # ── 논문 후보 4장 ──────────────────────────────────────────────────
    rows4 = [r for r in jsonl(RD / "exp004c_260813/train.jsonl")
             if r["task"] == "doc_parsing/paper"
             and Path(r["images"][0]).stem not in seen3]
    rows4.sort(key=lambda r: len(r["messages"][-1]["content"]))
    n = 0
    for r in rows4:
        gt = r["messages"][-1]["content"]
        if not (300 <= len(gt) <= 1400):
            continue
        p = ROOT / r["images"][0]
        if not p.exists():
            continue
        if add("논문", "doc_parsing/paper", f"08_논문_x{n+1}", p, gt,
               "exp_003 학습 미사용 (exp_004 학습분)", "데이터셋 원본 정답"):
            n += 1
            if n == 4:
                break

    (OUT / "_gt.json").write_text(
        json.dumps(made, ensure_ascii=False, indent=1), encoding="utf-8")
    for m in made:
        print(f"  {m['file']:26} {str(m['size']):14} gt {len(m['gt']):>6,}자")
    print(f"\n추가 후보 {len(made)}장 → {OUT/'_gt.json'}")


if __name__ == "__main__":
    main()
