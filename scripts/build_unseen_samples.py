"""학습에 쓰지 않은 이미지로 포맷별 샘플 후보를 고른다.

기존 build_dataset_samples.py 는 exp004c train.jsonl 한 곳에서만 뽑아, exp_003 이
이미 학습한 이미지가 절반 넘게 섞였다. 여기서는 소스마다 미학습 풀을 따로 만든다.

  영수증 (텍스트/표 크롭·페이지)  eval 978건 / 90장 — val·test, 학습에 미사용
  aihub 페이지 OCR               36,598장 중 학습에 안 쓴 것
  pubtabnet·multimodal·kogovdoc  exp_003 미학습 (exp_004 학습분, 정답이 여기에만 있음)

영수증은 기록된 평가 결과(review/eval_crops_exp003.json)에 크롭별 CER/TEDS 가
있으므로 잘 나온 것을 바로 고를 수 있다. 나머지 소스는 점수가 없어 슬롯마다 후보를
여러 장 담고, 추론 뒤 pick_unseen_best.py 가 제일 좋은 것을 남긴다.

usage: python3 scripts/build_unseen_samples.py <출력폴더>
"""
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "receipt_data"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

EXP003 = RD / "upload_mix/train.jsonl"
EXP004 = RD / "exp004c_260813/train.jsonl"
EVAL_MAN = RD / "eval_crops/manifest.jsonl"
EVAL_REC = RD / "review/eval_crops_exp003.json"
AIHUB = ROOT / "vlm_dataset_upload/train/public/aihub-71299-ocr/train.jsonl"
PUBROOT = ROOT / "vlm_dataset_upload/train/public"

# 좌표계가 90도 뒤집혀 평가셋에서 뺀 문서. 978건 기준을 그대로 따른다.
DROP_DOCS = {"receipt34", "receipt67"}


def jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n")
            if l.strip()]


def trained_stems():
    """exp_003 / exp_004 학습 입력에 들어간 이미지 stem 을 모은다."""
    seen3, seen4 = set(), set()
    for path, acc in ((EXP003, seen3), (EXP004, seen4)):
        for r in jsonl(path):
            for im in r.get("images", []):
                acc.add(Path(im).stem)
    return seen3, seen4


def eval_pool():
    rows = [r for r in jsonl(EVAL_MAN) if r["doc_id"] not in DROP_DOCS]
    assert len({r["doc_id"] for r in rows}) == 90, "평가셋 90장이 아니다"
    rec = {r["crop_id"]: r["upload-mix-qwen36"]
           for r in json.loads(EVAL_REC.read_text(encoding="utf-8"))["rows"]}
    for r in rows:
        r["score"] = rec.get(r["crop_id"], {})
    return rows


def spread(rows, n, key):
    """같은 영수증에서 몰아 뽑으면 대표성이 없다. 문서를 겹치지 않게 고른다."""
    out, used = [], set()
    for r in sorted(rows, key=key):
        if r["doc_id"] in used:
            continue
        out.append(r)
        used.add(r["doc_id"])
        if len(out) == n:
            break
    return out


def mid_picks(pool, n, key):
    """점수를 모르는 소스용. 가장 짧은/긴 것을 피해 중간 구간에서 고른다."""
    pool = sorted(pool, key=key)
    if not pool:
        return []
    idx = [len(pool) * (i + 1) // (n + 1) for i in range(n)]
    return [pool[i] for i in idx]


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
    el = d["result"]["elements"][0]
    return el.get("markdown") or d["result"].get("md")


def main():
    seen3, seen4 = trained_stems()
    ev = eval_pool()
    made = []

    def add(slot, task, name, src, gt, note, gt_src, crop=None, known=None):
        im = Image.open(src)
        made.append({
            "slot": slot,                  # 같은 슬롯 후보 중 하나만 남는다
            "task": task,
            "file": f"{name}.png",
            "size": list(im.size),
            "gt": gt,
            "src": str(Path(src).relative_to(ROOT)),
            "note": note,                  # 미학습 근거를 패널에 그대로 찍는다
            "gt_src": gt_src,              # 정답 출처 (검수본인지 파서 출력인지)
            "crop": crop,                  # 평가셋 크롭이면 원본 위치를 그릴 정보
            "known": known,                # 기록된 평가 점수 (있으면)
        })

    # ── 영수증 텍스트 크롭 — 기록된 CER 이 좋은 것 중에서 ──────────────────
    txt = [r for r in ev
           if r["task"] == "text" and len(r["gt"]) >= 60      # 너무 짧으면 볼 게 없다
           and r["score"].get("cer") is not None and r["score"]["cer"] <= 0.02]
    for i, r in enumerate(spread(txt, 2, lambda r: -len(r["gt"])), 1):
        add(f"텍스트크롭_영수증_{i}", "receipt_crop_text",
            f"01_텍스트크롭_영수증_{i}", RD / "eval_crops" / r["image"], r["gt"],
            f"평가셋 {r['split']} · {r['doc_id']} · 학습 미사용",
            "검수 정정본 (fix_gt.py 반영)", crop=r,
            known={"CER": round(r["score"]["cer"], 4)})

    # ── 영수증 표 크롭 — 기록된 TEDS 가 좋은 것 중에서 ────────────────────
    tbl = [r for r in ev
           if r["task"] == "table" and len(r["gt"]) >= 300
           and r["score"].get("teds") is not None and r["score"]["teds"] >= 0.9]
    for i, r in enumerate(spread(tbl, 2, lambda r: -r["score"]["teds"]), 1):
        add(f"표크롭_영수증_{i}", "receipt_crop_table",
            f"02_표크롭_영수증_{i}", RD / "eval_crops" / r["image"], r["gt"],
            f"평가셋 {r['split']} · {r['doc_id']} · 학습 미사용",
            "검수 정정본 (fix_gt.py 반영)", crop=r,
            known={"TEDS": round(r["score"]["teds"], 4)})

    # ── 영수증 페이지 — 크롭 평균 점수가 좋은 문서부터 후보 3장 ────────────
    by_doc = {}
    for r in ev:
        s = r["score"].get("cer")
        s = (1 - min(s, 1.0)) if s is not None else r["score"].get("teds")
        if s is not None:
            by_doc.setdefault(r["doc_id"], []).append(s)
    rank = sorted(by_doc, key=lambda d: -sum(by_doc[d]) / len(by_doc[d]))
    n = 0
    for doc in rank:
        md, page = page_markdown(doc), page_of(doc)[0]
        if md and page and 600 <= len(md) <= 2000:
            n += 1
            add("영수증_페이지", "receipt_markdown", f"03_영수증_페이지_c{n}",
                page, md, f"평가셋 90장 중 {doc} · 학습 미사용",
                "라벨링 파서 출력 — 페이지 단위는 미검수")
            if n == 3:
                break

    # ── aihub 페이지 OCR — 두 실험 모두 미학습분에서 후보 3장 ─────────────
    pool = [r for r in jsonl(AIHUB)
            if Path(r["images"][0]).stem not in seen3
            and Path(r["images"][0]).stem not in seen4
            and 300 <= len(r["messages"][-1]["content"]) <= 1200]
    for i, r in enumerate(mid_picks(pool, 3,
                                    lambda r: len(r["messages"][-1]["content"])), 1):
        add("페이지OCR_aihub", "page_ocr", f"04_페이지OCR_aihub_c{i}",
            PUBROOT / r["images"][0], r["messages"][-1]["content"],
            f"aihub 36,598장 중 학습 미사용분 {len(pool):,}장에서 선정",
            "aihub-71299-ocr 원본 정답")

    # ── exp_003 미학습 공개 데이터 (정답이 exp004c 에만 있어 거기서 뽑는다) ──
    rows4 = {}
    for r in jsonl(EXP004):
        rows4.setdefault(r["task"], []).append(r)

    PUB = [("html_table", "05_표크롭_pubtabnet", "표크롭_pubtabnet", 3, 2000),
           ("parsing/전체텍스트", "06_문서전사_multimodal", "문서전사_multimodal", 2, 1500),
           ("doc_parsing/government", "07_공공문서", "공공문서", 2, 1500),
           ("doc_parsing/paper", "08_논문", "논문", 2, 2000)]
    for task, name, slot, k, cap in PUB:
        pool = [r for r in rows4.get(task, [])
                if Path(r["images"][0]).stem not in seen3
                and 200 <= len(r["messages"][-1]["content"]) <= cap]
        for i, r in enumerate(mid_picks(pool, k,
                                        lambda r: len(r["messages"][-1]["content"])), 1):
            p = ROOT / r["images"][0]
            if not p.exists():
                print(f"  ✗ 이미지 없음 {r['images'][0]}")
                continue
            add(slot, task, f"{name}_c{i}", p, r["messages"][-1]["content"],
                "exp_003 학습 미사용 (exp_004 학습분)", "데이터셋 원본 정답")

    (OUT / "_gt.json").write_text(
        json.dumps(made, ensure_ascii=False, indent=1), encoding="utf-8")
    for m in made:
        k = f"  기록 {m['known']}" if m["known"] else ""
        print(f"  {m['file']:30} {str(m['size']):14} gt {len(m['gt']):>6,}자{k}")
    slots = len({m["slot"] for m in made})
    print(f"\n후보 {len(made)}장 / 슬롯 {slots}개 → {OUT/'_gt.json'}")


if __name__ == "__main__":
    main()
