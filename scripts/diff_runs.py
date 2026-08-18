"""
설정 A/B 비교. 같은 문서를 다른 설정으로 재추출한 두 JSON 묶음을 비교한다.

용도:
  - Model Quality: Fast vs 상위
  - Image Captioning: OFF vs ON
  - threshold 변경 전/후 (설치 가이드 나온 뒤)

usage:
    python3 scripts/diff_runs.py baseline_dir/ variant_dir/
    python3 scripts/diff_runs.py . runs_quality_high/ --label-a Fast --label-b High
"""
import argparse
import json
import re
from pathlib import Path


def load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    blocks = d.get("parsing_res_list", [])
    boxes = d.get("layout_det_res", {}).get("boxes", [])
    scores = {}
    for b in boxes:
        key = tuple(round(v) for v in b["coordinate"])
        scores[key] = b["score"]
    items = []
    for p in blocks:
        bb = tuple(round(v) for v in (p.get("block_bbox") or []))
        s = next((v for k, v in scores.items()
                  if len(bb) == 4 and max(abs(x - y) for x, y in zip(k, bb)) < 6), None)
        items.append({
            "label": p.get("block_label"),
            "order": p.get("block_order"),
            "score": s,
            "content": str(p.get("block_content", "")),
            "bbox": bb,
        })
    return {"settings": d.get("model_settings", {}), "size": (d.get("width"), d.get("height")),
            "items": items}


def norm(t):
    return re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", t))


def match(a_items, b_items):
    """bbox 근접으로 블록 짝짓기. 좌표가 크게 달라지면 매칭 실패로 남긴다."""
    used, pairs = set(), []
    for i, a in enumerate(a_items):
        best, bd = None, 1e9
        for j, b in enumerate(b_items):
            if j in used or len(a["bbox"]) != 4 or len(b["bbox"]) != 4:
                continue
            d = sum(abs(x - y) for x, y in zip(a["bbox"], b["bbox"]))
            if d < bd:
                best, bd = j, d
        if best is not None and bd < 60:
            used.add(best)
            pairs.append((i, best))
        else:
            pairs.append((i, None))
    for j in range(len(b_items)):
        if j not in used:
            pairs.append((None, j))
    return pairs


def diff_doc(pa, pb, la, lb):
    A, B = load(pa), load(pb)
    out = {"doc": Path(pa).stem, "settings_a": A["settings"], "settings_b": B["settings"],
           "blocks_a": len(A["items"]), "blocks_b": len(B["items"]),
           "text_changed": [], "score_changed": [], "order_changed": [],
           "filled": [], "emptied": [], "only_a": [], "only_b": []}

    for i, j in match(A["items"], B["items"]):
        if j is None:
            out["only_a"].append(A["items"][i]["label"]); continue
        if i is None:
            out["only_b"].append(B["items"][j]["label"]); continue
        a, b = A["items"][i], B["items"][j]
        ca, cb = norm(a["content"]), norm(b["content"])
        if not ca and cb:
            out["filled"].append({"label": a["label"], "chars": len(cb)})
        elif ca and not cb:
            out["emptied"].append({"label": a["label"], "chars": len(ca)})
        elif ca != cb:
            out["text_changed"].append({"label": a["label"],
                                        la: a["content"][:120], lb: b["content"][:120]})
        if a["score"] is not None and b["score"] is not None and abs(a["score"] - b["score"]) > 0.02:
            out["score_changed"].append({"label": a["label"],
                                         la: round(a["score"], 4), lb: round(b["score"], 4),
                                         "delta": round(b["score"] - a["score"], 4)})
        if a["order"] != b["order"]:
            out["order_changed"].append({"label": a["label"], la: a["order"], lb: b["order"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a"); ap.add_argument("dir_b")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", default="eval_dataset/diff")
    args = ap.parse_args()
    la, lb = args.label_a, args.label_b

    a_files = {p.name: p for p in Path(args.dir_a).glob("*.json")}
    b_files = {p.name: p for p in Path(args.dir_b).glob("*.json")}
    common = sorted(set(a_files) & set(b_files))
    if not common:
        print(f"공통 파일 없음. {args.dir_a} 와 {args.dir_b} 의 파일명이 같아야 합니다.")
        return 1

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    agg = {"score_up": 0, "score_down": 0, "text_changed": 0,
           "filled": 0, "emptied": 0, "order_changed": 0}
    results = []
    print(f"\n{'문서':<20}{'텍스트변경':>10}{'score↑':>8}{'score↓':>8}{'내용채움':>9}{'내용소실':>9}{'order변경':>10}")
    print("-" * 76)
    for name in common:
        r = diff_doc(a_files[name], b_files[name], la, lb)
        up = sum(1 for s in r["score_changed"] if s["delta"] > 0)
        dn = len(r["score_changed"]) - up
        agg["score_up"] += up; agg["score_down"] += dn
        agg["text_changed"] += len(r["text_changed"])
        agg["filled"] += len(r["filled"]); agg["emptied"] += len(r["emptied"])
        agg["order_changed"] += len(r["order_changed"])
        results.append(r)
        print(f"{r['doc'][:19]:<20}{len(r['text_changed']):>10}{up:>8}{dn:>8}"
              f"{len(r['filled']):>9}{len(r['emptied']):>9}{len(r['order_changed']):>10}")
    print("-" * 76)
    print(f"{'합계':<20}{agg['text_changed']:>10}{agg['score_up']:>8}{agg['score_down']:>8}"
          f"{agg['filled']:>9}{agg['emptied']:>9}{agg['order_changed']:>10}")

    # 주목할 변화 상세
    for r in results:
        notable = r["filled"] + r["emptied"]
        if notable or r["order_changed"]:
            print(f"\n[{r['doc']}]")
            for f in r["filled"]:
                print(f"   ✅ 내용 채워짐: {f['label']} ({f['chars']}자)")
            for e in r["emptied"]:
                print(f"   ❌ 내용 사라짐: {e['label']} ({e['chars']}자)")
            for o in r["order_changed"]:
                print(f"   ↕ order 변경: {o['label']}  {o[la]} -> {o[lb]}")

    (out / "diff.json").write_text(
        json.dumps({"labels": [la, lb], "summary": agg, "docs": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}/diff.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
