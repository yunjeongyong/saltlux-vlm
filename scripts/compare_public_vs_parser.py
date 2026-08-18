"""
공개 데이터셋(CORD / SROIE)의 사람 GT 와 document-parse 결과를 대조한다.

두 데이터셋은 GT 성격이 다르므로 재는 축도 다르다.

  SROIE : 필드 4종(company/date/address/total)만 있고 bbox 가 없다.
          → GT 값이 파서 텍스트에 살아있는지(값 축)만 볼 수 있다.
  CORD  : CLOVA 가 사람 손으로 줄 단위 bbox + 카테고리를 붙였다.
          → 값 축과 영역 축(IoU/덮임)을 모두 볼 수 있다. 공개셋 중 유일.

"어느 쪽이 좋은가"는 한 숫자로 답할 수 없다. GT 는 값이 정확하고 파서는
값이 없는 자리까지 다 읽는다. 그래서 두 축을 따로 낸다.

  값 재현율   : GT 필드값이 파서 텍스트에 있는가          (파서가 GT 를 커버)
  영역 덮임   : GT 박스가 파서 박스 안에 들어가는가        (CORD 만)
  IoU 매칭    : 같은 입도로 잡았는가                       (CORD 만)
  파서 추가분 : GT 에 없는데 파서가 읽어낸 줄              (GT 미라벨 영역)

usage:
    python3 scripts/compare_public_vs_parser.py --ds sroie
    python3 scripts/compare_public_vs_parser.py --ds cord
"""
import argparse
import json
import re
import statistics as st
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/data/workspace/yjyong/receipt_data/public")


def norm(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("(주)", "주").replace("㈜", "주")
    return re.sub(r"[^0-9A-Za-z가-힣]", "", s).lower()


def parse_text(doc):
    els = (doc.get("result") or {}).get("elements") or []
    if els:
        return els[0].get("markdown") or ""
    return doc.get("md") or ""


def parse_page(doc):
    els = (doc.get("result") or {}).get("elements") or []
    return els[0].get("json") if els else None


def parse_boxes(doc):
    """파서 박스를 원본 이미지 좌표로 되돌린다 (parse_to_bbox.py 와 같은 방식)."""
    page = parse_page(doc)
    sw, sh = doc.get("_source_width"), doc.get("_source_height")
    if not page or not all((sw, sh, page.get("width"), page.get("height"))):
        return []
    sx, sy = page["width"] / sw, page["height"] / sh
    out = []
    for b in page.get("layout_det_res", {}).get("boxes", []):
        c = b.get("coordinate") or b.get("bbox")
        if not c or len(c) < 4:
            continue
        out.append({"label": b.get("label", "?"), "score": b.get("score", 0.0),
                    "bbox": [c[0]/sx, c[1]/sy, c[2]/sx, c[3]/sy]})
    return out


def parse_lines(doc):
    """파서가 읽어낸 줄 단위 텍스트+박스. GT 미라벨 영역을 세는 데 쓴다."""
    page = parse_page(doc)
    sw, sh = doc.get("_source_width"), doc.get("_source_height")
    if not page or not all((sw, sh, page.get("width"), page.get("height"))):
        return []
    sx, sy = page["width"] / sw, page["height"] / sh
    out = []
    for r in page.get("parsing_res_list", []):
        c = r.get("block_bbox") or r.get("bbox")
        t = r.get("block_content") or r.get("content") or ""
        if not c or len(c) < 4:
            continue
        out.append({"text": t, "bbox": [c[0]/sx, c[1]/sy, c[2]/sx, c[3]/sy]})
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def covered(gt, pred):
    ix = max(0.0, min(gt[2], pred[2]) - max(gt[0], pred[0]))
    iy = max(0.0, min(gt[3], pred[3]) - max(gt[1], pred[1]))
    area = (gt[2]-gt[0]) * (gt[3]-gt[1])
    return (ix*iy) / area if area > 0 else 0.0


def field_axis(pairs):
    """GT 필드값이 파서 텍스트에 등장하는가. exact / 토큰전부 / 부분 / 없음."""
    stat = defaultdict(lambda: Counter())
    misses = []
    for doc_id, g, p in pairs:
        text = norm(parse_text(p))
        if not text:
            continue
        for k, v in (g.get("fields") or {}).items():
            if k.endswith("__num") or not norm(v):
                continue
            stat[k]["n"] += 1
            if norm(v) in text:
                stat[k]["exact"] += 1
                continue
            toks = [norm(x) for x in re.split(r"[\s,]+", str(v)) if len(norm(x)) >= 2]
            hit = sum(1 for t in toks if t in text)
            if toks and hit == len(toks):
                stat[k]["tokens"] += 1
            elif hit:
                stat[k]["partial"] += 1
            else:
                stat[k]["none"] += 1
                misses.append({"doc_id": doc_id, "field": k, "gt": v})
    return stat, misses


def item_axis(pairs):
    """CORD 품목명·금액이 파서 텍스트에 있는가."""
    c = Counter()
    for _, g, p in pairs:
        text = norm(parse_text(p))
        if not text:
            continue
        for it in (g.get("items") or []):
            if it.get("name"):
                c["name_n"] += 1
                c["name_hit"] += norm(it["name"]) in text
            if it.get("price_raw"):
                c["price_n"] += 1
                c["price_hit"] += norm(it["price_raw"]) in text
    return c


def region_axis(pairs, iou_thr, gt_region=True):
    """CORD 전용. GT 줄박스 vs 파서 박스."""
    TP = FP = FN = 0
    covs, best = [], []
    per_cat = Counter()
    per_cat_cov = Counter()
    per_cat_hit = Counter()
    extra_lines, extra_above = 0, 0
    n_doc = 0
    ratio = []
    for doc_id, g, p in pairs:
        gt = [(b["category"], b["bbox"]) for b in (g.get("layout") or [])]
        if not gt:
            continue
        pb = [b["bbox"] for b in parse_boxes(p)]
        if not pb:
            continue
        n_doc += 1
        ys = [b[1] for _, b in gt] + [b[3] for _, b in gt]
        lo, hi = min(ys), max(ys)
        if gt_region:
            pb_eval = [b for b in pb if not (b[3] < lo - 5 or b[1] > hi + 5)]
        else:
            pb_eval = pb
        used = set()
        for cat, gb in gt:
            per_cat[cat] += 1
            bi, bv, bc = -1, 0.0, 0.0
            for i, q in enumerate(pb_eval):
                v = iou(gb, q)
                if v > bv:
                    bv, bi = v, i
                bc = max(bc, covered(gb, q))
            covs.append(bc)
            best.append(bv)
            if bc >= 0.9:
                per_cat_cov[cat] += 1
            if bv >= iou_thr and bi not in used:
                TP += 1
                used.add(bi)
                per_cat_hit[cat] += 1
            else:
                FN += 1
        FP += len(pb_eval) - len(used)
        ratio.append(len(pb_eval) / len(gt))
        # GT 세로 구간 밖에서 파서가 읽어낸 줄 = GT 미라벨 영역
        for ln in parse_lines(p):
            if ln["bbox"][3] < lo - 5 or ln["bbox"][1] > hi + 5:
                extra_lines += 1
                if ln["bbox"][3] < lo - 5:
                    extra_above += 1
    prec = TP/(TP+FP) if TP+FP else 0
    rec = TP/(TP+FN) if TP+FN else 0
    return dict(n_doc=n_doc, TP=TP, FP=FP, FN=FN, precision=prec, recall=rec,
                f1=2*prec*rec/(prec+rec) if prec+rec else 0,
                cov_mean=st.mean(covs) if covs else 0,
                cov90=sum(1 for c in covs if c >= 0.9)/len(covs) if covs else 0,
                iou_mean=st.mean(best) if best else 0,
                pred_per_gt=st.mean(ratio) if ratio else 0,
                extra_lines=extra_lines, extra_above=extra_above,
                per_cat={c: (per_cat_cov[c], per_cat_hit[c], n) for c, n in per_cat.most_common()})


def load(ds):
    d = ROOT / ds
    pairs = []
    for gf in sorted((d / "gt").glob("*.json")):
        pf = d / "parse" / f"{gf.stem}.json"
        if not pf.exists():
            continue
        pairs.append((gf.stem, json.loads(gf.read_text(encoding="utf-8")),
                      json.loads(pf.read_text(encoding="utf-8"))))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="sroie", choices=["sroie", "cord"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    pairs = load(args.ds)
    if not pairs:
        raise SystemExit(f"{args.ds}: GT+parse 짝이 없다. 파싱을 먼저 돌려라.")
    print(f"[{args.ds}] GT+파서 짝 {len(pairs)}장\n")

    res = {"ds": args.ds, "n": len(pairs)}

    stat, misses = field_axis(pairs)
    print("값 축 — GT 필드값이 파서 텍스트에 등장하는가")
    print(f"  {'필드':<14}{'exact':>8}{'토큰전부':>9}{'부분':>7}{'없음':>7}{'전체':>7}   실질")
    tot = Counter()
    for k, c in sorted(stat.items(), key=lambda x: -x[1]["n"]):
        eff = 100*(c['exact']+c['tokens'])/c['n']
        print(f"  {k:<14}{c['exact']:>8}{c['tokens']:>9}{c['partial']:>7}{c['none']:>7}{c['n']:>7}  {eff:5.1f}%")
        for kk in ("exact", "tokens", "partial", "none", "n"):
            tot[kk] += c[kk]
    eff = 100*(tot['exact']+tot['tokens'])/tot['n']
    print(f"  {'합계':<14}{tot['exact']:>8}{tot['tokens']:>9}{tot['partial']:>7}{tot['none']:>7}{tot['n']:>7}  {eff:5.1f}%")
    res["fields"] = {k: dict(c) for k, c in stat.items()}
    res["fields_total"] = dict(tot)

    it = item_axis(pairs)
    if it.get("name_n"):
        print(f"\n품목 축 — 품목명 {it['name_hit']}/{it['name_n']} "
              f"({100*it['name_hit']/it['name_n']:.1f}%)  "
              f"금액 {it['price_hit']}/{it['price_n']} ({100*it['price_hit']/it['price_n']:.1f}%)")
        res["items"] = dict(it)

    if any(g.get("layout") for _, g, _ in pairs):
        r = region_axis(pairs, args.iou)
        print(f"\n영역 축 — GT 줄박스 {r['TP']+r['FN']}개 vs 파서 박스 (문서 {r['n_doc']}장)")
        print(f"  GT 1개당 파서 박스 {r['pred_per_gt']:.2f}개")
        print(f"  덮임 평균 {r['cov_mean']*100:.1f}%  /  90% 이상 덮임 {r['cov90']*100:.1f}%")
        print(f"  IoU>={args.iou}  precision {r['precision']*100:.1f}%  recall {r['recall']*100:.1f}%  F1 {r['f1']*100:.1f}%")
        print(f"  최고 IoU 평균 {r['iou_mean']*100:.1f}%")
        print(f"  GT 라벨 구간 밖에서 파서가 읽은 줄 {r['extra_lines']}개 (그중 상단 {r['extra_above']}개)")
        print(f"\n  카테고리별 (덮임90 / IoU매칭 / 전체)")
        for c, (c90, hit, n) in list(r["per_cat"].items())[:14]:
            print(f"    {c:22} {c90:5}/{hit:5}/{n:<6} 덮임 {100*c90/n:5.1f}%  매칭 {100*hit/n:5.1f}%")
        res["region"] = r

    print(f"\n값 미검출 {len(misses)}건")
    for m in misses[:8]:
        print(f"    {m['doc_id']:20} {m['field']:10} {m['gt']!r}")

    out = args.json_out or str(ROOT.parent / "review" / f"public_{args.ds}_vs_parser.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    miss_out = Path(out).with_name(f"public_{args.ds}_field_miss.jsonl")
    miss_out.write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in misses), encoding="utf-8")
    print(f"\n저장: {out}\n      {miss_out}")


if __name__ == "__main__":
    main()
