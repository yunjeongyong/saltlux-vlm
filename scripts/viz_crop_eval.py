"""
크롭 평가 결과(eval_crops.py 산출 JSON)를 눈으로 보는 HTML 리포트로 만든다.

CER 평균 0.45 같은 숫자만 보면 '나쁘다'까지만 안다. 무엇이 어떻게 틀렸는지는
크롭 이미지와 출력을 나란히 봐야 보인다.

  text  블록 -> 크롭 이미지 · 정답 · 모델별 출력을 나란히 (틀린 글자 강조)
  table 블록 -> 정답 HTML 과 예측 HTML 을 각각 **렌더링**해서 나란히
                (표는 코드로 보면 구조가 안 보인다)

여러 결과 JSON 을 같이 넘기면 모델을 나란히 비교한다. 같은 crop_id 로 붙인다.

실패 유형은 자동 분류한다 — 빈 출력 / 반복 루프 / 폭주 / 수식 폭주 / 언어 오분류.

usage:
    # 한 개
    python3 scripts/viz_crop_eval.py --eval receipt_data/review/eval_crops_base.json

    # 여러 모델 비교 (이름:경로)
    python3 scripts/viz_crop_eval.py \
        --eval "LUXIA(serve)=receipt_data/review/eval_crops_luxia.json" \
               "base=receipt_data/review/eval_crops_base.json" \
        --out receipt_data/review/crop_eval_report.html

    # 나쁜 것부터 60건만
    python3 scripts/viz_crop_eval.py --eval ... --worst 60
"""
import argparse
import base64
import html
import io
import json
import re
import statistics as st
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CROPS = ROOT / "receipt_data/eval_crops"

# 표 안에서만 허용할 태그. 예측 HTML 을 그대로 렌더링하므로 화이트리스트로 거른다.
TAG_OK = re.compile(r"</?(table|thead|tbody|tr|td|th|br)\b[^>]*>", re.I)
TAG_ANY = re.compile(r"<[^>]+>")


# ─────────────────────────── 실패 유형 분류 ───────────────────────────

def repeat_ratio(s, n=12):
    """가장 많이 반복된 n-gram 이 전체에서 차지하는 비율. 루프 탐지용."""
    if len(s) < n * 3:
        return 0.0
    grams = {}
    for i in range(len(s) - n + 1):
        g = s[i:i + n]
        grams[g] = grams.get(g, 0) + 1
    top = max(grams.values())
    return top * n / len(s)


def hangul_ratio(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "\uac00" <= c <= "\ud7a3") / len(letters)


def classify(gt, pred):
    """왜 틀렸는지 한 가지로 이름 붙인다. 정상 오류면 None."""
    p = (pred or "").strip()
    if not p:
        return "빈 출력"
    if repeat_ratio(p) > 0.35:
        return "반복 루프"
    if re.search(r"\\begin\{|\\frac|\$\$|\\hline", p) and not re.search(
            r"\\begin\{|\\frac|\$\$", gt):
        return "수식(LaTeX) 폭주"
    if len(p) > max(60, len(gt) * 3):
        return "폭주(정답보다 3배 이상 김)"
    if hangul_ratio(gt) > 0.5 and hangul_ratio(p) < 0.2:
        return "언어 오분류"
    return None


# ─────────────────────────── 표시 도우미 ───────────────────────────

def thumb(rel, max_w=520, max_h=900):
    p = CROPS / rel
    if not p.exists():
        return None
    im = Image.open(p).convert("RGB")
    if im.width > max_w or im.height > max_h:
        s = min(max_w / im.width, max_h / im.height)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def safe_table(s):
    """예측 HTML 을 렌더링하되 표 태그만 남긴다. 스크립트·스타일 주입 방지."""
    s = re.sub(r"(?is)<(script|style)\b.*?</\1>", "", s or "")
    out, last = [], 0
    for m in TAG_ANY.finditer(s):
        out.append(html.escape(s[last:m.start()]))
        out.append(m.group(0) if TAG_OK.fullmatch(m.group(0)) else "")
        last = m.end()
    out.append(html.escape(s[last:]))
    r = "".join(out)
    return r if "<table" in r.lower() else f'<p class="notable">표 태그 없음<br>{r[:400]}</p>'


def diff_marks(gt, pred):
    """예측 패널 하나에 정답 대비 차이를 전부 담는다.

    빠뜨린 부분(정답에만 있음)을 예측 쪽에 안 그리면 '무엇을 못 읽었나'가
    안 보인다. 덧붙인 것과 빠뜨린 것을 같은 흐름에 섞어 그린다.
    """
    import difflib
    sm = difflib.SequenceMatcher(None, gt, pred, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        gs, ps = html.escape(gt[i1:i2]), html.escape(pred[j1:j2])
        if tag == "equal":
            out.append(ps)
            continue
        if gs:
            out.append(f'<mark class="miss">{gs}</mark>')
        if ps:
            out.append(f'<mark class="add">{ps}</mark>')
    return "".join(out)


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


# ─────────────────────────── 데이터 적재 ───────────────────────────

def load(specs):
    """[이름=경로] 목록을 읽어 crop_id 로 합친다."""
    models, merged, order = [], {}, []
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            path, name = name, None
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        keys = [k for k in d["summary"]]
        for k in keys:
            label = f"{name}·{k}" if name and len(keys) > 1 else (name or k)
            models.append({"label": label, "key": k, "summary": d["summary"][k],
                           "src": path})
            for r in d["rows"]:
                cid = r["crop_id"]
                if cid not in merged:
                    merged[cid] = {k2: r[k2] for k2 in
                                   ("crop_id", "doc_id", "split", "task", "label",
                                    "image", "gt")}
                    merged[cid]["preds"] = {}
                    order.append(cid)
                if k in r:
                    merged[cid]["preds"][label] = r[k]
    return models, [merged[c] for c in order]


# ─────────────────────────── HTML ───────────────────────────

CSS = """
:root{color-scheme:light;--ground:#f1f4f2;--surface:#fbfcfb;--surface-2:#f4f6f4;
--ink:#14171a;--ink-2:#535a5e;--muted:#7d8487;--hairline:#e0e4e1;--rule:#c8cdc9;
--gt:#0f7a56;--good:#0ca30c;--bad:#d03b3b;--warn:#b8860b;
--miss:#ffd9d9;--add:#d9f0ff;
--sans:system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",sans-serif;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,"D2Coding",monospace;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--ground:#0d0f0e;--surface:#191b1a;--surface-2:#202322;--ink:#fff;--ink-2:#c1c6c2;
--muted:#8b918d;--hairline:#2b2f2c;--rule:#3b403c;--gt:#3fbf95;--good:#4cc44c;
--bad:#f06a6a;--warn:#e0b34a;--miss:#5c2b2b;--add:#1e3b52;}}
:root[data-theme="dark"]{color-scheme:dark;--ground:#0d0f0e;--surface:#191b1a;
--surface-2:#202322;--ink:#fff;--ink-2:#c1c6c2;--muted:#8b918d;--hairline:#2b2f2c;
--rule:#3b403c;--gt:#3fbf95;--good:#4cc44c;--bad:#f06a6a;--warn:#e0b34a;
--miss:#5c2b2b;--add:#1e3b52;}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
line-height:1.55;font-size:14px}
.wrap{max-width:1400px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--muted);font-size:13px;margin-bottom:24px}
.card{background:var(--surface);border:1px solid var(--hairline);border-radius:10px;
padding:16px 18px;margin-bottom:18px}
h2{font-size:16px;margin:0 0 12px}
table.sum{border-collapse:collapse;width:100%;font-size:13px}
table.sum th,table.sum td{border-bottom:1px solid var(--hairline);padding:7px 10px;
text-align:right}
table.sum th:first-child,table.sum td:first-child{text-align:left}
table.sum th{color:var(--ink-2);font-weight:600;background:var(--surface-2)}
.scroll{overflow-x:auto}
.crop{background:var(--surface);border:1px solid var(--hairline);border-radius:10px;
margin-bottom:16px;overflow:hidden}
.chead{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px 14px;
background:var(--surface-2);border-bottom:1px solid var(--hairline);font-size:12px}
.cid{font-family:var(--mono);font-weight:600}
.tag{border:1px solid var(--rule);border-radius:99px;padding:1px 9px;color:var(--ink-2)}
.tag.fail{border-color:var(--bad);color:var(--bad)}
.body{display:grid;grid-template-columns:minmax(180px,300px) 1fr;gap:16px;padding:14px}
@media(max-width:820px){.body{grid-template-columns:1fr}}
.shot img{max-width:100%;border:1px solid var(--rule);border-radius:6px;
background:#fff;display:block}
.shot .cap{color:var(--muted);font-size:11px;margin-top:5px}
.cols{display:grid;gap:12px}
.pane{border:1px solid var(--hairline);border-radius:8px;overflow:hidden}
.ptop{display:flex;justify-content:space-between;gap:8px;padding:6px 10px;
background:var(--surface-2);border-bottom:1px solid var(--hairline);font-size:12px}
.pname{font-weight:600}
.pname.gtc{color:var(--gt)}
.score{font-family:var(--mono);font-size:11px;color:var(--ink-2)}
pre.txt{margin:0;padding:10px;white-space:pre-wrap;word-break:break-word;
font-family:var(--mono);font-size:12px;max-height:340px;overflow:auto}
mark.miss{background:var(--miss);color:inherit;border-radius:2px}
mark.add{background:var(--add);color:inherit;border-radius:2px}
.tbl{padding:10px;overflow-x:auto}
.tbl table{border-collapse:collapse;font-size:11.5px;min-width:100%}
.tbl td,.tbl th{border:1px solid var(--rule);padding:3px 6px;text-align:left}
.notable{color:var(--bad);font-family:var(--mono);font-size:11.5px;margin:0}
.legend{color:var(--muted);font-size:12px}
.legend mark{padding:0 4px}
"""


def render(models, rows, args, stats):
    P = []
    A = P.append
    A(f"<title>크롭 평가 리포트 — {len(rows)}건</title><style>{CSS}</style>")
    A('<div class="wrap">')
    A("<h1>크롭 평가 리포트</h1>")
    A(f'<div class="sub">평가 크롭 {stats["n_all"]:,}건 중 {len(rows):,}건 표시'
      f' · 영수증 {stats["n_doc"]}장 · 모델 {len(models)}종'
      f' · 정렬: {stats["sort_ko"]}</div>')

    # 요약
    A('<div class="card"><h2>지표 요약</h2><div class="scroll"><table class="sum">')
    A("<tr><th>모델</th><th>text n</th><th>CER 평균</th><th>CER 중앙값</th>"
      "<th>CER&gt;1</th><th>table n</th><th>TEDS</th><th>AVG 매크로</th></tr>")
    for m in models:
        s = m["summary"]
        t, tb = s.get("text", {}), s.get("table", {})
        cs = stats["cer_by_model"].get(m["label"], [])
        A(f'<tr><td>{html.escape(m["label"])}</td>'
          f'<td>{t.get("n","—")}</td><td>{fmt(t.get("CER"))}</td>'
          f'<td>{fmt(st.median(cs)) if cs else "—"}</td>'
          f'<td>{sum(1 for c in cs if c>1)}</td>'
          f'<td>{tb.get("n","—")}</td><td>{fmt(tb.get("TEDS"))}</td>'
          f'<td>{fmt(s.get("AVG",{}).get("매크로"))}</td></tr>')
    A("</table></div></div>")

    # 실패 유형
    if stats["fails"]:
        A('<div class="card"><h2>실패 유형</h2><div class="scroll"><table class="sum">')
        A("<tr><th>유형</th>" + "".join(
            f"<th>{html.escape(m['label'])}</th>" for m in models) + "</tr>")
        for kind in sorted(stats["fails"]):
            A(f"<tr><td>{kind}</td>" + "".join(
                f'<td>{stats["fails"][kind].get(m["label"],0)}</td>'
                for m in models) + "</tr>")
        A("</table></div>")
        A('<p class="legend">정상 범위 오류(오탈자 등)는 유형에 넣지 않는다 — '
          "여기 잡히는 건 구조적 실패다.</p></div>")

    A('<div class="card legend">비교 표시 — '
      '<mark class="miss">정답에만 있음(빠뜨림)</mark> · '
      '<mark class="add">예측에만 있음(덧붙임)</mark></div>')

    # 크롭별
    for r in rows:
        A('<div class="crop"><div class="chead">')
        A(f'<span class="cid">{html.escape(r["crop_id"])}</span>')
        A(f'<span class="tag">{r["task"]}</span>')
        A(f'<span class="tag">{html.escape(r["label"])}</span>')
        A(f'<span class="tag">{r["split"]}</span>')
        for m in models:
            k = classify(r["gt"], (r["preds"].get(m["label"]) or {}).get("pred", ""))
            if k:
                A(f'<span class="tag fail">{html.escape(m["label"])}: {k}</span>')
        A("</div>")

        A('<div class="body">')
        src = None if args.no_images else thumb(r["image"])
        A('<div class="shot">')
        A(f'<img src="{src}" alt="{html.escape(r["crop_id"])}">' if src
          else '<div class="cap">이미지 없음</div>')
        A(f'<div class="cap">{html.escape(r["image"])}</div></div>')

        A('<div class="cols">')
        if r["task"] == "table":
            A('<div class="pane"><div class="ptop"><span class="pname gtc">정답</span>'
              "</div>" + f'<div class="tbl">{safe_table(r["gt"])}</div></div>')
            for m in models:
                pr = r["preds"].get(m["label"])
                if not pr:
                    continue
                A(f'<div class="pane"><div class="ptop">'
                  f'<span class="pname">{html.escape(m["label"])}</span>'
                  f'<span class="score">TEDS {fmt(pr.get("teds"))}</span></div>'
                  f'<div class="tbl">{safe_table(pr.get("pred",""))}</div></div>')
        else:
            A('<div class="pane"><div class="ptop"><span class="pname gtc">정답</span>'
              "</div>" + f'<pre class="txt">{html.escape(r["gt"])}</pre></div>')
            for m in models:
                pr = r["preds"].get(m["label"])
                if not pr:
                    continue
                marked = diff_marks(r["gt"], pr.get("pred", ""))
                A(f'<div class="pane"><div class="ptop">'
                  f'<span class="pname">{html.escape(m["label"])}</span>'
                  f'<span class="score">CER {fmt(pr.get("cer"))} · '
                  f'{pr.get("sec","—")}s</span></div>'
                  f'<pre class="txt">{marked}</pre></div>')
        A("</div></div></div>")

    A("</div>")
    return "\n".join(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", nargs="+", required=True,
                    help="[이름=]결과JSON. 여러 개면 나란히 비교")
    ap.add_argument("--worst", type=int, default=40,
                    help="CER 나쁜 순 상위 N건만 (0=전량)")
    ap.add_argument("--task", nargs="+", default=["text", "table"])
    ap.add_argument("--sort", choices=["worst", "best", "id"], default="worst")
    ap.add_argument("--by", default=None,
                    help="정렬 기준 모델 라벨 (기본: 첫 모델)")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "receipt_data/review/crop_eval_report.html"))
    args = ap.parse_args()

    models, rows = load(args.eval)
    if not models:
        raise SystemExit("결과가 비었다")
    n_all = len(rows)
    rows = [r for r in rows if r["task"] in args.task]

    by = args.by or models[0]["label"]

    def key(r):
        pr = r["preds"].get(by) or {}
        v = pr.get("cer")
        if v is None:                      # 표는 TEDS 를 뒤집어 같은 축으로
            t = pr.get("teds")
            v = None if t is None else 1 - t
        return -1 if v is None else v

    if args.sort == "worst":
        rows.sort(key=key, reverse=True)
    elif args.sort == "best":
        rows.sort(key=key)

    # 통계는 자르기 전 전량으로 낸다
    cer_by_model, fails = {}, {}
    for m in models:
        lab = m["label"]
        cer_by_model[lab] = [p["cer"] for r in rows
                             if (p := r["preds"].get(lab)) and p.get("cer") is not None]
        for r in rows:
            pr = r["preds"].get(lab)
            if not pr:
                continue
            k = classify(r["gt"], pr.get("pred", ""))
            if k:
                fails.setdefault(k, {}).setdefault(lab, 0)
                fails[k][lab] += 1

    stats = {"n_all": n_all, "n_doc": len({r["doc_id"] for r in rows}),
             "cer_by_model": cer_by_model, "fails": fails,
             "sort_ko": {"worst": f"나쁜 순 ({by} 기준)",
                         "best": f"좋은 순 ({by} 기준)",
                         "id": "crop_id 순"}[args.sort]}

    shown = rows[:args.worst] if args.worst else rows
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(models, shown, args, stats), encoding="utf-8")

    print(f"모델 {len(models)}종: {', '.join(m['label'] for m in models)}")
    for kind, per in sorted(fails.items()):
        print(f"  실패 [{kind}]: " + " / ".join(f"{k} {v}건" for k, v in per.items()))
    print(f"\n{len(shown):,}건 표시 (전체 {len(rows):,}건) -> {out}")
    print(f"크기 {out.stat().st_size/1e6:.1f}MB")


if __name__ == "__main__":
    main()
