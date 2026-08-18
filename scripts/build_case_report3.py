"""exp_003 3단 비교 리포트 (베이스 / 출발점 / 학습 후).

앞서 만든 2단 리포트는 LUXIA 를 "학습 전"으로 불러 오해를 만들었다. LUXIA 는
우리가 학습시키기 전의 상태가 아니라 별개의 서비스 모델이다. 우리 학습의
순수 효과는 Qwen 맨몸(출발점) → 학습 후로만 잴 수 있으므로 세 줄을 모두 놓는다.

입력: scratchpad/s3.json (이미지까지 base64 로 박아둔 것)
"""
import html
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])

# 검증 통과 팔레트(3슬롯, all-pairs). scripts/validate_palette.js 로 확인.
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # light
D1, D2, D3 = "#3987e5", "#d95926", "#199e70"          # dark

CATS = [
    ("A", "긴 오답이 해소된 경우", 26, "text",
     "출발점에서 CER이 1을 넘던 건이다. 크롭에 없는 내용까지 길게 뱉던 것이 "
     "학습 후 범위 안으로 들어왔다."),
    ("B", "긴 오답이 남은 경우", 18, "text",
     "학습 후에도 CER이 1을 넘는다. 다음 학습이 노려야 할 지점이다."),
    ("C", "학습 후 나빠진 경우", 122, "text",
     "출발점에서는 맞히던 것을 학습 후 틀린다. 이번 분석의 핵심 발견이 여기 있다."),
    ("D", "완전 정답으로 바뀐 경우", 88, "text",
     "글자 단위로 틀리던 것이 학습 후 완전히 일치한다."),
    ("E", "서비스 모델은 맞고 우리는 틀린 경우", 94, "text",
     "베이스 모델이 정확히 맞힌 것을 학습 모델이 틀렸다. 서비스 대체를 논하려면 "
     "여기를 줄여야 한다."),
    ("F", "표 — 좋아진 경우", 33, "table",
     "출발점 대비 TEDS가 0.1 이상 올랐다. 첫 사례는 베이스와 출발점이 표 구조를 "
     "통째로 놓친(TEDS 0.0) 것을 학습 모델만 잡아낸 건이다."),
    ("G", "표 — 나빠진 경우", 20, "table", "출발점 대비 TEDS가 0.1 이상 떨어졌다."),
    ("H", "표 — 학습 후에도 낮은 경우", 16, "table",
     "학습 후에도 TEDS가 0.5 미만이다."),
]

ROWS = [("정답", "gt", None), ("베이스 (LUXIA)", "luxia", "s1"),
        ("출발점 (Qwen 맨몸)", "qwen", "s2"), ("학습 후", "ft", "s3")]


def clean(s):
    """모델이 뱉은 대체문자·제어문자를 표시 가능한 기호로 바꾼다.

    그대로 두면 배포가 인코딩 오류로 거부된다. 지우지 않고 □ 로 남기는 이유는
    "모델이 여기서 글자를 못 만들었다"는 것 자체가 관찰 대상이라서다.
    """
    return "".join("□" if c == "�" or (ord(c) < 32 and c not in "\n\t")
                   else c for c in s)


def diff_html(gt, pred):
    """정답 대비 어긋난 부분만 표시한다.

    예측에만 있는 글자는 붉게, 정답에만 있는 글자는 자리(‸)만 남긴다 —
    누락을 글자로 펼치면 예측 문장 자체가 읽히지 않는다.
    """
    out = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, gt, pred).get_opcodes():
        if tag == "equal":
            out.append(html.escape(pred[j1:j2]))
        elif tag in ("replace", "insert"):
            out.append(f'<mark>{html.escape(pred[j1:j2])}</mark>')
        else:
            out.append('<i class="gap" title="누락: '
                       + html.escape(gt[i1:i2][:60]) + '">‸</i>')
    return "".join(out) or '<i class="gap">(빈 출력)</i>'


def score(row, which):
    if row["task"] == "table":
        v = {"luxia": row["lt"], "qwen": row["qt"], "ft": row["ftt"]}[which]
        return f"TEDS {v:.3f}"
    v = {"luxia": row["lc"], "qwen": row["qc"], "ft": row["fc"]}[which]
    return f"CER {v:.3f}"


def delta(row):
    """출발점 → 학습 후. 우리 학습이 이 크롭에 무엇을 했는지."""
    if row["task"] == "table":
        d = row["ftt"] - row["qt"]
        good = d > 0
    else:
        d = row["fc"] - row["qc"]
        good = d < 0
    arrow = "▼" if d < 0 else "▲"
    cls = "good" if good else "bad"
    return f'<span class="delta {cls}">{arrow} {abs(d):.3f}</span>'


def case_html(row):
    lines = []
    for label, key, cls in ROWS:
        body = (html.escape(clean(row["gt"])) if key == "gt"
                else diff_html(clean(row["gt"]), clean(row[key])))
        sc = "" if key == "gt" else f'<em>{score(row, key)}</em>'
        lines.append(
            f'<dt class="{cls or "ref"}">{label}{sc}</dt>'
            f'<dd class="{cls or "ref"}">{body}</dd>')
    return f"""
<article class="case">
  <header><span class="cid">{html.escape(row['id'])}</span>
    <span class="dl">출발점 → 학습 후 {delta(row)}</span></header>
  <div class="shot"><img src="data:image/png;base64,{row['img']}"
    alt="{html.escape(row['id'])} 크롭 이미지"
    width="{row['w']}" height="{row['h']}" loading="lazy"></div>
  <dl>{''.join(lines)}</dl>
</article>"""


# ── 차트 (인라인 SVG, 외부 라이브러리 없음) ────────────────────────────
def slope(vals, title, unit, lower_better, n):
    """세 모델의 값을 한 줄로 잇는다. CER 과 TEDS 는 척도도 방향도 달라
    한 축에 겹치지 않고 각각 별도 차트로 그린다."""
    W, H, PL, PR, PT, PB = 560, 210, 116, 128, 30, 34
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.55 or 0.1
    lo, hi = lo - pad, hi + pad
    xs = [PL + i * (W - PL - PR) / 2 for i in range(3)]
    ys = [PT + (hi - v) / (hi - lo) * (H - PT - PB) for v in vals]
    names = ["베이스", "출발점", "학습 후"]
    cols = ["var(--s1)", "var(--s2)", "var(--s3)"]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    g = [f'<polyline class="slope" points="{pts}"/>']
    for i, (x, y, v) in enumerate(zip(xs, ys, vals)):
        anc = "end" if i == 2 else ("start" if i == 0 else "middle")
        dx = -13 if i == 2 else (13 if i == 0 else 0)
        dy = 0 if i != 1 else (-16 if lower_better else 18)
        g.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{cols[i]}" '
                 f'class="dot"><title>{names[i]} {v:.4f}</title></circle>')
        g.append(f'<text class="vlab" x="{x+dx:.1f}" y="{y+dy+5:.1f}" '
                 f'text-anchor="{anc}">{v:.4f}</text>')
        g.append(f'<text class="nlab" x="{x:.1f}" y="{H-12}" '
                 f'text-anchor="middle">{names[i]}</text>')
    return f"""<figure class="fig">
<figcaption><b>{title}</b><span>{unit} · {n}건 · {'낮을수록 좋음' if lower_better else '높을수록 좋음'}</span></figcaption>
<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title} 3모델 비교">{''.join(g)}</svg>
</figure>"""


def grouped_bars(bins, series, title, note):
    W, H, PL, PR, PT, PB = 620, 260, 40, 14, 18, 62
    mx = max(max(s) for s in series.values())
    n = len(bins)
    gw = (W - PL - PR) / n
    bw = min(15, (gw - 16) / 3)
    keys = [("luxia", "베이스", "var(--s1)"), ("qwen", "출발점", "var(--s2)"),
            ("ft", "학습 후", "var(--s3)")]
    g = []
    for gi in range(n):
        x0 = PL + gi * gw + (gw - bw * 3 - 4) / 2
        for si, (k, nm, c) in enumerate(keys):
            v = series[k][gi]
            h = v / mx * (H - PT - PB)
            x = x0 + si * (bw + 2)
            y = H - PB - h
            g.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                     f'height="{max(h,1):.1f}" rx="3" fill="{c}">'
                     f'<title>{nm} · {bins[gi]} · {v}건</title></rect>')
        g.append(f'<text class="tick" x="{PL+gi*gw+gw/2:.1f}" y="{H-PB+17}" '
                 f'text-anchor="middle">{bins[gi]}</text>')
        tot = " / ".join(str(series[k][gi]) for k, _, _ in keys)
        g.append(f'<text class="tick sm" x="{PL+gi*gw+gw/2:.1f}" y="{H-PB+33}" '
                 f'text-anchor="middle">{tot}</text>')
    g.append(f'<line class="axis" x1="{PL-6}" y1="{H-PB}" x2="{W-PR}" '
             f'y2="{H-PB}"/>')
    return f"""<figure class="fig wide">
<figcaption><b>{title}</b><span>{note}</span></figcaption>
<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">{''.join(g)}</svg>
<p class="legend"><span class="sw s1"></span>베이스
  <span class="sw s2"></span>출발점<span class="sw s3"></span>학습 후</p>
</figure>"""


def move_bar(move, tmove):
    def one(vals, label, total):
        W, H = 620, 46
        segs = [("좋아짐", vals[0], "var(--s3)"), ("변화 없음", vals[1], "var(--mid)"),
                ("나빠짐", vals[2], "var(--s2)")]
        x = 0.0
        g = []
        for nm, v, c in segs:
            w = v / total * W
            g.append(f'<rect x="{x+1:.1f}" y="8" width="{max(w-2,1):.1f}" '
                     f'height="26" rx="3" fill="{c}"><title>{nm} {v}건</title></rect>')
            if w > 54:
                g.append(f'<text class="inbar" x="{x+w/2:.1f}" y="26" '
                         f'text-anchor="middle">{v}</text>')
            x += w
        return (f'<div class="mrow"><span class="mlab">{label}</span>'
                f'<svg viewBox="0 0 {W} {H}" role="img" '
                f'aria-label="{label} 좋아짐 {vals[0]} 변화없음 {vals[1]} '
                f'나빠짐 {vals[2]}">{"".join(g)}</svg></div>')
    return f"""<figure class="fig wide">
<figcaption><b>학습으로 각 크롭이 어느 쪽으로 움직였나</b>
  <span>출발점 → 학습 후 · 텍스트 CER ±0.05, 표 TEDS ±0.1 기준</span></figcaption>
{one(move, '텍스트 933건', sum(move))}
{one(tmove, '표 103건', sum(tmove))}
<p class="legend"><span class="sw s3"></span>좋아짐
  <span class="sw mid"></span>변화 없음<span class="sw s2"></span>나빠짐</p>
</figure>"""


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    c, cases = d["charts"], d["cases"]
    figs = (slope(c["cer"], "CER 평균", "텍스트 크롭", True, c["n_txt"])
            + slope(c["teds"], "TEDS", "표 크롭", False, c["n_tab"])
            + grouped_bars(c["bins"], c["hist"], "텍스트 크롭 933건의 CER 분포",
                           "막대 아래 숫자는 베이스 / 출발점 / 학습 후 건수")
            + move_bar(c["move"], c["tmove"]))
    nav, secs = [], []
    for key, title, total, task, blurb in CATS:
        rows = cases.get(key, [])
        if not rows:
            continue
        nav.append(f'<a href="#c{key.lower()}">{html.escape(title)}'
                   f'<span>{total}</span></a>')
        secs.append(f"""
<section id="c{key.lower()}" class="cat">
  <div class="ch"><h2>{html.escape(title)}</h2>
    <p class="cnt">{total}건 중 {len(rows)}건</p></div>
  <p class="blurb">{html.escape(blurb)}</p>
  {''.join(case_html(r) for r in rows)}
</section>""")
    OUT.write_text(TPL.replace("{{FIGS}}", figs)
                   .replace("{{NAV}}", "".join(nav))
                   .replace("{{SECTIONS}}", "".join(secs)), encoding="utf-8")
    print(f"저장: {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


TPL = r"""<title>세 모델의 영수증 크롭 오답</title>
<style>
:root{
  --paper:#f6f6f3; --card:#fffffe; --ink:#191b1e; --dim:#63686e;
  --rule:#dfdfd9; --rail:#eceae5; --shot:#f0efe9; --mid:#b9b7ae;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --wrong:#9d3a27; --wrongbg:#f7e8e4; --goodink:#1f6b4a;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#131519; --card:#1a1d22; --ink:#e4e5e2; --dim:#959aa1;
  --rule:#2c3038; --rail:#22262c; --shot:#0d0f12; --mid:#5b6069;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --wrong:#e08a72; --wrongbg:#2e1a15; --goodink:#6cc296;
}}
:root[data-theme="dark"]{
  --paper:#131519; --card:#1a1d22; --ink:#e4e5e2; --dim:#959aa1;
  --rule:#2c3038; --rail:#22262c; --shot:#0d0f12; --mid:#5b6069;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --wrong:#e08a72; --wrongbg:#2e1a15; --goodink:#6cc296;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"Pretendard","Apple SD Gothic Neo","Malgun Gothic",system-ui,
    -apple-system,"Segoe UI",sans-serif;
  font-size:16px;line-height:1.65;-webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:0 24px 96px}
header.top{padding:64px 0 36px;border-bottom:2px solid var(--ink)}
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--dim);font-weight:700;margin:0 0 14px}
h1{font-size:clamp(30px,4.5vw,46px);line-height:1.15;letter-spacing:-.022em;
  font-weight:800;margin:0 0 16px;text-wrap:balance}
.lede{font-size:18px;color:var(--dim);margin:0;max-width:64ch}

.who{display:grid;gap:1px;background:var(--rule);border:1px solid var(--rule);
  grid-template-columns:repeat(auto-fit,minmax(230px,1fr));margin:34px 0 0}
.who div{background:var(--card);padding:15px 18px;border-top:3px solid}
.who div:nth-child(1){border-color:var(--s1)}
.who div:nth-child(2){border-color:var(--s2)}
.who div:nth-child(3){border-color:var(--s3)}
.who b{display:block;font-size:14px;margin-bottom:3px}
.who code{font-size:12px;color:var(--dim);word-break:break-all}
.who p{margin:6px 0 0;font-size:13.5px;color:var(--dim)}

.figs{display:grid;gap:18px;grid-template-columns:1fr 1fr;margin:40px 0 0}
.fig{margin:0;border:1px solid var(--rule);background:var(--card);padding:16px 18px 12px}
.fig.wide{grid-column:1/-1}
figcaption{display:flex;flex-wrap:wrap;gap:4px 12px;align-items:baseline;
  margin-bottom:8px}
figcaption b{font-size:15px;font-weight:700}
figcaption span{font-size:12.5px;color:var(--dim)}
.fig svg{display:block;width:100%;height:auto;overflow:visible}
.slope{fill:none;stroke:var(--mid);stroke-width:2}
.dot{stroke:var(--card);stroke-width:2}
.vlab{font-size:14px;font-weight:700;fill:var(--ink);
  font-variant-numeric:tabular-nums}
.nlab{font-size:12px;fill:var(--dim)}
.tick{font-size:11.5px;fill:var(--dim)}
.tick.sm{font-size:10.5px;font-variant-numeric:tabular-nums}
.axis{stroke:var(--rule);stroke-width:1}
.inbar{font-size:12px;font-weight:700;fill:#fff;
  font-variant-numeric:tabular-nums}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;
  margin:10px 0 0;font-size:12.5px;color:var(--dim)}
.sw{width:11px;height:11px;border-radius:2px;display:inline-block;
  margin-right:5px;vertical-align:-1px}
.sw.s1{background:var(--s1)}.sw.s2{background:var(--s2)}
.sw.s3{background:var(--s3)}.sw.mid{background:var(--mid)}
.mrow{display:flex;align-items:center;gap:12px;margin:6px 0}
.mlab{font-size:12.5px;color:var(--dim);white-space:nowrap;min-width:82px}
.mrow svg{flex:1}

.finding{border-left:3px solid var(--wrong);background:var(--rail);
  padding:22px 26px;margin:40px 0}
.finding h2{font-size:19px;font-weight:800;margin:0 0 10px}
.finding p{margin:0 0 10px;max-width:66ch}
.finding p:last-child{margin:0}
.finding pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:13px;background:var(--card);border:1px solid var(--rule);
  padding:12px 14px;margin:12px 0;overflow-x:auto}

nav.toc{display:flex;flex-wrap:wrap;gap:8px;margin:36px 0 0}
nav.toc a{display:inline-flex;align-items:baseline;gap:7px;
  border:1px solid var(--rule);background:var(--card);padding:7px 13px;
  font-size:14px;color:var(--ink);text-decoration:none}
nav.toc a span{font-size:12px;color:var(--dim);
  font-variant-numeric:tabular-nums}
nav.toc a:hover{border-color:var(--ink)}
nav.toc a:focus-visible{outline:2px solid var(--s1);outline-offset:2px}

.cat{margin:64px 0 0;scroll-margin-top:16px}
.ch{display:flex;align-items:baseline;justify-content:space-between;gap:16px;
  border-bottom:1px solid var(--ink);padding-bottom:8px}
.ch h2{font-size:23px;font-weight:800;margin:0;letter-spacing:-.015em}
.cnt{font-size:13px;color:var(--dim);margin:0;white-space:nowrap;
  font-variant-numeric:tabular-nums}
.blurb{font-size:15px;color:var(--dim);margin:14px 0 0;max-width:66ch}

.case{border:1px solid var(--rule);background:var(--card);margin:22px 0}
.case>header{display:flex;justify-content:space-between;gap:12px;
  flex-wrap:wrap;padding:11px 16px;border-bottom:1px solid var(--rule)}
.cid,.dl{font-size:12.5px;color:var(--dim)}
.cid{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.delta{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-weight:700;font-variant-numeric:tabular-nums}
.delta.good{color:var(--goodink)}.delta.bad{color:var(--wrong)}
.shot{background:var(--shot);padding:14px 16px;overflow-x:auto;
  border-bottom:1px solid var(--rule)}
.shot img{display:block;max-width:100%;height:auto}
.case dl{display:grid;grid-template-columns:150px 1fr;margin:0}
.case dt{padding:11px 0 11px 16px;font-size:12px;color:var(--dim);
  border-top:1px solid var(--rule);display:flex;flex-direction:column;gap:2px}
.case dt em{font-style:normal;font-size:11.5px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
.case dd{margin:0;padding:11px 16px 11px 12px;border-top:1px solid var(--rule);
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,
    "Apple SD Gothic Neo",monospace;
  font-size:13.5px;line-height:1.6;white-space:pre-wrap;word-break:break-all}
.case dt.ref,.case dd.ref{border-top:none}
.case dd.ref{color:var(--dim)}
.case dt.s1{box-shadow:inset 3px 0 var(--s1)}
.case dt.s2{box-shadow:inset 3px 0 var(--s2)}
.case dt.s3{box-shadow:inset 3px 0 var(--s3)}
mark{background:var(--wrongbg);color:var(--wrong);font-weight:600}
.gap{color:var(--wrong);font-weight:700;font-style:normal;cursor:help}

.next{margin:72px 0 0;border-top:2px solid var(--ink);padding-top:32px}
.next h2{font-size:23px;font-weight:800;margin:0 0 20px}
.next ol{margin:0;padding-left:22px}
.next li{margin:0 0 18px;max-width:68ch}
.next li p{margin:5px 0 0;color:var(--dim);font-size:15px}
footer{margin:56px 0 0;padding-top:20px;border-top:1px solid var(--rule);
  font-size:13px;color:var(--dim)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:.92em}
@media (max-width:760px){.figs{grid-template-columns:1fr}}
@media (max-width:620px){
  .case dl{grid-template-columns:1fr}
  .case dt{padding:11px 16px 0;flex-direction:row;gap:10px;align-items:baseline}
  .case dd{padding-left:16px;border-top:none}
}
</style>

<div class="wrap">
<header class="top">
  <p class="eyebrow">exp_003 · 영수증 크롭 1,036건 · 동일 평가셋</p>
  <h1>세 모델이 같은 크롭을 어떻게 읽었나</h1>
  <p class="lede">지표는 올랐다. 그런데 무엇이 좋아졌고 무엇이 나빠졌는지는
    평균값이 말해주지 않는다. 같은 크롭에 세 모델을 나란히 놓고 글자 단위로 비교한다.</p>
  <div class="who">
    <div><b>베이스 (LUXIA)</b>
      <code>luxia-document-parsing-high</code>
      <p>지금 서비스 중인 모델. 넘어야 할 목표선이지 우리 학습의 출발점이 아니다.</p></div>
    <div><b>출발점 (Qwen 맨몸)</b>
      <code>Qwen3.6-35B-A3B</code>
      <p>어댑터 없이 잰 공개 가중치. 우리 학습의 실제 출발점이다.</p></div>
    <div><b>학습 후</b>
      <code>Qwen3.6-35B-A3B + LoRA</code>
      <p>exp_003 최종 어댑터(step 1124). 출발점과의 차이가 곧 학습의 효과다.</p></div>
  </div>
  <div class="figs">{{FIGS}}</div>
</header>

<div class="finding">
  <h2>핵심 발견 — 나빠진 건의 대부분은 구두점과 띄어쓰기다</h2>
  <p>출발점에서는 맞히던 것을 학습 후 틀리는 건이 122건 있다. 글자를 잘못 읽은 것이
     아니라 콜론과 공백을 생략한다. 출력이 <b>더 짧아지는</b> 방향의 실수다.</p>
  <pre>정답     전표 번호 : 6075        학습 후   전표 번호 6075     콜론 누락
정답     조제 의약품 :           학습 후   조제 의약품        콜론 누락
정답     가족사랑의 날           학습 후   가족사랑의날       공백 누락</pre>
  <p>학습 데이터의 정답 표기 규약이 평가셋과 다르기 때문으로 보인다. 영수증 페이지
     마크다운은 콜론과 구분자를 정규화해 적는데, 평가셋 정답은 인쇄된 그대로를
     옮긴다. 모델은 학습 데이터 쪽 규약을 배웠고 평가에서 감점됐다.</p>
  <p>즉 <b>데이터 정제만으로 되돌릴 수 있는 손실</b>이다. 모델 능력의 문제가 아니므로
     다음 학습에서 가장 먼저 손볼 곳이다.</p>
</div>

<nav class="toc">{{NAV}}</nav>
{{SECTIONS}}

<section class="next">
  <h2>다음 학습에서 할 것</h2>
  <ol>
    <li><b>학습 데이터의 구두점·공백 표기를 평가 정답 규약에 맞춘다</b>
      <p>나빠진 122건의 대부분이 여기서 나왔다. 영수증 페이지 마크다운의 정답을
         인쇄면 그대로로 되돌리면 모델을 건드리지 않고 CER이 더 내려간다.</p></li>
    <li><b>평가셋 크롭 패딩을 다시 잡는다</b>
      <p>긴 오답 중 상당수는 크롭에 이웃 줄이 딸려 들어가 생긴 것이다
         (<code>pad_ratio=0.02</code>). 지금은 모델이 "크롭 하나 = 한 덩어리"를
         학습해 점수가 오른 것처럼 보이지만, 평가가 실제 능력을 재고 있지 않다.
         패딩을 줄여 다시 뜨면 기준선 재측정이 필요하다.</p></li>
    <li><b>베이스는 맞고 우리는 틀린 94건을 유형별로 나눈다</b>
      <p>서비스 대체를 논하려면 이 격차를 줄여야 한다. 반대 방향(우리는 맞고
         베이스가 틀린)이 97건으로 거의 같아, 두 모델이 서로 다른 것을 놓치고 있다.</p></li>
    <li><b>표는 출발점 대비로 확실히 올랐다</b>
      <p>TEDS 0.6603 → 0.7229. 좋아진 33건 대 나빠진 20건이다. 베이스 대비로는
         +0.0198로 작아 보이지만 그건 서비스 설정이 이미 표를 끌어올려 둔 탓이다.
         표 크롭 5,593건을 넣는 exp_004에서 이 흐름이 이어지는지 본다.</p></li>
  </ol>
</section>

<footer>
  평가셋 영수증 크롭 1,036건(text 933 / table 103), transpose 2장 제외 ·
  근거 <code>eval_crops_luxia.json</code> / <code>eval_crops_base.json</code> /
  <code>eval_crops_exp003.json</code> ·
  붉은 글자는 정답과 어긋난 부분, <i class="gap">‸</i>는 빠뜨린 자리(마우스를 올리면 내용이 보인다)
</footer>
</div>
"""

if __name__ == "__main__":
    main()
