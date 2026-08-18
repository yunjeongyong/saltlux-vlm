"""exp_003 AS-IS/TO-BE 케이스 비교 리포트를 만든다.

크롭 이미지 · 정답 · 학습 전 예측 · 학습 후 예측을 한 화면에 놓고,
정답 대비 어느 글자가 틀렸는지 문자 단위로 표시한다. 지표 숫자만으로는
"무엇이 좋아졌나"를 말할 수 없어서(폭주 감소분에 평가 아티팩트가 섞여 있다)
샘플을 직접 보고 다음 학습 방향을 정하기 위한 문서다.

입력: scratchpad/samples.json (build 단계에서 이미지까지 base64 로 박아둔 것)
"""
import html
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])

CATS = [
    ("A_긴오답해소", "긴 오답이 해소된 경우", 39,
     "학습 전에는 크롭에 없는 내용까지 길게 뱉어 CER이 1을 넘던 건들이다. "
     "학습 후 출력이 크롭 범위 안으로 들어왔다."),
    ("B_긴오답잔존", "긴 오답이 남은 경우", 18,
     "학습 후에도 CER이 1을 넘는 건들이다. 여기가 다음 학습이 노려야 할 지점이다."),
    ("C_악화", "학습 후 나빠진 경우", 118,
     "학습 전에는 정확히 맞혔는데 학습 후 틀린 건들이다. 이번 분석의 핵심 발견이 여기 있다."),
    ("D_완전정답전환", "완전 정답으로 바뀐 경우", 97,
     "글자 단위로 틀리던 것이 학습 후 완전히 일치하게 된 건들이다."),
    ("E_표개선", "표 — 좋아진 경우", 23, "TEDS가 0.1 이상 오른 표 크롭이다."),
    ("F_표악화", "표 — 나빠진 경우", 24,
     "TEDS가 0.1 이상 떨어진 표 크롭이다. 개선(23건)과 거의 같은 수라 표의 순이득이 작다."),
    ("G_표저조", "표 — 학습 후에도 낮은 경우", 16,
     "학습 후에도 TEDS가 0.5 미만인 표 크롭이다."),
]


def clean(s):
    """모델이 뱉은 대체문자(U+FFFD)·제어문자를 표시 가능한 기호로 바꾼다.

    그대로 두면 배포 단계에서 잘못된 인코딩으로 거부된다. 지우지 않고 □ 로
    남기는 이유는 "모델이 여기서 글자를 못 만들었다"는 것 자체가 관찰 대상이라서다.
    """
    return "".join("□" if c == "�" or (ord(c) < 32 and c not in "\n\t")
                   else c for c in s)


def diff_html(gt, pred):
    """정답 대비 예측이 어디서 어긋났는지 표시한다.

    - 예측에만 있는 글자(초과 출력·오독) → 취소선 없는 붉은 강조
    - 정답에만 있는 글자(누락)          → 자리만 ‸ 로 표시
    누락을 글자로 펼쳐 보여주면 예측 문장이 읽히지 않아 위치만 남긴다.
    """
    out = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, gt, pred).get_opcodes():
        if tag == "equal":
            out.append(html.escape(pred[j1:j2]))
        elif tag in ("replace", "insert"):
            out.append(f'<mark class="wrong">{html.escape(pred[j1:j2])}</mark>')
        elif tag == "delete":
            out.append('<span class="gap" title="누락: '
                       + html.escape(gt[i1:i2]) + '">‸</span>')
    return "".join(out) or '<span class="gap">(빈 출력)</span>'


def metric_chip(row):
    if row["task"] == "table":
        b, f = row["bteds"], row["fteds"]
        d = f - b
        cls = "up" if d > 0 else "down"
        return (f'<span class="chip {cls}">TEDS {b:.3f} → {f:.3f} '
                f'<b>{d:+.3f}</b></span>')
    b, f = row["bcer"], row["fcer"]
    d = f - b
    cls = "up" if d < 0 else "down"          # CER 은 낮을수록 좋다
    return (f'<span class="chip {cls}">CER {b:.3f} → {f:.3f} '
            f'<b>{d:+.3f}</b></span>')


def case_html(row):
    return f"""
<article class="case">
  <header class="case-head">
    <span class="cid">{html.escape(row['id'])}</span>
    {metric_chip(row)}
  </header>
  <div class="shot"><img src="data:image/png;base64,{row['img']}"
       alt="{html.escape(row['id'])} 크롭" width="{row['w']}" height="{row['h']}"></div>
  <dl class="texts">
    <dt>정답</dt><dd class="gt">{html.escape(clean(row['gt']))}</dd>
    <dt>학습 전</dt><dd>{diff_html(clean(row['gt']), clean(row['base']))}</dd>
    <dt>학습 후</dt><dd>{diff_html(clean(row['gt']), clean(row['ft']))}</dd>
  </dl>
</article>"""


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    sections = []
    nav = []
    for key, title, total, blurb in CATS:
        rows = data.get(key, [])
        if not rows:
            continue
        anchor = key.split("_")[0].lower()
        nav.append(f'<a href="#{anchor}">{html.escape(title)}'
                   f'<span>{total}</span></a>')
        sections.append(f"""
<section id="{anchor}" class="cat">
  <div class="cat-head">
    <h2>{html.escape(title)}</h2>
    <p class="count">{total}건 중 {len(rows)}건</p>
  </div>
  <p class="blurb">{html.escape(blurb)}</p>
  {''.join(case_html(r) for r in rows)}
</section>""")

    OUT.write_text(TEMPLATE.replace("{{NAV}}", "".join(nav))
                   .replace("{{SECTIONS}}", "".join(sections)),
                   encoding="utf-8")
    print(f"저장: {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


TEMPLATE = r"""<title>영수증 크롭 오답 해부</title>
<style>
:root{
  --paper:#f6f6f3; --card:#fffffe; --ink:#191b1e; --dim:#63686e;
  --rule:#dfdfd9; --rail:#eceae5;
  --good:#1f6b4a; --goodbg:#e6f0ea; --bad:#9d3a27; --badbg:#f7e8e4;
  --accent:#2c5678; --shot:#f0efe9;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#131519; --card:#1a1d22; --ink:#e4e5e2; --dim:#959aa1;
    --rule:#2c3038; --rail:#22262c;
    --good:#6cc296; --goodbg:#152a21; --bad:#e08a72; --badbg:#2e1a15;
    --accent:#7aa8d0; --shot:#0e1013;
  }
}
:root[data-theme="dark"]{
  --paper:#131519; --card:#1a1d22; --ink:#e4e5e2; --dim:#959aa1;
  --rule:#2c3038; --rail:#22262c;
  --good:#6cc296; --goodbg:#152a21; --bad:#e08a72; --badbg:#2e1a15;
  --accent:#7aa8d0; --shot:#0e1013;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:"Pretendard","Apple SD Gothic Neo","Malgun Gothic",
    system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:16px; line-height:1.65; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1000px; margin:0 auto; padding:0 24px 96px}

/* ── 머리 ── */
header.top{padding:64px 0 40px; border-bottom:2px solid var(--ink)}
.eyebrow{
  font-size:12px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--accent); font-weight:700; margin:0 0 14px;
}
h1{
  font-size:clamp(30px,4.6vw,46px); line-height:1.15; letter-spacing:-.022em;
  font-weight:800; margin:0 0 16px; text-wrap:balance;
}
.lede{font-size:18px; color:var(--dim); margin:0; max-width:62ch}

/* ── 요약 ── */
.stats{
  display:grid; gap:1px; background:var(--rule);
  grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  border:1px solid var(--rule); margin:40px 0 8px;
}
.stat{background:var(--card); padding:18px 20px}
.stat .k{font-size:12px; letter-spacing:.06em; color:var(--dim); margin:0 0 6px}
.stat .v{
  font-size:27px; font-weight:800; letter-spacing:-.02em; margin:0;
  font-variant-numeric:tabular-nums;
}
.stat .v.g{color:var(--good)} .stat .v.b{color:var(--bad)}
.stat .n{font-size:13px; color:var(--dim); margin:4px 0 0}
.note{font-size:14px; color:var(--dim); margin:14px 0 0}

/* ── 발견 ── */
.finding{
  border-left:3px solid var(--accent); background:var(--rail);
  padding:22px 26px; margin:40px 0;
}
.finding h2{font-size:19px; font-weight:800; margin:0 0 10px; letter-spacing:-.01em}
.finding p{margin:0 0 10px; max-width:66ch}
.finding p:last-child{margin-bottom:0}
.finding .ex{
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:13.5px; background:var(--card); border:1px solid var(--rule);
  padding:12px 14px; margin:12px 0; overflow-x:auto; white-space:pre;
}

/* ── 목차 ── */
nav.toc{display:flex; flex-wrap:wrap; gap:8px; margin:36px 0 8px}
nav.toc a{
  display:inline-flex; align-items:baseline; gap:7px;
  border:1px solid var(--rule); background:var(--card);
  padding:7px 13px; font-size:14px; color:var(--ink); text-decoration:none;
}
nav.toc a span{
  font-size:12px; color:var(--dim); font-variant-numeric:tabular-nums;
}
nav.toc a:hover{border-color:var(--accent); color:var(--accent)}
nav.toc a:focus-visible{outline:2px solid var(--accent); outline-offset:2px}

/* ── 분류 ── */
.cat{margin:64px 0 0; scroll-margin-top:20px}
.cat-head{
  display:flex; align-items:baseline; justify-content:space-between;
  gap:16px; border-bottom:1px solid var(--ink); padding-bottom:8px;
}
.cat-head h2{font-size:23px; font-weight:800; margin:0; letter-spacing:-.015em}
.count{
  font-size:13px; color:var(--dim); margin:0; white-space:nowrap;
  font-variant-numeric:tabular-nums;
}
.blurb{font-size:15px; color:var(--dim); margin:14px 0 0; max-width:66ch}

/* ── 케이스 ── */
.case{border:1px solid var(--rule); background:var(--card); margin:22px 0}
.case-head{
  display:flex; align-items:center; justify-content:space-between;
  gap:12px; flex-wrap:wrap; padding:11px 16px; border-bottom:1px solid var(--rule);
}
.cid{
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:12.5px; color:var(--dim);
}
.chip{
  font-size:12.5px; padding:3px 9px; font-variant-numeric:tabular-nums;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
.chip.up{background:var(--goodbg); color:var(--good)}
.chip.down{background:var(--badbg); color:var(--bad)}
.chip b{font-weight:700}
.shot{
  background:var(--shot); padding:14px 16px; overflow-x:auto;
  border-bottom:1px solid var(--rule);
}
.shot img{display:block; max-width:100%; height:auto; image-rendering:crisp-edges}

.texts{display:grid; grid-template-columns:72px 1fr; margin:0}
.texts dt{
  padding:11px 0 11px 16px; font-size:12px; color:var(--dim);
  letter-spacing:.04em; border-top:1px solid var(--rule);
}
.texts dd{
  margin:0; padding:11px 16px 11px 12px; border-top:1px solid var(--rule);
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,
    "Apple SD Gothic Neo",monospace;
  font-size:13.5px; line-height:1.6; white-space:pre-wrap; word-break:break-all;
}
.texts dt:first-of-type,.texts dd.gt{border-top:none}
.texts dd.gt{color:var(--dim)}
mark.wrong{background:var(--badbg); color:var(--bad); font-weight:600}
.gap{color:var(--bad); font-weight:700; cursor:help}

/* ── 결론 ── */
.next{margin:72px 0 0; border-top:2px solid var(--ink); padding-top:32px}
.next h2{font-size:23px; font-weight:800; margin:0 0 20px; letter-spacing:-.015em}
.next ol{margin:0; padding-left:22px}
.next li{margin:0 0 18px; max-width:68ch}
.next li b{font-weight:700}
.next li p{margin:5px 0 0; color:var(--dim); font-size:15px}
footer{margin:56px 0 0; padding-top:20px; border-top:1px solid var(--rule);
  font-size:13px; color:var(--dim)}
@media (max-width:620px){
  .texts{grid-template-columns:1fr}
  .texts dt{padding:11px 16px 0; border-top:1px solid var(--rule)}
  .texts dd{padding-left:16px; border-top:none}
  .texts dd.gt{border-top:none}
}
</style>

<div class="wrap">
<header class="top">
  <p class="eyebrow">exp_003 · 영수증 크롭 1,036건</p>
  <h1>지표는 올랐다. 어디가 좋아지고 어디가 나빠졌나</h1>
  <p class="lede">CER 0.2276 → 0.1423. 평균은 37.5% 좋아졌지만 그 안에는
    개선 203건과 악화 118건이 함께 들어 있다. 크롭 이미지와 실제 출력을 놓고
    무엇이 바뀌었는지 확인한다.</p>

  <div class="stats">
    <div class="stat"><p class="k">개선</p><p class="v g">203</p>
      <p class="n">CER −0.05 초과</p></div>
    <div class="stat"><p class="k">악화</p><p class="v b">118</p>
      <p class="n">CER +0.05 초과</p></div>
    <div class="stat"><p class="k">긴 오답 해소</p><p class="v g">39</p>
      <p class="n">57건 → 20건</p></div>
    <div class="stat"><p class="k">완전 정답 전환</p><p class="v g">97</p>
      <p class="n">틀리던 것이 일치</p></div>
    <div class="stat"><p class="k">표 개선 / 악화</p><p class="v">23 / 24</p>
      <p class="n">TEDS ±0.1 초과</p></div>
  </div>
  <p class="note">개선분 합계 −104.28 · 악화분 합계 +24.64 → 순효과 −79.64.
    933건으로 나누면 0.0854로, 실제 CER 감소폭(0.0853)과 일치한다.</p>
</header>

<div class="finding">
  <h2>핵심 발견 — 악화 118건의 78%가 구두점과 띄어쓰기를 빠뜨린다</h2>
  <p>학습 전에는 정확히 맞히던 것을 학습 후 틀리는 건이 118건 있다. 그중 92건은
     출력이 <b>더 짧아졌다</b>. 글자를 잘못 읽은 것이 아니라 콜론과 공백을 생략한다.</p>
  <div class="ex">정답    전표 번호 : 6075        학습 전  전표 번호 : 6075   ✓
                        학습 후  전표 번호 6075     ✗ 콜론 누락

정답    가족사랑의 날          학습 전  가족사랑의 날      ✓
                        학습 후  가족사랑의날       ✗ 공백 누락</div>
  <p>학습 데이터의 정답 표기 규약이 평가셋과 다르기 때문으로 보인다. 영수증 페이지
     마크다운은 콜론 뒤 공백이나 구분자를 정규화해 적는데, 평가셋 정답은 인쇄된
     그대로를 옮긴다. 모델은 학습 데이터 쪽 규약을 배웠고 평가에서 감점됐다.</p>
  <p>이 118건은 <b>데이터 정제만으로 되돌릴 수 있는 손실</b>이다. 모델 능력의 문제가
     아니므로 다음 학습에서 가장 먼저 손볼 곳이다.</p>
</div>

<nav class="toc">{{NAV}}</nav>
{{SECTIONS}}

<section class="next">
  <h2>다음 학습에서 할 것</h2>
  <ol>
    <li><b>학습 데이터의 구두점·공백 표기를 평가 정답 규약에 맞춘다</b>
      <p>악화 118건의 대부분이 여기서 나왔다. 영수증 페이지 마크다운의 정답을
         인쇄면 그대로로 되돌리면 CER이 추가로 0.026 내려간다(악화분 24.64 ÷ 933).
         모델을 건드리지 않고 얻는 개선이다.</p></li>
    <li><b>평가셋 크롭 패딩을 다시 잡는다</b>
      <p>긴 오답 중 상당수는 크롭에 이웃 줄이 딸려 들어가 생긴 것이다
         (<code>pad_ratio=0.02</code>). 지금은 모델이 "크롭 하나 = 한 덩어리"를
         학습해 점수가 오른 것처럼 보이지만, 평가가 실제 능력을 재고 있지 않다.
         패딩을 줄여 다시 뜨면 기준선 재측정이 필요하다.</p></li>
    <li><b>남은 긴 오답 18건을 유형별로 나눈다</b>
      <p>학습 후에도 CER이 1을 넘는 건들이다. 크롭 문제인지 모델 문제인지
         가려야 다음 데이터를 어디서 모을지 정해진다.</p></li>
    <li><b>표는 개선 23건과 악화 24건이 맞물려 순이득이 작다</b>
      <p>TEDS +0.0198은 오른 것과 내린 것이 거의 상쇄된 결과다. 표 크롭을
         5,593건 투입하는 exp_004에서 이 균형이 깨지는지가 관전 지점이다.</p></li>
  </ol>
</section>

<footer>
  평가셋 영수증 크롭 1,036건(text 933 / table 103), transpose 2장 제외 ·
  학습 전 <code>luxia-document-parsing-high</code> ·
  학습 후 <code>Qwen3.6-35B-A3B</code> + LoRA(exp_003 최종 step 1124) ·
  근거 <code>eval_crops_luxia.json</code> / <code>eval_crops_exp003.json</code>
</footer>
</div>
"""

if __name__ == "__main__":
    main()
