"""평가셋 정제 기록 — 고친 정답과 뺀 문서만 보여준다.

평가셋 정답이 사람 검수본이 아니라 Document Studio 파서 출력 그대로였던 것이
확인돼(status:200 · documentId 가 남아 있고, 검수본 수집 스크립트는 train 만
대상으로 한다), 8/13 에 오독을 고치고 문서 8장을 뺐다. 남은 90장은 정상이므로
싣지 않고, 무엇을 왜 바꿨는지만 이미지와 함께 남긴다.

입력: scratchpad/fixes.json (정정 크롭) · scratchpad/removed8.json (제외 문서)
"""
import html
import json
import sys
from pathlib import Path

FIXES_JSON, REMOVED_JSON, OUT = (Path(sys.argv[1]), Path(sys.argv[2]),
                                 Path(sys.argv[3]))

DROP_WHY = {
    "receipt34": "좌표계가 90도 뒤집힘 — 이미지는 가로인데 json 은 1470×1960 세로",
    "receipt67": "같은 transpose 문제",
    "receipt73": "손글씨가 섞여 있어 전사 정답을 신뢰할 수 없음",
    "receipt53": "정답 누락 다수 · 원본이 한 번 분실됐다 복구된 건",
    "receipt36": "검수 판정으로 제외",
    "receipt81": "검수 판정으로 제외 (spc → SPC 정정 이후)",
    "receipt91": "검수 판정으로 제외",
    "receipt38": "검수 판정으로 제외",
}
BASE = [("베이스 (LUXIA 서빙)", 0.2276, 0.7031, 0.2009, 0.7032),
        ("출발점 (Qwen 맨몸)", 0.1963, 0.6603, 0.1689, 0.6628),
        ("학습 후 (exp_003)", 0.1423, 0.7229, 0.1210, 0.7286)]


def clean(s):
    """파서가 남긴 대체문자·제어문자를 표시용 기호로. 그대로 두면 배포가 거부된다."""
    return "".join("□" if c == "�" or (ord(c) < 32 and c not in "\n\t") else c
                   for c in s)


def mark(ctx, needle, cls):
    """문맥 안에서 문제가 된 조각만 강조한다."""
    ctx, needle = clean(ctx), clean(needle)
    i = ctx.find(needle)
    if i < 0:
        return html.escape(ctx)
    return (html.escape(ctx[:i]) + f'<mark class="{cls}">'
            + html.escape(needle) + "</mark>" + html.escape(ctx[i+len(needle):]))


def fix_card(f):
    tag = ("" if f["applied"] else
           '<span class="pend">미적용</span>')
    after = (f["ctx"].replace(f["old"], f["new"]) if f["applied"] else f["ctx"])
    after_html = (mark(after, f["new"], "ins") if f["applied"]
                  else '<span class="muted">— 아직 반영하지 않음</span>')
    return f"""
<article class="card">
  <header><span class="cid">{html.escape(f['cid'])}</span>
    <span class="why">{html.escape(f['why'])}{tag}</span></header>
  <div class="shot"><img src="data:image/png;base64,{f['img']}"
    alt="{html.escape(f['cid'])} 크롭" width="{f['w']}" height="{f['h']}"
    loading="lazy"></div>
  <dl>
    <dt>수정 전</dt><dd>{mark(f['ctx'], f['old'], 'del')}</dd>
    <dt>수정 후</dt><dd>{after_html}</dd>
  </dl>
</article>"""


def drop_card(d):
    tx = sum(1 for i in d["items"] if i["task"] == "text")
    rot = ' · 회전 보정 후 표시' if d.get("transposed") else ""
    lis = "".join(
        f'<li class="{"tb" if i["task"]=="table" else "tx"}">'
        f'<span class="num">{i["n"]}</span><div>'
        + (f'<details><summary>표 정답 {len(clean(i["gt"])):,}자</summary>'
           f'<pre>{html.escape(clean(i["gt"]))}</pre></details>'
           if i["task"] == "table"
           else f'<p class="gt">{html.escape(clean(i["gt"]))}</p>')
        + "</div></li>" for i in d["items"])
    return f"""
<section class="doc" id="{html.escape(d['doc'])}">
  <header><h3>{html.escape(d['doc'])}</h3>
    <span class="meta">{d['orig']} · 크롭 {len(d['items'])}
      (텍스트 {tx} / 표 {len(d['items'])-tx}){rot}</span></header>
  <p class="reason">{html.escape(DROP_WHY.get(d['doc'], ''))}</p>
  <div class="pair">
    <div class="page"><img src="data:image/jpeg;base64,{d['img']}"
      alt="{html.escape(d['doc'])} 크롭 영역" width="{d['w']}" height="{d['h']}"
      loading="lazy"></div>
    <ol class="items">{lis}</ol>
  </div>
</section>"""


def main():
    fixes = json.loads(FIXES_JSON.read_text(encoding="utf-8"))
    drops = json.loads(REMOVED_JSON.read_text(encoding="utf-8"))
    n_applied = sum(1 for f in fixes if f["applied"])
    n_crop = sum(len(d["items"]) for d in drops)
    base_rows = "".join(
        f'<tr><td>{html.escape(n)}</td><td class="n">{c0:.4f}</td>'
        f'<td class="n">{t0:.4f}</td><td class="n s">{c1:.4f}</td>'
        f'<td class="n s">{t1:.4f}</td></tr>' for n, c0, t0, c1, t1 in BASE)
    nav = "".join(f'<a href="#{html.escape(d["doc"])}">{html.escape(d["doc"])}'
                  f'<span>{len(d["items"])}</span></a>' for d in drops)
    OUT.write_text(
        TPL.replace("{{FIXCARDS}}", "".join(fix_card(f) for f in fixes))
           .replace("{{DROPCARDS}}", "".join(drop_card(d) for d in drops))
           .replace("{{BASE}}", base_rows).replace("{{NAV}}", nav)
           .replace("{{NFIX}}", str(n_applied))
           .replace("{{NPEND}}", str(len(fixes) - n_applied))
           .replace("{{NDROP}}", str(len(drops)))
           .replace("{{NDROPCROP}}", str(n_crop)),
        encoding="utf-8")
    print(f"저장: {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


TPL = r"""<title>평가 데이터셋 수정</title>
<style>
:root{--paper:#f6f6f3;--card:#fffffe;--ink:#191b1e;--dim:#63686e;--rule:#dfdfd9;
  --rail:#eceae5;--shot:#e9e8e2;--tx:#1baf7a;--tb:#eb6834;--del:#9d3a27;
  --delbg:#f7e8e4;--ins:#1f6b4a;--insbg:#e6f0ea}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#131519;--card:#1a1d22;--ink:#e4e5e2;--dim:#959aa1;--rule:#2c3038;
  --rail:#22262c;--shot:#0d0f12;--tx:#199e70;--tb:#d95926;--del:#e08a72;
  --delbg:#2e1a15;--ins:#6cc296;--insbg:#152a21}}
:root[data-theme="dark"]{--paper:#131519;--card:#1a1d22;--ink:#e4e5e2;--dim:#959aa1;
  --rule:#2c3038;--rail:#22262c;--shot:#0d0f12;--tx:#199e70;--tb:#d95926;
  --del:#e08a72;--delbg:#2e1a15;--ins:#6cc296;--insbg:#152a21}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-size:16px;line-height:1.6;
  font-family:"Pretendard","Apple SD Gothic Neo","Malgun Gothic",system-ui,
  -apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px 96px}
header.top{padding:56px 0 28px;border-bottom:2px solid var(--ink)}
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim);
  font-weight:700;margin:0 0 12px}
h1{font-size:clamp(28px,4vw,42px);line-height:1.15;letter-spacing:-.02em;
  font-weight:800;margin:0 0 14px;text-wrap:balance}
.lede{font-size:17px;color:var(--dim);margin:0;max-width:68ch}
.flow{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline;margin:26px 0 0;
  font-variant-numeric:tabular-nums}
.flow b{font-size:26px;font-weight:800;letter-spacing:-.01em}
.flow span{font-size:13px;color:var(--dim)}
.flow i{font-style:normal;color:var(--dim);font-size:19px;margin:0 2px}
h2{font-size:22px;font-weight:800;margin:56px 0 6px;letter-spacing:-.015em;
  border-bottom:1px solid var(--ink);padding-bottom:7px}
h2+p{font-size:14.5px;color:var(--dim);margin:12px 0 18px;max-width:72ch}
.why-box{border-left:3px solid var(--del);background:var(--rail);padding:18px 22px;
  margin:26px 0 0}
.why-box h3{font-size:15px;font-weight:800;margin:0 0 8px}
.why-box p{margin:0 0 8px;font-size:14.5px;max-width:74ch}
.why-box p:last-child{margin:0}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em}

.card{border:1px solid var(--rule);background:var(--card);margin:0 0 16px}
.card>header{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;
  padding:9px 14px;border-bottom:1px solid var(--rule);align-items:baseline}
.cid{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
  color:var(--dim)}
.why{font-size:12.5px;color:var(--dim)}
.pend{background:var(--delbg);color:var(--del);font-size:11px;font-weight:700;
  padding:1px 6px;margin-left:8px}
.shot{background:var(--shot);padding:12px 14px;overflow-x:auto;
  border-bottom:1px solid var(--rule)}
.shot img{display:block;max-width:100%;height:auto}
.card dl{display:grid;grid-template-columns:74px 1fr;margin:0}
.card dt{padding:9px 0 9px 14px;font-size:12px;color:var(--dim);
  border-top:1px solid var(--rule)}
.card dd{margin:0;padding:9px 14px 9px 10px;border-top:1px solid var(--rule);
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Apple SD Gothic Neo",
  monospace;font-size:13.5px;white-space:pre-wrap;word-break:break-all}
.card dt:first-of-type,.card dd:nth-of-type(1){border-top:none}
mark.del{background:var(--delbg);color:var(--del);font-weight:700}
mark.ins{background:var(--insbg);color:var(--ins);font-weight:700}
.muted{color:var(--dim)}

table{width:100%;border-collapse:collapse;font-size:14px;background:var(--card);
  border:1px solid var(--rule)}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--rule)}
th{font-size:12px;color:var(--dim);font-weight:700;background:var(--rail);
  white-space:nowrap}
tr:last-child td{border-bottom:none}
td.n{text-align:right;font-variant-numeric:tabular-nums}
td.n.s{font-weight:700}
.wrapt{overflow-x:auto}

nav.jump{display:flex;flex-wrap:wrap;gap:6px;margin:16px 0 0}
nav.jump a{display:inline-flex;gap:5px;align-items:baseline;background:var(--card);
  border:1px solid var(--rule);padding:4px 10px;font-size:12.5px;color:var(--ink);
  text-decoration:none;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
nav.jump a span{color:var(--dim);font-size:11px}
nav.jump a:hover{border-color:var(--ink)}
nav.jump a:focus-visible{outline:2px solid var(--tx);outline-offset:1px}

.doc{margin:40px 0 0;scroll-margin-top:12px}
.doc>header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;
  border-bottom:1px solid var(--ink);padding-bottom:6px}
.doc h3{font-size:19px;font-weight:800;margin:0;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.meta{font-size:12.5px;color:var(--dim)}
.reason{font-size:14px;color:var(--del);margin:10px 0 0;font-weight:600}
.pair{display:grid;grid-template-columns:minmax(240px,360px) 1fr;gap:20px;
  margin-top:14px;align-items:start}
.page{background:var(--shot);border:1px solid var(--rule);padding:10px;
  position:sticky;top:12px;max-height:86vh;overflow-y:auto}
.page img{display:block;width:100%;height:auto}
ol.items{list-style:none;margin:0;padding:0;border:1px solid var(--rule);
  background:var(--card)}
ol.items li{display:grid;grid-template-columns:32px 1fr;gap:10px;padding:8px 12px;
  border-bottom:1px solid var(--rule);align-items:start}
ol.items li:last-child{border-bottom:none}
.num{display:inline-flex;align-items:center;justify-content:center;width:23px;
  height:23px;border-radius:50%;color:#fff;font-size:12px;font-weight:700;
  font-variant-numeric:tabular-nums}
li.tx .num{background:var(--tx)}li.tb .num{background:var(--tb)}
p.gt{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,
  "Apple SD Gothic Neo",monospace;font-size:13.5px;line-height:1.55;
  white-space:pre-wrap;word-break:break-word}
details summary{cursor:pointer;font-size:13px;color:var(--dim);padding:2px 0}
details summary:focus-visible{outline:2px solid var(--tb);outline-offset:2px}
details pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;
  white-space:pre-wrap;word-break:break-all;background:var(--rail);
  border:1px solid var(--rule);padding:10px;margin:8px 0 0;max-height:300px;
  overflow:auto}
footer{margin:64px 0 0;padding-top:20px;border-top:1px solid var(--rule);
  font-size:13px;color:var(--dim)}
@media (max-width:800px){.pair{grid-template-columns:1fr}
  .page{position:static;max-height:none}
  .card dl{grid-template-columns:1fr}
  .card dt{padding:9px 14px 0}.card dd{padding-left:14px;border-top:none}}
</style>

<div class="wrap">
<header class="top">
  <p class="eyebrow">2026-08-13 · 영수증 크롭 평가셋</p>
  <h1>평가 데이터셋 수정</h1>
  <p class="lede">평가셋의 정답이 사람 검수본이 아니라 파서 출력 그대로라는 것이
    확인돼 오독을 고치고 문서를 덜어냈다. 무엇을 왜 바꿨는지만 모았다 —
    남은 90장은 정상이라 싣지 않았다.</p>
  <p class="flow"><b>98</b><span>장</span><i>−</i><b>{{NDROP}}</b><span>장 제외</span>
    <i>=</i><b>90</b><span>장 · 크롭 978건</span>
    <i>·</i><b>{{NFIX}}</b><span>곳 정답 정정</span></p>
</header>

<div class="why-box">
  <h3>왜 정답을 의심하게 됐나</h3>
  <p>크롭 정답은 라벨링 파일의 <code>parsing_res_list[].block_content</code>에서
     온다. 그런데 그 파일이 <code>status: 200</code>, <code>documentId</code>를 가진
     <b>Document Studio API 응답 그대로</b>였다. 검수본을 골라내는
     <code>collect_labeled.py</code>는 <code>train</code>만 대상으로 하고,
     val/test 파일의 수정 시각은 검수 시작(8/10 03:00)보다 앞선다.
     <b>학습 데이터는 검수본, 평가 데이터는 파서 출력</b>이라는 비대칭이 있었다.</p>
  <p>결정적인 증거는 <code>블루데몬에이드</code>가 markdown·html·md·
     <code>parsing_res_list</code> 다섯 필드에 똑같이 복제돼 있던 것이다. 사람이 한
     곳이라도 고쳤다면 필드끼리 달라졌을 텐데 전부 같았다.</p>
  <p>오독은 <b>세 모델(LUXIA · Qwen 맨몸 · 학습 후)이 서로 일치하는데 정답만 다른</b>
     크롭을 골라 찾았다. 독립적인 세 모델이 같게 읽었다면 정답 쪽을 의심하는 것이
     맞다. 그렇게 72건을 추려 사람이 판정했다.</p>
</div>

<h2>정답 수정 {{NFIX}}곳</h2>
<p>수정은 <code>scripts/fix_gt.py</code>로 했다. 문서를 지정해 정답 필드만 바꾸고
   모델 예측(<code>pred</code>)은 건드리지 않는다 — 전역 치환을 쓰면 같은 문자열이
   들어간 다른 영수증과 과거 평가 스냅샷까지 오염된다(한 번 40개 파일로 번져
   되돌린 적이 있다). 아래 {{NPEND}}건은 판단이 남아 아직 반영하지 않았다.</p>
{{FIXCARDS}}

<h2>문서 제외 {{NDROP}}장 · 크롭 {{NDROPCROP}}건</h2>
<p>지우지 않고 <code>receipt_data/_removed/&lt;문서&gt;/</code>로 옮겼다.
   원본·라벨·크롭이 그대로 있어 되돌릴 수 있다. 이미지 위 번호와 오른쪽 목록의
   번호가 같은 크롭을 가리킨다.</p>
<nav class="jump">{{NAV}}</nav>
{{DROPCARDS}}

<h2>기준선이 어떻게 바뀌었나</h2>
<p>정제 전 1,036건과 정제 후 978건을 같은 어댑터로 비교한 값이다. 세 모델이 모두
   같은 방향으로 좋아졌다 — 정답이 부실해 부당하게 감점되던 부분이 사라진 결과이고,
   모델 사이의 우열은 바뀌지 않는다.</p>
<div class="wrapt"><table>
<thead><tr><th>모델</th><th>CER 1,036</th><th>TEDS 1,036</th>
  <th>CER 978</th><th>TEDS 978</th></tr></thead>
<tbody>{{BASE}}</tbody></table></div>

<footer>
  평가셋 <code>receipt_data/exp004b_260813/val.jsonl</code> ·
  크롭 정의 <code>receipt_data/eval_crops/manifest.jsonl</code> ·
  정답 출처 <code>receipt_data/labeled/{split}/json/*.json</code> ·
  제외 문서 <code>receipt_data/_removed/</code>
</footer>
</div>
"""

if __name__ == "__main__":
    main()
