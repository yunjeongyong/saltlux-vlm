"""평가셋 96장의 정답 라벨 검수 시트.

크롭 정답(parsing_res_list[].block_content)이 사람 검수본이 아니라 파서 출력
그대로일 가능성이 확인돼, 눈으로 대조할 수 있게 원본 이미지와 정답 텍스트를
나란히 놓는다. 이미지 위 번호와 오른쪽 목록의 번호가 같은 크롭을 가리킨다.

입력: scratchpad/audit96.json (이미지에 번호 오버레이까지 그려 base64 로 박아둔 것)
"""
import html
import json
import sys
from pathlib import Path

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def clean(s):
    """모델·파서가 남긴 대체문자와 제어문자를 표시용 기호로 바꾼다.
    그대로 두면 배포가 인코딩 오류로 거부된다."""
    return "".join("□" if c == "�" or (ord(c) < 32 and c not in "\n\t")
                   else c for c in s)


def item_html(it):
    gt = clean(it["gt"])
    cls = "tb" if it["task"] == "table" else "tx"
    # 표 정답은 HTML 태그가 길어 접어 둔다. 텍스트는 한 줄이라 그대로 보인다.
    if it["task"] == "table":
        body = (f'<details><summary>표 정답 {len(gt):,}자 — 펼치기</summary>'
                f'<pre>{html.escape(gt)}</pre></details>')
    else:
        body = f'<p class="gt">{html.escape(gt)}</p>'
    return (f'<li class="{cls}"><span class="num">{it["n"]}</span>'
            f'<div class="body">{body}</div></li>')


def doc_html(d):
    n_tx = sum(1 for i in d["items"] if i["task"] == "text")
    n_tb = len(d["items"]) - n_tx
    return f"""
<section class="doc" id="{html.escape(d['doc'])}">
  <header>
    <h2>{html.escape(d['doc'])}</h2>
    <span class="meta">{html.escape(d['split'])} · 원본 {d['orig']} ·
      크롭 {len(d['items'])}건 (텍스트 {n_tx} / 표 {n_tb})</span>
  </header>
  <div class="pair">
    <div class="page"><img src="data:image/jpeg;base64,{d['img']}"
      alt="{html.escape(d['doc'])} 원본에 크롭 영역 표시"
      width="{d['w']}" height="{d['h']}" loading="lazy"></div>
    <ol class="items">{''.join(item_html(i) for i in d['items'])}</ol>
  </div>
</section>"""


def main():
    docs = json.loads(SRC.read_text(encoding="utf-8"))
    n_crop = sum(len(d["items"]) for d in docs)
    n_tb = sum(1 for d in docs for i in d["items"] if i["task"] == "table")
    nav = "".join(f'<a href="#{html.escape(d["doc"])}">{html.escape(d["doc"])}'
                  f'<span>{len(d["items"])}</span></a>' for d in docs)
    OUT.write_text(
        TPL.replace("{{NAV}}", nav)
           .replace("{{DOCS}}", "".join(doc_html(d) for d in docs))
           .replace("{{NDOC}}", str(len(docs)))
           .replace("{{NCROP}}", f"{n_crop:,}")
           .replace("{{NTB}}", str(n_tb))
           .replace("{{NTX}}", f"{n_crop - n_tb:,}"),
        encoding="utf-8")
    print(f"저장: {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


TPL = r"""<title>평가셋 정답 검수 시트</title>
<style>
:root{
  --paper:#f6f6f3; --card:#fffffe; --ink:#191b1e; --dim:#63686e;
  --rule:#dfdfd9; --rail:#eceae5; --shot:#e9e8e2;
  --tx:#1baf7a; --tb:#eb6834; --warn:#9d3a27;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#131519; --card:#1a1d22; --ink:#e4e5e2; --dim:#959aa1;
  --rule:#2c3038; --rail:#22262c; --shot:#0d0f12;
  --tx:#199e70; --tb:#d95926; --warn:#e08a72;
}}
:root[data-theme="dark"]{
  --paper:#131519; --card:#1a1d22; --ink:#e4e5e2; --dim:#959aa1;
  --rule:#2c3038; --rail:#22262c; --shot:#0d0f12;
  --tx:#199e70; --tb:#d95926; --warn:#e08a72;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"Pretendard","Apple SD Gothic Neo","Malgun Gothic",system-ui,
    -apple-system,"Segoe UI",sans-serif;font-size:16px;line-height:1.6}
.wrap{max-width:1240px;margin:0 auto;padding:0 24px 96px}
header.top{padding:56px 0 30px;border-bottom:2px solid var(--ink)}
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--dim);font-weight:700;margin:0 0 12px}
h1{font-size:clamp(28px,4vw,42px);line-height:1.15;letter-spacing:-.02em;
  font-weight:800;margin:0 0 14px;text-wrap:balance}
.lede{font-size:17px;color:var(--dim);margin:0;max-width:66ch}
.stats{display:flex;flex-wrap:wrap;gap:26px;margin:24px 0 0}
.stats div{font-size:13px;color:var(--dim)}
.stats b{display:block;font-size:24px;font-weight:800;color:var(--ink);
  font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.key{display:flex;flex-wrap:wrap;gap:18px;margin:20px 0 0;font-size:13px;
  color:var(--dim);align-items:center}
.sw{width:12px;height:12px;border-radius:3px;display:inline-block;
  margin-right:6px;vertical-align:-1px}
.sw.tx{background:var(--tx)}.sw.tb{background:var(--tb)}

.warn{border-left:3px solid var(--warn);background:var(--rail);
  padding:18px 22px;margin:30px 0 0}
.warn h2{font-size:16px;font-weight:800;margin:0 0 8px}
.warn p{margin:0 0 8px;font-size:14.5px;max-width:74ch}
.warn p:last-child{margin:0}

nav.jump{display:flex;flex-wrap:wrap;gap:5px;margin:30px 0 0;
  max-height:132px;overflow-y:auto;padding:4px;background:var(--rail);
  border:1px solid var(--rule)}
nav.jump a{display:inline-flex;gap:5px;align-items:baseline;
  background:var(--card);border:1px solid var(--rule);padding:3px 8px;
  font-size:12px;color:var(--ink);text-decoration:none;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
nav.jump a span{color:var(--dim);font-size:11px}
nav.jump a:hover{border-color:var(--ink)}
nav.jump a:focus-visible{outline:2px solid var(--tx);outline-offset:1px}

.doc{margin:52px 0 0;scroll-margin-top:12px}
.doc>header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;
  border-bottom:1px solid var(--ink);padding-bottom:7px}
.doc h2{font-size:20px;font-weight:800;margin:0;letter-spacing:-.01em;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.meta{font-size:12.5px;color:var(--dim)}
.pair{display:grid;grid-template-columns:minmax(280px,420px) 1fr;gap:22px;
  margin-top:16px;align-items:start}
.page{background:var(--shot);border:1px solid var(--rule);padding:10px;
  position:sticky;top:12px;max-height:88vh;overflow-y:auto}
.page img{display:block;width:100%;height:auto}

ol.items{list-style:none;margin:0;padding:0;border:1px solid var(--rule);
  background:var(--card)}
ol.items li{display:grid;grid-template-columns:34px 1fr;gap:10px;
  padding:9px 12px;border-bottom:1px solid var(--rule);align-items:start}
ol.items li:last-child{border-bottom:none}
.num{display:inline-flex;align-items:center;justify-content:center;
  width:24px;height:24px;border-radius:50%;color:#fff;font-size:12px;
  font-weight:700;font-variant-numeric:tabular-nums}
li.tx .num{background:var(--tx)}
li.tb .num{background:var(--tb)}
.body{min-width:0}
p.gt{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,
  "Apple SD Gothic Neo",monospace;font-size:13.5px;line-height:1.55;
  white-space:pre-wrap;word-break:break-word}
details summary{cursor:pointer;font-size:13px;color:var(--dim);
  padding:2px 0;user-select:none}
details summary:focus-visible{outline:2px solid var(--tb);outline-offset:2px}
details pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:12px;line-height:1.5;white-space:pre-wrap;word-break:break-all;
  background:var(--rail);border:1px solid var(--rule);padding:10px;
  margin:8px 0 0;max-height:340px;overflow:auto}
footer{margin:64px 0 0;padding-top:20px;border-top:1px solid var(--rule);
  font-size:13px;color:var(--dim)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em}
@media (max-width:860px){
  .pair{grid-template-columns:1fr}
  .page{position:static;max-height:none}
}
</style>

<div class="wrap">
<header class="top">
  <p class="eyebrow">평가셋 · val {{NDOC}}장 · 크롭 {{NCROP}}건</p>
  <h1>정답이 실제로 맞는지 눈으로 대조하기</h1>
  <p class="lede">이미지 위의 번호와 오른쪽 목록의 번호가 같은 크롭을 가리킨다.
    사각형이 잘라낸 영역이고, 목록의 글자가 그 영역의 정답으로 쓰이고 있는 값이다.
    둘이 어긋나면 모델이 아니라 평가가 틀린 것이다.</p>
  <div class="stats">
    <div><b>{{NDOC}}</b>영수증</div>
    <div><b>{{NCROP}}</b>크롭</div>
    <div><b>{{NTX}}</b>텍스트 (CER 채점)</div>
    <div><b>{{NTB}}</b>표 (TEDS 채점)</div>
  </div>
  <p class="key"><span><span class="sw tx"></span>텍스트 크롭</span>
    <span><span class="sw tb"></span>표 크롭</span>
    <span>번호 원은 크롭의 왼쪽 위 모서리에 찍혀 있다</span></p>
</header>

<div class="warn">
  <h2>이 시트를 만든 이유</h2>
  <p>크롭 정답은 라벨링 파일의 <code>parsing_res_list[].block_content</code>에서
     가져온다. 그런데 그 파일이 <code>status: 200</code>, <code>documentId</code>를
     가진 <b>Document Studio API 응답 그대로</b>이고, 검수본을 골라내는
     <code>collect_labeled.py</code>는 <code>train</code>만 대상으로 한다.
     val/test 파일의 수정 시각도 검수 시작(8/10 03:00)보다 앞선다.</p>
  <p>즉 지금 지표는 "정답을 얼마나 맞혔나"가 아니라 <b>"파서와 얼마나 같은가"</b>일
     수 있다. <code>블루데몬에이드</code>(→ 블루레몬에이드) 같은 오독이 정답에
     남아 있던 것이 그 사례다.</p>
  <p>세 가지를 보면서 확인해 주시면 된다 — 사각형이 글자를 제대로 감쌌는가,
     이웃 줄이 딸려 들어오지 않았는가, 그리고 목록의 글자가 이미지와 일치하는가.</p>
</div>

<nav class="jump">{{NAV}}</nav>
{{DOCS}}

<footer>
  이미지는 폭 660px로 축소했고 좌표는 원본 기준으로 환산해 표시했다.
  좌우 비율이 뒤집힌 장(receipt34·67)은 회전 보정 후 그렸다 ·
  정답 출처 <code>labeled/{split}/json/*.json</code> ·
  크롭 정의 <code>eval_crops/manifest.jsonl</code> ·
  평가셋 <code>exp004b_260813/val.jsonl</code>
</footer>
</div>
"""

if __name__ == "__main__":
    main()
