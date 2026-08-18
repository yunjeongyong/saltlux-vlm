"""
dataset_samples.json 을 사람이 넘겨보는 HTML 한 장으로 굽는다.

이미지는 이미 data URI 로 들어 있으므로 이 파일 하나만 열면 된다.
정답은 원문 그대로(이스케이프) 보여주고, 그 안의 표는 따로 렌더링해서
행·열·병합이 실제로 어떻게 잡혔는지 눈으로 확인할 수 있게 한다.

usage:
    python3 scripts/viz_dataset_samples.py     # 먼저 표본 추출
    python3 scripts/viz_dataset_page.py
"""
import argparse
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TASK_META = {
    "receipt_markdown": ("Document Parsing", "parse"),
    "page_ocr": ("OCR", "ocr"),
    "crop_table": ("TSR (평가)", "tsr"),
    "crop_text": ("OCR (평가)", "ocr"),
}

CSS = """
:root{
  --ground:#FAF8F4; --surface:#FFFFFF; --surface-2:#F3F0E9;
  --ink:#1C1A17; --ink-2:#4A453E; --muted:#7A736A; --rule:#E4DFD6;
  --accent:#2E5A78; --accent-soft:#E3ECF2;
  --parse:#8A5A2B; --parse-soft:#F5EADC;
  --ocr:#2F6157; --ocr-soft:#E1EFEA;
  --tsr:#6A4A7C; --tsr-soft:#EEE7F3;
  --warn:#8C4A32;
  --shadow:0 1px 2px rgba(28,26,23,.06), 0 8px 24px -12px rgba(28,26,23,.18);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#17161A; --surface:#1F1E23; --surface-2:#26252B;
    --ink:#EAE6DF; --ink-2:#C3BDB4; --muted:#948D84; --rule:#33313A;
    --accent:#8FB8D4; --accent-soft:#22303B;
    --parse:#D6A470; --parse-soft:#332720;
    --ocr:#7FBFAE; --ocr-soft:#1D2E2A;
    --tsr:#B99BCB; --tsr-soft:#2A2333;
    --warn:#D98F72;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
  }
}
:root[data-theme="dark"]{
  --ground:#17161A; --surface:#1F1E23; --surface-2:#26252B;
  --ink:#EAE6DF; --ink-2:#C3BDB4; --muted:#948D84; --rule:#33313A;
  --accent:#8FB8D4; --accent-soft:#22303B;
  --parse:#D6A470; --parse-soft:#332720;
  --ocr:#7FBFAE; --ocr-soft:#1D2E2A;
  --tsr:#B99BCB; --tsr-soft:#2A2333;
  --warn:#D98F72;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI","Noto Sans KR",sans-serif;
  font-size:15px; line-height:1.6;
  -webkit-font-smoothing:antialiased;
}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}
.wrap{max-width:1180px; margin:0 auto; padding:48px 24px 96px}

header{border-bottom:2px solid var(--ink); padding-bottom:20px; margin-bottom:40px}
.eyebrow{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:11px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--accent); margin:0 0 10px;
}
h1{font-size:clamp(26px,3.4vw,38px); line-height:1.15; margin:0 0 12px; text-wrap:balance; letter-spacing:-.015em}
.sub{color:var(--muted); margin:0; max-width:64ch}

h2{
  font-size:19px; margin:56px 0 4px; letter-spacing:-.01em;
  display:flex; align-items:baseline; gap:12px;
}
h2 .num{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:12px; color:var(--accent); letter-spacing:.08em;
}
h2 + .lede{color:var(--muted); margin:0 0 20px; font-size:14px; max-width:70ch}

.tblwrap{overflow-x:auto; border:1px solid var(--rule); border-radius:6px; background:var(--surface)}
table.spec{border-collapse:collapse; width:100%; font-size:13.5px; min-width:600px}
table.spec th, table.spec td{
  text-align:left; padding:11px 14px; border-bottom:1px solid var(--rule);
  vertical-align:top;
}
table.spec thead th{
  background:var(--surface-2); font-size:11px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted); font-weight:600;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
}
table.spec tbody tr:last-child td{border-bottom:none}
table.spec td.k{font-weight:600; white-space:nowrap; width:1%}
table.spec .num{font-variant-numeric:tabular-nums}

.badge{
  display:inline-block; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:10.5px; letter-spacing:.07em; text-transform:uppercase;
  padding:3px 8px; border-radius:3px; font-weight:600; white-space:nowrap;
}
.b-parse{background:var(--parse-soft); color:var(--parse)}
.b-ocr{background:var(--ocr-soft); color:var(--ocr)}
.b-tsr{background:var(--tsr-soft); color:var(--tsr)}
.b-split{background:var(--accent-soft); color:var(--accent)}

.cards{display:flex; flex-direction:column; gap:20px; margin-top:22px}
.card{
  background:var(--surface); border:1px solid var(--rule); border-radius:8px;
  box-shadow:var(--shadow); overflow:hidden;
}
.card-hd{
  display:flex; flex-wrap:wrap; align-items:center; gap:10px;
  padding:12px 16px; border-bottom:1px solid var(--rule); background:var(--surface-2);
}
.card-hd .id{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:13px; font-weight:600;
}
.card-hd .meta{
  margin-left:auto; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:11.5px; color:var(--muted); font-variant-numeric:tabular-nums;
}
.card-bd{display:grid; grid-template-columns:minmax(0,300px) minmax(0,1fr); gap:0}
.pane-img{
  padding:16px; border-right:1px solid var(--rule);
  display:flex; flex-direction:column; gap:10px; align-items:center;
  background:var(--surface-2);
}
.pane-img img{
  max-width:100%; height:auto; border-radius:4px;
  border:1px solid var(--rule); background:#fff;
}
.pane-img .cap{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:10.5px; color:var(--muted); text-align:center; word-break:break-all;
}
.pane-io{padding:16px; display:flex; flex-direction:column; gap:14px; min-width:0}

.io-label{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:10.5px; letter-spacing:.12em; text-transform:uppercase;
  color:var(--muted); margin:0 0 6px; display:flex; align-items:center; gap:8px;
}
.io-label::after{content:""; flex:1; height:1px; background:var(--rule)}
.prompt{
  background:var(--accent-soft); border-left:3px solid var(--accent);
  padding:10px 12px; border-radius:0 4px 4px 0; font-size:13.5px; color:var(--ink-2);
}
pre.answer{
  margin:0; padding:12px; background:var(--surface-2); border:1px solid var(--rule);
  border-radius:4px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:12px; line-height:1.55; white-space:pre-wrap; word-break:break-word;
  max-height:340px; overflow:auto; color:var(--ink-2);
}
.rendered{border:1px dashed var(--rule); border-radius:4px; padding:12px; overflow-x:auto}
.rendered table{border-collapse:collapse; font-size:12px; min-width:100%}
.rendered th,.rendered td{
  border:1px solid var(--rule); padding:4px 8px; text-align:left;
  font-variant-numeric:tabular-nums;
}
.rendered th{background:var(--surface-2); font-weight:600}

.note{
  border-left:3px solid var(--warn); background:var(--surface); padding:12px 16px;
  border-radius:0 6px 6px 0; font-size:13.5px; color:var(--ink-2); margin-top:18px;
  border-top:1px solid var(--rule); border-right:1px solid var(--rule);
  border-bottom:1px solid var(--rule);
}
.note b{color:var(--warn)}
footer{
  margin-top:72px; padding-top:20px; border-top:1px solid var(--rule);
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:11.5px; color:var(--muted);
}
@media (max-width:820px){
  .card-bd{grid-template-columns:1fr}
  .pane-img{border-right:none; border-bottom:1px solid var(--rule)}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
"""


def esc(s):
    return html.escape(str(s or ""))


def spec_table(headers, rows):
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = ""
    for r in rows:
        tds = "".join(
            f'<td class="{c}">{v}</td>' for c, v in r)
        body += f"<tr>{tds}</tr>"
    return (f'<div class="tblwrap"><table class="spec"><thead><tr>{h}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "receipt_data/review/dataset_samples.json"))
    ap.add_argument("--out", default=str(ROOT / "receipt_data/review/dataset_samples.html"))
    args = ap.parse_args()

    samples = json.loads(Path(args.data).read_text(encoding="utf-8"))

    cards = []
    for s in samples:
        label, kind = TASK_META.get(s["task"], (s["task"], "parse"))
        rendered = ""
        if s["tables"]:
            rendered = ('<div><p class="io-label">표 렌더링 — 행·열·병합 확인</p>'
                        + '<div class="rendered">' + "".join(s["tables"]) + "</div></div>")
        src = f' · {esc(s["source"])}' if s["source"] else ""
        cards.append(f"""
<article class="card">
  <div class="card-hd">
    <span class="id">{esc(s['doc_id'])}</span>
    <span class="badge b-{kind}">{esc(label)}</span>
    <span class="badge b-split">{esc(s['split'])}</span>
    <span class="meta">{s['w']}×{s['h']}px · 정답 {s['chars']:,}자 · 표 {len(s['tables'])}개{src}</span>
  </div>
  <div class="card-bd">
    <div class="pane-img">
      <img src="{s['img']}" alt="{esc(s['doc_id'])} 입력 이미지" loading="lazy">
      <span class="cap">{esc(s['path'])}</span>
    </div>
    <div class="pane-io">
      <div><p class="io-label">입력 — 프롬프트</p>
        <div class="prompt">{esc(s['prompt'])}</div></div>
      <div><p class="io-label">출력 — 정답 원문</p>
        <pre class="answer">{esc(s['answer'])}</pre></div>
      {rendered}
    </div>
  </div>
</article>""")

    io_rows = [
        [("k", "입력 이미지"),
         ("", "영수증 또는 문서 <b>1장</b>. 100만 픽셀을 넘으면 비율을 지켜 축소한다"
              "(<span class='mono'>--max-px 1000000</span>). 원본 규격은 470×652 ~ 2475×3497px 로 편차가 크다.")],
        [("k", "입력 프롬프트"),
         ("", "태스크별 고정 문장. chat template 으로 감싸며 <b>사고모드는 끈다</b>"
              "(<span class='mono'>enable_thinking=False</span>) — 학습과 추론 조건을 같게 두기 위해서다.")],
        [("k", "출력 정답"),
         ("", "assistant 메시지 하나. 태스크별 형식은 아래 2번 표 참고.")],
        [("k", "loss 구간"),
         ("", "<b>정답 구간에만</b> 건다. 프롬프트 토큰은 <span class='mono'>-100</span> 으로 마스킹한다.")],
        [("k", "시퀀스 길이"),
         ("num", "이미지 토큰 포함 <b>중간값 1,092</b> · 최대 1,246 (표본 40건 실측)")],
        [("k", "정답 토큰"),
         ("num", "영수증 중간값 <b>466</b> · 평균 508 · 최대 1,894<br>공공문서 중간값 <b>123</b> · 평균 134 · 최대 361")],
    ]

    task_rows = [
        [("k", '<span class="badge b-parse">Document Parsing</span>'),
         ("mono", "receipt_markdown"),
         ("", "이 영수증의 내용을 마크다운으로 정리해줘."),
         ("", "마크다운 본문 + <b>HTML &lt;table&gt;</b><br>"
              "<span class='mono' style='font-size:11px'>H1=doc_title, H2=paragraph_title, 표는 원본 HTML 유지</span>"),
         ("num", "1,496행")],
        [("k", '<span class="badge b-ocr">OCR</span>'),
         ("mono", "page_ocr"),
         ("", "이 문서에 적힌 텍스트를 순서대로 옮겨 적어줘."),
         ("", "평문 텍스트 (구조 없음)"),
         ("num", "3,000행")],
        [("k", '<span class="badge b-tsr">TSR (평가)</span>'),
         ("mono", "crop_table"),
         ("", "TABLE 프롬프트 — 허용 태그 table/thead/tbody/tr/th/td/br, 속성은 rowspan·colspan 만"),
         ("", "HTML &lt;table&gt; 하나 → <b>TEDS</b> 채점"),
         ("num", "104건")],
        [("k", '<span class="badge b-ocr">OCR (평가)</span>'),
         ("mono", "crop_text"),
         ("", "OCR 프롬프트 — 보이는 내용을 읽기 순서대로 한 번만 전사"),
         ("", "평문 텍스트 → <b>CER</b> 채점"),
         ("num", "973건")],
    ]

    page = f"""<title>영수증 VLM 학습셋 검수</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{CSS}</style>
<div class="wrap">
<header>
  <p class="eyebrow">upload_mix · Qwen3.6-35B-A3B · 학습 전 검수</p>
  <h1>학습셋 입력과 출력, 무작위로 열어보기</h1>
  <p class="sub">학습을 걸기 전에 정답이 엉뚱한 문서에 붙었거나 표가 깨지지 않았는지
  사람이 직접 확인하는 페이지입니다. 표본 {len(samples)}건은 무작위로 뽑았고
  (seed 42), 이미지는 실제 학습에 들어가는 파일 그대로입니다.</p>
</header>

<h2><span class="num">01</span> 입출력 규격</h2>
<p class="lede">모든 태스크가 공유하는 형식입니다. 수치는 프로세서로 실제 토크나이즈해서 측정했습니다.</p>
{spec_table(["항목", "내용"], io_rows)}

<h2><span class="num">02</span> 태스크별 입출력</h2>
<p class="lede">학습 2종, 평가 2종입니다. 학습은 페이지 전체를 넣고, 평가는 블록 크롭 하나만 넣습니다 —
호출 단위가 다르다는 점을 감안해서 점수를 읽어야 합니다.</p>
{spec_table(["태스크", "내부 이름", "입력 프롬프트", "출력 형식", "수량"], task_rows)}

<div class="note"><b>표를 HTML로 유지한 이유</b> — 마크다운 표 문법에는 rowspan·colspan이
없습니다. 이 데이터셋 표의 절반 이상(학습 53%, 평가 55%)이 병합 셀을 쓰기 때문에,
마크다운으로 변환하면 병합이 빈 칸으로 펴지고 TEDS는 104건 전부 0점이 됩니다.</div>

<h2><span class="num">03</span> 무작위 표본 {len(samples)}건</h2>
<p class="lede">왼쪽이 모델에 들어가는 이미지, 오른쪽이 프롬프트와 정답입니다.
표가 있는 정답은 아래에 실제로 렌더링해서 행·열·병합이 제대로 잡혔는지 볼 수 있게 했습니다.</p>
<div class="cards">{''.join(cards)}</div>

<footer>
  생성 {esc(Path(args.data).name)} · seed 42 ·
  train 4,496행 / val·test 98행 / 평가 크롭 1,077건 ·
  누수 검사 doc_id 0건, 이미지 해시 0건
</footer>
</div>
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"저장: {out} ({out.stat().st_size/2**20:.1f}MB)")


if __name__ == "__main__":
    main()
