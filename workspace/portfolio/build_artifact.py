import sys
sys.path.insert(0, ".")
from _img_b64 import IMGS

FIG = """<figure class="fig">
  <img src="{src}" alt="{alt}">
  <figcaption>{cap}</figcaption>
</figure>"""

def fig(key, alt, cap):
    return FIG.format(src=IMGS[key], alt=alt, cap=cap)

HTML = """<title>영수증 파싱 모델 개발기</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@400;700&family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#f6f7f5; --panel:#fdfdfc; --ink:#16181a; --ink-2:#4e5359;
  --muted:#8b8f95; --rule:#dfe1dd; --rule-2:#eceee9;
  --accent:#1f5fb0; --accent-2:#12407d; --accent-wash:rgba(31,95,176,.07);
  --flag:#a4402c;
  --measure:70ch;
}
:root:not([data-theme="light"]){ color-scheme:light; }
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --ground:#131619; --panel:#191d21; --ink:#e9ecee; --ink-2:#a8aeb4;
    --muted:#7d848b; --rule:#2b3037; --rule-2:#22272c;
    --accent:#7aa9e8; --accent-2:#a3c4f0; --accent-wash:rgba(122,169,232,.10);
    --flag:#e08a76;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --ground:#131619; --panel:#191d21; --ink:#e9ecee; --ink-2:#a8aeb4;
  --muted:#7d848b; --rule:#2b3037; --rule-2:#22272c;
  --accent:#7aa9e8; --accent-2:#a3c4f0; --accent-wash:rgba(122,169,232,.10);
  --flag:#e08a76;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans KR","Apple SD Gothic Neo","Malgun Gothic",sans-serif;
  font-size:16px; line-height:1.78; letter-spacing:-.003em;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:calc(var(--measure) + 15rem); margin:0 auto; padding:0 1.5rem 7rem}
@media(max-width:900px){ .wrap{padding:0 1.15rem 4rem} }

/* ── 기록 헤더 ── */
.rec{padding:4.5rem 0 0}
.rec .kicker{
  font-family:"IBM Plex Mono",monospace; font-size:.72rem; font-weight:500;
  letter-spacing:.16em; text-transform:uppercase; color:var(--accent);
}
.rec h1{
  font-family:"Gowun Batang",serif; font-weight:700;
  font-size:clamp(2.1rem,5.2vw,3.15rem); line-height:1.16; margin:.55rem 0 0;
  text-wrap:balance; letter-spacing:-.02em;
}
.rec .sub{
  margin:1.1rem 0 0; max-width:var(--measure); color:var(--ink-2);
  font-size:1.02rem;
}
.meta{
  margin:2.2rem 0 0; border-top:1px solid var(--ink); border-bottom:1px solid var(--rule);
  display:grid; grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));
}
.meta div{padding:.85rem 0; border-right:1px solid var(--rule-2)}
.meta div:last-child{border-right:0}
.meta dt{
  font-family:"IBM Plex Mono",monospace; font-size:.68rem; letter-spacing:.13em;
  text-transform:uppercase; color:var(--muted); margin:0 0 .3rem;
}
.meta dd{margin:0; font-size:.93rem; font-weight:500; line-height:1.45}

/* ── 요약 수치 ── */
.figs{
  margin:2.6rem 0 0; display:grid; gap:1px; background:var(--rule);
  grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));
  border:1px solid var(--rule);
}
.figs div{background:var(--panel); padding:1.15rem 1.15rem 1.05rem}
.figs b{
  display:block; font-family:"IBM Plex Mono",monospace; font-weight:600;
  font-size:1.62rem; line-height:1.1; letter-spacing:-.02em;
  font-variant-numeric:tabular-nums; color:var(--accent);
}
.figs span{display:block; margin-top:.42rem; font-size:.83rem; color:var(--ink-2); line-height:1.5}

/* ── 섹션 ── */
section{margin:4.6rem 0 0; display:grid; grid-template-columns:8.5rem 1fr; gap:2.2rem}
@media(max-width:900px){ section{grid-template-columns:1fr; gap:.5rem} }
.rail{
  font-family:"IBM Plex Mono",monospace; font-size:.7rem; letter-spacing:.13em;
  text-transform:uppercase; color:var(--muted); padding-top:.72rem;
  border-top:2px solid var(--ink);
}
@media(max-width:900px){ .rail{border-top:1px solid var(--rule); padding-top:.6rem} }
.body{max-width:var(--measure); min-width:0}
.body h2{
  font-family:"Gowun Batang",serif; font-weight:700; font-size:1.62rem;
  line-height:1.32; margin:0 0 .9rem; text-wrap:balance; letter-spacing:-.015em;
}
.body h3{
  font-size:.95rem; font-weight:600; margin:2.1rem 0 .5rem; color:var(--ink);
}
.body p{margin:0 0 1.05rem}
.body p:last-child{margin-bottom:0}
strong{font-weight:600}
.lede{font-size:1.06rem; color:var(--ink-2)}

/* ── 영수증 품목 줄: 점선 리더 ── */
.ledger{list-style:none; margin:1.1rem 0 1.4rem; padding:0}
.ledger li{
  display:flex; align-items:baseline; gap:.6rem; padding:.42rem 0;
  border-bottom:1px dotted var(--rule); font-size:.93rem;
}
.ledger li span:first-child{color:var(--ink-2)}
.ledger li span:last-child{
  margin-left:auto; font-family:"IBM Plex Mono",monospace; font-weight:500;
  font-variant-numeric:tabular-nums; white-space:nowrap;
}
.ledger .hit span:last-child{color:var(--accent); font-weight:600}

pre{
  margin:1.15rem 0; padding:.95rem 1.05rem; background:var(--panel);
  border:1px solid var(--rule); border-left:2px solid var(--accent);
  overflow-x:auto; font-family:"IBM Plex Mono",monospace; font-size:.8rem;
  line-height:1.7; color:var(--ink-2); font-variant-numeric:tabular-nums;
}
code{
  font-family:"IBM Plex Mono",monospace; font-size:.86em;
  background:var(--accent-wash); padding:.1em .34em; border-radius:2px;
}
pre code{background:none; padding:0; font-size:1em}

.fig{margin:1.7rem 0; }
.fig img{width:100%; display:block; border:1px solid var(--rule); background:#fcfcfb}
.fig figcaption{
  margin-top:.6rem; font-size:.8rem; color:var(--muted); line-height:1.6;
}

.tw{overflow-x:auto; margin:1.4rem 0}
table{border-collapse:collapse; width:100%; font-size:.86rem; min-width:34rem}
th,td{padding:.55rem .7rem; text-align:left; border-bottom:1px solid var(--rule)}
th{
  font-family:"IBM Plex Mono",monospace; font-size:.7rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); font-weight:500;
  border-bottom:1px solid var(--ink);
}
td.n{text-align:right; font-family:"IBM Plex Mono",monospace;
     font-variant-numeric:tabular-nums}
tr.now td{color:var(--accent); font-weight:600}
.note{
  margin:1.3rem 0; padding:.85rem 1.05rem; border-left:2px solid var(--flag);
  background:var(--panel); font-size:.9rem; color:var(--ink-2);
}
.note b{color:var(--ink)}

.close{
  margin:5.5rem 0 0; padding-top:2.2rem; border-top:2px solid var(--ink);
  max-width:var(--measure);
}
.close h2{
  font-family:"Gowun Batang",serif; font-weight:700; font-size:1.5rem;
  margin:0 0 1.1rem;
}
.close ul{margin:0; padding-left:1.1rem}
.close li{margin:0 0 .85rem}
</style>

<div class="wrap">

<header class="rec">
  <div class="kicker">2026.08 — 진행중 · 솔트룩스</div>
  <h1>영수증을 데이터로 바꾸는 모델을 만들며<br>배운 것</h1>
  <p class="sub">문서·영수증 이미지를 구조화된 텍스트로 변환하는 VLM을 개발한다.
  Qwen3.6-35B-A3B에 LoRA 파인튜닝하며 데이터 설계부터 학습·평가까지 담당한다.
  이 기록은 <strong>성능을 움직인 것이 모델 구조가 아니었다</strong>는 이야기다.</p>

  <dl class="meta">
    <div><dt>역할</dt><dd>학습 데이터 구축<br>파인튜닝 · 성능 평가</dd></div>
    <div><dt>기술</dt><dd>VLM · PyTorch<br>LoRA · FSDP</dd></div>
    <div><dt>환경</dt><dd>H100 80GB × 8<br>Qwen3.6-35B-A3B</dd></div>
    <div><dt>유형</dt><dd>사내 프로젝트<br>삼성SDS 협업 과제</dd></div>
  </dl>

  <div class="figs">
    <div><b>48,400</b><span>최종 학습셋 행수<br>7차 실험까지 4차 개편</span></div>
    <div><b>1,019</b><span>평가셋 블록 크롭<br>텍스트 921 / 표 98</span></div>
    <div><b>1.93 → 0.15</b><span>평균 CER<br>학습 없이 디코딩만으로</span></div>
    <div><b>2.25×</b><span>학습 처리량<br>FSDP 전환</span></div>
  </div>
</header>

<section>
  <div class="rail">데이터 생성</div>
  <div class="body">
    <h2>라벨이 확정적인 데이터를 만들었다</h2>
    <p class="lede">실물 영수증이 95장뿐이었다. 학습용 크롭이 913개 중 892개까지 소진되어
    재료가 바닥났고, 사람 라벨링을 늘리는 것은 비용이 맞지 않았다.</p>

    <p>영수증을 렌더링하는 생성기를 직접 만들었다. 핵심은 <strong>라벨을 렌더링 좌표에서
    기하 변환으로 얻는 구조</strong>로 설계한 것이다. 회전·원근·말림을 적용해도 좌표가
    정확히 따라오므로, 어절·줄·블록·표셀 4단계 bbox와 정답 텍스트가 사람 손을 거치지 않고
    확정된다.</p>

    <ul class="ledger">
      <li><span>v1 스캔풍 — 흰 종이, 책상 배경</span><span>1,000장</span></li>
      <li><span>v2 촬영풍 — 감열지, 바코드, 손에 든 각도</span><span>1,000장</span></li>
      <li><span>파생 크롭</span><span>12,000장</span></li>
      <li class="hit"><span>전수 검사 — 금액 정합 · bbox 이탈</span><span>불일치 0건</span></li>
    </ul>

    <h3>그런데 손가락이 글씨를 덮고 있었다</h3>
    <p>촬영풍의 손가락 가림 798장 중 <strong>172장에서 손가락이 어절을 90% 이상 덮고
    있었다.</strong> 정답에는 있는데 이미지에는 안 보이는 텍스트 — 모델에 환각을 가르치는
    데이터다. 좌표 겹침을 계산해 걸러내고 학습셋에서 1,204행을 제외했다. 종이 가장자리만
    가리는 것은 실제 촬영과 같은 조건이라 남겼다.</p>
  </div>
</section>

<section>
  <div class="rail">진단 · 데이터</div>
  <div class="body">
    <h2>데이터를 5배 늘렸는데 성능이 오르지 않았다</h2>
    <p class="lede">학습셋을 11,173행에서 50,457행으로 늘렸다. 텍스트 완전일치율 64.4%는
    5배 적은 데이터로 학습한 실험과 같았고, 표 지표는 오히려 떨어졌다.</p>

    {FIG1}

    <p>모델 출력을 직접 열어봤다. 텍스트 크롭인데 <code>&lt;td&gt;1,000&lt;/td&gt;</code>
    처럼 <strong>표 태그를 붙여 답하고</strong> 있었다. 정답이 <code>합계</code> 두 글자인
    크롭에도 같은 형태로 나왔다.</p>

    <p>학습 정답을 역추적한 결과가 이것이다.</p>

    {FIG4}

    <p>외부 후보 데이터를 편입할 때 <strong>표에서 떼어낸 크롭의 셀 태그를 벗기지 않아</strong>
    정답 1,757건(4.9%)이 오염돼 있었다. 기존 학습셋에는 0건이었다.</p>

    <div class="note"><b>모델이 지어낸 게 아니라 배운 대로 한 것이었다.</b>
    태그 제거 904행, 정답이 비어 있던 행 853행 삭제. 더 중요한 것은 원인이 데이터를 늘리는
    절차에 <b>정답 형식 검증이 빠져 있었다</b>는 점이라, 편입 파이프라인 앞단에 태스크별
    검사를 넣는 것을 제안했다.</div>
  </div>
</section>

<section>
  <div class="rail">진단 · 측정</div>
  <div class="body">
    <h2>지표가 모델을 잘못 판정하고 있었다</h2>
    <p class="lede">평균 CER이 1.93으로 나왔다. 문자 오류율이 1을 넘는다는 건 정답보다 긴
    오답을 쏟아낸다는 뜻이라, 모델이 망가진 것처럼 보였다.</p>

    <p>분포를 뜯어보니 <strong>중앙값은 0.0000</strong>이었다. 텍스트 921건 중 26건(2.8%)이
    평균을 전부 끌어올리고 있었다.</p>

<pre><code>GT     '합계'                                2자
PRED   '00000000000000000000000…'      1,024자 = 생성 상한
                                       → CER 512</code></pre>

    <p>같은 20자 조각이 1,003회 반복된 <strong>생성 반복 루프</strong>였다. 폭주 26건 중
    14건이 GT 10자 이하의 짧은 크롭이었고, 크기를 재보니 정상 크롭보다 5배 넓었다 —
    표의 한 행처럼 보이는데 정답은 두 글자인 크롭이다.</p>

    {FIG2}

    <h3>부작용 검증을 같이 설계했다</h3>
    <p>반복 억제는 <code>10,000</code> 같은 숫자나 반복되는 품목 줄처럼 원래 반복이 정답인
    텍스트를 망가뜨릴 수 있다. 그래서 폭주군 35건과 함께 <strong>정상 대조군 40건</strong>을
    같이 측정했다. 네 설정 모두 대조군 완전일치 40/40을 유지했고,
    <code>repetition_penalty</code>만 1건을 망가뜨려 탈락했다.</p>

    <p>전량 921건 환산 시 평균 CER <strong>1.9313 → 0.1519.</strong> 학습 없이 40분이었다.</p>

    <div class="note">다만 표 크롭은 아직 검증하지 않았다. HTML 표는
    <code>&lt;/td&gt;&lt;td&gt;</code>가 반복되는 구조라 같은 설정이 표 지표를 무너뜨릴 수
    있어, <b>서빙 반영 전 확인이 필요하다</b>고 기록했다.</div>
  </div>
</section>

<section>
  <div class="rail">측정 설계</div>
  <div class="body">
    <h2>무엇을 재는지부터 다시 정했다</h2>
    <p>문서 전체 정확도 하나로는 개선 방향이 안 잡힌다. 평가를 <strong>블록 크롭 단위
    1,019건</strong>으로 쪼개고 텍스트는 CER, 표는 TEDS로 나눠 재도록 했다.</p>

    <p>실제로 온라인 증강 실험에서 <strong>텍스트 완전일치율은 오르고(64.2% → 65.3%) 표
    구조는 떨어지는(0.7079 → 0.6831)</strong> 상반된 결과가 나왔다. 회전·원근 증강이 글자
    인식은 단단하게 만들었지만 표의 행렬 정렬을 흐트러뜨린 것으로 보인다. 통합 지표만
    봤다면 “차이 없음”으로 묻혔을 내용이다.</p>

    <h3>주지표를 교체했다</h3>
    <p>평균 CER은 소수 폭주 건에 좌우되어 대표값으로 쓸 수 없다. 완전일치율, CER 중앙값,
    TEDS, 폭주율로 판정하도록 정리했다.</p>

    <h3>측정의 함정도 기록했다</h3>
    <p>eval_loss는 실험 간 비교가 성립하지 않는다. 학습셋 토큰 중앙값이 510인데 검증셋은
    1,331이라, 학습셋 구성을 바꾸면 eval_loss가 함께 움직인다. <strong>모델 저하로
    오독하기 쉬운 지점</strong>이다.</p>
  </div>
</section>

<section>
  <div class="rail">인프라</div>
  <div class="body">
    <h2>GPU 8장 중 1장만 쓰고 있었다</h2>
    <p class="lede">H100 8장으로 돌리는데 사용률이 1~12%였다. <code>device_map="auto"</code>가
    데이터 병렬이 아니라 파이프라인 병렬이라, 모델을 레이어 단위로 쪼개 얹고 한 번에 한 장만
    계산하고 있었다. 네 실험이 전부 이 상태로 돌았다.</p>

    {FIG3}

    <p>FSDP는 이전에 두 차례 실패한 이력이 있었다. 원인을 하나씩 규명했다.</p>

    <ul class="ledger">
      <li><span>dtype 불일치 — Trainer 혼합정밀도가 fp32 마스터 사본 생성</span><span>순수 bf16 학습으로 해결</span></li>
      <li><span>활성값 OOM — 24스텝에서 장당 79.1GB</span><span>gradient checkpointing</span></li>
      <li class="hit"><span>결과</span><span>13.43 → 6.38 s/it</span></li>
    </ul>

    <p>중간에 DDP도 시도했으나 가중치 65.5GiB를 랭크마다 통째로 들어야 해서
    <strong>0.3GB 차이로 OOM</strong> 났다. 스모크 20스텝은 통과했는데 본학습에서 죽었고,
    원인은 스모크가 짧은 시퀀스만 만났기 때문이었다. <strong>검증 설계가 부실하면 통과해도
    의미가 없다</strong>는 것을 기록으로 남겼다.</p>
  </div>
</section>

<section>
  <div class="rail">실험 이력</div>
  <div class="body">
    <h2>변인을 하나만 바꾸며 쌓았다</h2>
    <p>전량 1,019건 동일 조건 측정. 각 실험은 앞선 실험에서 나온 질문에 답하도록 설계했다.</p>

    <div class="tw">
    <table>
      <thead><tr>
        <th>실험</th><th>학습셋</th><th style="text-align:right">완전일치</th>
        <th style="text-align:right">TEDS</th><th style="text-align:right">폭주</th>
        <th>확인한 것</th>
      </tr></thead>
      <tbody>
        <tr><td>exp_003</td><td class="n">4,496</td><td class="n">57.2%</td><td class="n">0.7287</td><td class="n">16</td><td>출력 형식을 평가에 맞추면 개선되는가</td></tr>
        <tr><td>exp_004</td><td class="n">11,173</td><td class="n">64.2%</td><td class="n">0.7079</td><td class="n">26</td><td>크롭 단위 학습 + 신규 데이터</td></tr>
        <tr><td>exp_005</td><td class="n">+증강</td><td class="n">65.3%</td><td class="n">0.6831</td><td class="n">25</td><td>증강은 텍스트↑ 표↓ 로 갈림</td></tr>
        <tr><td>exp_006</td><td class="n">50,457</td><td class="n">64.4%</td><td class="n">0.6916</td><td class="n">37</td><td>5배 증량 효과 없음 → 오염 발견</td></tr>
        <tr class="now"><td>exp_007</td><td class="n">48,400</td><td class="n">평가중</td><td class="n">—</td><td class="n">—</td><td>정제 효과 분리 측정</td></tr>
        <tr><td>exp_008</td><td class="n">동일</td><td class="n">계획</td><td class="n">—</td><td class="n">—</td><td>베이스 교체 (Qwen3.8-27B)</td></tr>
      </tbody>
    </table>
    </div>
  </div>
</section>

<div class="close">
  <h2>정리</h2>
  <p>이 프로젝트에서 성능을 움직인 것은 모델 구조가 아니었다.</p>
  <ul>
    <li><strong>데이터를 5배 늘려도 정답이 오염돼 있으면 효과가 없다.</strong>
    양을 늘리는 절차에 품질 검증이 없었던 것이 실제 원인이었다.</li>
    <li><strong>지표가 이상하면 모델보다 측정을 먼저 의심해야 한다.</strong>
    평균 CER 1.93은 모델의 문제가 아니라 2.8%의 폭주 건이 만든 착시였고, 중앙값은 0이었다.</li>
    <li><strong>통합 지표 하나로는 개선 방향이 안 잡힌다.</strong>
    텍스트와 표를 나눠 재고 나서야 증강이 한쪽만 올린다는 것이 보였다.</li>
    <li><strong>검증 설계가 부실하면 통과해도 의미가 없다.</strong>
    짧은 시퀀스만 본 스모크는 본학습을 보증하지 못했다.</li>
  </ul>
  <p style="margin-top:1.6rem">문서 파싱에서 정의를 정확히 세우고 그것을 검증할 수 있는
  측정 체계를 갖추는 일이, 성능 개선의 출발점이라고 본다.</p>
</div>

</div>
"""

HTML = (HTML
    .replace("{FIG1}", fig("01_experiment_metrics", "실험별 완전일치율과 TEDS 비교",
        "전량 1,019건 평가. 척도가 다른 두 지표를 한 축에 겹치지 않도록 패널을 나눴다."))
    .replace("{FIG4}", fig("04_label_contamination", "출처별 정답 오염률",
        "외부 후보 데이터 편입 시 정답에 HTML 표 태그가 섞인 비율. 기존 학습셋은 0건이었다."))
    .replace("{FIG2}", fig("02_runaway_suppression", "디코딩 설정별 폭주군 평균 CER",
        "폭주군 35건 기준. 정상 대조군 40건을 함께 측정해 부작용이 없음을 확인했다."))
    .replace("{FIG3}", fig("03_training_throughput", "병렬 방식별 학습 속도",
        "같은 유효 배치 8 기준. 파이프라인 병렬은 8장 중 1장만 계산에 참여한다."))
)
open("portfolio.html", "w", encoding="utf-8").write(HTML)
print("portfolio.html", len(HTML)//1024, "KB")
