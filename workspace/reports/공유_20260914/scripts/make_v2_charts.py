"""v2 보고서용 차트 — 타임라인 / 실험 계보 / 데이터 카탈로그 / CER 분포·정체 원인 / 오류 유형."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch, FancyBboxPatch

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
SCR = Path(__file__).parent
BLUE, ORANGE, GRAY, INK, DIM, SURF = "#2a78d6", "#eb6834", "#9a9a95", "#0b0b0b", "#52514e", "#fcfcfb"
LBLUE, LORANGE, GREEN, LGREEN = "#a9c7ee", "#f5c3ad", "#1f6b4a", "#bcd9c6"
rcParams.update({"font.family": ["NanumGothic", "DejaVu Sans"], "axes.unicode_minus": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": "#d8d8d2", "axes.grid": True, "grid.color": "#e6e6e0", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "xtick.color": DIM, "ytick.color": INK, "text.color": INK,
    "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left"})

def save(fig, name, note=None):
    if note: fig.text(0.01, -0.04, note, fontsize=9, color=DIM, ha="left", va="top")
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", pad_inches=0.3); plt.close(fig); print("saved", name)

# ═══════════ 1. 작업 흐름 타임라인
import textwrap as _tw
fig, ax = plt.subplots(figsize=(15, 6.2)); ax.grid(False); ax.set_xlim(0, 100); ax.set_ylim(0, 10); ax.axis("off")
phases = [
    (0, 15,  "8/12~13", "① 기준선·정렬",   "exp_001 기준선 → exp_003 출력형식 정렬 → exp_004 크롭 도입\nCER 0.201→0.106 · 현 배포", BLUE),
    (15, 31, "8/18~24", "② 4연속 실패",    "exp_005 증강 · 006 5배 증량 · 007 오염 제거 · 008 27B 교체\n전부 exp_004 미달 → '무엇을 늘리나'가 관건", GRAY),
    (31, 48, "8/25~31", "③ 요인 분리",     "exp_009 크롭↑페이지 209회 · 010 경량 · 011 크롭↑반복 24회\nexp_011 TEDS 목표 달성 · 세 지표 최상위", BLUE),
    (48, 65, "9/1~5",   "④ 신규 도메인",   "SS 명세서·계산서 대응 — exp_012·013 서식 10,491행 투입\nSS 85.6 → 74.7% 급락", ORANGE),
    (65, 83, "9/7~12",  "⑤ 원인 규명·수정", "FATURA2 4,000행 라벨 누락 규명 → exp_015 복원 · 014 NC 제외\n텍소노미 v1.1 · 평가 vLLM 전환(15h→15m)", GREEN),
    (83, 100,"9/14",    "⑥ 정체 원인 분석", "exp_015 표 출력 100/100 회복 확인\nCER 정체 원인: 회전 2장·GT 오류 26건이 평균의 54%", BLUE),
]
for x0, x1, d, t, body, c in phases:
    ax.add_patch(FancyBboxPatch((x0 + 0.4, 3.6), x1 - x0 - 0.8, 5.2, boxstyle="round,pad=0,rounding_size=0.4", fc="white", ec=c, lw=1.6))
    ax.add_patch(FancyBboxPatch((x0 + 0.4, 7.9), x1 - x0 - 0.8, 0.9, boxstyle="round,pad=0,rounding_size=0.3", fc=c, ec=c))
    ax.text(x0 + 0.9, 8.35, t, fontsize=10.5, fontweight="bold", color="white", va="center")
    ax.text(x1 - 0.9, 8.35, d, fontsize=8, color="white", va="center", ha="right")
    wrapped = "\n".join(_tw.fill(seg, 17) for seg in body.split("\n"))
    ax.text(x0 + 1.0, 7.4, wrapped, fontsize=8.8, va="top", linespacing=1.35)
ax.annotate("", xy=(100, 2.2), xytext=(0, 2.2), arrowprops=dict(arrowstyle="-|>", color=DIM, lw=1.5))
for x0, x1, d, t, body, c in phases: ax.plot([ (x0+x1)/2 ], [2.2], "o", color=c, ms=9)
ax.text(0, 1.2, "2026-08-12", fontsize=10, color=DIM); ax.text(100, 1.2, "2026-09-14", fontsize=10, color=DIM, ha="right")
ax.set_title("작업 흐름 — 5주 · 15개 실험 · 6단계")
save(fig, "v01_timeline.png", "출처: LUXIA_VLM_학습현황_3.xlsx 실험로그 · 팀즈보고 8/26·9/3·9/7 · 2026-09-14 분석")

# ═══════════ 2. 실험 계보 — 3지표 + 무엇을 바꿨나
ex = ["base","exp_003","exp_004","exp_005","exp_006","exp_007","exp_008","exp_009","exp_010","exp_011"]
cer = [0.1378,0.1302,0.1063,0.1081,0.1177,0.1155,0.1305,0.1155,0.1131,0.1091]
teds= [0.6647,0.7287,0.7079,0.6831,0.6916,0.6691,0.6879,0.7476,0.7335,0.7410]
em  = [57.9,57.2,64.2,65.3,64.4,63.1,60.8,63.5,63.7,65.6]
why = ["원본","출력형식\n정렬","크롭 도입\n(현 배포)","온라인\n증강","5배 증량","오염 제거","27B 교체","크롭 41k\n페이지 209회","경량판","크롭 12k\n페이지 24회"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, v, tgt, ttl, better in [(axes[0], cer, 0.097, "CER ↓ (목표 ≤0.097)", "min"), (axes[1], teds, 0.739, "TEDS ↑ (목표 ≥0.739)", "max"), (axes[2], em, 67, "완전일치 % ↑ (목표 ≥67)", "max")]:
    c = [GRAY if e=="base" else ORANGE if e=="exp_004" else GREEN if e=="exp_011" else BLUE for e in ex]
    ax.bar(range(len(ex)), v, color=c, width=0.62, edgecolor=SURF, linewidth=2)
    ax.axhline(tgt, color=INK, ls="--", lw=1.1); ax.text(-0.4, tgt, f"목표 {tgt}", fontsize=9, va="bottom", fontweight="bold")
    ax.set_xticks(range(len(ex))); ax.set_xticklabels([e.replace("exp_","") for e in ex], fontsize=9.5); ax.tick_params(axis="x", length=0); ax.grid(axis="x", visible=False)
    ax.set_title(ttl, fontsize=13)
    best = (min if better=="min" else max)(v[1:])
    for i, x in enumerate(v): ax.text(i, x + (0.001 if better=="min" and max(v)<1 else (0.003 if max(v)<1 else 0.3)), f"{x:.3f}" if max(v)<1 else f"{x:.1f}", ha="center", fontsize=8.5, fontweight="bold" if x==best else "normal")
    lo, hi = min(v), max(v); ax.set_ylim(lo - (hi-lo)*0.25, hi + (hi-lo)*0.35)
fig.suptitle("학습 실험 계보 — 무엇을 바꿨고 무엇이 움직였나 (자체 평가셋 1,019 크롭)", x=0.01, ha="left", fontsize=15, fontweight="bold")
fig.legend(handles=[Patch(color=GRAY, label="학습 전"), Patch(color=ORANGE, label="exp_004 현 배포"), Patch(color=GREEN, label="exp_011 종합 최적"), Patch(color=BLUE, label="기타 실험")], loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=4, frameon=False, fontsize=10)
fig.subplots_adjust(top=0.8, wspace=0.2, bottom=0.2)
save(fig, "v02_lineage_3metrics.png", "출처: review/_all_models.json · CER 0~1 유계 · exp_012·013은 SS 급락으로 자체 평가 생략, exp_015·014는 채점 대기")

# ═══════════ 3. 데이터 카탈로그 — 보유 vs 활용 (S-ID, 도메인 묶음)
cat = [  # (ID, 이름, 도메인, 보유, 활용(exp_013 기준), 상태)
 ("S01","train_crops(사내)","영수증",913,365,"사내"),("S03","labeled 검수본(사내)","영수증",188,182,"사내"),
 ("S09","KorIE ocr","영수증",8381,2885,"public·미확인"),("S10","KorIE kid","영수증",2250,501,"public·미확인"),("S11","CORD","영수증",4978,1734,"public"),
 ("S02","receipts3000 크롭","영수증(생성)",9520,1481,"생성"),("S04","synth_v1","영수증(생성)",7000,1000,"생성"),("S05","synth_v2","영수증(생성)",1000,828,"생성"),
 ("S07","synth_merged_crops","영수증(생성)",12000,5037,"생성"),("S27","정제 크롭","영수증(생성)",1883,1883,"생성"),("S06","synth_v2_nofinger","영수증(생성)",1000,0,"생성"),
 ("S18","std_hb(선임 제공)","서식(생성)",5000,5000,"생성"),("S26","ctg v2","서식(생성)",9000,3916,"생성"),("S17","ctg 구판","서식(생성)",8000,0,"생성·제외"),
 ("S21/22","synth_bl·mcip","서식(생성)",6000,0,"폐기"),
 ("S19","FATURA2","인보이스",4000,4000,"public"),("S20","invoices-donut","인보이스",491,491,"public"),
 ("S13","AI-Hub 71299","공공행정",3000,2000,"public·미확인"),("S25","multimodal-retrieval","공공행정",2499,2499,"public·미확인"),("S15","kogovdoc-bench","공공행정",139,139,"public"),
 ("S14","PubTabNet","학술 표",5461,5461,"public·NC"),
 ("S16","crawl_google","영수증",155,0,"미확인·제외"),("S23","research_only","레이아웃",1582,0,"미확인·격리"),
]
col = {"사내": GREEN, "생성": BLUE, "public": GRAY, "public·미확인": "#c9a227", "public·NC": ORANGE, "생성·제외": LBLUE, "폐기": "#d9d9d4", "미확인·제외": "#c9a227", "미확인·격리": "#c9a227"}
fig, ax = plt.subplots(figsize=(15, 8.2))
y = np.arange(len(cat))[::-1]
ax.barh(y, [c[3] for c in cat], color="#e9e9e4", height=0.7, edgecolor=SURF, linewidth=2)
ax.barh(y, [c[4] for c in cat], color=[col[c[5]] for c in cat], height=0.7, edgecolor=SURF, linewidth=2)
ax.set_yticks(y); ax.set_yticklabels([f"{c[0]}  {c[1]}  ·  {c[2]}" for c in cat], fontsize=9.5); ax.tick_params(axis="y", length=0); ax.grid(axis="y", visible=False)
for yy, c in zip(y, cat):
    ax.text(c[3] + 120, yy, f"{c[4]:,} / {c[3]:,}" + ("" if c[4] else f"  ({c[5]})"), va="center", fontsize=9, color=INK if c[4] else DIM)
ax.set_xlim(0, 14500); ax.set_xlabel("행(장) 수 — 회색 배경 = 보유, 색 = exp_013 기준 학습 활용", color=DIM)
ax.set_title("원본 데이터셋 카탈로그 — 25종 · 보유 94,840 · 활용 39,402 (S-ID 순서는 도메인별)")
ax.legend(handles=[Patch(color=GREEN, label="사내 실문서"), Patch(color=BLUE, label="자체 생성"), Patch(color=GRAY, label="public (상용 가능)"), Patch(color="#c9a227", label="public (라이선스 미확인)"), Patch(color=ORANGE, label="public NC (상용 불가)"), Patch(color="#d9d9d4", label="폐기·제외")], loc="lower right", frameon=False, fontsize=9.5, ncol=2)
save(fig, "v03_data_catalog.png", "출처: LUXIA_VLM_01_데이터셋_카탈로그.xlsx 「1.원본 데이터셋」 · 활용 열은 exp_013_v2 기준 · S08·S12는 파생 원본이라 생략")

# ═══════════ 4. 학습셋 버전별 구성 (task 그룹 × 실험)
vers = ["exp_004","exp_009","exp_010","exp_011","exp_012","exp_013","exp_015","exp_014"]
groups = {  # 행 수
 "영수증 크롭(텍스트·표)": [892,36291,892,13900,13900,13886,13886,13886],
 "영수증 페이지":          [182,39866,4368,6196,6196,3284,3284,3284],
 "서식·인보이스(신규 도메인)": [0,0,0,0,15091,10491,9825,9825],
 "학술 표 PubTabNet(NC)": [5461,5461,0,5461,5461,5461,5461,0],
 "공공행정(OCR·파싱)":     [4638,4638,4638,4638,4638,4638,4638,4638],
}
tot = [11173,86256,9898,30195,47686,40676,40010,34549]
# 합이 맞도록 잔차를 '기타'로
other = [t - sum(g[i] for g in groups.values()) for i, t in enumerate(tot)]
groups["기타·검증분"] = [max(0, o) for o in other]
cols = [BLUE, LBLUE, ORANGE, "#c9a227", GRAY, "#d9d9d4"]
fig, ax = plt.subplots(figsize=(14, 5.6)); x = np.arange(len(vers)); bottom = np.zeros(len(vers))
for (k, v), c in zip(groups.items(), cols):
    ax.bar(x, v, bottom=bottom, color=c, width=0.62, edgecolor=SURF, linewidth=2, label=k); bottom += np.array(v)
for i, t in enumerate(tot): ax.text(i, t + 1500, f"{t:,}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([v.replace("exp_","exp_") for v in vers]); ax.tick_params(axis="x", length=0); ax.grid(axis="x", visible=False)
ax.set_ylabel("학습 행 수", color=DIM); ax.set_ylim(0, 95000)
ax.legend(loc="upper right", frameon=False, fontsize=10, ncol=2)
ax.set_title("학습셋 버전별 구성 — exp_009의 86k는 영수증 반복, exp_012부터 서식 도메인 투입")
save(fig, "v04_trainset_versions.png", "출처: 카탈로그 「2.학습 활용 데이터」 태스크별 구성 · exp_015/014는 manifest 기준 · '기타'는 val·소량 task")

# ═══════════ 5. CER 분포 — 평균은 꼬리가 만든다
A = json.load(open(SCR / "cer/cer_analysis.json")); D = A["dist"]
order = ["base","exp_004","exp_005","exp_006","exp_007","exp_008","exp_009","exp_010","exp_011"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.5, 1]})
x = np.arange(len(order)); b = np.zeros(len(order))
for key, lab, c in [("zero","CER = 0 (완전일치)",LGREEN),("le01","0 < CER ≤ 0.1",LBLUE),("le03","0.1 < CER ≤ 0.3",LORANGE),("gt03","CER > 0.3",ORANGE)]:
    v = np.array([D[k][key] for k in order]); a1.bar(x, v, bottom=b, color=c, width=0.62, edgecolor=SURF, linewidth=2, label=lab)
    for i, (vv, bb) in enumerate(zip(v, b)):
        if key in ("zero","gt03"): a1.text(i, bb + vv/2, str(vv), ha="center", va="center", fontsize=9, color=INK if key=="zero" else "white", fontweight="bold")
    b += v
a1.set_xticks(x); a1.set_xticklabels([k.replace("exp_","") for k in order]); a1.tick_params(axis="x", length=0); a1.grid(axis="x", visible=False)
a1.set_ylabel("텍스트 크롭 수 (921)", color=DIM); a1.legend(loc="upper center", frameon=False, fontsize=9.5, ncol=4, bbox_to_anchor=(0.5, -0.1))
a1.set_title("크롭별 CER 분포 — 중앙값은 전 실험 0, 완전일치 57~65%", fontsize=13)
m = [D[k]["mean"] for k in order]; ts = [D[k]["tail_share"]*100 for k in order]
a2.bar(x, ts, color=ORANGE, width=0.62, edgecolor=SURF, linewidth=2)
for i, t in enumerate(ts): a2.text(i, t + 1, f"{t:.0f}%", ha="center", fontsize=9.5, fontweight="bold")
a2.set_xticks(x); a2.set_xticklabels([k.replace("exp_","") for k in order], fontsize=9.5); a2.tick_params(axis="x", length=0); a2.grid(axis="x", visible=False)
a2.set_ylim(0, 100); a2.set_ylabel("%", color=DIM)
a2.set_title("평균 CER 중 'CER>0.3 크롭 107~160건'이 차지하는 몫", fontsize=13)
fig.suptitle("CER이 안 내려가는 이유 ① — 평균의 3/4를 상위 12%의 크롭이 만들고, 그 꼬리는 어떤 실험도 못 줄였다", x=0.01, ha="left", fontsize=15, fontweight="bold")
fig.subplots_adjust(top=0.8, wspace=0.22)
save(fig, "v05_cer_distribution.png", "출처: receipt_data/review/*_full1019.json · 921 텍스트 크롭 · 유계 CER 재계산 (2026-09-14)")

# ═══════════ 6. 지속 실패 76건 분류 + 반사실 CER
CF = json.load(open(SCR / "cer/cer_counterfactual.json"))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1.3]})
man = CF["manual"]; labs = ["회전 이미지(원본 2장)", "평가셋 GT/bbox 오류", "모델 오독·누락(흐림·잘림 포함)", "형식 차이"]; vals = [man[l] for l in labs]
cs = [ORANGE, "#c9a227", BLUE, GRAY]
a1.barh(range(4)[::-1], vals, color=cs, height=0.62, edgecolor=SURF, linewidth=2)
a1.set_yticks(range(4)[::-1]); a1.set_yticklabels(["회전 90° 영수증 2장\n(val_receipt34·67의 크롭)", "평가셋 GT·bbox 오류\n(GT가 크롭 밖 줄 포함 등)", "진짜 모델 오독·누락\n(흐림·잘림 포함)", "형식 차이\n(【】 vs [])"], fontsize=10); a1.tick_params(axis="y", length=0); a1.grid(axis="y", visible=False)
for i, v in zip(range(4)[::-1], vals): a1.text(v + 0.8, i, f"{v}건 ({v/76*100:.0f}%)", va="center", fontsize=11, fontweight="bold")
a1.set_xlim(0, 52); a1.set_xticks([])
a1.set_title("4개 실험(004·009·010·011) 모두 CER>0.3인 크롭 76건 — 전수 육안 분류", fontsize=12.5)
k = "exp_011"; a, bb, c, d = CF["counterfactual"][k]
steps = [("원본\n(921 크롭)", a), ("회전 2장\n제외", bb), ("+ GT 오류\n26건 수정", c), ("+ 공백·기호\n정규화", d)]
xs = np.arange(4); vs = [s[1] for s in steps]
a2.bar(xs, vs, color=[ORANGE, "#c9a227", BLUE, GREEN], width=0.6, edgecolor=SURF, linewidth=2)
a2.axhline(0.097, color=INK, ls="--", lw=1.1); a2.text(3.35, 0.097, "목표 0.097", fontsize=9.5, va="bottom", ha="right", fontweight="bold")
for i, v in enumerate(vs): a2.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
for i in range(3): a2.annotate("", xy=(i+1, vs[i+1]+0.012), xytext=(i, vs[i]+0.012), arrowprops=dict(arrowstyle="->", color=DIM, lw=1))
a2.set_xticks(xs); a2.set_xticklabels([s[0] for s in steps], fontsize=10); a2.tick_params(axis="x", length=0); a2.grid(axis="x", visible=False)
a2.set_ylim(0, 0.135); a2.set_ylabel("exp_011 텍스트 CER", color=DIM)
a2.set_title("같은 exp_011을 정상 문서 기준으로 다시 재면 — 회전 2장만 빼도 목표 달성", fontsize=12.5)
fig.suptitle("CER이 안 내려가는 이유 ② — 안 고쳐지던 76건의 87%는 모델이 아니라 평가셋(회전 문서·라벨 오류) 몫", x=0.01, ha="left", fontsize=15, fontweight="bold")
fig.subplots_adjust(top=0.8, wspace=0.35)
save(fig, "v06_persistent_failures.png", "출처: 2026-09-14 분석 · 76건 전수 육안 확인 · 반사실 CER은 exp_004 0.107→0.050, exp_009 0.116→0.059도 동일 패턴")

# ═══════════ 7. exp_011 오류 330건 유형 — 건수 vs CER 기여
T = CF["all_err_types"]
order_t = sorted(T.items(), key=lambda kv: -kv[1][1])
labs = [k for k, _ in order_t]; cnt = [v[0] for _, v in order_t]; share = [v[1]*100 for _, v in order_t]
def color_of(l):
    if "회전" in l: return ORANGE
    if "GT" in l: return "#c9a227"
    if "형식" in l: return GRAY
    return BLUE
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
y = np.arange(len(labs))[::-1]
a1.barh(y, cnt, color=[color_of(l) for l in labs], height=0.62, edgecolor=SURF, linewidth=2)
for yy, v in zip(y, cnt): a1.text(v + 2, yy, f"{v}", va="center", fontsize=10)
a1.set_yticks(y); a1.set_yticklabels(labs, fontsize=10); a1.tick_params(axis="y", length=0); a1.grid(axis="y", visible=False); a1.set_xlim(0, 150)
a1.set_title("건수 — 오류 크롭 330건 중", fontsize=13)
a2.barh(y, share, color=[color_of(l) for l in labs], height=0.62, edgecolor=SURF, linewidth=2)
for yy, v in zip(y, share): a2.text(v + 0.6, yy, f"{v:.1f}%", va="center", fontsize=10, fontweight="bold")
a2.tick_params(axis="y", length=0); a2.grid(axis="y", visible=False); a2.set_xlim(0, 46); a2.set_xlabel("exp_011 총 CER 중 비중 (%)", color=DIM)
a2.set_title("CER 기여 — 건수는 형식 차이가 1위, 비중은 회전 이미지가 1위", fontsize=13)
fig.suptitle("지속 발생 오류 유형 — exp_011 자체 평가셋 전 오류 (CER > 0인 330 크롭)", x=0.01, ha="left", fontsize=15, fontweight="bold")
fig.legend(handles=[Patch(color=ORANGE, label="평가셋 원본 문제(회전)"), Patch(color="#c9a227", label="평가셋 라벨 문제"), Patch(color=GRAY, label="채점 형식 문제"), Patch(color=BLUE, label="모델 인식 오류")], loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=4, frameon=False, fontsize=10)
fig.subplots_adjust(top=0.8, wspace=0.08)
save(fig, "v07_error_types_exp011.png", "출처: 2026-09-14 분석 · 유형 판정: 회전/GT오류는 육안 확정, 나머지는 규칙(정규화 후 일치=형식, 길이비=누락·추가, 문자 종류=한글·숫자 오독)")

# ═══════════ 8. churn — exp_004 → exp_011
ch = A["churn"]
fig, ax = plt.subplots(figsize=(9, 3.4)); ax.grid(False); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 10)
tot = ch["improved"] + ch["worsened"] + ch["same"]; x = 0
for lab, v, c in [("개선", ch["improved"], GREEN), ("악화", ch["worsened"], ORANGE), ("변화 없음 (±0.05 이내)", ch["same"], "#e3e3de")]:
    w = v / tot * 100; ax.add_patch(FancyBboxPatch((x, 3), w, 4, boxstyle="square,pad=0", fc=c, ec=SURF, lw=2))
    ax.text(x + w/2, 5, f"{lab}\n{v}건", ha="center", va="center", fontsize=11 if w > 12 else 9.5, color="white" if w < 30 else INK, fontweight="bold"); x += w
ax.text(0, 8.8, "exp_004 → exp_011 · 크롭 921건의 CER 변화 — 개선분 합 13.8 vs 악화분 합 16.7 → 상쇄", fontsize=13, fontweight="bold")
ax.text(0, 1.6, "학습셋을 바꾸면 어떤 크롭은 좋아지고 비슷한 수가 나빠진다. 남은 오류가 '데이터 부족'이 아니라 '평가셋 특성'에 묶여 있어 평균이 제자리다.", fontsize=10.5, color=DIM)
save(fig, "v08_churn_004_to_011.png", "출처: 2026-09-14 분석 · 임계 ±0.05")
