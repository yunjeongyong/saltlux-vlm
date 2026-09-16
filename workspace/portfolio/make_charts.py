"""LUXIA VLM 포트폴리오용 차트 4종. 전량 1,019건 평가와 8/20 실측치 기준."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# ── 디자인 토큰 (reference palette, light) ────────────────────────────
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
AXIS      = "#c3c2b7"
ACCENT    = "#2a78d6"   # categorical slot 1 (blue)
DEEMPH    = "#c9c8c2"   # 강조 아닌 것은 회색 — emphasis form
GOOD      = "#006300"

font_manager.fontManager.addfont if False else None
rcParams["font.family"] = "NanumGothic"
rcParams["axes.unicode_minus"] = False
rcParams["figure.facecolor"] = SURFACE
rcParams["axes.facecolor"] = SURFACE
rcParams["savefig.facecolor"] = SURFACE

OUT = "/data/workspace/yyj/data/portfolio/images"


def frame(ax, xlabel=None):
    """축·격자를 뒤로 물린다. 데이터가 앞, 크롬은 뒤."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    ax.xaxis.label.set_color(INK_2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=INK_2, labelpad=8)


def hbars(ax, labels, values, hi, fmt, xmax=None):
    """가로 막대 + 직접 라벨. hi 인덱스만 강조하고 나머지는 회색."""
    colors = [ACCENT if i == hi else DEEMPH for i in range(len(values))]
    y = range(len(values))
    ax.barh(y, values, height=0.52, color=colors, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=11,
                       color=INK)  # 텍스트는 잉크 토큰, 계열색 아님
    ax.invert_yaxis()
    top = xmax or max(values) * 1.24
    ax.set_xlim(0, top)
    ax.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for i, v in enumerate(values):
        ax.text(v + top * 0.015, i, fmt(v), va="center", fontsize=11,
                color=INK if i == hi else INK_2,
                fontweight="bold" if i == hi else "normal")
    ax.set_xticklabels([])
    frame(ax)


def header(fig, title, sub, top=0.80, left=0.26, right=0.97, bottom=0.10):
    """제목·부제를 figure 좌표에 고정하고 그만큼 축을 내린다.
    y축 라벨이 길어 왼쪽 여백을 차트마다 따로 준다."""
    fig.subplots_adjust(top=top, left=left, right=right, bottom=bottom)
    fig.text(0.012, 0.985, title, fontsize=15, color=INK,
             ha="left", va="top", fontweight="bold")
    fig.text(0.012, 0.905, sub, fontsize=10.5, color=INK_2,
             ha="left", va="top")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=200, pad_inches=0.3)
    plt.close(fig)
    print(f"  {name}")


# ── 1. 실험별 성능 — 척도가 다르므로 두 패널로 나눈다(이중축 금지) ──
exps = ["exp_003\n4,496행", "exp_004\n11,173행", "exp_005\n+온라인증강", "exp_006\n50,457행"]
exact = [57.2, 64.2, 65.3, 64.4]
teds = [0.7287, 0.7079, 0.6831, 0.6916]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"wspace": 0.42})
hbars(axes[0], exps, exact, 2, lambda v: f"{v:.1f}%")
axes[0].set_title("텍스트 완전일치율 — 921건", fontsize=12, color=INK_2,
                  loc="left", pad=10)
hbars(axes[1], exps, teds, 0, lambda v: f"{v:.4f}")
axes[1].set_title("표 구조 TEDS — 98건", fontsize=12, color=INK_2,
                  loc="left", pad=10)
header(fig, "학습 데이터를 5배 늘려도 지표는 오르지 않았다",
       "증강은 텍스트를 올리고 표를 내렸다. 통합 지표 하나만 봤다면 '차이 없음'으로 묻혔을 결과다.",
       top=0.74, left=0.13, right=0.985, bottom=0.08)
save(fig, "01_experiment_metrics.png")

# ── 2. 폭주 억제 — 강조형(하나가 답, 나머지는 맥락) ──
opts = ["현행 (억제 없음)", "repetition_penalty 1.05", "반복 감지 조기종료", "no_repeat_ngram_size 8"]
cer = [48.62, 47.93, 3.40, 1.80]
fig, ax = plt.subplots(figsize=(9.6, 4.1))
hbars(ax, opts, cer, 3, lambda v: f"{v:.2f}")

header(fig, "디코딩 설정만 바꿔 폭주군 평균 CER 48.62 → 1.80", "학습 없이 40분. 정상 대조군 40건은 네 설정 모두 완전일치 40/40 유지 — 부작용 없음을 같이 검증했다.", top=0.76, left=0.245)
save(fig, "02_runaway_suppression.png")

# ── 3. 학습 처리량 ──
ways = ["파이프라인 batch 2×4\n+ grad ckpt", "파이프라인 batch 2×4", "파이프라인 batch 8×1", "FSDP 8랭크"]
sit = [20.63, 13.43, 8.49, 6.38]
fig, ax = plt.subplots(figsize=(9.6, 4.3))
hbars(ax, ways, sit, 3, lambda v: f"{v:.2f} s/it")

header(fig, "GPU 8장 중 1장만 쓰던 구조를 FSDP로 전환 — 처리량 2.25배", "두 차례 실패 원인을 규명해 해결: Trainer 혼합정밀도가 만드는 fp32 마스터 사본, 그리고 활성값 메모리.", top=0.76, left=0.245)
save(fig, "03_training_throughput.png")

# ── 4. 데이터 오염 ──
srcs = ["exp004c (기존 학습셋)", "korie", "crawl_google"]
rate = [0.0, 12.0, 47.1]
fig, ax = plt.subplots(figsize=(9.6, 3.5))
hbars(ax, srcs, rate, 2, lambda v: f"{v:.1f}%", xmax=58)

header(fig, "외부 후보 데이터 편입 시 정답 오염률 — 전체 1,757건 (4.9%)", "텍스트 전사 태스크인데 정답이 <td>…</td> 로 감싸져 있었다. 모델이 지어낸 게 아니라 배운 대로 한 것이었다.", top=0.72, left=0.215, bottom=0.12)
save(fig, "04_label_contamination.png")
print("완료")
