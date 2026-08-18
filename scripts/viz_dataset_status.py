"""
지금 무엇이 어디까지 됐는지 한 장으로 본다.

세 가지를 붙인다.
  1) 71299 필터링 퍼널 — 64,228 장에서 무엇이 왜 빠졌는가
  2) 배치 현황 — 팀 공용 데이터셋(train/) 에 무엇이 올라갔고 무엇이 대기 중인가
  3) GPU 점유 — 학습을 돌릴 수 있는 자리가 있는지

usage:
    python3 scripts/viz_dataset_status.py
"""
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"
KEEP, DROP = "#2a78d6", "#eb6834"
OK, WAIT = "#1baf7a", "#eda100"
INK, MUTED, RULE = "#14171c", "#5d6674", "#dce0e6"

fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

FUNNEL = [
    ("짝이 맞는 문서", 64228, None),
    ("손글씨 10% 초과", -19124, "typeface 2·3·4 가 10%↑ 인 장"),
    ("표 서식", -8500, "열 정렬이 뚜렷한 장 — 정답이 평문이라 충돌"),
    ("박스 부족", -6, "박스 5개 미만"),
    ("최종 채택", 36598, None),
]


def gpu():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        out = []
        for ln in r.stdout.strip().splitlines():
            i, tot, used, util = [x.strip() for x in ln.split(",")]
            out.append((int(i), int(tot), int(used), int(util)))
        return out
    except Exception:
        return []


def funnel_chart(path, w=760, h=380):
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    labels = [t for t, _, _ in FUNNEL]
    running, bottoms, vals, cols = 0, [], [], []
    for k, (t, v, _) in enumerate(FUNNEL):
        if k == 0:
            running = v
            bottoms.append(0); vals.append(v); cols.append(KEEP)
        elif t == "최종 채택":
            bottoms.append(0); vals.append(v); cols.append(KEEP)
        else:
            running += v
            bottoms.append(running); vals.append(-v); cols.append(DROP)
    y = range(len(labels))[::-1]
    ax.barh(list(y), vals, left=bottoms, color=cols, height=.6,
            edgecolor="white", linewidth=2)
    for k, (yy, b, v) in enumerate(zip(list(y), bottoms, vals)):
        ax.text(b + v + 900, yy, f"{v:,}", va="center", fontsize=11,
                color=MUTED)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=12, color=INK)
    ax.set_xlabel("문서 수", fontsize=10, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.grid(axis="x", color=RULE, linewidth=.8)
    ax.set_axisbelow(True)
    ax.set_title("AIHub 71299 — 무엇이 왜 빠졌나 (파랑=남김 / 주황=제외)",
                 fontsize=14, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def gpu_chart(g, path, w=760, h=380):
    if not g:
        return None
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    idx = [f"GPU {i}" for i, _, _, _ in g]
    used = [u / 1024 for _, _, u, _ in g]
    free = [(t - u) / 1024 for _, t, u, _ in g]
    ax.bar(idx, used, color="#c3c9d2", edgecolor="white", linewidth=2, label="사용 중")
    ax.bar(idx, free, bottom=used, color=OK, edgecolor="white", linewidth=2, label="여유")
    for k, (u, f) in enumerate(zip(used, free)):
        if f >= 5:
            ax.text(k, u + f / 2, f"{f:.0f}GB", ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold")
    ax.set_ylabel("GiB", fontsize=10, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.grid(axis="y", color=RULE, linewidth=.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc="upper right", ncol=2)
    ax.set_title("GPU 점유 — 학습 자리가 있는가", fontsize=14, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return True


def place_panel(w, h):
    """배치 현황 표."""
    im = Image.new("RGB", (w, h), "white")
    dr = ImageDraw.Draw(im)
    f17 = ImageFont.truetype(FONT, 17)
    f14 = ImageFont.truetype(FONT, 14)
    f13 = ImageFont.truetype(FONT, 13)
    dr.text((4, 4), "팀 공용 데이터셋 배치 현황  /data/workspace/VLM/dataset/train/",
            fill=INK, font=f17)
    rows = [
        ("public/pubtabnet-html", "표 500,777", "기존", OK),
        ("public/kogovdoc-bench", "문서 2,961", "기존", OK),
        ("public/aihub-71299-ocr", "문서 36,598", "만듦 · 복사 대기", WAIT),
        ("human_annotated/receipt", "영수증 84", "만듦 · 복사 대기", WAIT),
        ("private", "—", "비어 있음", MUTED),
        ("api_generated", "—", "비어 있음", MUTED),
    ]
    y = 34
    dr.line([4, y, w - 4, y], fill=RULE, width=2)
    y += 8
    for name, size, state, col in rows:
        dr.rectangle([6, y + 4, 14, y + 12], fill=col)
        dr.text((24, y), name, fill=INK, font=f14)
        dr.text((int(w * 0.52), y), size, fill=MUTED, font=f13)
        dr.text((int(w * 0.72), y), state, fill=col, font=f13)
        y += 24
    y += 6
    dr.line([4, y, w - 4, y], fill=RULE, width=1)
    y += 10
    dr.text((6, y), "※ /data/workspace/VLM 이 작업 컨테이너에서 읽기 전용이라",
            fill=MUTED, font=f13)
    y += 19
    dr.text((6, y), "   복사(cp -a)는 직접 실행해야 반영된다.", fill=MUTED, font=f13)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/workspace/yjyong/receipt_data/review/dataset_status.png")
    args = ap.parse_args()

    S = Path("/tmp/claude-0/-data-workspace-yjyong/be45946a-421f-4383-9259-48c06f6b862d/scratchpad")
    funnel_chart(S / "f.png")
    g = gpu()
    has_gpu = gpu_chart(g, S / "g.png")

    f_img = Image.open(S / "f.png").convert("RGB")
    parts = [f_img]
    if has_gpu:
        parts.append(Image.open(S / "g.png").convert("RGB"))

    top_w = sum(p.width for p in parts) + 14 * (len(parts) - 1)
    top = Image.new("RGB", (top_w, max(p.height for p in parts)), "white")
    x = 0
    for p in parts:
        top.paste(p, (x, 0))
        x += p.width + 14

    pp = place_panel(top.width - 20, 220)
    head = 92
    out = Image.new("RGB", (top.width, head + top.height + pp.height + 30), "white")
    dr = ImageDraw.Draw(out)
    dr.text((14, 12), "데이터셋 구축 현황", fill=INK, font=ImageFont.truetype(FONT, 26))
    dr.text((14, 50), "AIHub 71299 필터링 · 배치 대기 · GPU 여유", fill=MUTED,
            font=ImageFont.truetype(FONT, 16))
    free = [(i, (t - u) / 1024) for i, t, u, _ in g] if g else []
    best = max(free, key=lambda x: x[1]) if free else None
    if best:
        dr.text((14, 72), f"학습 가능 자리: GPU {best[0]}번 여유 {best[1]:.0f}GB "
                          f"— Qwen3.5-4B(8.8GB) LoRA 는 들어간다",
                fill=OK, font=ImageFont.truetype(FONT, 15))
    dr.line([14, head - 8, out.width - 14, head - 8], fill=RULE, width=2)
    out.paste(top, (0, head))
    out.paste(pp, (14, head + top.height + 14))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"저장: {args.out}  {out.size}")


if __name__ == "__main__":
    main()
