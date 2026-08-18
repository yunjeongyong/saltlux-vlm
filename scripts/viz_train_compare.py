"""
학습 두 판(v1 300스텝 / v2 2,000스텝)을 나란히 놓고 무엇이 달라졌는지 본다.

  왼쪽   loss 곡선 — 두 판의 train/eval 을 한 축에 겹친다
  오른쪽 과제별 CER — base 대비 얼마나 내려갔나

loss 는 학습 로그에서 긁어온다. tqdm 진행바가 섞여 있어 정규식으로 발라낸다.

usage:
    python3 scripts/viz_train_compare.py
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/data/workspace/yjyong")
FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"
# dataviz 기본 categorical 1·2 — validate_palette.js 통과
C_V1, C_V2, C_BASE = "#eb6834", "#2a78d6", "#8B94A3"
INK, MUTED, RULE = "#14171c", "#5d6674", "#dce0e6"

fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

RUNS = [
    ("v1 · 300스텝", "ml/upload_v1.log", "receipt_data/review/upload_run_eval.json", C_V1),
    ("v2 · ckpt-1000", "ml/upload_v2.log", "receipt_data/review/upload_run_eval_ckpt1000.json", C_V2),
]
TASK_KO = {"page_ocr": "공공문서 (평문 전사)", "receipt_markdown": "영수증 (마크다운 표)"}


def parse_log(p):
    """tqdm 이 섞인 로그에서 loss / eval_loss 만 발라낸다."""
    txt = Path(p).read_text(errors="ignore")
    tr = [(float(e), float(l)) for l, e in
          re.findall(r"\{'loss': '([\d.]+)'.*?'epoch': '([\d.]+)'\}", txt)]
    ev = [(float(e), float(l)) for l, e in
          re.findall(r"\{'eval_loss': '([\d.]+)'.*?'epoch': '([\d.]+)'\}", txt)]
    return tr, ev


def loss_chart(path, w=760, h=440):
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for name, log, _, col in RUNS:
        f = ROOT / log
        if not f.exists():
            continue
        tr, ev = parse_log(f)
        if tr:
            # 진행바 때문에 같은 스텝이 여러 번 찍히므로 순서만 보고 그린다
            xs = [e for e, _ in tr]
            ys = [l for _, l in tr]
            ax.plot(xs, ys, color=col, linewidth=1.2, alpha=.35)
            # 이동평균으로 추세를 보인다
            k = max(1, len(ys) // 40)
            sm = [sum(ys[max(0, i - k):i + 1]) / len(ys[max(0, i - k):i + 1])
                  for i in range(len(ys))]
            ax.plot(xs, sm, color=col, linewidth=2.2, label=f"{name} train")
        if ev:
            ax.plot([e for e, _ in ev], [l for _, l in ev], "o--", color=col,
                    linewidth=2, markersize=7, label=f"{name} eval")
    ax.set_xlabel("epoch", fontsize=11, color=MUTED)
    ax.set_ylabel("loss", fontsize=11, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.grid(color=RULE, linewidth=.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("학습 곡선 — 옅은 선이 스텝별 loss, 진한 선이 이동평균",
                 fontsize=14, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def cer_chart(path, w=760, h=440):
    """과제별 CER: base / v1 / v2 를 나란히."""
    data = {}
    for name, _, ev, col in RUNS:
        f = ROOT / ev
        if not f.exists():
            continue
        rows = json.loads(f.read_text())
        for t in set(r["task"] for r in rows):
            g = [r for r in rows if r["task"] == t]
            data.setdefault(t, {})["base"] = sum(r["cer_base"] for r in g) / len(g)
            data[t][name] = sum(r["cer_lora"] for r in g) / len(g)
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    tasks = [t for t in ("page_ocr", "receipt_markdown") if t in data]
    series = ["base"] + [n for n, _, _, _ in RUNS if any(n in v for v in data.values())]
    cols = {"base": C_BASE, RUNS[0][0]: C_V1, RUNS[1][0]: C_V2}
    bw = .8 / len(series)
    for k, s in enumerate(series):
        xs = [i + k * bw - .4 + bw / 2 for i in range(len(tasks))]
        ys = [data[t].get(s, 0) for t in tasks]
        ax.bar(xs, ys, width=bw * .9, color=cols.get(s, "#999"),
               edgecolor="white", linewidth=2, label=s)
        for x, y in zip(xs, ys):
            ax.text(x, y + .008, f"{y:.3f}", ha="center", va="bottom",
                    fontsize=10, color=MUTED)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_KO.get(t, t) for t in tasks], fontsize=12, color=INK)
    ax.set_ylabel("CER (낮을수록 좋음)", fontsize=11, color=MUTED)
    ax.tick_params(axis="y", colors=MUTED, labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.grid(axis="y", color=RULE, linewidth=.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, ncol=3)
    ax.set_title("과제별 정확도 — 정답 대비 편집거리", fontsize=14,
                 color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "receipt_data/review/train_compare.png"))
    args = ap.parse_args()
    S = Path("/tmp/claude-0/-data-workspace-yjyong/be45946a-421f-4383-9259-48c06f6b862d/scratchpad")
    loss_chart(S / "lc.png")
    data = cer_chart(S / "cc.png")

    parts = [Image.open(S / "lc.png").convert("RGB")]
    if data:
        parts.append(Image.open(S / "cc.png").convert("RGB"))
    W = sum(p.width for p in parts) + 14 * (len(parts) - 1)
    head = 96
    foot = 108
    out = Image.new("RGB", (W, head + max(p.height for p in parts) + foot), "white")
    dr = ImageDraw.Draw(out)
    dr.text((14, 12), "Qwen3.5-4B LoRA 학습 비교", fill=INK, font=ImageFont.truetype(FONT, 26))
    dr.text((14, 50), "학습셋: 공공문서(AIHub 71299) + 영수증(사람 검수 + 합성) · GPU 1장 · LoRA 57.7M/4.60B",
            fill=MUTED, font=ImageFont.truetype(FONT, 15))
    dr.text((14, 72), "v1 3,472건 · 영수증 13.6%      v2 7,208건 · 영수증 16.8%",
            fill=MUTED, font=ImageFont.truetype(FONT, 15))
    dr.line([14, head - 10, W - 14, head - 10], fill=RULE, width=2)
    x = 0
    for p in parts:
        out.paste(p, (x, head))
        x += p.width + 14
    y = head + max(p.height for p in parts) + 12
    dr.line([14, y, W - 14, y], fill=RULE, width=2)
    y += 12
    f14 = ImageFont.truetype(FONT, 14)
    dr.text((14, y), "읽을 때 주의", fill=INK, font=ImageFont.truetype(FONT, 16))
    y += 24
    for t in ["· loss 는 내려갔는데 CER 은 올라갔다. 원인은 반복 생성 — LoRA 가 '처리기간/처리기관'을 무한 반복한다.",
              "· teacher forcing 으로 재는 loss 와 실제로 뽑아 쓰는 생성 품질은 다르다. loss 만 보면 안 된다.",
              "· CER 은 검증 표본 6건(과제별 3건) 기준이다. 표본이 작아 개별 값의 흔들림이 크다.",
              "· 영수증 CER 이 원래 높다 — 마크다운 표 형식까지 맞춰야 해서 평문 전사보다 어렵다."]:
        dr.text((16, y), t, fill=MUTED, font=f14)
        y += 21
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"저장: {args.out}  {out.size}")
    if data:
        for t, v in data.items():
            print(f"  {TASK_KO.get(t,t)}: " + " · ".join(f"{k} {x:.3f}" for k, x in v.items()))


if __name__ == "__main__":
    main()
