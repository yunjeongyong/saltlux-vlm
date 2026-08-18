"""
AIHub 71299 다운로드 원본(_raw) 이 무엇을 담고 있고 무엇이 빠졌는지 그린다.

압축을 풀어 놓은 extracted/ 만 보면 "라벨 55만 개"라는 숫자만 보이고,
그중 절반 이상이 짝 이미지가 없다는 사실이 안 보인다. 그 사실은 _raw 의
폴더 구조에 그대로 드러나 있다 — Training/01.원천데이터 폴더가 아예 없다.

파일명 규칙:
    {TL|VL|VS}_OCR(public)_{도메인}_{연대}_{기관코드}_{일련번호}.zip
    연대에는 b1980(1980년 이전) 이 섞여 있다.

usage:
    python3 scripts/viz_aihub_71299_raw.py
"""
import argparse
import collections
import glob
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

RAW = Path("/data/workspace/yjyong/aihub/71299/_raw/023.OCR_데이터(공공)/01-1.정식개방데이터")
FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"
PAT = re.compile(r"^(TL|VL|VS)_OCR\(public\)_([A-Z]+)_(b?\d{4})_(\d+)_(\d+)\.zip$")

# dataviz 기본 categorical 1~2 (light) — validate_palette.js 통과
C_HAVE, C_MISS = "#2a78d6", "#eb6834"
INK, MUTED, RULE = "#14171c", "#5d6674", "#dce0e6"

fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

# 도메인 코드(CST/EN/EV/CT/WF/AF/DI)의 뜻은 라벨 파일 어디에도 없다.
# 표본을 읽어봐도 민원서류·위생단속·자재관리 등이 섞여 있어 단정할 수 없다.
# 추측한 한글명을 붙이면 잘못된 정보가 퍼지므로 코드 그대로 쓴다.


def scan():
    rows = collections.Counter()
    size = collections.Counter()
    grid = collections.Counter()
    dom = collections.Counter()
    for p in glob.glob(str(RAW / "**" / "*.zip"), recursive=True):
        m = PAT.match(os.path.basename(p))
        if not m:
            continue
        pre, d, yr, _, _ = m.groups()
        rows[pre] += 1
        dom[d] += 1
        if pre == "TL":
            grid[(d, yr)] += 1
        size[pre] += sum(os.path.getsize(f) for f in glob.glob(p[:-4] + ".zip*"))
    return rows, size, grid, dom


def chart(grid, dom, path, w=1340, h=470):
    order = ["b1980", "1980", "1990", "2000", "2010"]
    doms = [d for d, _ in dom.most_common() if d != "DI"] + ["DI"]
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bottom = [0] * len(doms)
    # 연대는 순서가 있는 축이라 한 색의 명암으로 간다 (categorical 이 아님)
    shades = ["#c3d9f4", "#93bcec", "#5f9be2", "#2a78d6", "#1b5296"]
    for k, yr in enumerate(order):
        vals = [grid[(d, yr)] for d in doms]
        ax.bar(doms, vals, bottom=bottom,
               color=shades[k], edgecolor="white", linewidth=2,
               label=("~1979" if yr == "b1980" else f"{yr}년대"))
        for i, v in enumerate(vals):
            if v >= 8:
                ax.text(i, bottom[i] + v / 2, str(v), ha="center", va="center",
                        color="white" if k >= 3 else INK, fontsize=10, fontweight="bold")
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, b in enumerate(bottom):
        ax.text(i, b + 2, str(b), ha="center", va="bottom", color=MUTED, fontsize=11)
    ax.set_ylabel("압축파일 수", fontsize=11, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.grid(axis="y", color=RULE, linewidth=.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, ncol=5, loc="upper right")
    ax.set_title("학습 라벨(TL) 385개 묶음의 분야 · 연대 분포", fontsize=15,
                 color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def tree_panel(rows, size, w, h):
    im = Image.new("RGB", (w, h), "white")
    dr = ImageDraw.Draw(im)
    f20 = ImageFont.truetype(FONT, 19)
    f15 = ImageFont.truetype(FONT, 15)
    f14 = ImageFont.truetype(FONT, 14)
    y = 6
    dr.text((4, y), "다운로드 구조 — 무엇이 빠졌는가", fill=INK, font=f20)
    y += 32
    lines = [
        ("01-1.정식개방데이터/", None, 0),
        ("├── Training/", None, 0),
        (f"│   ├── 01.원천데이터/          없음 — 미다운로드 (약 133GB)", "miss", 0),
        (f"│   └── 02.라벨링데이터/        zip {rows['TL']}개 · {size['TL']/1e9:.1f}GB  →  라벨 489,681", "have", 0),
        ("└── Validation/", None, 0),
        (f"    ├── 01.원천데이터/          zip {rows['VS']}개 · {size['VS']/1e9:.1f}GB  →  이미지 64,228", "have", 0),
        (f"    └── 02.라벨링데이터/        zip {rows['VL']}개 · {size['VL']/1e9:.1f}GB  →  라벨 64,228", "have", 0),
    ]
    for txt, kind, _ in lines:
        col = C_MISS if kind == "miss" else (INK if kind == "have" else MUTED)
        dr.text((10, y), txt, fill=col, font=f14)
        y += 23
    y += 10
    dr.line([4, y, w - 4, y], fill=RULE, width=2)
    y += 12
    dr.text((4, y), "학습 라벨 489,681건은 짝 이미지가 없어 그대로는 못 쓴다.",
            fill=C_MISS, font=f15)
    y += 24
    dr.text((4, y), "실제 사용 가능: 검증셋 64,228쌍 (라벨↔이미지 100% 일치 확인)",
            fill="#0f7a4d", font=f15)
    y += 24
    dr.text((4, y), "파일명 = {TL|VL|VS}_OCR(public)_{분야코드}_{연대}_{기관}_{일련}.zip"
                    "   · 연대에 b1980(1979년 이전) 포함",
            fill=MUTED, font=f14)
    y += 22
    dr.text((4, y), "※ 분야코드(CST/EN/EV/CT/WF/AF/DI)의 뜻은 라벨 파일에 없다. AIHub 문서 확인 필요",
            fill=MUTED, font=f14)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/workspace/yjyong/receipt_data/review/aihub_71299_raw.png")
    args = ap.parse_args()

    rows, size, grid, dom = scan()
    tmp = Path("/tmp/claude-0/-data-workspace-yjyong/be45946a-421f-4383-9259-48c06f6b862d/scratchpad/raw_chart.png")
    chart(grid, dom, tmp)
    ch = Image.open(tmp).convert("RGB")

    W = max(ch.width, 1340)
    head = 96
    tp = tree_panel(rows, size, W - 28, 320)
    out = Image.new("RGB", (W, head + tp.height + ch.height + 30), "white")
    dr = ImageDraw.Draw(out)
    dr.text((14, 12), "AIHub 71299 대규모 OCR 데이터(공공) — 다운로드 원본", fill=INK,
            font=ImageFont.truetype(FONT, 26))
    total = sum(size.values())
    dr.text((14, 50), f"_raw 총 {total/1e9:.0f}GB · zip {sum(rows.values())}개  "
                      f"(TL {rows['TL']} / VL {rows['VL']} / VS {rows['VS']})",
            fill=MUTED, font=ImageFont.truetype(FONT, 16))
    dr.line([14, head - 12, W - 14, head - 12], fill=RULE, width=2)
    out.paste(tp, (14, head))
    out.paste(ch, (0, head + tp.height + 14))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"저장: {args.out}  {out.size}")


if __name__ == "__main__":
    main()
