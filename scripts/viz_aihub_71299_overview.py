"""
AIHub 71299(대규모 OCR 데이터, 공공) 라벨의 성격을 한 장으로 훑는다.

"OCR 데이터"라는 이름만으로는 무엇이 들어있는지 알 수 없다. 실제로는
**어절 단위 박스**(중앙값 3자)가 한 장에 46개씩 붙어 있고, 서체가 4종으로
나뉘며, 대부분이 300dpi 스캔이다. 그 성격이 우리 영수증 데이터와 어떻게
다른지가 이 그림의 목적이다.

usage:
    python3 scripts/viz_aihub_71299_overview.py
    python3 scripts/viz_aihub_71299_overview.py --sample 800
"""
import argparse
import collections
import glob
import json
import random
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/data/workspace/yjyong/aihub/71299/extracted")
FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"

# dataviz 기본 categorical 1~4 (light). validate_palette.js 통과 (대비 WARN 은 직접 라벨로 해소)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQ = "#2a78d6"
INK, MUTED, RULE = "#14171c", "#5d6674", "#dce0e6"

fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False


def scan(n, seed):
    fs = glob.glob(str(ROOT / "VL_*" / "*.json"))
    fs = random.Random(seed).sample(fs, min(n, len(fs)))
    nb, tl = [], []
    tf = collections.Counter()
    for f in fs:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        b = d.get("Bbox") or []
        nb.append(len(b))
        for x in b:
            tf[x.get("typeface")] += 1
            tl.append(len(str(x.get("data", ""))))
    return nb, tl, tf, len(fs)


def charts(nb, tl, tf, path, w=1340, h=380):
    fig, axes = plt.subplots(1, 3, figsize=(w / 100, h / 100), dpi=100,
                             gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(RULE)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.grid(axis="y", color=RULE, linewidth=.8)
        ax.set_axisbelow(True)

    # 1) 장당 박스 수 — 분포는 한 색(크기 축)
    ax = axes[0]
    ax.hist([v for v in nb if v <= 200], bins=28, color=SEQ, edgecolor="white", linewidth=1)
    med = st.median(nb)
    ax.axvline(med, color="#c0392b", linewidth=2)
    ax.text(med + 4, ax.get_ylim()[1] * .88, f"중앙값 {med:.0f}", color="#c0392b", fontsize=11)
    ax.set_title("장당 글자박스 수", fontsize=13, color=INK, loc="left", pad=10)
    ax.set_xlabel("박스 수 (200 이하만 표시)", fontsize=10, color=MUTED)

    # 2) 박스 텍스트 길이
    ax = axes[1]
    ax.hist([v for v in tl if v <= 20], bins=20, range=(0, 20), color=SEQ,
            edgecolor="white", linewidth=1)
    m2 = st.median(tl)
    ax.axvline(m2, color="#c0392b", linewidth=2)
    ax.text(m2 + .4, ax.get_ylim()[1] * .88, f"중앙값 {m2:.0f}자", color="#c0392b", fontsize=11)
    ax.set_title("박스 하나에 담긴 글자 수 — 어절 단위다", fontsize=13, color=INK, loc="left", pad=10)
    ax.set_xlabel("글자 수", fontsize=10, color=MUTED)

    # 3) typeface — 4종 (의미는 문서 미기재)
    ax = axes[2]
    keys = sorted(tf, key=lambda k: -tf[k])
    vals = [tf[k] for k in keys]
    tot = sum(vals)
    bars = ax.bar([f"typeface {k}" for k in keys], vals,
                  color=[CAT[i % 4] for i in range(len(keys))],
                  edgecolor="white", linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{100*v/tot:.0f}%",
                ha="center", va="bottom", color=MUTED, fontsize=11)
    ax.set_title("서체 구분 4종", fontsize=13, color=INK, loc="left", pad=10)
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def samples(per_dom=1, seed=7, w=300, h=380):
    """도메인별 이미지 한 장씩. 박스를 얹어 라벨 밀도를 보인다."""
    rng = random.Random(seed)
    out = []
    for dom in ["CST", "EN", "EV", "CT", "WF", "AF", "DI"]:
        imgs = glob.glob(str(ROOT / f"VS_*_{dom}_*" / "*.jpg"))
        if not imgs:
            continue
        lab = {Path(p).stem: p for p in glob.glob(str(ROOT / f"VL_*_{dom}_*" / "*.json"))}
        rng.shuffle(imgs)
        for ip in imgs[:60]:
            lp = lab.get(Path(ip).stem)
            if not lp:
                continue
            d = json.loads(Path(lp).read_text(encoding="utf-8"))
            boxes = d.get("Bbox") or []
            if len(boxes) < 12:
                continue
            im = Image.open(ip).convert("RGB")
            W, H = im.size
            s = h / H
            im = im.resize((max(1, int(W * s)), h))
            dr = ImageDraw.Draw(im)
            for b in boxes:
                try:
                    x1, y1 = min(b["x"]) * s, min(b["y"]) * s
                    x2, y2 = max(b["x"]) * s, max(b["y"]) * s
                except Exception:
                    continue
                dr.rectangle([x1, y1, x2, y2],
                             outline=CAT[(b.get("typeface", 1) - 1) % 4], width=1)
            cell = Image.new("RGB", (w, h + 46), "white")
            cell.paste(im, ((w - im.width) // 2, 42))
            c = ImageDraw.Draw(cell)
            c.text((3, 2), dom, fill=INK, font=ImageFont.truetype(FONT, 17))
            c.text((3, 22), f"박스 {len(boxes)}개 · {W}x{H}", fill=MUTED,
                   font=ImageFont.truetype(FONT, 12))
            out.append(cell)
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="/data/workspace/yjyong/receipt_data/review/aihub_71299_overview.png")
    args = ap.parse_args()

    nb, tl, tf, n = scan(args.sample, args.seed)
    tmp = Path("/tmp/claude-0/-data-workspace-yjyong/be45946a-421f-4383-9259-48c06f6b862d/scratchpad/c71299.png")
    charts(nb, tl, tf, tmp)
    ch = Image.open(tmp).convert("RGB")

    sm = samples()
    pad = 8
    srow = Image.new("RGB", (sum(s.width for s in sm) + pad * (len(sm) - 1),
                             max(s.height for s in sm)), "white")
    x = 0
    for s in sm:
        srow.paste(s, (x, 0))
        x += s.width + pad

    W = max(ch.width, srow.width, 1340)
    head = 118
    foot = 96
    out = Image.new("RGB", (W, head + ch.height + srow.height + foot + 30), "white")
    dr = ImageDraw.Draw(out)
    f26 = ImageFont.truetype(FONT, 25)
    f16 = ImageFont.truetype(FONT, 16)
    f14 = ImageFont.truetype(FONT, 14)
    dr.text((14, 12), "AIHub 71299 대규모 OCR 데이터(공공) — 라벨의 성격", fill=INK, font=f26)
    dr.text((14, 48), f"검증셋 표본 {n:,}장 · 전부 300dpi 스캔 · 이미지 중앙값 2450x3420px",
            fill=MUTED, font=f16)
    dr.text((14, 72), "실제 사용 가능 64,228쌍 (라벨↔이미지 100% 일치) · 학습 원천 385묶음 133GB 내려받는 중",
            fill=MUTED, font=f14)
    dr.text((14, 92), "AIHub  https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71299",
            fill="#2a78d6", font=f14)
    dr.line([14, head - 10, W - 14, head - 10], fill=RULE, width=2)

    y = head
    out.paste(ch, (0, y))
    y += ch.height + 8
    dr.text((14, y), "분야별 샘플 — 박스 색은 typeface (도메인 코드의 뜻은 라벨 파일에 없다)",
            fill=INK, font=f16)
    y += 26
    out.paste(srow, (14, y))
    y += srow.height + 16
    dr.line([14, y, W - 14, y], fill=RULE, width=2)
    y += 12
    dr.text((14, y), "우리 영수증 데이터와 무엇이 다른가", fill=INK, font=f16)
    y += 24
    for t, c in [
        ("· 박스가 어절 단위(중앙값 3자)다. 우리 파서는 블록 단위라 입도가 두 단계 다르다.", MUTED),
        ("· 좌표가 원본 픽셀이라 환산이 필요 없다. 우리 파서 출력은 처리 해상도 기준이라 환산해야 한다.", MUTED),
        ("· id 순서가 곧 읽기 순서다(표본 40장 전수 확인). 페이지 단위 정답을 만들 수 있다.", "#0f7a4d"),
    ]:
        dr.text((16, y), t, fill=c, font=f14)
        y += 22

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"저장: {args.out}  {out.size}")


if __name__ == "__main__":
    main()
