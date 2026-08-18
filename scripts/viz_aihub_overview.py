"""
AIHub 71845(학술논문 이해) 전체를 한 장으로 훑는다.

extracted/ 아래 12개 폴더가 무엇인지, 이미지가 어떤 종류인지, 요약 라벨이
실제로 어떻게 생겼는지를 한 화면에 붙인다. 폴더 이름만 봐서는 TL/TS/VL/VS 가
무엇인지 알 수 없고, 이미지 3만 장의 절반 이상이 표라는 것도 세어봐야 안다.

  TL / VL   라벨  — json + PPTX 에서 뽑은 조각 PNG
  TS / VS   원본  — pdf + pptx (born-digital)

집계는 전량 스캔이라 1~2분 걸린다. --stats 로 캐시를 재사용한다.

usage:
    python3 scripts/viz_aihub_overview.py
    python3 scripts/viz_aihub_overview.py --stats /tmp/.../aihub_stats.json
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

ROOT = Path("/data/workspace/yjyong/aihub/71845/extracted")
FONT = "/data/workspace/VLM/gitlab/documentai_api/PaddleOCR/doc/fonts/korean.ttf"

# dataviz 기본 팔레트 categorical 1~3 (light). validate_palette.js 통과 확인함.
C_TA, C_PI, C_CH = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, RULE = "#14171c", "#5d6674", "#dce0e6"

fm.fontManager.addfont(FONT)
KOR = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["font.family"] = KOR
plt.rcParams["axes.unicode_minus"] = False


def scan():
    rows = []
    for d in sorted(ROOT.glob("[TV]L_*")):
        cat = collections.Counter()
        nsec = 0
        files = sorted(glob.glob(str(d / "*.json")))
        for f in files:
            t = json.load(open(f))["training_data_info"]
            nsec += len(t.get("section_info") or [])
            for i in (t.get("image_info") or []):
                cat[i.get("image_category")] += 1
        rows.append(dict(folder=d.name, docs=len(files), sec=nsec,
                         TA=cat["TA"], PI=cat["PI"], CH=cat["CH"]))
    return rows


def short(name):
    """TL_과학기술(ST) -> 'TL 과학기술'"""
    return name.replace("_", " ").replace("인문학,예술체육학", "인문·예술").replace("(ST)", "").replace("(SS)", "").replace("(HA)", "").strip()


def chart(rows, path, w=1400, h=560):
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    labels = [short(r["folder"]) for r in rows][::-1]
    ta = [r["TA"] for r in rows][::-1]
    pi = [r["PI"] for r in rows][::-1]
    ch = [r["CH"] for r in rows][::-1]
    y = range(len(labels))
    # 2px 서피스 간격을 흉내내려고 edgecolor 를 흰색으로 준다
    b1 = ax.barh(y, ta, color=C_TA, height=.62, edgecolor="white", linewidth=2, label="표 (TA)")
    b2 = ax.barh(y, pi, left=ta, color=C_PI, height=.62, edgecolor="white", linewidth=2, label="그림 (PI)")
    b3 = ax.barh(y, ch, left=[a + b for a, b in zip(ta, pi)], color=C_CH, height=.62,
                 edgecolor="white", linewidth=2, label="차트 (CH)")
    # 값은 직접 라벨링한다 (색만으로 식별하지 않게 + 대비 WARN 해소)
    for i in y:
        segs = [(ta[i], 0, C_TA), (pi[i], ta[i], C_PI), (ch[i], ta[i] + pi[i], C_CH)]
        for v, off, _ in segs:
            if v > 700:
                ax.text(off + v / 2, i, f"{v:,}", ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")
        tot = ta[i] + pi[i] + ch[i]
        ax.text(tot + 250, i, f"{tot:,}", ha="left", va="center", color=MUTED, fontsize=11)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=12, color=INK)
    ax.set_xlabel("이미지 수", fontsize=11, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.grid(axis="x", color=RULE, linewidth=.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=11, ncol=3)
    ax.set_title("라벨 폴더별 이미지 구성 — 표가 절반을 넘는다", fontsize=15,
                 color=INK, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def samples(seed=11, per=1, w=430, ih=250):
    """분야별 표(TA) 이미지 한 장씩 + 캡션 앞부분."""
    rng = random.Random(seed)
    out = []
    for d in sorted(ROOT.glob("TL_*")):
        pool = []
        for f in rng.sample(sorted(glob.glob(str(d / "*.json"))), 40):
            j = json.load(open(f))
            for i in (j["training_data_info"].get("image_info") or []):
                if i.get("image_category") == "TA":
                    p = d / Path(i["image_file_name"]).name
                    if p.exists():
                        pool.append((p, i))
        for p, meta in rng.sample(pool, min(per, len(pool))):
            im = Image.new("RGB", (w, ih + 118), "white")
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, w - 1, 24], fill=C_TA)
            dr.text((6, 3), f"표 (TA)   {short(d.name)}", fill="white",
                    font=ImageFont.truetype(FONT, 15))
            pic = Image.open(p).convert("RGB")
            pw, ph = pic.size
            s = min((w - 14) / pw, ih / ph)
            pic = pic.resize((max(1, int(pw * s)), max(1, int(ph * s))))
            im.paste(pic, ((w - pic.width) // 2, 30 + (ih - pic.height) // 2))
            dr.rectangle([7, 30, w - 8, 30 + ih], outline=RULE)
            y = 30 + ih + 8
            dr.text((8, y), "GT = image_caption (서술형)", fill="#c0392b",
                    font=ImageFont.truetype(FONT, 13))
            y += 18
            import textwrap
            cap = (meta.get("image_caption") or "").strip()
            for ln in textwrap.wrap(cap, width=36)[:4]:
                dr.text((9, y), ln, fill=MUTED, font=ImageFont.truetype(FONT, 12))
                y += 15
            dr.rectangle([0, 0, w - 1, im.height - 1], outline="#bbbbbb", width=2)
            out.append(im)
    return out


def para_stats(n_doc=200):
    """단락이 제목인지 본문인지. 제목은 summary_text 가 original_text 와 같다."""
    d = ROOT / "TL_사회과학(SS)"
    same = tot = 0
    ratio = []
    for f in sorted(glob.glob(str(d / "*.json")))[:n_doc]:
        for s in json.load(open(f))["training_data_info"]["section_info"]:
            o = (s.get("original_text") or "").strip()
            m = (s.get("summary_text") or "").strip()
            tot += 1
            if o == m:
                same += 1
            elif o:
                ratio.append(len(m) / len(o))
    return same, tot, (st.median(ratio) if ratio else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="")
    ap.add_argument("--out", default="/data/workspace/yjyong/receipt_data/review/aihub_71845_overview.png")
    args = ap.parse_args()

    if args.stats and Path(args.stats).exists():
        rows = json.loads(Path(args.stats).read_text())
        rows = [{k: r[k] for k in ("folder", "docs", "sec", "TA", "PI", "CH")} for r in rows]
    else:
        rows = scan()

    tmp = Path("/tmp/claude-0/-data-workspace-yjyong/be45946a-421f-4383-9259-48c06f6b862d/scratchpad/aihub_chart.png")
    chart(rows, tmp)
    ch_img = Image.open(tmp).convert("RGB")

    sm = samples()
    pad = 12
    sw = sum(s.width for s in sm) + pad * (len(sm) - 1)
    srow = Image.new("RGB", (sw, max(s.height for s in sm)), "white")
    x = 0
    for s in sm:
        srow.paste(s, (x, 0))
        x += s.width + pad

    same, tot, med = para_stats()
    D = sum(r["docs"] for r in rows)
    S = sum(r["sec"] for r in rows)
    I = sum(r["TA"] + r["PI"] + r["CH"] for r in rows)
    TA = sum(r["TA"] for r in rows)

    W = max(ch_img.width, srow.width, 1400)
    head = 150
    info = 132
    out = Image.new("RGB", (W, head + ch_img.height + srow.height + info + 40), "white")
    dr = ImageDraw.Draw(out)
    f28 = ImageFont.truetype(FONT, 27)
    f17 = ImageFont.truetype(FONT, 17)
    f15 = ImageFont.truetype(FONT, 15)
    f13 = ImageFont.truetype(FONT, 13)

    dr.text((14, 12), "AIHub 71845 — 학술논문 이해 데이터 전체 구성", fill=INK, font=f28)
    dr.text((14, 50), f"논문 {D:,}편 · 단락 {S:,}개 · 이미지 {I:,}장 (표 {TA:,} = {100*TA/I:.0f}%)  ·  25GB",
            fill=MUTED, font=f17)
    dr.text((14, 78), "TL·VL = 라벨(json + PPTX 조각 PNG)   /   TS·VS = 원본(pdf + pptx, born-digital)",
            fill=MUTED, font=f15)
    dr.text((14, 102), "AIHub 공식: 활용 모델 Qwen2-VL-7B · 라벨링 유형 \"내용요약(자연어)\"",
            fill="#c0392b", font=f15)
    dr.line([14, head - 12, W - 14, head - 12], fill=RULE, width=2)

    y = head
    out.paste(ch_img, (0, y))
    y += ch_img.height + 10
    out.paste(srow, (14, y))
    y += srow.height + 18

    dr.line([14, y, W - 14, y], fill=RULE, width=2)
    y += 12
    dr.text((14, y), "요약 라벨의 실체", fill=INK, font=f17)
    y += 26
    pct = 100 * same / tot
    dr.text((16, y), f"· 단락의 {pct:.0f}% 는 제목·소제목이라 summary_text 가 original_text 와 똑같다 "
                     f"(표본 {tot:,}개 중 {same:,}개)", fill=MUTED, font=f15)
    y += 22
    dr.text((16, y), f"· 실제 본문 단락만 보면 요약률 중앙값 {100*med:.1f}% — 원문을 크게 줄인다",
            fill=MUTED, font=f15)
    y += 22
    dr.text((16, y), "· 표(TA)의 GT 는 표 내용이 아니라 서술형 캡션이다. 셀·행·열·값이 없다 → 표 구조 학습 불가",
            fill="#c0392b", font=f15)
    y += 22
    dr.text((16, y), "· 좌표는 EMU(PPTX 슬라이드 단위). 페이지 이미지가 없어 레이아웃 박스를 얹을 판이 없다",
            fill=MUTED, font=f13)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"저장: {args.out}  {out.size}")


if __name__ == "__main__":
    main()
