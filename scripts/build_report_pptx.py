"""exp_003 결과 + 평가셋 정제 보고 PPT.

두 주제를 한 덱으로 묶는다 — 평가셋을 정제한 덕에 exp_003 의 개선폭이 더 크게
드러났으므로 따로 떼면 오히려 설명이 어려워진다.

usage: python3 scripts/build_report_pptx.py <출력.pptx>
"""
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Pt

OUT = Path(sys.argv[1])

INK = RGBColor(0x19, 0x1B, 0x1E)
DIM = RGBColor(0x63, 0x68, 0x6E)
RULE = RGBColor(0xDF, 0xDF, 0xD9)
RAIL = RGBColor(0xEC, 0xEA, 0xE5)
PAPER = RGBColor(0xF6, 0xF6, 0xF3)
GOOD = RGBColor(0x1F, 0x6B, 0x4A)
BAD = RGBColor(0x9D, 0x3A, 0x27)
ACC = RGBColor(0x2C, 0x56, 0x78)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "맑은 고딕"

W, H = Emu(12192000), Emu(6858000)          # 16:9
M = Emu(760000)                              # 좌우 여백


def cm(v):
    return Emu(int(v * 360000))


def deck():
    p = Presentation()
    p.slide_width, p.slide_height = W, H
    return p


def blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(1, 0, 0, W, H)
    bg.fill.solid(); bg.fill.fore_color.rgb = PAPER
    bg.line.fill.background(); bg.shadow.inherit = False
    return s


def text(s, x, y, w, h, txt, size=18, bold=False, color=INK,
         align=PP_ALIGN.LEFT, space=0):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(txt.split("\n")):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(space)
        r = para.add_run(); r.text = line
        f = r.font
        f.name, f.size, f.bold, f.color.rgb = FONT, Pt(size), bold, color
    return tb


def rule(s, y, x=M, w=None):
    w = w or (W - 2 * M)
    ln = s.shapes.add_shape(1, x, y, w, Emu(12000))
    ln.fill.solid(); ln.fill.fore_color.rgb = INK
    ln.line.fill.background(); ln.shadow.inherit = False


def header(s, eyebrow, title):
    text(s, M, cm(1.5), W - 2 * M, cm(0.7), eyebrow, 12, True, ACC)
    text(s, M, cm(2.1), W - 2 * M, cm(1.4), title, 30, True, INK)
    rule(s, cm(3.5))


def table(s, x, y, cols, rows, widths, size=14, head_size=12,
          hi_col=None, hi_color=GOOD):
    """가벼운 표. pptx 기본 표는 테두리가 요란해서 도형으로 직접 그린다."""
    rh = cm(0.95)
    # 헤더
    hb = s.shapes.add_shape(1, x, y, sum(widths), rh)
    hb.fill.solid(); hb.fill.fore_color.rgb = RAIL
    hb.line.fill.background(); hb.shadow.inherit = False
    cx = x
    for c, wd in zip(cols, widths):
        al = PP_ALIGN.RIGHT if c.startswith("→") else PP_ALIGN.LEFT
        text(s, cx + cm(0.3), y + cm(0.26), wd - cm(0.6), rh,
             c.lstrip("→"), head_size, True, DIM, al)
        cx += wd
    # 본문
    for ri, row in enumerate(rows):
        ry = y + rh * (ri + 1)
        band = s.shapes.add_shape(1, x, ry, sum(widths), rh)
        band.fill.solid(); band.fill.fore_color.rgb = WHITE
        band.line.color.rgb = RULE; band.line.width = Pt(0.75)
        band.shadow.inherit = False
        cx = x
        for ci, (val, wd) in enumerate(zip(row, widths)):
            al = PP_ALIGN.RIGHT if cols[ci].startswith("→") else PP_ALIGN.LEFT
            col = hi_color if (hi_col is not None and ci == hi_col) else INK
            bold = hi_col is not None and ci == hi_col
            text(s, cx + cm(0.3), ry + cm(0.24), wd - cm(0.6), rh,
                 str(val), size, bold, col, al)
            cx += wd
    return y + rh * (len(rows) + 1)


def stat(s, x, y, w, label, value, note, color=INK):
    box = s.shapes.add_shape(1, x, y, w, cm(2.6))
    box.fill.solid(); box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = RULE; box.line.width = Pt(0.75)
    box.shadow.inherit = False
    text(s, x + cm(0.45), y + cm(0.32), w - cm(0.9), cm(0.5), label, 12, False, DIM)
    text(s, x + cm(0.45), y + cm(0.82), w - cm(0.9), cm(1.0), value, 30, True, color)
    text(s, x + cm(0.45), y + cm(1.95), w - cm(0.9), cm(0.5), note, 11, False, DIM)


def bullets(s, x, y, w, items, size=15, gap=0.95):
    for i, (head, body) in enumerate(items):
        yy = y + cm(gap * i * 1.75)
        bar = s.shapes.add_shape(1, x, yy + cm(0.08), Emu(30000), cm(0.55))
        bar.fill.solid(); bar.fill.fore_color.rgb = ACC
        bar.line.fill.background(); bar.shadow.inherit = False
        text(s, x + cm(0.35), yy, w, cm(0.6), head, size, True, INK)
        text(s, x + cm(0.35), yy + cm(0.72), w, cm(1.0), body, 13, False, DIM)


# ── 슬라이드 ────────────────────────────────────────────────────
def s_title(prs):
    s = blank(prs)
    text(s, M, cm(4.3), W - 2 * M, cm(0.7),
         "VLM 문서파싱 · 2026-08-13", 14, True, ACC)
    text(s, M, cm(5.1), W - 2 * M, cm(2.4),
         "파인튜닝으로 베이스 모델을 넘었습니다", 42, True, INK)
    text(s, M, cm(7.9), cm(20), cm(1.4),
         "그리고 평가 데이터셋의 정답 자체가 검수되지 않았다는 것을 발견해 정제했습니다.",
         17, False, DIM)
    rule(s, cm(9.6), M, cm(6))
    text(s, M, cm(10.1), cm(20), cm(0.6),
         "exp_003 결과 · 평가셋 정제 · exp_004 계획", 13, False, DIM)


def s_result(prs):
    s = blank(prs)
    header(s, "실험 결과", "전 지표에서 베이스 모델을 넘었습니다")
    y = cm(4.2)
    wcol = [cm(6.2), cm(4.2), cm(4.2), cm(5.4)]
    y2 = table(s, M, y, ["지표", "→베이스 모델", "→학습 모델", "→개선"], [
        ["긴 오답 (CER>1) ↓", "45건", "11건", "−75.6%"],
        ["CER 평균 ↓", "0.2009", "0.1210", "−39.8%"],
        ["TEDS (표) ↑", "0.7032", "0.7286", "+3.6%"],
        ["종합 정확도 (AVG) ↑", "0.8246", "0.8715", "+5.7%"],
        ["완전일치 % ↑", "58.1", "58.5", "+0.4p"],
    ], wcol, hi_col=3)
    text(s, M, y2 + cm(0.5), W - 2 * M, cm(1.2),
         "평가셋 영수증 크롭 978건 (텍스트 881 / 표 97) · 세 모델 동일 조건\n"
         "베이스 luxia-document-parsing-high (사내 서빙)  ·  "
         "학습 Qwen3.6-35B-A3B + LoRA", 13, False, DIM, space=4)


def s_interpret(prs):
    s = blank(prs)
    header(s, "해석", "무엇이 좋아졌나 — 그리고 무엇이 아닌가")
    bullets(s, M, cm(4.3), cm(28), [
        ("핵심은 긴 오답이 75% 줄어든 것입니다",
         "정답보다 길게 뱉어 쓸 수 없던 출력이 45건에서 11건으로 줄었습니다. "
         "실사용에서 가장 치명적인 실패가 사라졌습니다."),
        ("다만 “더 잘 읽게” 된 것은 아닙니다",
         "중앙값 CER은 두 모델 모두 0.0이고 완전일치율은 58.1 → 58.5로 제자리입니다. "
         "“불필요한 출력을 덜 뱉게” 된 개선에 가깝습니다."),
        ("표는 소폭 올랐고, 효과는 아직 미검증입니다",
         "표를 크롭 단위로 학습한 적이 없는데도 TEDS가 올랐습니다. 출력 형식을 평가에 "
         "맞춘 효과로 보이며, 표 크롭 자체의 효과는 exp_004에서 확인합니다."),
        ("체크포인트는 loss가 아니라 CER/TEDS로 골라야 합니다",
         "eval_loss는 학습 후반에 계속 올랐지만 실제 생성 품질은 유지됐습니다. "
         "teacher-forcing loss가 생성 품질을 대변하지 못한다는 것이 확인됐습니다."),
    ])


def s_dataset(prs):
    s = blank(prs)
    header(s, "평가 데이터셋 수정", "정답이 검수되지 않은 상태였습니다")
    box = s.shapes.add_shape(1, M, cm(4.2), W - 2 * M, cm(3.5))
    box.fill.solid(); box.fill.fore_color.rgb = RAIL
    box.line.fill.background(); box.shadow.inherit = False
    bar = s.shapes.add_shape(1, M, cm(4.2), Emu(38000), cm(3.5))
    bar.fill.solid(); bar.fill.fore_color.rgb = BAD
    bar.line.fill.background(); bar.shadow.inherit = False
    text(s, M + cm(0.6), cm(4.6), W - 2 * M - cm(1.2), cm(2.8),
         "평가셋 정답 파일이 status: 200, documentId 를 가진 "
         "Document Studio API 응답 그대로였습니다.\n"
         "검수본을 수집하는 collect_labeled.py 는 train 만 대상으로 하고, "
         "val/test 파일의 수정 시각은 검수 시작보다 앞섭니다.\n"
         "즉 학습 데이터는 검수본, 평가 데이터는 파서 출력이라는 비대칭이 있었습니다.",
         14, False, INK, space=6)
    x = M
    wd = (W - 2 * M - cm(1.2)) // 3
    stat(s, x, cm(8.4), wd, "발견 방법",
         "3모델 합의", "세 모델이 일치하는데 정답만 다른 크롭 72건", ACC)
    stat(s, x + wd + cm(0.6), cm(8.4), wd, "정답 정정", "22곳",
         "오독 · 자모 파편 · 어미 누락", GOOD)
    stat(s, x + (wd + cm(0.6)) * 2, cm(8.4), wd, "문서 제외", "8장",
         "98장 → 90장 · 크롭 978건", BAD)


def s_drops(prs):
    s = blank(prs)
    header(s, "평가 데이터셋 수정", "제외한 문서 8장과 사유")
    wcol = [cm(4.6), cm(3.0), cm(12.4)]
    y = table(s, M, cm(4.2), ["문서", "→크롭", "제외 사유"], [
        ["receipt34 · 67", "41", "좌표계가 90도 뒤집힘 — 이미지는 가로, json 은 세로"],
        ["receipt73", "10", "손글씨가 섞여 있어 전사 정답을 신뢰할 수 없음"],
        ["receipt53", "6", "정답 누락 다수 · 원본이 한 번 분실됐다 복구된 건"],
        ["receipt36 · 38 · 81 · 91", "42", "검수 판정으로 제외"],
    ], wcol, size=13)
    text(s, M, y + cm(0.5), W - 2 * M, cm(1.6),
         "지우지 않고 receipt_data/_removed/ 로 옮겨 두었습니다 — 원본·라벨·크롭이 "
         "그대로 있어 되돌릴 수 있습니다.\n"
         "정정은 scripts/fix_gt.py 로 했습니다. 문서를 지정해 정답 필드만 바꾸고 "
         "모델 예측은 건드리지 않습니다.", 13, False, DIM, space=5)


def s_effect(prs):
    s = blank(prs)
    header(s, "평가 데이터셋 수정", "정제하니 개선폭이 더 크게 드러났습니다")
    wcol = [cm(6.4), cm(4.6), cm(4.6), cm(4.4)]
    y = table(s, M, cm(4.2),
              ["지표", "→정제 전 1,036건", "→정제 후 978건", "→차이"], [
        ["베이스 CER", "0.2276", "0.2009", "−0.0267"],
        ["학습 모델 CER", "0.1423", "0.1210", "−0.0213"],
        ["CER 개선폭", "−37.5%", "−39.8%", "+2.3p"],
        ["긴 오답 개선폭", "−64.9%", "−75.6%", "+10.7p"],
        ["TEDS 개선폭", "+2.8%", "+3.6%", "+0.8p"],
    ], wcol, hi_col=2, hi_color=GOOD)
    text(s, M, y + cm(0.5), W - 2 * M, cm(1.4),
         "세 모델이 모두 같은 방향으로 좋아졌습니다. 정답이 부실해 부당하게 "
         "감점되던 부분이 사라진 결과이고, 모델 사이의 우열은 바뀌지 않습니다.\n"
         "앞으로는 978건을 기준으로 고정합니다.", 13, False, DIM, space=5)


def s_next(prs):
    s = blank(prs)
    header(s, "다음 실험", "exp_004 — 데이터만 바꿉니다")
    wcol = [cm(9.0), cm(4.4), cm(4.4)]
    y = table(s, M, cm(4.2), ["소스", "→exp_003", "→exp_004"], [
        ["표 크롭 (pubtabnet + 영수증)", "0", "5,533"],
        ["문서 전사 (multimodal)", "0", "2,499"],
        ["페이지 OCR (aihub)", "3,000", "2,000"],
        ["영수증 텍스트 크롭", "0", "820"],
        ["영수증 페이지 (반복 8회 → 1회)", "1,496", "182"],
        ["합계", "4,496", "11,173"],
    ], wcol, size=13, hi_col=2, hi_color=ACC)
    text(s, M, y + cm(0.45), W - 2 * M, cm(1.8),
         "학습 데이터의 절반(49.5%)이 표 크롭입니다. exp_003에는 한 건도 없던 "
         "데이터이고, 표 크롭이 실제로 TEDS를 올리는지 확인합니다.\n"
         "유효 배치와 epoch는 exp_003과 동일하게 유지해 데이터 효과만 드러나게 "
         "했습니다. 평가는 같은 978건으로 합니다.", 13, False, DIM, space=5)


def s_todo(prs):
    s = blank(prs)
    header(s, "이후 계획", "평가가 실력을 재도록 만드는 일이 남았습니다")
    bullets(s, M, cm(4.3), cm(28), [
        ("학습 데이터의 구두점·공백 표기를 평가 규약에 맞춥니다",
         "학습 후 나빠진 122건의 대부분이 콜론·공백을 빠뜨린 것이었습니다. "
         "학습 데이터의 정답 표기가 평가셋과 달라 생긴 일이라, 데이터 정제만으로 되돌립니다."),
        ("크롭 패딩을 다시 잡습니다",
         "긴 오답 중 상당수는 크롭에 이웃 줄이 딸려 들어가 생긴 것입니다. "
         "패딩을 줄여 다시 뜨면 평가가 실제 능력을 재게 됩니다 — 기준선 재측정이 필요합니다."),
        ("독립 평가셋을 분리합니다",
         "현재 val과 test가 같은 파일입니다. 체크포인트를 고른 셋으로 최종 성능을 "
         "재고 있어 과적합을 걸러낼 수 없습니다."),
        ("한국어 영수증·계산서를 확대합니다",
         "평가셋이 90장까지 줄었습니다. 검수를 거친 데이터를 늘려 평가의 신뢰도를 높입니다."),
    ])


def main():
    prs = deck()
    for f in (s_title, s_result, s_interpret, s_dataset, s_drops, s_effect,
              s_next, s_todo):
        f(prs)
    prs.save(OUT)
    print(f"저장: {OUT} ({OUT.stat().st_size/1e6:.2f} MB, {len(prs.slides.__iter__.__self__._sldIdLst)}장)")


if __name__ == "__main__":
    main()
