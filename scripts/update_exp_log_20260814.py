"""실험로그 xlsx 를 현행 평가셋(978건) 재채점 결과로 갱신한다.

시트에 적힌 수치는 평가셋 정제 전(1,036건 / 정답 미정정) 기준이라 지금 보고서와
어긋난다. review/asis_tobe.json 으로 다시 잰 값으로 바꾸고, 빠져 있던 exp_002
(출발점 맨몸)를 채운다.

usage: python3 scripts/update_exp_log_20260814.py <xlsx>
"""
import shutil
import sys
from copy import copy
from datetime import datetime
from pathlib import Path

import openpyxl

SRC = Path(sys.argv[1])
BAK = SRC.with_suffix(f".bak_{datetime.now():%m%d_%H%M}.xlsx")

EVALSET = "영수증 크롭 978 (text 881 / table 97) — 90장. 문서 8장 제외" \
          "(receipt34·67 좌표뒤집힘, 36·38·53·73·81·91 검수판정)"

# 현행 978건 정답으로 재채점한 값 (scripts/score_asis_tobe.py)
M = {
    "exp_001": dict(cer_med=0, cer=0.1990, long=45, teds=0.7032, exact=59.1,
                    macro=0.7715, micro=0.8263),
    "exp_002": dict(cer_med=0, cer=0.1671, long=33, teds=0.6628, exact=60.6,
                    macro=0.7614, micro=0.8405),
    "exp_003": dict(cer_med=0, cer=0.1191, long=11, teds=0.7286, exact=59.6,
                    macro=0.8089, micro=0.8732),
}
COL = {}          # 헤더명 -> 열번호


def setc(ws, row, header, value):
    c = ws.cell(row, COL[header])
    old = c.value
    c.value = value
    if str(old) != str(value):
        print(f"    {header:16} {str(old)[:40]!r} → {str(value)[:60]!r}")


def main():
    shutil.copy2(SRC, BAK)
    print(f"백업: {BAK.name}\n")
    wb = openpyxl.load_workbook(SRC)
    ws = wb["실험로그"]
    for i, c in enumerate(ws[2], 1):
        if c.value:
            COL[str(c.value)] = i

    rowof = {ws.cell(r, 1).value: r for r in range(3, ws.max_row + 1)
             if ws.cell(r, 1).value}

    # ── exp_002 (출발점 맨몸) 행 추가 — exp_001 바로 아래 ──────────────
    if "exp_002" not in rowof:
        at = rowof["exp_001"] + 1
        ws.insert_rows(at)
        for c in range(1, ws.max_column + 1):                # 서식 물려받기
            src = ws.cell(rowof["exp_001"], c)
            dst = ws.cell(at, c)
            dst._style = copy(src._style)
        print(f"[exp_002] 행 추가 (행 {at}) — 출발점 Qwen3.6-35B-A3B 맨몸")
        vals = {
            "exp_id": "exp_002", "날짜": "2026-08-12",
            "가설/목적": "학습의 실제 출발점(맨몸) 기준선. 베이스와 같은 가중치를 "
                     "서빙 설정 없이 호출했을 때의 성능",
            "상태": "완료",
            "베이스모델": "Qwen3.6-35B-A3B (맨몸, 어댑터 없음)",
            "태스크": "VLM(측정만)", "학습데이터(구성)": "— (측정만)",
            "GPU": "H100 x2", "소요시간": "—",
            "checkpoint경로": "ml/models/Qwen3.6-35B-A3B",
            "평가결과_json": "receipt_data/review/eval_crops_base.json",
            "메모": "exp_001 과 같은 가중치를 서빙 설정 없이 호출한 것. 두 행의 차이가 "
                  "서빙 설정(프롬프트·디코딩·후처리) 효과다. 매크로에서는 exp_001 보다 "
                  "낮다(0.7614 < 0.7715) — 서빙 설정이 표를 올리고 텍스트를 낮춘다.",
        }
        for k, v in vals.items():
            ws.cell(at, COL[k]).value = v
        rowof = {ws.cell(r, 1).value: r for r in range(3, ws.max_row + 1)
                 if ws.cell(r, 1).value}

    # ── 지표·평가셋 갱신 ───────────────────────────────────────────────
    for exp, m in M.items():
        r = rowof.get(exp)
        if not r:
            print(f"  ✗ {exp} 행 없음")
            continue
        print(f"\n[{exp}] 행 {r}")
        setc(ws, r, "평가셋", EVALSET)
        setc(ws, r, "평가건수", 978)
        setc(ws, r, "CER_중앙값", m["cer_med"])
        setc(ws, r, "CER_평균(참고)", m["cer"])
        setc(ws, r, "CER>1(폭주)", m["long"])
        setc(ws, r, "TEDS", m["teds"])
        setc(ws, r, "완전일치%", m["exact"])
        setc(ws, r, "AVG_매크로", m["macro"])
        setc(ws, r, "AVG_마이크로", m["micro"])

    # ── exp_003 — base 대비 / 체크포인트 ──────────────────────────────
    r = rowof["exp_003"]
    print(f"\n[exp_003] 보정")
    setc(ws, r, "base대비Δ",
         "베이스(exp_001) 대비 CER −40.2% / TEDS +3.6% / 긴오답 45→11건. "
         "출발점(exp_002) 대비 CER −28.7% / TEDS +9.9%")
    setc(ws, r, "checkpoint경로",
         "ml/runs/upload-mix-qwen36 (최종 step1124). "
         "ckpt-1000 재채점: CER 0.1163 / TEDS 0.7243 / 긴오답 10 / "
         "완전일치 60.4% / 매크로 0.8075 / 마이크로 0.8742")
    setc(ws, r, "메모",
         "★베이스(exp_001) 전 지표 초과. 지표 최고 체크포인트는 마이크로 기준 "
         "ckpt-1000(0.8742), 매크로 기준 최종(0.8089) — 차이 0.001 로 사실상 동률. "
         "eval_loss 는 step250 이후 단조상승(0.1628→0.2294)했으나 생성 품질은 "
         "ckpt-1000 이 최종보다 좋아, loss 로 과적합을 판정하면 안 된다. "
         "표 크롭 학습은 0건(표는 영수증 페이지 마크다운 안 1,424행에만 존재).")

    # ── exp_004 — 실제 학습 파일과 진행률 ─────────────────────────────
    r = rowof["exp_004"]
    print(f"\n[exp_004] 보정")
    setc(ws, r, "학습데이터(구성)",
         "exp004c_260813 11,173행 — exp004b 11,298행에서 5,000토큰 초과 125행 제외. "
         "표크롭 5,533(pubtabnet 5,461+영수증72) / 문서전사 2,499 / 페이지OCR 2,000 / "
         "영수증 텍스트크롭 820 / 영수증 페이지 182 / 공공문서·논문 139")
    setc(ws, r, "소요시간",
         "진행중 2,106/2,794 step (75%, 31.24s/it). 8/13 06:54 UTC 시작, "
         "잔여 약 6시간 (8/14 12:00 UTC 전후 완료 예상)")
    setc(ws, r, "평가셋", EVALSET)

    wb.save(SRC)
    print(f"\n저장: {SRC}")


if __name__ == "__main__":
    main()
