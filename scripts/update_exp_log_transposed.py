"""실험로그 시트를 transpose 2장 제외 기준으로 갱신한다.

receipt34/receipt67 은 이미지와 라벨의 좌표계가 90° 어긋나 크롭 41건이 엉뚱한
영역을 잘랐다. 그 41건의 CER 이 5.7 로 폭주해 평균을 지배했으므로, 파트장 지시대로
두 장을 빼고 재계산한 값으로 바꾼다.

지표는 scripts/exp_log_rows.py 가 계산한다 (fill_exp_log.py 와 같은 정의).
학습 설정값은 로그 실측치로 맞춘다 — 시트에 예상치가 들어가 있던 칸들이다.

usage:
    python3 scripts/update_exp_log_transposed.py --dry-run
    python3 scripts/update_exp_log_transposed.py
"""
import argparse
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from exp_log_rows import dropped_ids, metrics  # noqa: E402

XLSX = ROOT / "receipt_data/LUXIA_VLM_학습현황 (2).xlsx"
HEAD_ROW = 2

EVAL_SET = "영수증 크롭 1,036 (text 933 / table 103) — transpose 2장 제외"

# 지표 외 정정. 시트에 '예상' 으로 들어가 있던 칸을 로그 실측치로 바꾼다.
FIXES = {
    "exp_001": {
        "소요시간": "4분",
        "메모": ("enable_thinking=False로 호출(안 끄면 추론과정 뱉어 CER 붕괴). "
               "transpose 2장(receipt34/67, 크롭 41건) 제외 후 재계산 — "
               "제외 전 CER 0.4532 / TEDS 0.6991 / 완전일치 54.8%. "
               "중앙값 0이나 짧은 한글 크롭 LaTeX 폭주가 평균을 끌어올림 → 대표값은 중앙값."),
    },
    "exp_002": {
        "상태": "완료",
        "가설/목적": ("파인튜닝 전 출발점 측정 (exp_003 개선폭 기준). "
                  "파트장 정의에 따라 이 공개 웨이트를 '베이스 모델'로 확정"),
        "소요시간": "2시간 18분",
        "메모": ("어댑터 없이 base 맨몸 측정. transpose 2장 제외 후 재계산 — "
               "제외 전 CER 0.4221 / TEDS 0.6567 / 완전일치 56.0%. "
               "텍스트는 LUXIA 우세(0.1963 vs 0.2276), 표는 열세(0.6603 vs 0.7031) "
               "→ 파인튜닝 목표를 표(TEDS)로 좁히는 근거."),
    },
    "exp_003": {
        "상태": "학습 완료 / 평가 진행중",
        # 시트에는 2×4 로 적혀 있었으나 실행 스크립트는 --batch 1 --accum 4 다.
        "batch×accum": "1×4",
        "소요시간": "6시간 24분 (1124 step, 20.51s/it)",
        "eval_loss": "0.2294 (최종) — 0.1628→0.1851→0.2163→0.2294 상승",
        "checkpoint경로": "ml/runs/upload-mix-qwen36 (ckpt-500 / 1000 / 1124)",
        "메모": ("train_vlm.py 이미지 버그 수정 후 재학습(이전엔 pixel_values v[0]만 "
               "들어가 패치 1/3848만 반영 → 이미지 못 보고 학습. v3·upload-v1~v3 전부 해당). "
               "eval 250step / ckpt 500step. "
               "★eval_loss가 epoch 0.44부터 단조 상승 — 과적합. 최적 지점은 step 250 부근이나 "
               "ckpt는 500부터만 저장됨. 평가 대상 ckpt-1000은 eval_loss 최악 구간이라 "
               "ckpt-500 병행 평가 필요."),
    },
}

# exp_id -> (결과 JSON, summary 키). None 이면 마지막 키를 쓴다.
SRC = {
    "exp_001": ("receipt_data/review/eval_crops_luxia.json", "luxia-document-parsing-high"),
    "exp_002": ("receipt_data/review/eval_crops_base.json", "base"),
    "exp_003": ("receipt_data/review/eval_crops_exp003.json", None),
}

METRIC_COLS = ["평가셋", "평가건수", "CER_중앙값", "CER_평균(참고)", "CER>1(폭주)",
               "TEDS", "완전일치%", "AVG_매크로", "AVG_마이크로", "base대비Δ"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    wb = openpyxl.load_workbook(XLSX)
    ws = wb["실험로그"]
    col = {ws.cell(HEAD_ROW, c).value: c
           for c in range(1, ws.max_column + 1) if ws.cell(HEAD_ROW, c).value}
    row_of = {ws.cell(r, 1).value: r for r in range(HEAD_ROW + 1, ws.max_row + 1)
              if ws.cell(r, 1).value}

    drop = dropped_ids()
    print(f"제외 크롭 {len(drop)}건 (receipt34/receipt67 소속 전량)\n")

    # base(exp_002) 를 먼저 계산해야 base대비Δ 를 낼 수 있다.
    calc = {}
    for exp, (path, key) in SRC.items():
        p = ROOT / path
        calc[exp] = metrics(p, key, drop) if p.exists() else None
    base_macro = calc["exp_002"]["AVG_매크로"] if calc["exp_002"] else None

    changes = []

    def put(exp, name, val):
        r, c = row_of[exp], col[name]
        old = ws.cell(r, c).value
        if old == val:
            return
        changes.append((exp, name, old, val))
        if not args.dry_run:
            ws.cell(r, c, val)

    for exp in ("exp_001", "exp_002", "exp_003"):
        if exp not in row_of:
            print(f"{exp}: 시트에 행이 없다 — 건너뜀")
            continue
        for name, val in FIXES.get(exp, {}).items():
            put(exp, name, val)

        m = calc[exp]
        if m is None:
            for name in METRIC_COLS:
                put(exp, name, "평가 진행중")
            continue

        put(exp, "평가셋", EVAL_SET)
        put(exp, "평가건수", m["평가건수"])
        put(exp, "CER_중앙값", m["CER_중앙값"])
        put(exp, "CER_평균(참고)", m["CER_평균"])
        put(exp, "CER>1(폭주)", m["CER>1"])
        put(exp, "TEDS", m["TEDS"])
        put(exp, "완전일치%", m["완전일치%"])
        put(exp, "AVG_매크로", m["AVG_매크로"])
        put(exp, "AVG_마이크로", m["AVG_마이크로"])
        if exp == "exp_002":
            put(exp, "base대비Δ", "기준(0)")
        elif base_macro is not None:
            put(exp, "base대비Δ", round(m["AVG_매크로"] - base_macro, 4))

    for exp, name, old, new in changes:
        o = str(old)[:38] + ("…" if old and len(str(old)) > 38 else "")
        n = str(new)[:38] + ("…" if len(str(new)) > 38 else "")
        print(f"  {exp}  {name:<14} {o!r:<42} -> {n!r}")
    print(f"\n총 {len(changes)}칸 변경")

    if args.dry_run:
        print("(--dry-run 이라 저장 안 함)")
        return
    wb.save(XLSX)
    print(f"저장: {XLSX}")


if __name__ == "__main__":
    main()
