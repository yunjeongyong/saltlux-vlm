#!/usr/bin/env bash
# 2026-09-10 수집 재개 — 05:53 에 세션이 끊겨 중단된 다운로드를 이어받는다.
# 두 스크립트 모두 _collect*_log.json 의 status=ok 를 건너뛰고,
# snapshot_download 는 이미 받은 파일을 건너뛰므로 파일 단위로 이어받는다.
set -u
cd /data/workspace/yyj/scripts

export HF_HUB_DOWNLOAD_TIMEOUT=60      # 기본 10초 → 60초
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_MAX_WORKERS=4                # 기본 8 → 4 (429 회피)
export PYTHONUNBUFFERED=1

LOGDIR=/data/workspace/yyj/data/external/public_260910
TS=$(date +%Y%m%d-%H%M%S)

echo "=== 재개 시작 $(date '+%F %T') (workers=$HF_MAX_WORKERS) ==="

echo "--- [1/2] 2차 공개 데이터 (collect_public2) ---"
python3 collect_public2_260910.py 2>&1 | tee -a "$LOGDIR/collect2_resume_$TS.log"

echo "--- [2/2] 한국어 (collect_korean) ---"
python3 collect_korean_260910.py 2>&1 | tee -a "$LOGDIR/korean/collect_ko_resume_$TS.log"

echo "=== 전체 종료 $(date '+%F %T') ==="
