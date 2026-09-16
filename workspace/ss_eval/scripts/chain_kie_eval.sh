#!/usr/bin/env bash
# SS 평가셋: 우리 VLM 파싱이 끝난 도메인부터 KIE → 채점까지 자동 진행.
#
# 파싱(parse_ss_docs.py)은 GPU 4~6 에서 receipt → quotation → invoice → bill 순으로
# 돌고 있다. 이 스크립트는 도메인별로 .md 100개가 다 차면 그 즉시 KIE 를 태운다.
#
# KIE 엔드포인트: 강한빛 선임 기준선(base 0.848 / alt 0.860)과 동일한 서버다.
#   실행 리포트에 luxia3.5-120b-sft-v1.4 로 남아 있으나, 그 필드는 --api-model 인자를
#   그대로 기록한 값이고 서버는 모델명을 무시하고 gemma-4-31B-it 로 응답한다.
#   (존재하지 않는 모델명으로 호출해도 동일 응답 확인) 따라서 기준선과 직접 비교 가능.
set -u
cd /data/workspace/yyj/data/ss_eval || exit 1
API=(--api-url http://124.198.30.179:15006/v1 --api-model public/gemma-4-31B-it --api-key "")
VLM=vlm_output/exp009
KIE=vlm_output/exp009_kie

for T in receipt quotation invoice bill; do
  echo; echo "=== [$T] 파싱 완료 대기 ($(date -u '+%m/%d %H:%M:%S UTC')) ==="
  while true; do
    n=$(ls $VLM/$T/*.md 2>/dev/null | wc -l)
    [ "$n" -ge 100 ] && break
    if ! pgrep -f "parse_ss_docs.py" >/dev/null 2>&1 && ! pgrep -f "chain_ss_parse" >/dev/null 2>&1; then
      echo "[중단] 파싱 프로세스 없음. $T 현재 $n/100"; break
    fi
    sleep 120
  done
  n=$(ls $VLM/$T/*.md 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && { echo "[건너뜀] $T 파싱 결과 없음"; continue; }
  echo "=== [$T] KIE 시작 · 파싱 $n건 ($(date -u '+%H:%M:%S UTC')) ==="
  python3 scripts/kie_from_parsing_vlm.py --parsing-dir "$VLM" --kie-dir "$KIE" --types "$T" "${API[@]}"
  echo "=== [$T] KIE 종료 rc=$? ($(date -u '+%H:%M:%S UTC')) ==="

  echo "--- [$T] 채점 (우리 모델) ---"
  python3 scripts/eval_kie.py --types "$T" --pred-dir "$KIE" 2>&1 | tail -20
done
echo; echo "=== 전체 종료 ($(date -u '+%m/%d %H:%M:%S UTC')) ==="
