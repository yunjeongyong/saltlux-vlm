#!/usr/bin/env bash
# 모든 구성을 같은 추출기(gemma-4-31B-it, :14006)로 재채점한다.
# 강한빛 선임 저장 KIE 는 8/22 다른 추출기로 뽑혀 있어 우리 수치와 직접 비교가 안 된다.
# 파싱 결과는 zip 에 그대로 있으므로 채점만 다시 한다.
set -uo pipefail
cd /data/workspace/yyj/data/ss_eval
G=(--api-url http://localhost:14006/v1 --api-key "" --api-model public/gemma-4-31B-it)
LOG(){ echo "[$(date -u '+%F %T UTC')] $*"; }

# 진행 중인 ds_exp009 KIE 가 끝나기를 기다린다 (엔드포인트 경합 방지)
LOG "ds_exp009 KIE 대기"
while pgrep -f "kie-dir vlm_output/gm_ds_exp009" >/dev/null; do sleep 60; done

LOG "① DocStudio + exp_004 — 나머지 300건"
python3 scripts/kie_from_parsing_vlm.py --types quotation invoice bill \
  --parsing-dir test_output_alt/parsing --kie-dir vlm_output/gm_ds_exp004 "${G[@]}" 2>&1 | tail -1

LOG "② DocStudio + 원본 — 400건"
python3 scripts/kie_from_parsing_vlm.py --types receipt quotation invoice bill \
  --parsing-dir test_output/parsing --kie-dir vlm_output/gm_ds_base "${G[@]}" 2>&1 | tail -1

LOG "③ 최종 채점 — 동일 추출기 5구성"
for n in "DocStudio + 원본|vlm_output/gm_ds_base" \
         "DocStudio + exp_004|vlm_output/gm_ds_exp004" \
         "DocStudio + exp_009|vlm_output/gm_ds_exp009" \
         "exp_004 단독|vlm_output/gm_engine_exp004" \
         "exp_009 단독|vlm_output/gm_engine_exp009"; do
  echo "───── ${n%%|*} ─────"
  python3 scripts/eval_kie.py --types receipt quotation invoice bill --pred-dir "${n##*|}" 2>/dev/null | grep -E "^\["
done
LOG "완료"
