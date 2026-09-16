#!/usr/bin/env bash
# KIE 추출기를 GPU7 엔진으로 대체해 전 후보를 같은 조건으로 재채점한다.
# 외부 gemma 엔드포인트(124.198.30.179:15006)가 다운이라 비교 자체가 막혀 있었다.
# 절대값은 강한빛 선임 수치와 달라지지만, 같은 추출기로 전부 다시 뽑으므로
# 표 안에서의 우열 비교는 성립한다.
set -uo pipefail
cd /data/workspace/yyj/data/ss_eval
API=(--api-url http://localhost:60150/v1 --api-key "" --api-model /home/model/merged_model)
TYPES="${TYPES:-receipt}"

run() {   # $1 라벨  $2 파싱디렉터리
  local name=$1 src=$2
  [ -d "$src" ] || { echo "  [skip] $name — $src 없음"; return; }
  echo "=== KIE: $name  ($TYPES) ==="
  python3 scripts/kie_from_parsing_vlm.py --types $TYPES \
    --parsing-dir "$src" --kie-dir "vlm_output/g7_${name}" "${API[@]}" 2>&1 | tail -4
}

run ds_base   test_output/parsing
run ds_alt    test_output_alt/parsing
run exp004    vlm_output/engine_exp004
run exp009    vlm_output/exp009

echo
for n in ds_base ds_alt exp004 exp009; do
  [ -d "vlm_output/g7_$n" ] || continue
  echo "───── $n ─────"
  python3 scripts/eval_kie.py --types $TYPES --pred-dir "vlm_output/g7_$n" 2>&1 | tail -12
done
echo "ALL DONE $(date -u '+%F %T UTC')"
