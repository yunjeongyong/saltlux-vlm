#!/usr/bin/env bash
# DocStudio 파이프라인 + exp_009 어댑터 — 400건 파싱 후 gemma 로 KIE 채점.
# 워커 :12200 은 HIGH_VLM_SERVER_URL=:60101(GPU5, exp009_merged) 을 본다.
# 한빛 선임 document_parsing_alt.py 와 같은 호출(modelQuality=high)을 쓴다.
set -uo pipefail
cd /data/workspace/yyj/data/ss_eval
LOG(){ echo "[$(date -u '+%F %T UTC')] $*"; }

LOG "① DocStudio+exp_009 400건 파싱 (:12200)"
python3 /data/workspace/yyj/data/vlm_exp34/scripts/parse_ss_via_worker.py \
  --base http://localhost:12200 --types receipt quotation invoice bill \
  --out vlm_output/ds_exp009 --workers 3
LOG "파싱 $(find vlm_output/ds_exp009 -name '*.md'|wc -l)/400"

LOG "② gemma 로 KIE"
python3 scripts/kie_from_parsing_vlm.py --types receipt quotation invoice bill \
  --parsing-dir vlm_output/ds_exp009 --kie-dir vlm_output/gm_ds_exp009 \
  --api-url http://localhost:14006/v1 --api-key "" --api-model public/gemma-4-31B-it 2>&1 | tail -3

LOG "③ 채점 — 5개 구성"
for n in "DocStudio+원본|test_output/kie" "DocStudio+exp_004|test_output_alt/kie" \
         "DocStudio+exp_009|vlm_output/gm_ds_exp009" \
         "엔진 exp_004|vlm_output/gm_engine_exp004" "엔진 exp_009|vlm_output/gm_engine_exp009"; do
  echo "───── ${n%%|*} ─────"
  python3 scripts/eval_kie.py --types receipt quotation invoice bill --pred-dir "${n##*|}" 2>/dev/null | grep -E "^\["
done
LOG "완료"
