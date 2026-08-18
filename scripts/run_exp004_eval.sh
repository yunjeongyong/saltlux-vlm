#!/bin/bash
# exp_004 평가 — 크롭 978건(정제 후) 기준 CER/TEDS 측정.
#
# 베이스는 한 번만 올리고 어댑터를 갈아끼운다. exp_003 을 함께 주는 이유는
# 기록된 exp_003 예측이 크롭 재생성(v2/v3 회전·배율 보정) 이전 것이라,
# 같은 크롭에서 재측정해야 TEDS 0.7286 초과 여부를 공정하게 볼 수 있어서다.
#
#   FAIR=1 (기본)  exp_003 + exp_004 재측정  — 약 4시간
#   FAIR=0         exp_004 만 측정           — 약 2시간 (기록된 exp_003 과 비교)
#
# usage:
#   bash scripts/run_exp004_eval.sh
#   GPUS=0,1 FAIR=0 bash scripts/run_exp004_eval.sh
set -u
cd /workspace

GPUS="${GPUS:-0,1}"
FAIR="${FAIR:-1}"
MODEL="${MODEL:-ml/models/Qwen3.6-35B-A3B}"
OUT="${OUT:-receipt_data/review/eval_crops_exp004.json}"
LOG="${LOG:-ml/eval_exp004.log}"

echo "== GPU 접근 확인 =="
python3 - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("GPU 사용 불가 — 컨테이너 device cgroup 이 막혀 있다(/dev/nvidia* 가 "
             "EPERM). 호스트에서 컨테이너를 재시작한 뒤 다시 실행할 것.")
n = torch.cuda.device_count()
print(f"  GPU {n}장 인식")
for i in range(n):
    free, total = torch.cuda.mem_get_info(i)
    print(f"   GPU{i} {torch.cuda.get_device_name(i)}  "
          f"여유 {free/2**30:.0f}GiB / 전체 {total/2**30:.0f}GiB")
PY
[ $? -eq 0 ] || exit 1

ADAPTERS="ml/runs/exp004-260813"
if [ "$FAIR" = "1" ]; then
  ADAPTERS="ml/runs/upload-mix-qwen36/checkpoint-1000 ml/runs/exp004-260813"
fi

echo
echo "== 평가 시작 (GPU ${GPUS}) =="
echo "  어댑터: ${ADAPTERS}"
echo "  로그:   ${LOG}"

CUDA_VISIBLE_DEVICES="${GPUS}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 scripts/eval_crops.py \
  --model "${MODEL}" \
  --adapter ${ADAPTERS} \
  --manifest receipt_data/eval_crops/manifest.jsonl \
  --skip-base \
  --out "${OUT}" "$@" \
  > "${LOG}" 2>&1 &

echo "  PID $!"
echo
echo "진행 보기:  tail -f ${LOG} | tr '\\r' '\\n' | grep -v it/s"
echo "끝나면:     python3 scripts/score_exp004.py"
