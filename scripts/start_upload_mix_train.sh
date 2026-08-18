#!/usr/bin/env bash
# vlm_dataset_upload 기반 신규 학습 시작.
#
# 컨테이너 재시작 직후 이것만 실행하면 된다. GPU 접근이 실제로 복구됐는지
# 먼저 확인하고, 안 됐으면 학습을 띄우지 않고 멈춘다 — 35B 로드에 수십 분을
# 쓰고 나서 CUDA 없음으로 죽는 것을 막는다.
#
# usage:
#   bash scripts/start_upload_mix_train.sh              # 기본 (GPU 0,1)
#   GPUS=2,3 bash scripts/start_upload_mix_train.sh     # GPU 지정
#   bash scripts/start_upload_mix_train.sh --resume auto # 중단분 이어받기
set -euo pipefail

cd /workspace

GPUS="${GPUS:-0,1}"
OUT="${OUT:-ml/runs/upload-mix-qwen36}"
LOG="${LOG:-ml/train_upload_mix.log}"
MODEL="${MODEL:-ml/models/Qwen3.6-35B-A3B}"

echo "== GPU 접근 확인 =="
python3 - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("GPU 사용 불가 — 컨테이너 device cgroup 이 아직 복구되지 않았다. "
             "호스트에서 컨테이너를 재시작한 뒤 다시 실행할 것.")
n = torch.cuda.device_count()
print(f"  GPU {n}장 인식")
for i in range(n):
    free, total = torch.cuda.mem_get_info(i)
    print(f"   GPU{i} {torch.cuda.get_device_name(i)}  "
          f"여유 {free/2**30:.0f}GiB / 전체 {total/2**30:.0f}GiB")
PY

echo
echo "== 데이터셋 확인 =="
for f in train val test; do
  p="receipt_data/upload_mix/${f}.jsonl"
  [ -s "$p" ] || { echo "없거나 비어 있음: $p"; exit 1; }
  echo "  ${f}: $(wc -l < "$p")건"
done

echo
echo "== 학습 시작 (GPU ${GPUS}) =="
echo "  로그: ${LOG}"
echo "  출력: ${OUT}"

CUDA_VISIBLE_DEVICES="${GPUS}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 scripts/train_vlm.py \
  --model "${MODEL}" \
  --train receipt_data/upload_mix/train.jsonl \
  --val   receipt_data/upload_mix/val.jsonl \
  --epochs 2 --batch 1 --accum 4 \
  --device-map auto --no-grad-checkpoint --max-px 1000000 \
  --save-steps 500 --eval-steps 250 \
  --out "${OUT}" "$@" \
  > "${LOG}" 2>&1 &

echo "  PID $!"
echo
echo "진행 보기:  tail -f ${LOG} | tr '\\r' '\\n'"
