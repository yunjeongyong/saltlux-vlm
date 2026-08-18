#!/usr/bin/env bash
# merged_260813 (12,694건) 기반 학습 시작.
#
# start_upload_mix_train.sh 와 같은 구조이되 두 가지가 다르다.
#   1. 데이터가 upload_mix 4,496 -> merged_260813 12,694 로 늘었다.
#      (upload_mix 4,496 + kdy_260813 8,198. 후자는 경로를 ROOT 상대로 재작성한 것)
#   2. exp_003 에서 eval_loss 가 epoch 0.44 부터 단조 상승했는데 best 지점이
#      저장되지 않았다(save_steps=500 이라 step 250 은 애초에 안 찍혔고,
#      step 500 은 save_total_limit=2 로 지워졌다). 그래서 여기서는
#      save_steps 를 eval_steps 와 같은 250 으로 맞추고 --load-best 를 켠다.
#
# usage:
#   bash scripts/start_merged_train.sh                    # 기본 (GPU 0,1)
#   GPUS=2,3 bash scripts/start_merged_train.sh           # GPU 지정
#   EPOCHS=1 bash scripts/start_merged_train.sh           # 1 epoch
#   bash scripts/start_merged_train.sh --resume auto      # 중단분 이어받기
set -euo pipefail

cd /workspace

GPUS="${GPUS:-0,1}"
EPOCHS="${EPOCHS:-1}"
OUT="${OUT:-ml/runs/merged-260813-qwen36}"
LOG="${LOG:-ml/train_merged_260813.log}"
MODEL="${MODEL:-ml/models/Qwen3.6-35B-A3B}"
DATA="${DATA:-receipt_data/merged_260813}"

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
  p="${DATA}/${f}.jsonl"
  [ -s "$p" ] || { echo "없거나 비어 있음: $p"; exit 1; }
  echo "  ${f}: $(wc -l < "$p")건"
done

# 이미지 경로는 ROOT(=/workspace) 상대다. 35B 로드에 수십 분 쓰고 나서
# 파일 없음으로 죽지 않도록 학습 전에 전수 확인한다.
echo
echo "== 이미지 경로 전수 확인 =="
python3 - "${DATA}" <<'PY'
import json, os, sys
d = sys.argv[1]
for name in ("train", "val", "test"):
    miss = []
    with open(f"{d}/{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            for p in json.loads(line)["images"]:
                if not os.path.exists(p):
                    miss.append(p)
    print(f"  {name}: 누락 {len(miss)}")
    if miss:
        for p in miss[:5]:
            print(f"    {p}")
        sys.exit(f"이미지 누락 — 학습 중단")
PY

echo
echo "== 학습 시작 (GPU ${GPUS}, ${EPOCHS} epoch) =="
echo "  로그: ${LOG}"
echo "  출력: ${OUT}"

CUDA_VISIBLE_DEVICES="${GPUS}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 scripts/train_vlm.py \
  --model "${MODEL}" \
  --train "${DATA}/train.jsonl" \
  --val   "${DATA}/val.jsonl" \
  --epochs "${EPOCHS}" --batch 1 --accum 4 \
  --device-map auto --no-grad-checkpoint --max-px 1000000 \
  --save-steps 250 --eval-steps 250 --load-best --save-total-limit 3 \
  --out "${OUT}" "$@" \
  > "${LOG}" 2>&1 &

echo "  PID $!"
echo
echo "진행 보기:  tail -f ${LOG} | tr '\\r' '\\n'"
