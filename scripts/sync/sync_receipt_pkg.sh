#!/usr/bin/env bash
# 영수증 패키지 동기화: 176건 → 187건 (사람 검수 84 → 95)
#   신규 11장 (json+png = 22개 파일) 추가
#   kid_IMG00001.json 갱신 1건 (08-12 02:08 재검수분 — bbox 5개 블록 수정)
# /data/workspace/VLM 이 rw 로 마운트된 컨테이너/호스트에서 실행하세요.
set -euo pipefail

S=/data/workspace/yjyong/vlm_dataset_upload/train/human_annotated/receipt
D=/data/workspace/VLM/dataset/train/human_annotated/receipt

NEW=(kid_IMG00100 kid_IMG00101 kid_IMG00110 kid_IMG00111 kid_IMG00112
     kid_IMG00113 kid_IMG00115 kid_IMG00116 kid_IMG00117 kid_IMG00118 kid_IMG00119)

[ -w "$D" ] || { echo "ERROR: $D 에 쓰기 권한이 없습니다 (read-only 마운트)"; exit 1; }

echo "[1/4] 기존 manifest·train.jsonl 백업"
cp -p "$D/manifest.json" "$D/manifest.json.bak-176"
cp -p "$D/train.jsonl"   "$D/train.jsonl.bak-176"

echo "[2/4] 신규 22개 파일 복사"
for n in "${NEW[@]}"; do
  cp -p "$S/data/$n.json" "$D/data/$n.json"
  cp -p "$S/data/$n.png"  "$D/data/$n.png"
done

echo "[3/4] 재검수분 갱신 (kid_IMG00001 — bbox 5개 블록)"
cp -p "$D/data/kid_IMG00001.json" "$D/data/kid_IMG00001.json.bak-0811"
cp -p "$S/data/kid_IMG00001.json" "$D/data/kid_IMG00001.json"

echo "[4/4] manifest.json · train.jsonl 갱신"
cp -p "$S/manifest.json" "$D/manifest.json"
cp -p "$S/train.jsonl"   "$D/train.jsonl"

echo "--- 검증 ---"
echo "train.jsonl 줄수: $(wc -l < "$D/train.jsonl")  (기대 187)"
echo "data/ 파일수:     $(ls "$D/data" | wc -l)      (기대 374)"
python3 - <<'PY'
import json
m=json.load(open('/data/workspace/VLM/dataset/train/human_annotated/receipt/manifest.json'))
print('manifest composition:', m['info']['composition'], '| data 항목:', len(m['data']))
PY
echo "완료."
