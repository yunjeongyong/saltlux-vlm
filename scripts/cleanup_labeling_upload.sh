#!/bin/bash
# 라벨링 툴 데이터 폴더를 데이터셋별로 분리한다.
#
#   data/
#   ├── ss_receipt/          SS 영수증 평가 99건 (용윤정·강한빛·김동영 검수 중)
#   │   ├── images/          99
#   │   └── json/            99      ※ 원래 이름이 jsons/ 라 json/ 으로 통일
#   └── trainset_original/   VLM 학습셋 1,019건
#       ├── images/          1019
#       └── json/            1019
#
# ⚠ yjyong 컨테이너에서는 /data/workspace/VLM 이 읽기 전용(ro)이라 실패한다.
#    쓰기 권한이 있는 툴 서버/컨테이너에서 실행할 것.
#      확인:  mount | grep VLM     →  ro 로 나오면 실행 불가
set -euo pipefail

D=/data/workspace/VLM/labling-tool/luxia-labeling-tool/data
SS="$D/ss_receipt"
TR="$D/trainset_original"

echo "== 분리 전 =="
printf "   images/ %s   json/ %s   jsons/ %s\n" \
  "$(ls "$D/images" | wc -l)" "$(ls "$D/json" | wc -l)" "$(ls "$D/jsons" | wc -l)"

mkdir -p "$SS/images" "$SS/json" "$TR/images" "$TR/json"

# ── 99건: receipt* ─────────────────────────────────────────────
find "$D/images" -maxdepth 1 -name 'receipt*' -exec mv -t "$SS/images" {} +
find "$D/jsons"  -maxdepth 1 -name 'receipt*.json' -exec mv -t "$SS/json" {} +
# 검수 상태·락도 같이 옮겨야 이어서 작업할 수 있다
mkdir -p "$SS/locks" "$SS/reviews"
find "$D/locks"   -maxdepth 1 -name 'receipt*' -exec mv -t "$SS/locks" {} + 2>/dev/null || true
[ -d "$D/reviews" ] && cp -a "$D/reviews/." "$SS/reviews/" 2>/dev/null || true

# ── 1,019건: kid_* / gcse_* / p1_gcse_* ────────────────────────
# glob 특성상 'gcse_*' 는 'p1_gcse_00026' 을 매치하지 않으므로 따로 적는다
for pat in 'kid_*' 'gcse_*' 'p1_gcse_*'; do
  find "$D/json"   -maxdepth 1 -name "$pat.json" -exec mv -t "$TR/json" {} +
  find "$D/images" -maxdepth 1 \
       \( -name "$pat.png" -o -name "$pat.jpg" -o -name "$pat.jpeg" \) \
       -exec mv -t "$TR/images" {} +
done

# ── 라벨 목록 ──────────────────────────────────────────────────
# 라벨은 dataRoot 폴더마다 따로 관리된다.
# SS 쪽은 원래 4개, 학습셋 쪽은 업로드로 생긴 15개를 그대로 둔다.
mkdir -p "$SS/labels" "$TR/labels"
cat > "$SS/labels/labels.json" <<'JSON'
{
  "labels" : [ {
    "id" : "text",
    "name" : "text",
    "color" : "#2563EB"
  }, {
    "id" : "table",
    "name" : "table",
    "color" : "#2563EB"
  }, {
    "id" : "doc_title",
    "name" : "doc_title",
    "color" : "#2563EB"
  }, {
    "id" : "image",
    "name" : "image",
    "color" : "#2563EB"
  } ]
}
JSON
cp "$D/labels/labels.json" "$TR/labels/labels.json"

echo "== 분리 후 =="
printf "   ss_receipt/        images %s   json %s\n" \
  "$(ls "$SS/images" | wc -l)" "$(ls "$SS/json" | wc -l)"
printf "   trainset_original/ images %s   json %s\n" \
  "$(ls "$TR/images" | wc -l)" "$(ls "$TR/json" | wc -l)"
printf "   남은 공용          images %s   json %s   jsons %s\n" \
  "$(ls "$D/images" | wc -l)" "$(ls "$D/json" | wc -l)" "$(ls "$D/jsons" | wc -l)"
echo
echo "   기대값: ss_receipt 99/99   trainset_original 1019/1019   공용 0/0/0"
echo
echo "== 마지막 단계 =="
echo "   툴 화면 좌측 상단 '파일 경로 입력' 칸에 아래를 넣고 [적용]"
echo "     SS 검수 계속하려면 :  /data/ss_receipt"
echo "     학습셋 보려면      :  /data/trainset_original"
echo "   ※ 한 번에 한 폴더만 보인다. 지금 검수 중이면 /data/ss_receipt 로 두 것."
