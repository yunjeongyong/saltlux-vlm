#!/usr/bin/env bash
# 소화기 내시경 공개 데이터 수집 — 2026-09-10
# 라이선스/이용조건은 _CATALOG.md 참조. 상업적 이용 제한이 있는 것은 [NC] 표시.
set -u
ROOT=/data/workspace/yyj/data/endoscopy_260910
cd "$ROOT" || exit 1

get() { # name url
  local name=$1
  local url=$2
  local out="$ROOT/$name"
  if [ -s "$out" ]; then echo "[SKIP] $name (이미 있음)"; return; fi
  echo "[GET ] $name <- $url"
  curl -sk -L --retry 3 --retry-delay 5 -o "$out.part" "$url" \
    && mv "$out.part" "$out" \
    && echo "[OK  ] $name  $(du -h "$out" | cut -f1)" \
    || echo "[FAIL] $name"
}

B=https://datasets.simula.no/downloads

# --- CC BY 4.0 (상업적 이용 가능) ---
get hyper-kvasir-labeled-images.zip   "$B/hyper-kvasir/hyper-kvasir-labeled-images.zip"
get hyper-kvasir-segmented-images.zip "$B/hyper-kvasir/hyper-kvasir-segmented-images.zip"
get hyper-kvasir-unlabeled-images.zip "$B/hyper-kvasir/hyper-kvasir-unlabeled-images.zip"

# --- 연구/교육용 (상업적 이용 사전 서면 허가 필요) [NC] ---
get kvasir-seg.zip           "$B/kvasir-seg.zip"
get kvasir-sessile.zip       "$B/kvasir-sessile.zip"
get kvasir-instrument.zip    "$B/kvasir-instrument/kvasir-instrument.zip"
get nerthus.zip              "$B/nerthus/nerthus.zip"
get kvasir-capsule-labeled-images.zip "$B/kvasir-capsule/kvasir-capsule-labeled-images.zip"

echo "ALLDONE"
ls -la "$ROOT"
