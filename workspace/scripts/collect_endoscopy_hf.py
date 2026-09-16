#!/usr/bin/env python3
"""소화기 내시경 공개 데이터 수집 (HuggingFace 경유) — 2026-09-10
simula.no 직배포는 180KB/s로 느려 저자 공식 HF 미러를 사용한다."""
import json, os
from huggingface_hub import snapshot_download

ROOT = "/data/workspace/yyj/data/endoscopy_260910"
TOKEN = open("/data/workspace/yyj/.hf_token").read().strip()

JOBS = [
 ("hyperkvasir", "SimulaMet-HOST/HyperKvasir", "CC BY 4.0 (논문 기준)", "저자 공식 미러",
  ["*.md","hyper-kvasir-labeled-images.zip","hyper-kvasir-segmented-images.zip",
   "hyper-kvasir-unlabeled-images.zip","hyper-kvasir-videos.zip"]),
 ("kvasir_seg",  "berkaytrhn/kvasir-seg", "연구/교육용 [NC]", "폴립 마스크 1,000장", None),
 ("gastrovision","prabhashj07/Gastrovision", "CC BY 4.0", "27클래스 8,000장", None),
]

log=[]
for name, repo, lic, note, pats in JOBS:
    dst=os.path.join(ROOT, name)
    rec={"name":name,"repo":repo,"license":lic,"note":note}
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dst,
                          token=TOKEN, allow_patterns=pats, max_workers=8)
        sz=sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(dst) for f in fs)
        n=sum(len(fs) for _,_,fs in os.walk(dst))
        rec.update(status="ok", files=n, bytes=sz)
        print(f"[OK]   {name:14s} {n:6d} files {sz/1e9:6.2f} GB", flush=True)
    except Exception as e:
        rec.update(status="fail", error=repr(e)[:300])
        print(f"[FAIL] {name:14s} {repr(e)[:150]}", flush=True)
    log.append(rec)
    json.dump(log, open(os.path.join(ROOT,"_collect_log.json"),"w"), ensure_ascii=False, indent=1)
print("ALLDONE", flush=True)
