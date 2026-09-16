#!/usr/bin/env python3
"""한국어 문서 축 수집 — 2026-09-10 (3차)

2차 수집으로는 한국어가 실질적으로 안 늘어난다고 판단했는데, 다시 뒤져보니 있었다.
검색어를 "korean ocr"이 아니라 language:ko 태그 전수 + VDR(visual document retrieval)로
바꾸니 나왔다. VDR 데이터가 검색용으로 분류돼 있어서 문서 파싱 검색에 안 걸렸던 것이다.

핵심: ko-vdr-train-public 은 검색용 데이터인데 `image` + `markdown` + `elements` 를
모두 들고 있다. 즉 그대로 문서→마크다운 전사 학습셋으로 쓸 수 있다.
"""
import json, os, time
from huggingface_hub import snapshot_download

ROOT  = "/data/workspace/yyj/data/external/public_260910/korean"
TOKEN = open("/data/workspace/yyj/.hf_token").read().strip()
MAXW  = int(os.environ.get("HF_MAX_WORKERS", "4"))   # 429 회피용, 기본 4

JOBS = [
 ("ko_vdr_train", "NomaDamas/ko-vdr-train-public", "cc-by-4.0",
  ["*.md", "data/train-0000?-of-00271.parquet", "data/train-0001?-of-00271.parquet"],
  "한국어 실문서 310,226행(image+markdown+elements). 251GB 중 20샤드 표본 ~23,000행"),

 ("ko_kolmocr", "posicube/KolmOCR-traindataset", "odc-by", None,
  "한국어 문서→마크다운 11,866쌍 + PNG + bbox JSON. 13GB 전체"),

 ("ko_sds_kopub", "SamsungSDS-Research/SDS-KoPub-VDR-Benchmark", "cc-by-sa-4.0", None,
  "실제 한국 공공문서(정부) 이미지 + QA + 주석. 21.9GB 전체. SA 조건 주의"),

 ("ko_mdpbench", "Delores-Lin/MDPBench", "apache-2.0", None,
  "촬영 왜곡 강건성 — ko 160장(신문·교과서·시험지·재무보고서·슬라이드) × 구김/조명/기울기/블러 변형"),
]

def main():
    os.makedirs(ROOT, exist_ok=True)
    logp = os.path.join(ROOT, "_collect_ko_log.json")
    log = json.load(open(logp)) if os.path.exists(logp) else []
    done = {r["name"] for r in log if r.get("status") == "ok"}
    for name, repo, lic, pats, note in JOBS:
        if name in done:
            print(f"[SKIP] {name}", flush=True); continue
        dst = os.path.join(ROOT, name)
        rec = {"name": name, "repo": repo, "license": lic, "note": note,
               "sampled": pats is not None, "ts": time.strftime("%F %T")}
        try:
            snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dst,
                              token=TOKEN, allow_patterns=pats, max_workers=MAXW)
            n  = sum(len(f) for _, _, f in os.walk(dst))
            sz = sum(os.path.getsize(os.path.join(r, f))
                     for r, _, fs in os.walk(dst) for f in fs)
            rec.update(status="ok", files=n, bytes=sz)
            print(f"[OK]   {name:16s} {n:6d} files {sz/1e9:8.2f} GB  {repo}", flush=True)
        except Exception as e:
            rec.update(status="fail", error=repr(e)[:300])
            print(f"[FAIL] {name:16s} {repr(e)[:160]}", flush=True)
        log.append(rec)
        json.dump(log, open(logp, "w"), ensure_ascii=False, indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
