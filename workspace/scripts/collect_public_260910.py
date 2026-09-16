#!/usr/bin/env python3
"""공개 데이터 수집 — 2026-09-10
라이선스가 명시된 상업적 이용 가능 데이터셋만 받는다.
대용량은 일부 샤드만 표본으로 받는다."""
import json, os, sys, traceback
from huggingface_hub import snapshot_download, hf_hub_download, HfApi

ROOT = "/data/workspace/yyj/data/external/public_260910"
TOKEN = open("/data/workspace/yyj/.hf_token").read().strip()
api = HfApi(token=TOKEN)

# (로컬명, repo_id, 라이선스, 도메인, allow_patterns 또는 None=전체, 비고)
JOBS = [
 ("po_northwind",      "AyoubChLin/northwind_PurchaseOrders", "apache-2.0", "견적/발주", None, "Northwind DB 기반 생성 PDF 830장"),
 ("inv_voxel51_hq",    "Voxel51/high-quality-invoice-images-for-ocr", "odbl", "인보이스", None, "ODbL 동일조건변경허락 — 배포 시 주의"),
 ("inv_rvlcdip",       "Navneetkumar11/rvl-cdip-invoice-extracted", "mit", "인보이스", None, "RVL-CDIP 인보이스 추출본"),
 ("inv_d4rk3r",        "d4rk3r/invoices", "unlicense", "인보이스", None, "이미지 300 + XML 주석 600"),
 ("rcp_sroie2019",     "jsdnrs/ICDAR2019-SROIE", "cc-by-4.0", "영수증", None, "ICDAR2019 SROIE"),
 ("rcp_korean",        "HumynLabs/Korean_Receipts_Dataset", "cc-by-4.0", "영수증(한국어)", None, "한국어 영수증 20장"),
 ("rcp_voxel51_cons",  "Voxel51/consolidated_receipt_dataset", "cc-by-4.0", "영수증", None, "영수증 800장"),
 ("rcp_voxel51_scan",  "Voxel51/scanned_receipts", "cc-by-4.0", "영수증", None, "스캔 영수증 712장"),
 ("tbl_fintabnet_c",   "bsmock/FinTabNet.c", "cdla-permissive-2.0", "표(금융)", None, "금융보고서 표 — PubTabNet 대체 후보"),
 ("tbl_scitsr_pd",     "bevaya/SciTSR-pd", "cc0-1.0", "표(학술)", None, "SciTSR"),
 ("tbl_pubtables_v2",  "kensho/PubTables-v2", "cdla-permissive-2.0", "표", ["*.md","*.sh","PubTables-v2-*_000.tar.gz"], "대용량 218GB — 샤드 표본만"),
 ("doc_olmocr_bench",  "allenai/olmOCR-bench", "odc-by", "문서전사", None, "PDF 1,403 + 정답 jsonl"),
 ("doc_getomni_bench", "getomni-ai/ocr-benchmark", "mit", "문서전사(표 포함)", None, "이미지 1,000 + jsonl"),
 ("doc_olmocr_mix",    "allenai/olmOCR-mix-1025", "odc-by", "문서전사(마크다운)", ["*.md","*/*00000*.parquet","*00000*.parquet"], "대용량 77GB — 샤드 표본만"),
 ("doc_scanned_vlm",   "Voxel51/scanned-images-dataset-for-ocr-and-vlm-finetuning", "mit", "문서전사", None, "스캔 문서 3,482장"),
 ("doc_jp_pdf",        "HumynLabs/Japanese_Documents_Dataset_PDF", "cc-by-4.0", "문서(일본어)", None, "일본어 문서 PDF 63"),
 ("doc_omnidocbench",  "PaddlePaddle/Real5-OmniDocBench", "apache-2.0", "문서전사(벤치)", ["*.md","*.json","*.yaml","*.py"], "16.8GB — 주석/설정 먼저"),
]

def run():
    log = []
    for name, repo, lic, dom, pats, note in JOBS:
        dst = os.path.join(ROOT, name)
        rec = {"name": name, "repo": repo, "license": lic, "domain": dom, "note": note}
        try:
            p = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dst,
                                  token=TOKEN, allow_patterns=pats, max_workers=8)
            n = sum(len(f) for _, _, f in os.walk(dst))
            sz = sum(os.path.getsize(os.path.join(r, f))
                     for r, _, fs in os.walk(dst) for f in fs)
            rec.update(status="ok", files=n, bytes=sz, path=p)
            print(f"[OK]   {name:20s} {n:6d} files  {sz/1e9:6.2f} GB  {repo}", flush=True)
        except Exception as e:
            rec.update(status="fail", error=repr(e)[:300])
            print(f"[FAIL] {name:20s} {repr(e)[:160]}", flush=True)
        log.append(rec)
        json.dump(log, open(os.path.join(ROOT, "_collect_log.json"), "w"),
                  ensure_ascii=False, indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    run()
