#!/usr/bin/env python3
"""공개 데이터 다양성 보강 수집 — 2026-09-10 (2차)
1차(collect_public_260910.py)가 영수증/인보이스에 몰려 있어, 겹치지 않는 축으로 채운다.
  - 표 구조: PubTabNet(CC BY-NC-SA, exp014에서 제외 중) 대체용 상용 가능 표 데이터
  - 레이아웃 / 차트 / 수식: 학습셋에 아예 없는 축
  - 금융·행정 실문서: SS 계산서·명세서 계열 대응
  - 한국어: 공개분이 전부 영문이라 ko 축 확보
라이선스 미표기 저장소는 받지 않는다. 상용 불가/조건부는 research_only/ 로 분리한다.
"""
import json, os, time
from huggingface_hub import snapshot_download

ROOT   = "/data/workspace/yyj/data/external/public_260910"
RO     = os.path.join(ROOT, "research_only")
TOKEN  = open("/data/workspace/yyj/.hf_token").read().strip()
MAXW  = int(os.environ.get("HF_MAX_WORKERS", "4"))   # 429 회피용, 기본 4

def shards(prefix, n, total, width=5, suffix=".parquet"):
    return [f"{prefix}{i:0{width}d}-of-{total:05d}{suffix}" for i in range(n)]

# (로컬명, repo_id, 라이선스, 축, allow_patterns(None=전체), 비고)
JOBS = [
 # ── 표 구조 (PubTabNet 대체) ─────────────────────────────────────────────
 ("tbl_fintabnet_otsl",  "docling-project/FinTabNet_OTSL", "CDLA-Permissive-1.0(원본 FinTabNet 준거)",
  "표(금융)", None, "OTSL 표 구조 3.0GB — PubTabNet 직접 대체 1순위"),
 ("tbl_pubtables1m_otsl","docling-project/PubTables-1M_OTSL-v1.1", "CDLA-Permissive-2.0(원본 PubTables-1M 준거)",
  "표(학술·정부)", None, "24GB 전체"),
 ("tbl_synthtabnet_otsl","docling-project/SynthTabNet_OTSL", "CDLA-Permissive-1.0(원본 SynthTabNet 준거)",
  "표(합성 4스타일)", None, "35GB 전체 — 표 스타일 다양성"),
 ("tbl_tablebank",       "liminghao1630/TableBank", "apache-2.0",
  "표(Word/LaTeX)", None, "25GB zip 5분할"),
 ("tbl_semtabnet",       "docling-project/SemTabNet", "mit",
  "표(의미 주석)", None, "5.2GB"),
 ("tbl_finstmt_html",    "apoidea/financial-statement-table-html", "mit",
  "표(재무제표→HTML)", None, "소형 — 재무제표 표 HTML 쌍"),
 ("tbl_html_recon_bench","sfd-anonymous/html-table-reconstruction-benchmark", "cc-by-4.0",
  "표(HTML 복원 벤치)", None, "소형 평가셋"),

 # ── 레이아웃 ────────────────────────────────────────────────────────────
 ("lay_doclaynet_v12",   "docling-project/DocLayNet-v1.2", "CDLA-Permissive-1.0(README 명시)",
  "레이아웃(6개 도메인)", None, "40GB — _license/STATUS.md 의 DocLayNet '보류' 건 해소"),
 ("lay_icdar23_doclaynet","docling-project/icdar2023-doclaynet", "apache-2.0",
  "레이아웃(대회)", None, "주석만, 소형"),
 ("lay_publaynet",       "creative-graphic-design/PubLayNet", "cdla-permissive-1.0",
  "레이아웃(논문)", ["*.md","data/test-*","data/train-0000?-of-00199.parquet"], "107GB 중 표본 10샤드+test"),

 # ── 차트 / 수식 (학습셋에 없는 축) ───────────────────────────────────────
 ("cht_synthchartnet",   "docling-project/SynthChartNet", "cdla-permissive-2.0",
  "차트→데이터", ["*.md","train-0000?-of-00135.parquet","train-0001?-of-00135.parquet"], "70GB 중 표본 20샤드"),
 ("fml_synthformulanet", "docling-project/SynthFormulaNet", "cdla-permissive-2.0",
  "수식→LaTeX", ["*.md","test/*","train/train-0000?-of-00067.parquet"], "35GB 중 표본 10샤드+test"),

 # ── 문서 전사 / 문서 QA 다양성 ──────────────────────────────────────────
 ("doc_doclingmatix",    "HuggingFaceM4/DoclingMatix", "cdla-permissive-2.0",
  "문서 QA(Docmatix 재주석)", ["*.md","train-0000?-of-01106-*.parquet"], "932GB 중 표본 10샤드"),
 ("doc_olmocr_synthmix", "allenai/olmOCR-synthmix-1025", "odc-by",
  "문서 전사(합성 혼합)", None, "1.2GB 전체"),
 ("doc_docling_dpbench", "docling-project/docling-dpbench", "apache-2.0",
  "문서 파싱 벤치", None, "0.3GB"),
 ("doc_ocr_md_dense",    "prithivMLmods/OCR-Markdown-Dense-200x", "apache-2.0",
  "문서→마크다운(고밀도)", None, "0.2GB"),
 ("doc_docvqa_single",   "pixparse/docvqa-single-page-questions", "mit",
  "문서 QA(DocVQA)", None, "12.4GB 전체"),
 ("doc_ar_kitab_md",     "Misraj/KITAB_pdf_to_markdown_reviewed", "apache-2.0",
  "문서→마크다운(아랍어)", None, "비라틴 문자 강건성"),

 # ── KIE / 폼 ────────────────────────────────────────────────────────────
 ("kie_nanonets",        "nanonets/key_information_extraction", "apache-2.0",
  "KIE(항목 추출)", None, "0.2GB"),
 ("kie_wildreceipt",     "Theivaprakasham/wildreceipt", "apache-2.0",
  "KIE(실촬영 영수증)", None, "1,740장 실촬영 — 크롭 생성분과 성격 다름"),
 ("kie_sroie_du",        "arvindrajan92/sroie_document_understanding", "mit",
  "KIE(SROIE 재구성)", None, "0.2GB"),

 # ── 금융·행정 실문서 (SS 계산서·명세서 계열) ────────────────────────────
 ("fin_bank_stmt_in",    "AgamiAI/Indian-Bank-Statements", "apache-2.0",
  "거래명세(은행 명세서)", None, "1.4GB — 다열 명세표 실문서"),
 ("fin_jp_annual_sec",   "u-10bei/Annual_securities_report", "cc-by-4.0",
  "유가증권보고서(일본어)", None, "2.4GB — CJK 금융 서식"),
 ("gov_vn_doc",          "Tucker6742/Vietnamese-government-document", "apache-2.0",
  "행정문서(베트남어)", None, "9.9GB"),

 # ── 한국어 ──────────────────────────────────────────────────────────────
 ("ko_nvidia_ocr_multi", "nvidia/OCR-Synthetic-Multilingual-v1", "cc-by-4.0",
  "OCR(한국어)", ["*.md","ko/test/test_00?.h5","ko/train/train_00?.h5"], "ko 1TB 중 표본 ~25GB"),
 ("ko_handwritten_notes","HumynLabs/Korean_Handwritten_Notes_Dataset", "cc-by-4.0",
  "필기(한국어)", None, "소형 — 한국어 필기 강건성"),
]

# 상용 투입 금지 — research_only/ 로 격리
JOBS_RO = [
 ("ro_funsd_plus",   "Voxel51/form_understanding_in_noisy_scanned_documents_plus", "FUNSD+ 커스텀(개인식별 시도 금지)",
  "폼 이해", None, "STATUS.md 의 FUNSD '보류' 건 — 커스텀 약관이라 상용 제외"),
 ("ro_chartqa",      "ahmed-masry/ChartQA", "gpl-3.0",
  "차트 QA", None, "GPL 전염성 — 상용 납품 모델 투입 불가"),
 ("ro_korean_docs",  "HumynLabs/Korean-Documents-Dataset", "카드에 상업적 재사용 제한 명시",
  "문서(한국어)", None, "한국어지만 상용 불가"),
 ("ro_handwriting_ocr","lance-format/handwriting-ocr", "odbl(동일조건변경허락)",
  "필기 OCR", None, "ODbL — 보수적으로 격리"),
]

def run(jobs, base, log_name):
    logp = os.path.join(ROOT, log_name)
    log = json.load(open(logp)) if os.path.exists(logp) else []
    done = {r["name"] for r in log if r.get("status") == "ok"}
    for name, repo, lic, axis, pats, note in jobs:
        if name in done:
            print(f"[SKIP] {name}", flush=True); continue
        dst = os.path.join(base, name)
        rec = {"name": name, "repo": repo, "license": lic, "axis": axis, "note": note,
               "sampled": pats is not None, "ts": time.strftime("%F %T")}
        try:
            snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dst,
                              token=TOKEN, allow_patterns=pats, max_workers=MAXW)
            n  = sum(len(f) for _, _, f in os.walk(dst))
            sz = sum(os.path.getsize(os.path.join(r, f))
                     for r, _, fs in os.walk(dst) for f in fs)
            rec.update(status="ok", files=n, bytes=sz, path=dst)
            print(f"[OK]   {name:22s} {n:6d} files {sz/1e9:8.2f} GB  {repo}", flush=True)
        except Exception as e:
            rec.update(status="fail", error=repr(e)[:300])
            print(f"[FAIL] {name:22s} {repr(e)[:160]}", flush=True)
        log.append(rec)
        json.dump(log, open(logp, "w"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    os.makedirs(RO, exist_ok=True)
    run(JOBS,    ROOT, "_collect2_log.json")
    run(JOBS_RO, RO,   "_collect2_ro_log.json")
    print("DONE", flush=True)
