"""출처 원장 — 전부 디스크 실측 및 학습셋 역추적으로 확인한 값이다. (2026-09-01)

수량을 세 가지로 나눈다. 종전에 한 칸에 섞여 있어 추적이 되지 않았다.
  공개 규모  외부 데이터셋이 원래 배포하는 규모 (참고용, 우리가 가진 양이 아니다)
  보유       우리 디스크에 실제로 있는 파일 수
  활용       exp_012 학습셋에 실제로 들어간 고유 이미지 수
"""
# (S-ID, 데이터명, 태스크, 도메인, 수집방법, 공개규모, 보유, 경로, 출처/라이선스, 상태)
#   상태: use=활용 / new=exp_012 신규 / val=검증전용 / drop=미활용
SOURCES = [
 ("S01","train_crops","OCR (Text Recognition)","영수증","사내 수집",None,913,
  "vlm_exp34/receipt_data/train_crops","자체 라벨링","use"),
 ("S02","receipts3000 파생 크롭","OCR (Text Recognition)","영수증(합성)","생성형 (크롭)",None,9520,
  "vlm_exp34/receipt_data/cand_crops/images (cand_receipt3000_*)",
  "자체 생성 · S08 receipts3000 원본 1,904장에서 크롭 · 미검수 포함","use"),
 ("S03","labeled (검수본)","Document Parsing","영수증","사내 수집",None,188,
  "vlm_exp34/receipt_data/labeled","자체 검수 · val/test 원본","use"),
 ("S04","synth_v1","Document Parsing","영수증(합성)","생성형 (PIL 렌더링)",None,7000,
  "vlm_exp34/receipt_data/synth_v1","자체 생성 · 스캔풍","use"),
 ("S05","synth_v2","Document Parsing","영수증(합성)","생성형 (PIL 렌더링)",None,1000,
  "vlm_exp34/receipt_data/synth_v2","자체 생성 · 촬영풍","use"),
 ("S06","synth_v2_nofinger","Document Parsing","영수증(합성)","생성형 (PIL 렌더링)",None,1000,
  "vlm_exp34/receipt_data/synth_v2_nofinger","자체 생성 · 손가락 제거판","drop"),
 ("S07","synth_merged_crops","OCR / TSR","영수증(합성)","생성형 (크롭)",None,12000,
  "vlm_exp34/receipt_data/synth_merged_crops","자체 생성","use"),
 ("S09","KorIE ocr","OCR (Text Recognition)","영수증","공개 데이터",None,8381,
  "vlm_exp34/receipt_data/cand_crops/images (cand_korie_ocr_*)","KorIE","use"),
 ("S10","KorIE kid","Grounding / Detection","영수증","공개 데이터",None,2250,
  "vlm_exp34/receipt_data/cand_crops/images (cand_korie_kid_*)","KorIE · YOLO bbox","use"),
 ("S11","CORD","Document Parsing + KIE","영수증(인니)","공개 데이터",1000,4978,
  "vlm_exp34/receipt_data/cand_crops/images (cand_cord_cord_*)","CC BY-4.0","use"),
 ("S13","AI-Hub 71299 OCR","OCR (Full-page)","공공행정문서","공개 데이터",36598,3000,
  "vlm_exp34/vlm_dataset_upload/train/public/aihub-71299-ocr","AI-Hub · 표본 추출 보유","use"),
 ("S14","PubTabNet","TSR","학술논문 표(영문)","공개 데이터",500777,5461,
  "vlm_exp34/vlm_dataset_upload/train/public/pubtabnet-html","PubTabNet · 표본 추출 보유","use"),
 ("S15","kogovdoc-bench","Document Parsing","정부문서·논문","공개 데이터",294,139,
  "vlm_exp34/vlm_dataset_upload/train/public/kogovdoc-bench","Wigtn/KoGovDoc-Bench","use"),
 ("S16","crawl_google","Document Parsing","영수증","크롤링 (웹 수집)",None,155,
  "vlm_exp34/receipt_data/cand_crops/images (cand_crawl_google_gcse_*)","라이선스 확인 필요 · 일부만 검수 투입","use"),
 ("S17","complex-table-generator (자체 구동)","Document Parsing + TSR","무역·금융·행정 서식",
  "생성형 (typst+augraphy)",None,8000,"external/commercial_ok/ctg_main",
  "자체 생성 · 코드는 강한빛 선임 제공","new"),
 ("S18","synthetic_table_dataset (선임 제공)","Document Parsing + TSR","무역·금융·행정 서식",
  "생성형 (typst+augraphy)",None,5000,"external/commercial_ok/std_hb","강한빛 선임 제공","new"),
 ("S19","FATURA2-invoices","Document Parsing + KIE","인보이스(영문)","공개 데이터",10000,4000,
  "external/commercial_ok/pub_invoice/images (pubinv_fatura_*)","CC BY-4.0 · 표본 추출 보유","new"),
 ("S20","invoices-donut-data-v1","Document Parsing + KIE","인보이스(영문)","공개 데이터",501,491,
  "external/commercial_ok/pub_invoice/images (pubinv_donut_*)","MIT","new"),
 ("S21","synth_bl (폐기)","Document Parsing","선하증권(합성)","생성형 (PIL 렌더링)",None,3000,
  "external/synthetic/synth_bl","자체 생성 · 글꼴 결함으로 폐기, S17 로 대체","drop"),
 ("S22","synth_mcip (폐기)","Document Parsing","보험증권(합성)","생성형 (PIL 렌더링)",None,3000,
  "external/synthetic/synth_mcip","자체 생성 · 폐기, S17 로 대체","drop"),
 ("S23","research_only (HF 캐시)","Layout / Document Parsing","문서 레이아웃·인보이스","공개 데이터",None,1582,
  "external/research_only/hf_cache","라이선스 미확인 · 격리 · 학습 금지","drop"),
 ("S24","ctg_val (검증 전용)","Document Parsing","무역·금융·행정 서식(합성)","생성형 (typst 렌더링)",None,400,
  "external/commercial_ok/ctg_val","자체 생성 · 시드 90001 · 학습셋과 겹침 0건","val"),
 ("S25","multimodal-retrieval","Document Parsing","공공행정문서","공개 데이터",None,2499,
  "vlm_exp34/vlm_dataset_upload/train/public/multimodal-retrieval","팀 공용 업로드분","use"),
]

# 버전별 출처 사용량 — train.jsonl 을 역추적한 고유 이미지 수
VER_SRC = {
 "upload_mix":       {"S03":187,"S13":3000},
 "exp004c_260813":   {"S01":892,"S03":182,"S13":2000,"S14":5461,"S15":139,"S25":2499},
 "exp007_all":       {"S01":892,"S02":9520,"S03":182,"S04":1000,"S05":1000,"S07":12000,"S09":8381,
                      "S10":2250,"S11":4978,"S13":2000,"S14":5461,"S15":139,"S16":155,"S25":2499},
 "exp007_all_final": {"S01":892,"S02":9520,"S03":182,"S04":1000,"S05":828,"S07":10968,"S09":8381,
                      "S10":1437,"S11":4978,"S13":2000,"S14":5461,"S15":139,"S16":115,"S25":2499},
 "exp009_full":      {"S01":892,"S02":9520,"S03":182,"S04":1000,"S05":828,"S07":10968,"S09":8381,
                      "S10":1437,"S11":4978,"S13":2000,"S14":5461,"S15":139,"S16":115,"S25":2499},
 "exp010_lean":      {"S01":25,"S02":224,"S03":182,"S07":308,"S09":171,"S10":43,"S11":114,
                      "S13":2000,"S15":139,"S16":7,"S25":2499},
 "exp011_mix":       {"S01":374,"S02":3318,"S03":182,"S04":1000,"S05":828,"S07":5037,"S09":2885,
                      "S10":507,"S11":1734,"S13":2000,"S14":5461,"S15":139,"S16":45,"S25":2499},
 "exp012_mix":       {"S01":374,"S02":3318,"S03":182,"S04":1000,"S05":828,"S07":5037,"S09":2885,
                      "S10":507,"S11":1734,"S13":2000,"S14":5461,"S15":139,"S16":45,
                      "S17":8000,"S18":5000,"S19":4000,"S20":491,"S25":2499},
}
# (버전, 사용 실험, train행, val행, 경로, 상태)
VERSIONS = [
 ("upload_mix","exp_003",4496,92,"vlm_exp34/receipt_data/upload_mix","완료"),
 ("exp004c_260813","exp_004 / exp_005",11173,90,"vlm_exp34/receipt_data/exp004c_260813","완료"),
 ("exp007_all","exp_006",50457,90,"vlm_exp34/receipt_data/exp007_all","완료"),
 ("exp007_all_final","exp_007 / exp_008",48400,90,"vlm_exp34/receipt_data/exp007_all_final","완료"),
 ("exp009_full","exp_009",86256,90,"vlm_exp34/receipt_data/exp009_full","완료"),
 ("exp010_lean","exp_010",9898,90,"vlm_exp34/receipt_data/exp010_lean","완료"),
 ("exp011_mix","exp_011",30195,90,"vlm_exp34/receipt_data/exp011_mix","완료"),
 ("exp012_mix","exp_012",47686,90,"vlm_exp34/receipt_data/exp012_mix","학습 중"),
]
def used(sid):
    """exp_012 기준 활용 고유 이미지 수"""
    return VER_SRC["exp012_mix"].get(sid, 0)


# ── 출처별 언어 분포 (exp_012 학습셋 행 기준 · HTML 태그 제거 후 문자 종류로 판정) ──
LANG_CATS = ['KOR', '라틴문자', 'JPN', 'ZHO', 'RUS', '숫자·기호']
LANG = {
  "S01": {"KOR":255, "라틴문자":50, "JPN":6, "ZHO":10, "RUS":0, "숫자·기호":53},
  "S02": {"KOR":897, "라틴문자":0, "JPN":0, "ZHO":0, "RUS":0, "숫자·기호":2421},
  "S03": {"KOR":4080, "라틴문자":192, "JPN":24, "ZHO":72, "RUS":0, "숫자·기호":0},
  "S04": {"KOR":1000, "라틴문자":0, "JPN":0, "ZHO":0, "RUS":0, "숫자·기호":0},
  "S05": {"KOR":828, "라틴문자":0, "JPN":0, "ZHO":0, "RUS":0, "숫자·기호":0},
  "S07": {"KOR":4442, "라틴문자":589, "JPN":0, "ZHO":0, "RUS":0, "숫자·기호":6},
  "S09": {"KOR":434, "라틴문자":53, "JPN":0, "ZHO":0, "RUS":0, "숫자·기호":2398},
  "S10": {"KOR":261, "라틴문자":20, "JPN":0, "ZHO":0, "RUS":0, "숫자·기호":226},
  "S11": {"KOR":0, "라틴문자":1148, "JPN":0, "ZHO":0, "RUS":0, "숫자·기호":586},
  "S13": {"KOR":1998, "라틴문자":2, "JPN":0, "ZHO":0, "RUS":0, "숫자·기호":0},
  "S14": {"KOR":0, "라틴문자":5461, "JPN":0, "ZHO":0, "RUS":0, "숫자·기호":0},
  "S15": {"KOR":104, "라틴문자":35, "JPN":0, "ZHO":0, "RUS":0, "숫자·기호":0},
  "S16": {"KOR":22, "라틴문자":6, "JPN":0, "ZHO":3, "RUS":0, "숫자·기호":14},
  "S17": {"KOR":531, "라틴문자":5816, "JPN":2, "ZHO":975, "RUS":676, "숫자·기호":0},
  "S18": {"KOR":345, "라틴문자":3610, "JPN":2, "ZHO":632, "RUS":411, "숫자·기호":0},
  "S19": {"KOR":0, "라틴문자":4000, "JPN":0, "ZHO":0, "RUS":0, "숫자·기호":0},
  "S20": {"KOR":0, "라틴문자":491, "JPN":0, "ZHO":0, "RUS":0, "숫자·기호":0},
  "S25": {"KOR":2492, "라틴문자":5, "JPN":0, "ZHO":0, "RUS":0, "숫자·기호":2},
}

# ── 김덕기 주임 실험노트 포맷 대응 항목 ──
TRAIN_META = {  # exp_id: (desc, train_type, model_name, version, author, 학습서버, dtype, packing)
 "exp_001": ("baseline","-","luxia-document (기배포)","-","용윤정","52번","bf16","False"),
 "exp_003": ("format-align","LoRA","luxia-vlm-doc-parse","v0.1.0","용윤정","52번","bf16","False"),
 "exp_004": ("crop-intro","LoRA","luxia-vlm-doc-parse","v0.2.0","용윤정","52번","bf16","False"),
 "exp_005": ("aug-online","LoRA","luxia-vlm-doc-parse","v0.3.0","용윤정","52번","bf16","False"),
 "exp_006": ("scale-up","LoRA","luxia-vlm-doc-parse","v0.4.0","용윤정","52번","bf16","False"),
 "exp_007": ("clean-label","LoRA","luxia-vlm-doc-parse","v0.5.0","용윤정","52번","bf16","False"),
 "exp_008": ("base-swap","LoRA","luxia-vlm-doc-parse-27b","v0.6.0","용윤정","52번","bf16","False"),
 "exp_009": ("crop-max","LoRA","luxia-vlm-doc-parse","v0.7.0","용윤정","52번","bf16","False"),
 "exp_010": ("crop-lean","LoRA","luxia-vlm-doc-parse","v0.8.0","용윤정","52번","bf16","False"),
 "exp_011": ("crop-balanced","LoRA","luxia-vlm-doc-parse","v0.9.0","용윤정","52번","bf16","False"),
 "exp_012": ("new-domain","LoRA","luxia-vlm-doc-parse","v0.10.0","용윤정","52번","bf16","False"),
}


# ── 길이 분포 (exp_012 학습셋 · 문자 수) — 김덕기 주임 Length 시트 대응 ──
LEN_BINS = ['0-499', '500-999', '1000-1499', '1500-1999', '2000-2499', '2500-2999', '3000-3499', '3500-3999', '4000-4499', '4500-4999', '5000-5999', '6000-7999', '8000+']
LEN_COLS = ['신규 서식', '영수증 크롭', '영수증 전체', '표(HTML)', '공공행정 파싱', '공공행정 OCR']
LEN_COMPLETION = {'신규 서식': [3112, 1902, 1990, 2098, 3187, 3397, 788, 600, 469, 1, 86, 0, 0], '영수증 크롭': [12747, 1134, 16, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0], '영수증 전체': [701, 4463, 864, 144, 0, 24, 0, 0, 0, 0, 0, 0, 0], '표(HTML)': [310, 1443, 1212, 768, 562, 379, 223, 183, 111, 83, 154, 33, 0], '공공행정 파싱': [1154, 1195, 149, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], '공공행정 OCR': [1982, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
LEN_PROMPT = {'신규 서식': [17630, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '영수증 크롭': [0, 12000, 0, 0, 1900, 0, 0, 0, 0, 0, 0, 0, 0], '영수증 전체': [6196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '표(HTML)': [5461, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '공공행정 파싱': [2499, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '공공행정 OCR': [2000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}


# ── 이미지 해상도 분포 (exp_012 학습셋 행 기준) ──
RES_LABELS = ['~0.05MP', '0.05~0.2MP', '0.2~0.5MP', '0.5~1MP', '1~2MP', '2~4MP', '4MP+']
RES_COLS = ['신규 서식', '영수증 크롭', '영수증 전체', '표(HTML)', '공공행정 파싱', '공공행정 OCR']
RES = {'신규 서식': [0, 0, 117, 5409, 7089, 4524, 491], '영수증 크롭': [11346, 2487, 62, 5, 0, 0, 0], '영수증 전체': [0, 1128, 2684, 1710, 170, 168, 336], '표(HTML)': [1800, 3127, 534, 0, 0, 0, 0], '공공행정 파싱': [0, 0, 0, 0, 0, 0, 2499], '공공행정 OCR': [0, 0, 0, 4, 0, 2, 1994]}
