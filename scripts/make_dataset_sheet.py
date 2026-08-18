#!/usr/bin/env python3
"""데이터셋 현황 + 학습 기록 엑셀 시트를 만든다.

2026-08-12 미팅 요청분. 컬럼은 확정본이 아니라 시작점이다 —
논의하면서 계속 추가/수정하는 것을 전제로 만든다.

시트 구성
    1. 데이터셋_현황   데이터명 × 태스크 단위. 같은 이미지를 다른 태스크로 썼으면 행이 갈린다
    2. 태스크_정의     OCR/Parsing/VQA/KIE/Grounding 용어 정의 (합의 전 초안)
    3. 학습_기록       run 단위. 목적/가설 · 데이터 · 파라미터 · 결과 · 결론
    4. 컬럼_정의       각 컬럼을 무슨 기준으로 채우는지

usage:
    python3 scripts/make_dataset_sheet.py
"""
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

OUT = Path("/data/workspace/yjyong/VLM_데이터셋_현황_및_학습기록_20260812.xlsx")

HDR_FILL = PatternFill("solid", fgColor="1F3864")
HDR_FONT = Font(color="FFFFFF", bold=True, size=10)
SUB_FILL = PatternFill("solid", fgColor="D9E2F3")
WARN_FILL = PatternFill("solid", fgColor="FFF2CC")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

# ---------------------------------------------------------------------------
# 1. 데이터셋 현황
# ---------------------------------------------------------------------------
DS_COLS = [
    ("No", 5), ("공개구분", 9), ("데이터명", 22), ("도메인", 11), ("문서종류", 13),
    ("이미지 취득방식", 14), ("이미지 포맷", 11), ("수집방법", 26), ("라벨 생성방식", 14),
    ("태스크", 12), ("태스크 세부", 22), ("보유 이미지", 11), ("사용 이미지", 11),
    ("학습 row", 10), ("품질등급", 11), ("학습 사용", 22), ("평가 사용", 11),
    ("라이선스", 20), ("경로", 46), ("담당", 8), ("비고", 44),
]

Y, N = "O", "-"
DS_ROWS = [
    # --- private ------------------------------------------------------------
    ["private", "luxia-receipt (gcse)", "영수증", "영수증", "웹수집", "jpg/png",
     "document-parse API + 사람검수", "반자동(사람검수)", "Parsing", "영수증→마크다운",
     69, 69, 69, "골드셋", "upload-v1/v2/v3", N, "internal_only",
     "receipt_data/labeled/train (gcse_*)", "윤정",
     "구글 크롤 이미지를 파서로 초벌 파싱 후 라벨링 툴에서 사람 검수. 팀 업로드본에는 아직 84건만 반영(11건 미동기화)"],
    ["private", "luxia-receipt (kid)", "영수증", "영수증", "스캔/사진", "png",
     "document-parse API + 사람검수", "반자동(사람검수)", "Parsing", "영수증→마크다운",
     26, 26, 26, "골드셋", "upload-v1/v2/v3", N, "internal_only",
     "receipt_data/labeled/train (kid_*)", "윤정",
     "KorIE KID 이미지를 사람이 재검수. 08-12 kid_IMG00001 bbox 재검수 반영"],
    ["private", "luxia-receipt (val)", "영수증", "영수증", "웹수집/스캔", "jpg/png",
     "document-parse API + 사람검수", "반자동(사람검수)", "Parsing", "영수증→마크다운",
     49, 0, 0, "골드셋", N, Y, "internal_only",
     "receipt_data/labeled/val", "윤정", "평가 전용. 학습에 미투입"],
    ["private", "luxia-receipt (test)", "영수증", "영수증", "웹수집/스캔", "jpg/png",
     "document-parse API + 사람검수", "반자동(사람검수)", "Parsing", "영수증→마크다운",
     50, 0, 0, "골드셋", N, Y, "internal_only",
     "receipt_data/labeled/test", "윤정", "평가 전용. 학습에 미투입"],
    ["private", "holdout_eval", "영수증+공공문서", "혼합", "-", "-",
     "기존 패키지에서 추출", "-", "Parsing", "영수증 10 / 공공문서 10",
     20, 0, 0, "골드셋", N, Y, "internal_only",
     "receipt_data/review/holdout_eval.jsonl", "윤정",
     "학습셋과 교집합 0인 고정 홀드아웃. 08-11 이후 모든 평가는 이걸로 통일"],
    ["private", "receipts3000 (합성)", "영수증", "영수증", "합성렌더링", "png",
     "합성 생성기 (이미지+정답 동시 생성)", "합성(자동)", "KIE", "field_qa 7종",
     3000, 2400, 16800, "실버", "receipt-v3-qwen36", N, "internal_only",
     "receipt_data/receipts3000", "윤정",
     "markdown은 렌더링 원본값이라 정확. 좌표·블록은 파서 결과"],
    ["private", "receipts3000 (합성)", "영수증", "영수증", "합성렌더링", "png",
     "합성 생성기 (이미지+정답 동시 생성)", "합성(자동)", "KIE", "amount_qa 3종",
     3000, 2400, 7200, "실버", "receipt-v3-qwen36", N, "internal_only",
     "receipt_data/receipts3000", "윤정", ""],
    ["private", "receipts3000 (합성)", "영수증", "영수증", "합성렌더링", "png",
     "합성 생성기 (이미지+정답 동시 생성)", "합성(자동)", "KIE", "json_kie / merchant_info / items_only",
     3000, 2400, 7200, "실버", "receipt-v3-qwen36", N, "internal_only",
     "receipt_data/receipts3000", "윤정", ""],
    ["private", "receipts3000 (합성)", "영수증", "영수증", "합성렌더링", "png",
     "합성 생성기 (이미지+정답 동시 생성)", "합성(자동)", "Parsing", "md_table (표→마크다운)",
     3000, 2400, 2400, "실버", "receipt-v3-qwen36 / upload-v1~v3", N, "internal_only",
     "receipt_data/receipts3000", "윤정", "upload 패키지에는 92장만 투입"],
    ["private", "crawl_google", "영수증", "영수증", "웹수집", "jpg/png",
     "구글 검색 크롤링", "미라벨", "-", "원본 풀 (라벨링 대기)",
     246, 0, 0, "검증전", N, N, "확인필요",
     "receipt_data/crawl_google", "윤정",
     "pass1 759장에서 필터링해 246장 남김. 라벨링 대기 풀"],
    ["private", "eval_crops", "영수증", "영수증", "크롭", "jpg",
     "bbox 크롭", "자동", "OCR", "크롭 이미지 텍스트 인식",
     1077, 0, 0, "검증전", N, N, "internal_only",
     "receipt_data/eval_crops", "윤정", "블록 크롭 OCR 실험용. 현재 미사용"],
    # --- public -------------------------------------------------------------
    ["public", "aihub-71299-ocr", "공공문서", "공공문서(스캔)", "스캔", "jpg",
     "공개셋 원본 라벨", "공개셋 원본", "OCR", "페이지 전문 평문",
     64228, 36598, 36598, "실버", "upload-v1/v2/v3 (6,000장 샘플)", N,
     "AI Hub 이용약관 (국내 연구·개발)", "vlm_dataset_upload/train/public/aihub-71299-ocr", "윤정",
     "후보 64,228 중 27,630 제외(손글씨>10% 19,124 / 표서식 8,500 / 박스부족 6). "
     "표서식 제외는 영수증 표→마크다운 학습과 충돌해서 뺀 것 — 재논의 필요"],
    ["public", "kogovdoc-bench", "공공문서", "공공문서/논문", "디지털문서", "png",
     "공개셋 원본 라벨", "공개셋 원본", "Parsing", "doc_parsing (government 229 / paper 65)",
     294, 294, 294, "실버", N, N, "확인필요 (HF 카드 미명시)",
     "VLM/dataset/train/public/kogovdoc-bench", "확인필요",
     "라이선스 미확인 — 학습 투입 전 확인 필요"],
    ["public", "multimodal-retrieval", "일반문서", "혼합", "디지털문서", "jpg",
     "공개셋 원본 라벨", "공개셋 원본", "Parsing", "전체텍스트 파싱",
     60480, 14208, 14208, "실버", N, N, "AI Hub 이용약관",
     "VLM/dataset/train/public/multimodal-retrieval", "확인필요",
     "manifest 60,480 / train.jsonl 14,208 — 차이 확인 필요"],
    ["public", "pubtabnet-html", "표", "논문 표", "디지털문서", "jpg",
     "공개셋 원본 라벨", "공개셋 원본", "Parsing", "표→HTML 복원",
     500777, 500777, 500777, "실버", N, N, "CDLA-Permissive (PubTabNet)",
     "VLM/dataset/train/public/pubtabnet-html", "확인필요",
     "영어 데이터. 단일 데이터가 전체의 91% — 그대로 넣으면 다른 태스크가 묻힘"],
    ["public", "CORD", "영수증", "영수증", "사진촬영", "png",
     "공개셋 원본 라벨", "공개셋 원본", "KIE", "amount_qa / json_kie / items_only",
     1000, 800, 3241, "실버", "receipt-v3-qwen36", N, "CC BY 4.0 (확인필요)",
     "receipt_data/public/cord", "윤정", "인도네시아 영수증. 파서 대조 미실행"],
    ["public", "SROIE", "영수증", "영수증", "스캔", "jpg",
     "공개셋 원본 라벨", "공개셋 원본", "KIE", "field_qa / amount_qa / merchant_info",
     973, 500, 2498, "실버", "receipt-v3-qwen36", N, "research_only (ICDAR)",
     "receipt_data/public/sroie", "윤정",
     "파서 대조 결과 필드값 완전일치 91.6% (506장/2,022필드). 날짜가 가장 약함"],
    ["public", "KorIE KID", "영수증", "영수증", "스캔/사진", "png",
     "공개셋 원본 라벨", "공개셋 원본", "KIE", "field_qa 4종 / amount_qa 3종 / merchant_info",
     408, 408, 2693, "실버", "receipt-v3-qwen36", N, "확인필요",
     "receipt_data/korie/kid", "윤정",
     "파서 커버리지 98.4% / IoU0.5 F1 8.9% — 영역검출은 신뢰, 필드분할은 불가"],
    ["public", "KorIE OCR", "일반", "혼합", "확인필요", "jpg",
     "공개셋 원본 라벨", "공개셋 원본", "OCR", "미사용",
     5356, 0, 0, "검증전", N, N, "확인필요",
     "receipt_data/korie/ocr", "윤정", "train 5,356 / val 1,785 / test 1,786. 아직 미투입"],
    ["public", "KorIE IE", "일반", "혼합", "-", "csv",
     "공개셋 원본 라벨", "공개셋 원본", "KIE", "미사용",
     674, 0, 0, "검증전", N, N, "확인필요",
     "receipt_data/korie/ie", "윤정",
     "GT 문자열 파서 포함률 87.9%(실질 92.5%). csv 포맷이라 변환 필요"],
    ["public", "AI Hub 71845 (학술논문)", "논문", "논문", "디지털문서", "-",
     "공개셋 원본 라벨", "공개셋 원본", "-", "제외 판정",
     0, 0, 0, "제외", N, N, "AI Hub 이용약관",
     "(내려받았다가 제외)", "윤정",
     "24GB 받아 열어보니 요약·캡션 데이터. OCR 정답이 아니라 후보에서 제외"],
]

# ---------------------------------------------------------------------------
# 2. 태스크 정의 (합의 전 초안)
# ---------------------------------------------------------------------------
TASK_COLS = [("태스크", 13), ("한 줄 정의", 40), ("입력", 14), ("출력", 24),
             ("판별 기준", 44), ("우리 데이터 예", 30), ("헷갈리는 지점", 46)]
TASK_ROWS = [
    ["OCR", "이미지의 글자를 읽어 텍스트로 옮긴다", "이미지(전체/크롭)", "평문 텍스트",
     "구조를 복원하지 않는다. 읽은 순서대로 글자만 나온다",
     "aihub-71299 (페이지 전문), eval_crops",
     "표가 있는 문서를 평문으로 뱉게 학습하면 Parsing과 충돌한다"],
    ["Parsing", "문서의 구조까지 복원한다 (제목/문단/표/이미지 영역 구분)", "이미지(전체 페이지)",
     "마크다운 / HTML",
     "레이아웃 디텍션을 거쳐 블록 단위로 나눈 뒤 구조를 갖춘 텍스트를 만든다. "
     "표·차트·이미지를 다룰 수 있게 구축된 데이터는 모두 Parsing",
     "luxia-receipt, kogovdoc-bench, pubtabnet-html, receipts3000 md_table",
     "OCR과의 경계 — 출력에 구조(표/헤딩)가 있으면 Parsing으로 본다"],
    ["KIE", "정해진 필드의 값을 뽑는다 (핵심정보 추출)", "이미지 + 필드명",
     "필드값 / JSON",
     "무엇을 뽑을지 스키마가 미리 정해져 있다",
     "SROIE, CORD, KorIE KID, receipts3000 field_qa/amount_qa",
     "질문 형태로 물으면 VQA처럼 보이지만, 필드가 고정이면 KIE"],
    ["VQA", "이미지에 대한 자유 질문에 답한다", "이미지 + 자유 질문", "자유 텍스트",
     "질문이 미리 정해져 있지 않다", "(현재 보유 없음)",
     "KIE와의 경계 — 스키마 고정 여부로 가른다"],
    ["Grounding", "이 영역이 무엇인지 라벨을 붙인다", "이미지", "bbox + 라벨",
     "영역을 찾아 '본문/이미지/표'처럼 종류를 지정한다",
     "(레이아웃 디텍션 결과 — 현재 학습 미투입)",
     "'bbox 예측'·'레이아웃 디텍션'과 같은 말로 쓰이고 있어 용어 통일 필요"],
    ["Captioning", "이미지를 설명하는 문장을 만든다", "이미지", "자유 텍스트",
     "이미지에 없는 정보를 요약·서술한다", "(현재 보유 없음)",
     "AI Hub 71845가 여기에 해당해서 OCR 후보에서 제외했다"],
]

# ---------------------------------------------------------------------------
# 3. 학습 기록
# ---------------------------------------------------------------------------
RUN_COLS = [
    ("No", 5), ("run 이름", 20), ("일자", 11), ("상태", 10), ("베이스 모델", 18),
    ("학습 목적 / 가설", 40), ("학습 데이터", 30), ("train row", 10), ("val row", 9),
    ("학습 범위", 24), ("LoRA", 14), ("lr", 9), ("max step", 10), ("실제 step", 10),
    ("epoch", 7), ("train loss", 14), ("eval loss", 14), ("GPU", 9), ("소요", 10),
    ("평가 결과", 46), ("결론 / 문제점", 52), ("산출물 경로", 34),
]
RUN_ROWS = [
    ["qwen4b-block", "08-05", "완료", "Qwen3.5-4B",
     "블록 단위 크롭으로 소형 모델이 영수증을 읽을 수 있는지 탐색",
     "clean_v1 블록셋", "-", "-", "LoRA", "r=16 a=32", "1e-4", 98, 98, 2.0,
     "4.604 → 2.319", "-", "1장", "-",
     "미측정", "탐색용. loss가 2점대에서 안 내려가 블록 단위 접근은 접음", "ml/runs/qwen4b-block"],
    ["gemma4e4b-block", "08-05", "완료", "gemma-4-E4B-it",
     "같은 블록셋으로 gemma 계열 비교",
     "clean_v1 블록셋", "-", "-", "LoRA", "r=16 a=32", "1e-4", 98, 98, 2.0,
     "1.163 → 0.063", "0.1244", "1장", "-",
     "미측정", "loss는 가장 낮았으나 과적합 의심 (98step에 0.06). 모델 비교용", "ml/runs/gemma4e4b-block"],
    ["qwen36-clean-v1-300", "08-06", "완료", "Qwen3.6-35B-A3B",
     "35B로 정제셋(clean_v1) 300스텝 실험",
     "clean_v1", "26,347", "3,810", "LoRA + aligner", "r=16 a=32", "1e-4", 300, 300, 0.05,
     "2.604 → 0.748", "1.1881", "2장", "-",
     "미측정", "35B 첫 정상 학습. 이후 trainset_v3로 전환", "ml/runs/qwen36-clean-v1-300"],
    ["upload-v1", "08-11", "완료", "Qwen3.5-4B",
     "팀 규격 업로드 패키지(영수증+AI Hub)로 4B가 학습되는지 확인",
     "upload_mix", "-", "200", "LoRA", "r=16 a=32", "1e-4", 300, 300, 0.35,
     "3.097 → 2.941", "2.5801", "1장", "-",
     "홀드아웃 미적용 (검증 6건, 그중 2건 오염)",
     "검증셋이 학습셋과 겹쳐 결과를 신뢰할 수 없었음 → 고정 홀드아웃 20건 신설", "ml/runs/upload-v1"],
    ["upload-v2", "08-11", "완료", "Qwen3.5-4B",
     "영수증 비중을 오버샘플 8배로 올려 영수증 성능을 끌어올린다",
     "upload_mix (영수증 8× / 16.8%)", "7,208", "200", "LoRA", "r=16 a=32", "1e-4",
     2000, 2000, 1.11, "3.012 → 0.947", "2.0688", "1장", "-",
     "영수증 CER 0.349→0.663 악화(2/10) · 공공문서 0.079→0.376 악화(2/10)",
     "회귀. 원인은 영수증 151장을 8회 반복해 표 형식을 통째로 암기 — 이미지를 읽는 대신 "
     "외운 템플릿을 출력(없는 품목표 생성, 다른 서식 출력)", "ml/runs/upload-v2"],
    ["upload-v3", "08-11", "완료", "Qwen3.5-4B",
     "오버샘플만 8→2로 낮춰 암기 회귀가 사라지는지 확인 (단일 변수 변경)",
     "upload_mix (영수증 2× / 5.1%)", "6,324", "200", "LoRA", "r=16 a=32", "1e-4",
     2000, 2000, 1.27, "3.686 → 1.820", "2.1039", "1장", "8.5h",
     "ckpt250: 영수증 0.348→0.298(5/10) 공공문서 0.080→0.026(8/10) / "
     "ckpt2000: 영수증 0.302(8/10) 공공문서 0.169(4/10)",
     "회귀는 사라짐. 다만 두 태스크가 반대로 움직인다 — 영수증은 2,000스텝이 최고 승률, "
     "공공문서는 500스텝 이후 계속 악화. 단일 최적점이 없어 기준 ckpt 미확정",
     "ml/runs/upload-v3 / receipt_data/review/upload-v3_ckpt_sweep.json"],
    ["receipt-v3-qwen36", "08-09~", "진행중", "Qwen3.6-35B-A3B",
     "영수증 KIE·파싱 전 태스크를 35B로 본학습. 필드추출·금액·표를 한 모델에서 처리",
     "trainset_v3 (합성2400/CORD800/KID408/SROIE500)", "39,632", "5,844",
     "LoRA + aligner (ViT 동결)", "r=16 a=32, 49.8M(0.142%)", "1e-4",
     19816, 17500, 1.77, "2.197 → 1.024",
     "2500:1.039 → 15000:0.9038 → 17500:0.9006", "6·7번", "68h+ (잔여 ~8h)",
     "학습 중 — 미평가",
     "eval loss 계속 하강, 과적합 신호 없음. 완료 후 홀드아웃 20건으로 평가 예정",
     "ml/runs/receipt-v3-qwen36"],
]

# ---------------------------------------------------------------------------
# 4. 컬럼 정의
# ---------------------------------------------------------------------------
COLDEF_COLS = [("시트", 14), ("컬럼", 18), ("무엇을 적나", 52), ("허용값 / 예시", 46)]
COLDEF_ROWS = [
    ["데이터셋_현황", "공개구분", "외부 공개 데이터인지 사내 데이터인지", "public / private"],
    ["데이터셋_현황", "데이터명", "데이터셋 이름. 같은 이름이 태스크별로 여러 행에 나뉜다", "aihub-71299-ocr"],
    ["데이터셋_현황", "도메인", "문서의 큰 갈래", "영수증 / 계산서 / 견적서 / 공공문서 / 논문 / 표 / 일반"],
    ["데이터셋_현황", "문서종류", "도메인 안의 세부 서식", "영수증 / 세금계산서 / 보고서 / 논문 표"],
    ["데이터셋_현황", "이미지 취득방식", "그 이미지가 어떻게 만들어졌나", "사진촬영 / 스캔 / 디지털문서 / 합성렌더링 / 웹수집 / 크롭"],
    ["데이터셋_현황", "이미지 포맷", "실제 파일 확장자", "jpg / png / jpeg"],
    ["데이터셋_현황", "수집방법", "라벨(정답)을 어떤 경로로 얻었나",
     "VLM 파이프라인(레이아웃→파싱) / document-parse API / 외부 API / HTML / 사람 라벨링 / 공개셋 원본 / 합성 생성기"],
    ["데이터셋_현황", "라벨 생성방식", "사람이 얼마나 개입했나", "자동 / 반자동(사람검수) / 사람 / 합성(자동) / 미라벨"],
    ["데이터셋_현황", "태스크", "태스크_정의 시트의 값만 쓴다", "OCR / Parsing / KIE / VQA / Grounding / Captioning"],
    ["데이터셋_현황", "태스크 세부", "같은 태스크 안의 세부 구분", "field_qa 7종 / amount_qa 3종 / md_table"],
    ["데이터셋_현황", "보유/사용 이미지", "가지고 있는 장수와 실제로 학습에 넣은 장수를 따로 적는다",
     "보유 64,228 → 사용 36,598 처럼"],
    ["데이터셋_현황", "학습 row", "이미지 1장에서 여러 QA를 뽑으면 장수보다 크다", "2,400장 → 31,200 row"],
    ["데이터셋_현황", "품질등급", "골드셋 = 사람이 직접 검수. 평가에 항상 포함한다",
     "골드셋 / 실버 / 검증전 / 제외"],
    ["데이터셋_현황", "학습 사용", "실제로 투입된 run 이름. 안 썼으면 '-'", "upload-v3, receipt-v3-qwen36"],
    ["데이터셋_현황", "평가 사용", "평가셋으로 쓰이는지", "O / -"],
    ["학습_기록", "학습 목적 / 가설", "왜 이 학습을 돌렸나. 무엇이 좋아질 거라 봤나",
     "'오버샘플을 낮추면 암기 회귀가 사라진다'"],
    ["학습_기록", "평가 결과", "고정 홀드아웃 기준 수치. 어떤 ckpt인지 같이 적는다",
     "ckpt250 영수증 CER 0.348→0.298 (5/10 개선)"],
    ["학습_기록", "결론 / 문제점", "왜 그런 결과가 나왔는지. 다음 실험에 넘길 것",
     "회귀 원인은 8회 오버샘플에 의한 템플릿 암기"],
]


def style_sheet(ws, cols, n_rows, wrap_from=0):
    for i, (name, width) in enumerate(cols, 1):
        ws.column_dimensions[get_column_letter(i)].width = width
        c = ws.cell(row=1, column=i, value=name)
        c.fill, c.font = HDR_FILL, HDR_FONT
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BORDER
    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{n_rows + 1}"
    for row in ws.iter_rows(min_row=2, max_row=n_rows + 1, max_col=len(cols)):
        for c in row:
            c.alignment = Alignment(vertical="top", wrap_text=True)
            c.border = BORDER


def main():
    wb = Workbook()

    # 1. 데이터셋 현황
    ws = wb.active
    ws.title = "데이터셋_현황"
    for i, r in enumerate(DS_ROWS, 1):
        ws.append([i] + r)
    style_sheet(ws, DS_COLS, len(DS_ROWS))
    for row in ws.iter_rows(min_row=2, max_row=len(DS_ROWS) + 1):
        if row[1].value == "public":
            row[1].fill = SUB_FILL
        if "확인필요" in str(row[17].value) or row[14].value == "검증전":
            row[17].fill = WARN_FILL

    # 2. 태스크 정의
    ws2 = wb.create_sheet("태스크_정의")
    for r in TASK_ROWS:
        ws2.append(r)
    ws2.insert_rows(1)
    style_sheet(ws2, TASK_COLS, len(TASK_ROWS))
    ws2.append([])
    ws2.append(["※ 합의 전 초안입니다. 금요일 미팅에서 확정합니다. "
                "특히 OCR / Parsing 경계와 Grounding 용어가 미정입니다."])
    ws2.cell(row=ws2.max_row, column=1).font = Font(bold=True, color="C00000")

    # 3. 학습 기록
    ws3 = wb.create_sheet("학습_기록")
    for i, r in enumerate(RUN_ROWS, 1):
        ws3.append([i] + r)
    style_sheet(ws3, RUN_COLS, len(RUN_ROWS))
    for row in ws3.iter_rows(min_row=2, max_row=len(RUN_ROWS) + 1):
        if row[3].value == "진행중":
            row[3].fill = WARN_FILL

    # 4. 컬럼 정의
    ws4 = wb.create_sheet("컬럼_정의")
    for r in COLDEF_ROWS:
        ws4.append(r)
    ws4.insert_rows(1)
    style_sheet(ws4, COLDEF_COLS, len(COLDEF_ROWS))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"저장: {OUT}")
    print(f"  데이터셋_현황 {len(DS_ROWS)}행 / 태스크_정의 {len(TASK_ROWS)}행 / "
          f"학습_기록 {len(RUN_ROWS)}행 / 컬럼_정의 {len(COLDEF_ROWS)}행")


if __name__ == "__main__":
    main()
