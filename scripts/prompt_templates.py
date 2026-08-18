"""
영수증 VLM 학습용 프롬프트 템플릿 (과제 7종).

설계 근거는 문서 "영수증 VLM — 프롬프트 설정 방안" 을 따른다.
  - 다양화 대상은 '과제'다 (arXiv 2402.10891, 2410.04717).
    말바꾸기를 늘리는 것보다 과제 종류를 늘리는 편이 일반화에 효과적이다.
  - 고정 대상은 '출력 형식·범위'다. 이전 실험에서 완전일치 0.0% 가 나온 원인은
    정답에 없는 필드(주소·전화 등)를 모델이 함께 출력해 형식 불일치로 처리된 것이었다.
    형식 규칙(FORMAT_RULE)은 전 과제에 붙이고, 범위 규칙(SCOPE_RULE)은 여러
    필드를 한꺼번에 출력하는 과제에만 붙인다. 범위 규칙을 전 과제에 일괄
    부착하면 field_qa/사업자번호 같은 단일 필드 질의가 자기모순이 된다.
  - 표현 변형은 과제당 3~4개면 충분하다. 여기에 리소스를 더 쓰지 않는다.

프롬프트는 학습 jsonl 의 messages[0].content 에 그대로 기록된다. 즉 템플릿을
바꾸면 데이터셋을 다시 만들어야 한다. 데이터 생성보다 이 파일 확정이 먼저다.

usage:
    from prompt_templates import build_prompt, iter_field_tasks, TASKS

    build_prompt("md_table")                      # 랜덤 말투 + 공통 규칙
    build_prompt("md_table", variant=0)           # 말투 고정 (재현용)
    build_prompt("field_qa", field="상호명")
    for field, p in iter_field_tasks("amount_qa"): # 합계/공급가액/부가세 전개
        ...
"""
import random

# 공통 규칙은 두 조각이다.
#
#   FORMAT_RULE : 모든 과제에 붙는다. 군더더기 금지.
#   SCOPE_RULE  : '정답 범위 밖 필드를 함께 출력하지 마라'. 이건 전 과제에
#                 붙이면 안 된다. field_qa/사업자번호 처럼 그 필드를 묻는
#                 과제에 붙으면 "사업자번호를 물어놓고 사업자번호를 쓰지
#                 말라"는 자기모순이 된다. 전 과제 일괄 부착 시 52,380건 중
#                 14,117건(27%)이 모순이었다.
#
# 그래서 SCOPE_RULE 은 '여러 필드를 한꺼번에 출력하는 과제'에만 붙인다.
# 단일 값을 묻는 field_qa / amount_qa 에는 FORMAT_RULE 만 붙는다.
FORMAT_RULE = "설명이나 인사말 없이 결과만 출력하라."
SCOPE_RULE = (
    "정답에 포함되지 않은 항목(주소, 전화번호, 사업자번호, 결제수단 등)은 출력하지 마라."
)

# 하위호환용. 기존 코드가 참조할 수 있어 남긴다.
COMMON_RULES = f"{SCOPE_RULE} {FORMAT_RULE}"

# field_qa / amount_qa 에서 전개할 필드.
# GT JSON 에 실제로 보존된 키만 넣는다. 없는 필드를 물으면 학습이 오염된다.
FIELD_QA_FIELDS = ["상호명", "거래일시", "사업자번호", "전화번호", "주소", "결제수단"]
AMOUNT_QA_FIELDS = ["합계", "공급가액", "부가세"]


def _has_batchim(word):
    """마지막 글자에 받침이 있는가. 한글이 아니면 받침 있는 것으로 본다(숫자·영문 대비)."""
    ch = word[-1]
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    return True


def _josa(word, with_batchim, without_batchim):
    return with_batchim if _has_batchim(word) else without_batchim


def _apply_josa(text, field):
    """템플릿의 {field}이(가) / {field}을(를) 를 받침에 맞게 확정한다.

    학습 프롬프트에 그대로 기록되는 문장이므로 '합계이(가)' 같은 표기를 남기면 안 된다.
    """
    text = text.replace("{field}이(가)", field + _josa(field, "이", "가"))
    text = text.replace("{field}을(를)", field + _josa(field, "을", "를"))
    return text.replace("{field}", field)


TASKS = {
    # 1. 전체 -> 마크다운 표
    "md_table": {
        "output": "markdown",
        "needs_field": False,
        "scope_rule": True,
        "variants": [
            "이 영수증의 구매 내역을 표로 작성하라.",
            "영수증 내용을 마크다운 표로 정리해줘.",
            "아래 영수증을 표 형식으로 옮겨 적으세요.",
            "이 영수증을 마크다운으로 변환하라.",
        ],
        "rule": ("상품명·수량·금액 3개 열의 마크다운 표로 작성하고, "
                 "표 아래에 공급가액·부가세·합계를 적어라."),
    },
    # 2. 전체 -> JSON 추출 (KIE)
    "json_kie": {
        "output": "json",
        "needs_field": False,
        # 스키마 8키를 이미 명시했으므로 범위 규칙이 중복되지만, 스키마 밖
        # 필드를 덧붙이는 실패가 실제로 있었으므로 유지한다.
        "scope_rule": True,
        "variants": [
            "이 영수증에서 정보를 JSON으로 추출하라.",
            "영수증 내용을 JSON 형식으로 뽑아줘.",
            "아래 영수증을 구조화된 JSON으로 변환하세요.",
            "이 영수증의 항목들을 JSON으로 정리하라.",
        ],
        # 스키마를 프롬프트에 못박는다. CORD 를 참조해 품목은 객체 배열로 둔다.
        # 필드별 배열로 두면 한 항목만 누락돼도 인덱스 정렬이 통째로 어긋난다.
        "rule": ('다음 키만 사용하라: merchant_name, business_no, transaction_datetime, '
                 'items, subtotal, tax, total, payment_method. '
                 'items 는 각 품목을 {"name", "unit_price", "qty", "amount"} 객체로 담은 '
                 '배열이다. 값이 없으면 null 로 두어라.'),
    },
    # 3. 품목 목록만
    "items_only": {
        "output": "list",
        "needs_field": False,
        "scope_rule": True,
        "variants": [
            "이 영수증에서 구매한 상품 목록을 나열하라.",
            "영수증에 있는 품목들만 알려줘.",
            "아래 영수증의 상품명을 순서대로 적으세요.",
        ],
        "rule": "한 줄에 하나씩 '상품명 수량 금액' 형식으로 적어라. 합계나 결제 정보는 적지 마라.",
    },
    # 4. 단일 필드 질의
    "field_qa": {
        "output": "value",
        "needs_field": True,
        # 묻는 필드가 금지 목록에 들어 있어 모순이 된다. 붙이지 않는다.
        "scope_rule": False,
        "variants": [
            "이 영수증의 {field}이(가) 뭐야?",
            "영수증에서 {field}을(를) 찾아줘.",
            "아래 영수증의 {field}을(를) 알려주세요.",
            "{field}을(를) 추출하라.",
        ],
        "rule": "값 하나만 출력하라.",
    },
    # 5. 금액 계열 질의
    "amount_qa": {
        "output": "value",
        "needs_field": True,
        "scope_rule": False,
        "variants": [
            "이 영수증의 {field}이(가) 얼마야?",
            "영수증에서 {field} 금액을 알려줘.",
            "아래 영수증의 {field}을(를) 구하세요.",
            "{field} 금액을 추출하라.",
        ],
        "rule": "숫자만 출력하라. 통화 기호나 단위는 붙이지 마라.",
    },
    # 6. 크롭 -> 전사
    "crop_ocr": {
        "output": "text",
        "needs_field": False,
        # 크롭 조각에는 주소·전화가 실제로 적혀 있을 수 있다. 전사 과제이므로
        # 보이는 것을 다 써야 한다. 범위 규칙을 붙이면 안 된다.
        "scope_rule": False,
        "variants": [
            "이 이미지에 적힌 글자를 그대로 옮겨 적어라.",
            "잘린 영수증 조각의 텍스트를 읽어줘.",
            "아래 이미지의 내용을 전사하세요.",
        ],
        "rule": "보이는 텍스트만 그대로 출력하라. 추측하거나 보완하지 마라.",
    },
    # 7. 매장 정보만
    "merchant_info": {
        "output": "json",
        "needs_field": False,
        "scope_rule": True,
        "variants": [
            "이 영수증의 매장 정보를 추출하라.",
            "영수증에서 가게 정보만 뽑아줘.",
            "아래 영수증의 판매자 정보를 정리하세요.",
        ],
        "rule": ('merchant_name, business_no, transaction_datetime 세 키만 담은 '
                 'JSON 으로 출력하라. 값이 없으면 null 로 두어라.'),
    },
}


def _fields_for(key):
    if key == "field_qa":
        return FIELD_QA_FIELDS
    if key == "amount_qa":
        return AMOUNT_QA_FIELDS
    return []


def build_prompt(key, field=None, variant=None, rng=None):
    """과제 키 -> 완성된 프롬프트 문자열.

    variant 를 주면 말투를 고정한다. 데이터 재생성 시 동일 결과를 얻으려면 쓴다.
    """
    if key not in TASKS:
        raise KeyError(f"모르는 과제: {key}. 가능: {sorted(TASKS)}")
    t = TASKS[key]
    if t["needs_field"]:
        allowed = _fields_for(key)
        if field is None:
            raise ValueError(f"{key} 는 field 가 필요하다. 가능: {allowed}")
        if field not in allowed:
            raise ValueError(f"{key} 에서 쓸 수 없는 필드: {field}. 가능: {allowed}")

    variants = t["variants"]
    if variant is None:
        variant = (rng or random).randrange(len(variants))
    head = variants[variant % len(variants)]
    if t["needs_field"]:
        head = _apply_josa(head, field)

    tail = f"{SCOPE_RULE} {FORMAT_RULE}" if t.get("scope_rule") else FORMAT_RULE
    return f"{head} {t['rule']} {tail}"


def iter_field_tasks(key, variant=None, rng=None):
    """field_qa / amount_qa 를 필드별로 전개한다. -> [(field, prompt), ...]"""
    fields = _fields_for(key)
    if not fields:
        raise ValueError(f"{key} 는 필드 전개 대상이 아니다.")
    return [(f, build_prompt(key, field=f, variant=variant, rng=rng)) for f in fields]


def all_instructions():
    """실제로 생성 가능한 인스트럭션 전부. 문서의 '20종 이상' 을 실측으로 확인할 때 쓴다."""
    out = []
    for key, t in TASKS.items():
        if t["needs_field"]:
            for f in _fields_for(key):
                for v in range(len(t["variants"])):
                    out.append((key, f, build_prompt(key, field=f, variant=v)))
        else:
            for v in range(len(t["variants"])):
                out.append((key, None, build_prompt(key, variant=v)))
    return out


if __name__ == "__main__":
    import collections

    allp = all_instructions()
    by_task = collections.Counter(k for k, _, _ in allp)
    print(f"과제 {len(TASKS)}종 / 조합 {len(allp)}개\n")
    for k, n in by_task.items():
        nf = len(_fields_for(k))
        extra = f" (필드 {nf}개 x 말투 {len(TASKS[k]['variants'])}개)" if nf else \
                f" (말투 {len(TASKS[k]['variants'])}개)"
        print(f"  {k:14} {n:3}개{extra}")
    print("\n--- 예시 ---")
    for key in ["md_table", "json_kie", "items_only"]:
        print(f"\n[{key}]\n{build_prompt(key, variant=0)}")
    print(f"\n[field_qa / 상호명]\n{build_prompt('field_qa', field='상호명', variant=0)}")
    print(f"\n[amount_qa / 합계]\n{build_prompt('amount_qa', field='합계', variant=0)}")
