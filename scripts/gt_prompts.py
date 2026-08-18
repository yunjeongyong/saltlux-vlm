"""
GT 추출용 프롬프트 (교사 모델에게 보내는 것).

학습 프롬프트(prompt_templates.py)와 목적이 정반대다.
  학습용  : 말투와 과제를 다양화해서 학생 모델을 일반화시킨다.
  GT 추출 : 하나로 고정해서 6,243장에서 일관된 정답을 뽑는다.
            여기서 흔들리면 정답 자체가 오염되고, 학습 프롬프트를 아무리
            잘 짜도 복구되지 않는다.

GT 추출에만 있는 관심사가 셋이다.
  1. 불확실성 — 안 보이는 값을 추측하면 학생 모델이 '환각하는 법'을 배운다.
     읽히지 않으면 null. 이게 이 프롬프트에서 제일 중요한 규칙이다.
  2. 산술 검증 — 영수증은 total = Σ items[].amount, subtotal + tax = total 이
     성립한다. 모델에게 직접 검산시키고, 안 맞으면 숨기지 말고 표시하게 한다.
     사후 필터링만으로는 '어디가 틀렸는지'를 알 수 없다.
  3. 스키마 강제 — 5만 건 규모에서 "JSON만 출력해라"라는 부탁은 반드시 깨진다.
     Structured Outputs 로 API가 보장하게 한다. 프롬프트는 보조 수단이다.

품질 분기: clean_v1 합성본은 깨끗하지만 heavy/extreme 은 열화가 심하고,
크롤링 실사는 손·배경·기울기가 섞인다. 같은 프롬프트로 밀면 저품질에서
추측이 늘어난다. 그래서 품질 힌트를 덧붙인다.

usage:
    from gt_prompts import build_gt_prompt, PAGE_KIE_SCHEMA, build_crop_prompt

    build_gt_prompt(quality="heavy", source="receipts3000")
    build_crop_prompt(label="table")
"""

# ---------------------------------------------------------------- 스키마

# 채택 스키마 8키. CORD 를 따라 품목은 객체 배열로 둔다.
# 필드별 병렬 배열(상품명[], 단가[], 수량[])은 한 항목만 누락돼도 인덱스가
# 통째로 어긋나고, 어긋났는지 자동 검증할 방법이 없다.
PAGE_KIE_SCHEMA = {
    "type": "object",
    "properties": {
        "merchant_name": {"type": ["string", "null"]},
        "business_no": {"type": ["string", "null"]},
        "transaction_datetime": {"type": ["string", "null"]},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "unit_price": {"type": ["integer", "null"]},
                    "qty": {"type": ["integer", "null"]},
                    "amount": {"type": ["integer", "null"]},
                },
                "required": ["name", "unit_price", "qty", "amount"],
                "additionalProperties": False,
            },
        },
        "subtotal": {"type": ["integer", "null"]},
        "tax": {"type": ["integer", "null"]},
        "total": {"type": ["integer", "null"]},
        "payment_method": {"type": ["string", "null"]},
        # 모델이 스스로 검산한 결과. 사후 계산으로도 구할 수 있지만, 모델이
        # 검산하는 과정에서 오독을 스스로 잡아내는 효과가 더 크다.
        "checks": {
            "type": "object",
            "properties": {
                "items_sum_matches_total": {"type": ["boolean", "null"]},
                "subtotal_plus_tax_matches_total": {"type": ["boolean", "null"]},
                "unreadable_fields": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["items_sum_matches_total",
                         "subtotal_plus_tax_matches_total",
                         "unreadable_fields"],
            "additionalProperties": False,
        },
    },
    "required": ["merchant_name", "business_no", "transaction_datetime", "items",
                 "subtotal", "tax", "total", "payment_method", "checks"],
    "additionalProperties": False,
}

CROP_OCR_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "readable": {"type": "boolean"},
    },
    "required": ["text", "readable"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------- 페이지 KIE

_PAGE_BASE = """이 영수증 이미지에서 정보를 추출해 JSON으로 출력하라.

## 추출할 필드

merchant_name         상호명. 지점명이 있으면 함께 포함한다("이디야커피 서초점").
business_no           사업자등록번호. 하이픈을 포함해 인쇄된 그대로.
transaction_datetime  거래일시. "YYYY-MM-DD HH:MM" 으로 정규화한다.
                      날짜만 있으면 "YYYY-MM-DD", 시각만 있으면 null.
items                 구매 품목 배열. 품목 하나가 객체 하나다.
  name                상품명. 인쇄된 그대로.
  unit_price          단가(정수). 표기가 없으면 null. amount/qty 로 계산하지 마라.
  qty                 수량(정수).
  amount              그 품목의 금액(정수).
subtotal              공급가액.
tax                   부가세.
total                 합계.
payment_method        결제수단("현금", "신용카드", "체크카드", "카카오페이" 등).

## 숫자 표기

모든 금액은 정수로만 출력한다. 쉼표·통화기호·단위를 제거한다.
  "84,600원" -> 84600      "₩1,346,000" -> 1346000
음수는 음수로 둔다("-3,900" -> -3900). 소수점이 있으면 반올림하지 말고
가장 가까운 정수로 절사한다.

## 값이 없거나 읽히지 않을 때

이 규칙이 가장 중요하다. 지키지 않으면 이 데이터는 학습에 쓸 수 없다.

- 영수증에 그 항목이 인쇄되어 있지 않으면 null.
- 인쇄는 되어 있으나 글자가 뭉개져 읽을 수 없으면 null 로 두고,
  checks.unreadable_fields 에 그 필드명을 넣는다.
- 절대 추측하지 마라. 비슷한 영수증의 관례, 일반적인 상호명, 그럴듯한
  금액을 만들어내지 마라. 확신이 없으면 null 이 정답이다.
- 일부만 읽히면 읽힌 부분만 쓴다. "이디야커피 서○점" 처럼 뭉개진 글자를
  임의로 채우지 마라 — 읽힌 데까지만 쓰고 unreadable_fields 에 넣는다.
- 품목 일부가 안 보이면 보이는 품목만 items 에 넣고,
  unreadable_fields 에 "items" 를 넣는다. 개수를 맞추려 빈 항목을 만들지 마라.

## 검산

출력 전에 두 가지를 직접 계산해서 확인하라.

1. items 의 amount 를 모두 더한 값이 total 과 같은가.
   -> checks.items_sum_matches_total
2. subtotal + tax 가 total 과 같은가.
   -> checks.subtotal_plus_tax_matches_total

어느 하나라도 맞지 않으면, 먼저 이미지를 다시 보고 숫자를 잘못 읽지
않았는지 확인하라. 대개는 오독이다. 다시 봐도 맞지 않으면 읽은 값을
그대로 두고 해당 항목을 false 로 둔다. 숫자를 맞추려고 값을 고치지 마라.
계산에 필요한 값이 하나라도 null 이면 해당 항목은 null 로 둔다.

## 출력

위 스키마의 JSON 하나만 출력한다. 설명·인사말·마크다운 코드펜스를 붙이지 마라."""

# 품질별 덧붙임. 열화가 심한 이미지일수록 '추측하지 마라'를 다시 강조해야 한다.
_QUALITY_HINT = {
    "clean": "",
    "sharp_photo": "",
    "light": "",
    "screenshot": (
        "\n## 이 이미지에 대하여\n"
        "휴대폰 화면 캡처다. 상단 상태바(시각·배터리·통신사)와 앱 UI 요소는 "
        "영수증 내용이 아니다. 추출 대상에서 제외하라."
    ),
    "heavy": (
        "\n## 이 이미지에 대하여\n"
        "인쇄·촬영 상태가 좋지 않아 글자가 뭉개져 있을 수 있다. "
        "읽히지 않는 글자를 문맥으로 메우려 하지 마라. 읽히는 것만 쓰고 "
        "나머지는 null 로 두는 것이 옳다."
    ),
    "extreme": (
        "\n## 이 이미지에 대하여\n"
        "열화가 심한 이미지다. 상당수 필드가 읽히지 않을 수 있고, 그것이 "
        "정상적인 결과다. 억지로 값을 채우면 이 데이터는 폐기된다. "
        "확신이 서는 값만 쓰고 나머지는 전부 null 로 두어라."
    ),
}

# 소스별 덧붙임. 실사 크롤링본은 배경·손·기울기가 섞인다.
_SOURCE_HINT = {
    "google-crawl": (
        "\n## 이 이미지에 대하여\n"
        "실제로 촬영된 사진이다. 손·책상·배경이 함께 찍혀 있을 수 있고 "
        "영수증이 기울거나 휘어 있을 수 있다. 영수증 영역의 내용만 추출하고, "
        "배경에 있는 다른 물건이나 글자는 무시하라. 영수증이 여러 장 겹쳐 "
        "있으면 가장 크고 온전하게 보이는 한 장만 대상으로 한다."
    ),
    "sroie-2019": (
        "\n## 이 이미지에 대하여\n"
        "영문 영수증이다. 상호명·품목명을 번역하지 말고 인쇄된 그대로 쓴다."
    ),
    "cord-v2": (
        "\n## 이 이미지에 대하여\n"
        "인도네시아 영수증이다. 상호명·품목명을 번역하지 말고 인쇄된 그대로 쓴다. "
        "상단 일부가 흐림 처리되어 있을 수 있다. 그 영역은 읽으려 하지 말고 "
        "null 로 두어라."
    ),
}


def build_gt_prompt(quality=None, source=None):
    """페이지 KIE 프롬프트. 품질·소스 힌트만 덧붙고 본문은 항상 동일하다."""
    p = _PAGE_BASE
    hint = _SOURCE_HINT.get(source, "") or _QUALITY_HINT.get(quality, "")
    return p + hint


# ---------------------------------------------------------------- 크롭 전사

_CROP_BASE = """이 이미지는 영수증에서 잘라낸 조각이다. 여기에 인쇄된 글자를
있는 그대로 옮겨 적어라.

규칙.
- 보이는 텍스트만 쓴다. 문맥으로 보완하거나 추측하지 마라.
- 줄바꿈은 유지한다. 표라면 셀 사이를 공백으로 구분한다.
- 글자가 전혀 읽히지 않으면 text 를 빈 문자열로 두고 readable 을 false 로 한다.
- 일부만 읽히면 읽힌 부분만 쓰고 readable 을 false 로 한다.
- 요약하거나 정리하지 마라. 전사(transcription)이지 해석이 아니다.

JSON 하나만 출력한다."""

# 블록 라벨별 덧붙임. 파서가 붙인 라벨을 그대로 활용한다.
_CROP_LABEL_HINT = {
    "table": "\n이 조각은 표다. 행과 열 구조를 유지해서 옮겨 적어라.",
    "figure_title": "\n이 조각은 제목이다. 보통 한 줄이다.",
    "doc_title": "\n이 조각은 제목이다. 보통 한 줄이다.",
    "paragraph_title": "\n이 조각은 제목이다. 보통 한 줄이다.",
    "header": "\n이 조각은 머리말 영역이다.",
    "footer": "\n이 조각은 꼬리말 영역이다.",
    "image": "\n이 조각은 그림이다. 글자가 없으면 readable 을 false 로 둔다.",
    "seal": "\n이 조각은 도장·로고다. 글자가 없으면 readable 을 false 로 둔다.",
}


def build_crop_prompt(label=None):
    """크롭 전사 프롬프트. 파서가 붙인 블록 라벨을 힌트로 쓴다."""
    return _CROP_BASE + _CROP_LABEL_HINT.get(label, "")


# ---------------------------------------------------------------- 사후 검증

def verify_gt(gt):
    """모델이 낸 checks 를 믿지 않고 다시 계산한다.

    모델에게 검산시키는 목적은 '검산 과정에서 오독을 잡게 하는 것'이지
    그 결과를 그대로 신뢰하는 게 아니다. 최종 판정은 여기서 한다.
    """
    issues = []
    items = gt.get("items") or []
    total = gt.get("total")

    amounts = [i.get("amount") for i in items]
    if items and all(a is not None for a in amounts) and total is not None:
        s = sum(amounts)
        if s != total:
            issues.append(f"items 합계 {s} != total {total}")

    sub, tax = gt.get("subtotal"), gt.get("tax")
    if sub is not None and tax is not None and total is not None:
        if sub + tax != total:
            issues.append(f"subtotal+tax {sub + tax} != total {total}")

    # 품목의 unit_price * qty 가 amount 와 맞는지도 본다. 개별 품목의 오독은
    # 합계가 우연히 맞아도 남아 있을 수 있다.
    for n, it in enumerate(items):
        u, q, a = it.get("unit_price"), it.get("qty"), it.get("amount")
        if None not in (u, q, a) and u * q != a:
            issues.append(f"items[{n}] {u}x{q}={u * q} != amount {a}")

    # 모델 자체 판정과 실제 계산이 어긋나면 그 자체가 신호다.
    ck = gt.get("checks") or {}
    if ck.get("items_sum_matches_total") is True and any("items 합계" in i for i in issues):
        issues.append("모델이 검산을 통과했다고 했으나 실제로는 불일치")

    return {"passed": not issues, "issues": issues,
            "unreadable": (ck.get("unreadable_fields") or [])}


if __name__ == "__main__":
    print("=== 페이지 KIE (기본) ===")
    print(build_gt_prompt())
    print("\n\n=== extreme 품질 덧붙임 ===")
    print(build_gt_prompt(quality="extreme")[len(_PAGE_BASE):])
    print("\n=== 크롤링 실사 덧붙임 ===")
    print(build_gt_prompt(source="google-crawl")[len(_PAGE_BASE):])
    print("\n\n=== 크롭 전사 (table) ===")
    print(build_crop_prompt(label="table"))
    print("\n\n=== 사후 검증 ===")
    bad = {"items": [{"name": "A", "unit_price": 1000, "qty": 2, "amount": 2000},
                     {"name": "B", "unit_price": 500, "qty": 3, "amount": 1600}],
           "subtotal": 3600, "tax": 360, "total": 3600,
           "checks": {"items_sum_matches_total": True,
                      "subtotal_plus_tax_matches_total": False,
                      "unreadable_fields": ["business_no"]}}
    import json as _j
    print(_j.dumps(verify_gt(bad), ensure_ascii=False, indent=1))
