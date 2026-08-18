"""
출력 포맷 정규화 규칙 (canonical form).

GT 생성·학습 타겟·평가 세 곳에서 모두 이 함수를 통과시켜, 표기 차이가
내용 차이로 오인되는 것을 막는다. 규칙은 오늘 관측된 실제 차이에 근거한다.

관측된 차이와 결정:

  1. <th>/<thead> vs <td>      -> <td> 로 통일
     파서는 <td> 만 쓰고 모델마다 <th> 적용 위치가 제각각이었다.
     일관되지 않은 신호를 학습시키면 노이즈만 는다. 헤더는 첫 행 위치로 식별 가능.

  2. border/width/style/data-bbox 등 속성 -> 전부 제거
     내용이 아니다. 모델마다 다르게 붙인다(Qwen: border="1" data-bbox="0 0 999 999").

  3. rowspan/colspan                      -> 보존
     실제로 쓰이고 있고(연구개발계획6·8), 빠지면 열이 밀려 내용이 어긋난다.

  4. HTML 엔티티 (&#x27; &amp; 등)        -> 실제 문자로 복원
     fast/high 설정에 따라 인코딩이 달라졌다. 내용은 같다.

  5. 표 내부 개행/들여쓰기                 -> 제거
     모델마다 탭·줄바꿈 스타일이 다르다. 렌더링에 영향 없다.

  6. ```html 같은 마크다운 펜스           -> 제거

  7. <br> / <br/>                        -> <br/> 로 통일

  8. 글머리 기호(ㅇ ○ ● - 등)             -> 원문 유지
     구조 정보다. 지우면 목록 계층이 사라진다.
     단 '비교'할 때는 무시한다 (compare_key 참고).

  9. 본문의 띄어쓰기·따옴표                -> 원문 유지
     내용이므로 임의로 바꾸지 않는다.
"""
import re

# 유지할 표 속성 — 이것만 남기고 나머지는 버린다
KEEP_ATTRS = ("rowspan", "colspan")

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?|\n?```\s*$")
_ENTITY = {"&#x27;": "'", "&#39;": "'", "&quot;": '"', "&nbsp;": " ",
           "&lt;": "<", "&gt;": ">", "&amp;": "&"}   # &amp; 는 마지막에


def _clean_attrs(tag_body):
    keep = []
    for a in KEEP_ATTRS:
        m = re.search(rf'{a}\s*=\s*"(\d+)"', tag_body)
        if m:
            keep.append(f'{a}="{m.group(1)}"')
    return (" " + " ".join(keep)) if keep else ""


def canon(text):
    """표기 차이를 제거한 표준형. 내용은 건드리지 않는다."""
    if text is None:
        return ""
    t = str(text)

    t = _FENCE.sub("", t).strip()                    # 6. 마크다운 펜스
    for k, v in _ENTITY.items():                     # 4. 엔티티 복원
        t = t.replace(k, v)

    if "<" not in t:                                 # 표가 아니면 여기까지
        return t.strip()

    t = re.sub(r"<br\s*/?>", "<br/>", t, flags=re.I)          # 7.
    t = re.sub(r"</?(thead|tbody|tfoot)[^>]*>", "", t, flags=re.I)   # 1.
    t = re.sub(r"<th(\s[^>]*)?>", lambda m: "<td" + _clean_attrs(m.group(1) or "") + ">",
               t, flags=re.I)                                  # 1. th -> td
    t = re.sub(r"</th>", "</td>", t, flags=re.I)
    t = re.sub(r"<td(\s[^>]*)?>", lambda m: "<td" + _clean_attrs(m.group(1) or "") + ">",
               t, flags=re.I)                                  # 2. 속성 정리
    t = re.sub(r"<tr(\s[^>]*)?>", "<tr>", t, flags=re.I)
    t = re.sub(r"<table(\s[^>]*)?>", "<table>", t, flags=re.I)

    t = re.sub(r">\s+<", "><", t)                    # 5. 태그 사이 공백/개행 제거
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# 비교 전용 — 글머리 기호·공백 차이를 무시한다. 학습 타겟에는 쓰지 않는다.
_BULLET_HEAD = re.compile(r"^[ㅇoO0*\-–—·・•○●■□㉠①\s]+")
_BULLET_MAP = {"○": "ㅇ", "●": "ㅇ", "•": "ㅇ", "＊": "*", "※": "*",
               "（": "(", "）": ")", "－": "-", "–": "-", "—": "-",
               "“": '"', "”": '"', "‘": "'", "’": "'"}


def compare_key(text):
    t = canon(text)
    t = re.sub(r"<[^>]+>", "", t)
    for a, b in _BULLET_MAP.items():
        t = t.replace(a, b)
    t = re.sub(r"\s+", "", t)
    return _BULLET_HEAD.sub("", t)


if __name__ == "__main__":
    samples = [
        '<table border="1" data-bbox="0 0 999 999"><thead><tr><th>구분</th>'
        '<th rowspan="2">기간</th></tr></thead><tbody><tr><td>국내</td></tr></tbody></table>',
        '```html\n<table>\n  <tr>\n    <td>기관명<br>구분</td>\n  </tr>\n</table>\n```',
        "'차이나 플러스 원' &#x27;테스트&#x27; &amp; 기타",
        "● 미국 물류 (Logistics) Digital Twin",
    ]
    for s in samples:
        print("입력 :", s[:78].replace("\n", "⏎"))
        print("표준형:", canon(s)[:78])
        print("비교키:", compare_key(s)[:78])
        print()
