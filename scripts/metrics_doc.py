"""
문서 파싱 채점 지표 — CER (텍스트) / TEDS (표).

기존 eval 스크립트들은 CER 을 difflib.SequenceMatcher 로 근사했다. 그건 편집거리가
아니라 최장공통부분열 기반이라 값이 실제 CER 보다 낮게 나온다(특히 순서가 뒤섞일 때).
보고용 숫자는 표준 정의를 쓴다 — 편집거리 / 정답 길이.

TEDS 는 PubTabNet 정의를 따른다. HTML 표를 트리로 만들어 트리편집거리를 구하고
노드 수로 정규화한 뒤 1 에서 뺀다. 셀 내용도 비교에 넣는 원래 TEDS 이며,
구조만 보는 TEDS-S 는 셀 텍스트를 지워 같은 함수로 계산한다.
"""
import re
from collections import deque

import Levenshtein
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html


def norm_text(s):
    """공백만 정규화한다. 글자를 바꾸면 그 시점에서 CER 이 아니게 된다."""
    return re.sub(r"\s+", " ", str(s or "")).strip()


def cer(ref, hyp):
    """문자 오류율. 0 이 완벽, 1 이상도 나올 수 있다(삽입이 많으면)."""
    a, b = norm_text(ref), norm_text(hyp)
    if not a:
        return 0.0 if not b else 1.0
    return Levenshtein.distance(a, b) / len(a)


# ── TEDS ────────────────────────────────────────────────────────────────
class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        if self.tag == "td":
            r = ('"tag": %s, "colspan": %d, "rowspan": %d, "text": %s'
                 % (self.tag, self.colspan, self.rowspan, self.content))
        else:
            r = '"tag": %s' % self.tag
        for c in self.children:
            r += c.bracket()
        return "{{{}}}".format(r)


class TableConfig(Config):
    @staticmethod
    def maximum(*sequences):
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        return float(Levenshtein.distance(*sequences)) / self.maximum(*sequences)

    def rename(self, n1, n2):
        # 태그나 병합 정보가 다르면 다른 셀이다. 같으면 안의 글자로 부분점수를 준다.
        if (n1.tag != n2.tag or n1.colspan != n2.colspan
                or n1.rowspan != n2.rowspan):
            return 1.0
        if n1.tag == "td" and (n1.content or n2.content):
            return self.normalized_distance(n1.content or "", n2.content or "")
        return 0.0


def _tokens(node):
    """셀 안 텍스트를 문자 단위로 편다. <br> 등 태그는 그대로 토큰이 된다."""
    out = []
    if node.tag in ("td", "th"):
        pass
    if node.text is not None:
        out += list(node.text)
    for c in node.getchildren():
        if c.tag in ("b", "i", "u", "sup", "sub", "br", "span"):
            out += _tokens(c)
        if c.tail is not None:
            out += list(c.tail)
    return out


def _to_tree(node, parent=None):
    if node.tag in ("td", "th"):
        cell = "".join(_tokens(node))
        t = TableTree("td", int(node.get("colspan", "1") or 1),
                      int(node.get("rowspan", "1") or 1), norm_text(cell))
    else:
        t = TableTree(node.tag, None, None, None)
    if parent is not None:
        parent.children.append(t)
    if node.tag not in ("td", "th"):
        for c in node.getchildren():
            _to_tree(c, t)
    return t


def _n_nodes(t):
    n, q = 0, deque([t])
    while q:
        x = q.popleft()
        n += 1
        q.extend(x.children)
    return n


def _parse_table(s):
    """문자열에서 <table> 한 덩어리를 꺼내 lxml 노드로 만든다. 실패하면 None."""
    s = str(s or "").strip()
    if not s:
        return None
    s = re.sub(r"^```(?:html)?|```$", "", s, flags=re.M).strip()
    m = re.search(r"<table.*?</table>", s, flags=re.S | re.I)
    if not m:
        return None
    try:
        doc = html.fromstring(m.group(0))
    except (etree.ParserError, etree.XMLSyntaxError, ValueError):
        return None
    return doc if doc.tag == "table" else (doc.find(".//table"))


def teds(ref_html, hyp_html, structure_only=False):
    """표 유사도. 1 이 완벽, 0 이 완전 불일치. 파싱 실패한 예측은 0 점."""
    r = _parse_table(ref_html)
    if r is None:
        return None                      # 정답이 표가 아니면 채점 대상이 아니다
    h = _parse_table(hyp_html)
    if h is None:
        return 0.0                       # 모델이 표를 못 뱉으면 0 점이 맞다
    if structure_only:
        for n in list(r.iter()) + list(h.iter()):
            if n.tag in ("td", "th"):
                for ch in list(n):
                    n.remove(ch)
                n.text = None
    tr, th = _to_tree(r), _to_tree(h)
    n = max(_n_nodes(tr), _n_nodes(th))
    if n == 0:
        return 0.0
    d = APTED(tr, th, TableConfig()).compute_edit_distance()
    return max(0.0, 1.0 - d / n)


def avg_score(cer_v, teds_v):
    """요청 지표 AVG(CER, TEDS).

    CER 은 낮을수록, TEDS 는 높을수록 좋다. 그대로 평균내면 방향이 섞여 뜻이 없다.
    CER 을 정확도(1-CER)로 뒤집어 같은 방향으로 만든 뒤 평균낸다 — 높을수록 좋다.
    """
    parts = []
    if cer_v is not None:
        parts.append(max(0.0, 1.0 - cer_v))
    if teds_v is not None:
        parts.append(teds_v)
    return sum(parts) / len(parts) if parts else None
