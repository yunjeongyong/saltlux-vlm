"""
GT 라벨 3자 합의 생성.

문제: 학습 라벨 24건 중 사람이 검증한 건 1건뿐. 나머지는 파서 출력을 그대로 써서
     오류가 그대로 들어 있다(물류→블류, 시장분류→시상분류 등).
     전수 사람 검증은 비싸다.

방법: 파서 출력 + 독립 모델 2개, 총 3개 소스를 대조한다.
     - 3자 일치            -> 고신뢰 라벨. 사람이 안 봐도 됨
     - 2자 일치 (다수결)    -> 중신뢰. 다수 쪽 채택하되 표시
     - 전부 불일치          -> 사람 확인 필요

     "베이스 모델이 이미 한글 텍스트를 CER 0~6%로 읽는다"는 측정에 근거한다.
     대부분 일치할 것이므로 사람이 볼 물량이 크게 줄어든다.

usage:
    python3 scripts/gen_gt.py                      # val 21블록으로 시험
    python3 scripts/gen_gt.py --all --workers 4    # 전체
"""
import argparse
import base64
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval_dataset/gt_consensus"

MODELS = [
    ("gemma-4-31B", "http://localhost:15010/v1/chat/completions", "gemma-4-31B-it"),
    ("Qwen3.5-4B", "http://localhost:15030/v1/chat/completions", "Qwen3.5-4B"),
]

PROMPT_TABLE = ("이 표 이미지를 HTML <table>로 변환하세요. "
                "설명 없이 <table>...</table>만 출력하세요.")
PROMPT_TEXT = ("이 이미지에 적힌 텍스트를 그대로 옮겨 적으세요. "
               "설명·번역·추측 없이 보이는 글자만 출력하세요.")

from canon import canon, compare_key


def norm(t):
    """비교용 키. 표기 차이(태그 속성·글머리 기호·공백)를 모두 제거한다."""
    return compare_key(t)


def cer(a, b):
    if not a:
        return 0.0 if not b else 1.0
    sm = SequenceMatcher(None, a, b, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes()
               if t != "equal") / max(len(a), len(b))


def call(url, model, b64, prompt, timeout=300):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
            {"type": "text", "text": prompt}]}],
        "max_tokens": 4096, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    t = d["choices"][0]["message"]["content"]
    if "</think>" in t:
        t = t.split("</think>")[-1]
    return re.sub(r"^```\w*\n?|\n?```$", "", t.strip()).strip()


def judge(row, tol):
    """파서 라벨 + 모델 2개를 대조해 합의 등급을 매긴다."""
    img = ROOT / row["images"][0]
    b64 = base64.b64encode(img.read_bytes()).decode()
    is_table = "table" in row["id"] or "<table" in row["messages"][1]["content"]
    prompt = PROMPT_TABLE if is_table else PROMPT_TEXT

    parser = row["messages"][1]["content"]
    outs = {"parser": parser}
    for name, url, model in MODELS:
        try:
            outs[name] = call(url, model, b64, prompt)
        except Exception as exc:
            outs[name] = f"__ERROR__ {exc}"

    keys = list(outs)
    n = {k: norm(v) for k, v in outs.items()}
    # 쌍별 일치 여부 (tol 이내면 같다고 본다)
    same = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            same[(a, b)] = cer(n[a], n[b]) <= tol

    agree_all = all(same.values())
    if agree_all:
        # 정규화 기준으로 같으면 파서 원문을 유지한다. 모델 출력을 채택하면
        # 불릿(●, ○)·구두점 같은 서식이 소실된다 (실측: "● 미국 물류…" -> "미국 물류…").
        grade, label, note = "high", parser, "3자 일치 (파서 원문 유지)"
    else:
        # 다수결: 가장 많은 다른 소스와 일치하는 것
        score = {k: sum(1 for (a, b), v in same.items() if v and k in (a, b))
                 for k in keys}
        best = max(score, key=score.get)
        if score[best] >= 1:
            grade = "mid"
            agreed = [k for k in keys if k != best
                      and same.get((min(k, best), max(k, best)),
                                   same.get((best, k), False))]
            # 다수 쪽에 파서가 포함되면 파서 원문을 쓴다 (서식 보존).
            label = parser if ("parser" == best or "parser" in agreed) else outs[best]
            note = f"2자 일치 ({best} + {','.join(agreed) or '?'})"
        else:
            grade, label, note = "low", parser, "전부 불일치 — 사람 확인 필요"

    return {"id": row["id"], "image": row["images"][0], "grade": grade,
            "note": note, "label": canon(label), "sources": outs,
            "parser_changed": norm(label) != norm(parser)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tol", type=float, default=0.02, help="이 CER 이내면 일치로 간주")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = ["eval_dataset/train/val_block.jsonl"]
    if args.all:
        files.append("eval_dataset/train/train_block.jsonl")
    rows = []
    for f in files:
        rows += [json.loads(l) for l in open(ROOT / f, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    print(f"{len(rows)}블록 / 소스 {len(MODELS)+1}개 (파서 + {', '.join(m[0] for m in MODELS)})")
    OUT.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(lambda r: judge(r, args.tol), rows))

    cnt = {"high": 0, "mid": 0, "low": 0}
    changed = 0
    for r in res:
        cnt[r["grade"]] += 1
        changed += r["parser_changed"]

    n = len(res)
    print(f"\n{'등급':<8}{'건수':>6}{'비율':>8}  설명")
    print("-" * 58)
    print(f"{'high':<8}{cnt['high']:>6}{cnt['high']/n:>8.1%}  3자 일치 — 사람 확인 불필요")
    print(f"{'mid':<8}{cnt['mid']:>6}{cnt['mid']/n:>8.1%}  2자 일치 — 다수결 채택")
    print(f"{'low':<8}{cnt['low']:>6}{cnt['low']/n:>8.1%}  전부 불일치 — 사람 확인 필요")
    print("-" * 58)
    # low 뿐 아니라 '파서 라벨이 바뀐 건'도 전부 사람이 봐야 한다.
    # 두 모델이 같은 실수를 하면(상관된 오류) 다수결이 오답을 확정하기 때문.
    # 실측: 두 모델이 함께 '공통'->'공동'으로 바꿈 (정답 불명).
    need = [r for r in res if r["grade"] == "low" or r["parser_changed"]]
    print(f"사람이 봐야 할 물량: {len(need)}/{n} ({len(need)/n:.1%})")
    print(f"  - 전부 불일치        {cnt['low']}건")
    print(f"  - 파서 라벨 교정 발생 {changed}건  <- 상관된 오류 가능성, 반드시 확인")

    (OUT / "consensus.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in res), encoding="utf-8")
    review = need
    (OUT / "needs_review.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in review), encoding="utf-8")
    print(f"\n전체 -> {OUT}/consensus.jsonl")
    print(f"사람 확인 대상 -> {OUT}/needs_review.jsonl")

    print(f"\n=== 파서 라벨이 바뀐 예시 (최대 5건) ===")
    for r in [x for x in res if x["parser_changed"]][:5]:
        print(f"\n[{r['id']}] {r['note']}")
        print(f"  파서 : {r['sources']['parser'][:90]}")
        print(f"  채택 : {r['label'][:90]}")


if __name__ == "__main__":
    main()
