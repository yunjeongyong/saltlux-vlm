"""
파인튜닝 전/후 비교: gemma-4-31B-it(15010) vs luxia4-31b-sft(15011)

같은 재무제표 이미지를 양쪽에 넣고, GT에 기록된 10개 한글 오인식 케이스가
각 모델 출력에서 정답으로 나오는지 / 오답으로 나오는지 자동 판정한다.
"""
import base64
import json
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "eval_dataset/images/doc001_saltlux_bs_page000.jpg"
GT = ROOT / "eval_dataset/annotations/doc001_saltlux_bs_page000.json"
OUT = ROOT / "eval_dataset/runs"

ENDPOINTS = [
    ("gemma4_base", "http://localhost:15010/v1/chat/completions", "gemma-4-31B-it"),
    ("luxia4_sft", "http://localhost:15011/v1/chat/completions", "luxia4-31b-sft-v0.1.0_3ep"),
]

PROMPT = (
    "이 재무제표 이미지를 HTML <table>로 변환하세요.\n"
    "- 모든 행과 열을 빠짐없이 포함할 것\n"
    "- 계정과목명은 이미지에 적힌 그대로 정확히 옮길 것\n"
    "- 금액의 쉼표와 음수 부호를 유지할 것\n"
    "- 설명 없이 <table>...</table>만 출력할 것"
)


def call(url, model, b64, timeout=900):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            {"type": "text", "text": PROMPT},
        ]}],
        "max_tokens": 8192,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"], d.get("usage", {})


def grade(text, errors):
    """GT의 각 오류 케이스에 대해 정답형/오답형 중 무엇이 나왔는지 판정."""
    rows = []
    for e in errors:
        truth, wrong = e["ground_truth"], e["parser_output"]
        has_truth, has_wrong = truth in text, wrong in text
        if has_truth and not has_wrong:
            verdict = "correct"
        elif has_wrong and not has_truth:
            verdict = "same_error"
        elif has_truth and has_wrong:
            verdict = "mixed"
        else:
            verdict = "missing"
        rows.append({"ground_truth": truth, "parser_error": wrong, "verdict": verdict})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b64 = base64.b64encode(IMG.read_bytes()).decode()
    errors = json.loads(GT.read_text(encoding="utf-8"))["errors"]

    summary = {}
    for name, url, model in ENDPOINTS:
        print(f"[{name}] 요청 중... ({model})", flush=True)
        try:
            text, usage = call(url, model, b64)
        except Exception as exc:
            print(f"[{name}] 실패: {exc}", flush=True)
            summary[name] = {"error": str(exc)}
            continue

        (OUT / f"{name}_raw.txt").write_text(text, encoding="utf-8")
        graded = grade(text, errors)
        counts = {}
        for g in graded:
            counts[g["verdict"]] = counts.get(g["verdict"], 0) + 1
        summary[name] = {"model": model, "usage": usage, "counts": counts, "detail": graded}
        print(f"[{name}] 완료 — {counts}", flush=True)

    (OUT / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 표로 출력
    print("\n" + "=" * 78)
    print(f"{'계정과목(정답)':<20}{'파서오류':<18}{'gemma4_base':<16}{'luxia4_sft':<16}")
    print("=" * 78)
    g = {d["ground_truth"]: d["verdict"] for d in summary.get("gemma4_base", {}).get("detail", [])}
    l = {d["ground_truth"]: d["verdict"] for d in summary.get("luxia4_sft", {}).get("detail", [])}
    for e in errors:
        t = e["ground_truth"]
        print(f"{t:<20}{e['parser_output']:<18}{g.get(t,'-'):<16}{l.get(t,'-'):<16}")
    print("=" * 78)
    for name in ("gemma4_base", "luxia4_sft"):
        c = summary.get(name, {}).get("counts", {})
        print(f"{name:<16} correct={c.get('correct',0)}  same_error={c.get('same_error',0)}  "
              f"missing={c.get('missing',0)}  mixed={c.get('mixed',0)}")


if __name__ == "__main__":
    main()
