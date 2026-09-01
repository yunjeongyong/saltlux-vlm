"""
VLM 문서파싱 성능 비교.

GT가 있는 재무제표(계정명 60개 / 금액 161셀)로 모델을 정량 비교한다.
측정 항목은 앞선 분석에서 실제로 문제가 됐던 것들만 고른다:

  - 행 recall      : 60행 중 몇 행을 뽑았나 (VLM은 긴 표에서 행을 흘린다)
  - 계정명 CER     : 한글 유사자형 오인식 (파서의 약점)
  - 금액 정확도    : 161셀 중 몇 개가 정확한가 (VLM의 약점, 가장 중요)
  - 회계 항등식    : 자산총계 == 부채와자본총계
  - 소요 시간

usage:
    python3 scripts/bench_vlm.py                     # ENDPOINTS 전체
    python3 scripts/bench_vlm.py --only qwen3.5-2b
"""
import argparse
import base64
import json
import re
import sys
import time
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "eval_dataset/images/doc001_saltlux_bs_page000.jpg"
GT = ROOT / "eval_dataset/annotations/doc001_saltlux_bs_page000.json"
OUT = ROOT / "eval_dataset/bench"

# (표시명, 엔드포인트, 모델ID, 파라미터수 표기)
ENDPOINTS = [
    ("gemma-4-31B",  "http://localhost:15010/v1/chat/completions", "gemma-4-31B-it",            "31B"),
    ("luxia4-31B-sft", "http://localhost:15011/v1/chat/completions", "luxia4-31b-sft-v0.1.0_3ep", "31B"),
    ("qwen3.5-2b",   "http://localhost:15030/v1/chat/completions", "Qwen3.5-2B",                "2B"),
    ("qwen3.5-4b",   "http://localhost:15031/v1/chat/completions", "Qwen3.5-4B",                "4B"),
]

PROMPT = (
    "이 재무제표 이미지를 HTML <table>로 변환하세요.\n"
    "- 모든 행을 빠짐없이 포함할 것\n"
    "- 계정과목명은 이미지에 적힌 그대로 정확히 옮길 것\n"
    "- 금액의 쉼표와 음수 부호를 유지할 것\n"
    "- 설명 없이 <table>...</table>만 출력할 것"
)


def call(url, model, b64, timeout=1800):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            {"type": "text", "text": PROMPT},
        ]}],
        "max_tokens": 8192, "temperature": 0,
    }).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"], d.get("usage", {}), time.time() - t


def parse_table(html):
    """rowspan/colspan 전개. 병합셀 표에서 열 밀림 오탐을 막는다."""
    grid, pending = [], {}
    for r, tr in enumerate(re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)):
        row, col = [], 0
        for attrs, inner in re.findall(r"<t[dh]([^>]*)>(.*?)</t[dh]>", tr, re.S):
            while (r, col) in pending:
                row.append(pending.pop((r, col))); col += 1
            text = re.sub(r"<[^>]+>", "", inner).strip()
            rs = int((re.search(r'rowspan="(\d+)"', attrs) or [0, 1])[1])
            cs = int((re.search(r'colspan="(\d+)"', attrs) or [0, 1])[1])
            for c in range(cs):
                row.append(text)
                for e in range(1, rs):
                    pending[(r + e, col + c)] = text
            col += cs
        while (r, col) in pending:
            row.append(pending.pop((r, col))); col += 1
        if row:
            grid.append(row)
    return grid


def align(a, b):
    sm = SequenceMatcher(None, a, b, autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pairs += list(zip(range(i1, i2), range(j1, j2)))
        elif tag == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                pairs.append((i1 + k if i1 + k < i2 else None,
                              j1 + k if j1 + k < j2 else None))
        elif tag == "delete":
            pairs += [(i, None) for i in range(i1, i2)]
        else:
            pairs += [(None, j) for j in range(j1, j2)]
    return pairs


def cer(ref, hyp):
    if not ref:
        return 0.0
    sm = SequenceMatcher(None, ref, hyp, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes()
               if t != "equal") / max(len(ref), len(hyp))


def grade(text, gt):
    rows = gt["ground_truth_table"]["rows"]
    cols = [c for c in gt["ground_truth_table"]["columns"] if c != "계정과목"]
    gt_names = [r["계정과목"] for r in rows]

    pr = [r for r in parse_table(text) if len(r) >= 2 and r[0].strip()
          and not re.match(r"^제\s*\d+\s*기", r[0])]
    pr_names = [r[0] for r in pr]
    pairs = align(gt_names, pr_names)
    matched = [(i, j) for i, j in pairs if i is not None and j is not None]

    ref_chars = err_chars = 0
    num_tot = num_ok = 0
    for i, j in matched:
        ref, hyp = gt_names[i], pr_names[j]
        ref_chars += len(ref)
        err_chars += cer(ref, hyp) * len(ref)
        for ci, col in enumerate(cols, start=1):
            g = rows[i].get(col, "").strip()
            p = pr[j][ci].strip() if ci < len(pr[j]) else ""
            if not g and not p:
                continue
            num_tot += 1
            num_ok += (g == p)

    # 회계 항등식
    def find(*keys):
        for r in pr:
            if any(k in r[0].replace(" ", "") for k in keys):
                return r[1:]
        return None
    a, b = find("자산총계"), find("부채와자본총계", "부채및자본총계")
    ident = (a is not None and b is not None
             and all(x == y for x, y in zip(a, b)))

    return {
        "rows_out": len(pr),
        "row_recall": len(matched) / len(gt_names),
        "name_cer": err_chars / ref_chars if ref_chars else None,
        "num_cells": num_tot,
        "num_acc": num_ok / num_tot if num_tot else None,
        "identity_ok": ident,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    b64 = base64.b64encode(IMG.read_bytes()).decode()
    gt = json.loads(GT.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for name, url, model, params in ENDPOINTS:
        if args.only and args.only != name:
            continue
        print(f"[{name}] 추론 중...", flush=True)
        try:
            text, usage, secs = call(url, model, b64)
        except Exception as exc:
            print(f"[{name}] 실패: {exc}", flush=True)
            results.append({"name": name, "params": params, "error": str(exc)})
            continue
        (OUT / f"{name}.txt").write_text(text, encoding="utf-8")
        m = grade(text, gt)
        m.update(name=name, params=params, secs=round(secs, 1),
                 out_tokens=usage.get("completion_tokens"))
        results.append(m)
        print(f"[{name}] 완료 {secs:.0f}s — 행 {m['rows_out']}, "
              f"금액 {m['num_acc']:.1%}" if m.get("num_acc") is not None else "", flush=True)

    (OUT / "bench.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                    encoding="utf-8")

    print(f"\n{'모델':<18}{'파라미터':>8}{'행수':>6}{'행recall':>10}"
          f"{'계정명CER':>11}{'금액정확도':>11}{'항등식':>8}{'시간':>8}")
    print("-" * 82)
    print(f"{'Document Studio':<18}{'—':>8}{60:>6}{'100.0%':>10}{'3.78%':>11}"
          f"{'100.00%':>11}{'통과':>8}{'—':>8}   ← 현행 파서")
    for r in results:
        if "error" in r:
            print(f"{r['name']:<18}{r['params']:>8}   {r['error'][:44]}")
            continue
        print(f"{r['name']:<18}{r['params']:>8}{r['rows_out']:>6}{r['row_recall']:>9.1%}"
              f"{r['name_cer']:>10.2%}{r['num_acc']:>10.2%}"
              f"{'통과' if r['identity_ok'] else '실패':>8}{r['secs']:>7.0f}s")
    print(f"\n-> {OUT}/bench.json")


if __name__ == "__main__":
    sys.exit(main())
