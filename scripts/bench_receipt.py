"""
영수증 OCR 베이스라인 측정.

평균 CER 하나로는 진단이 안 된다. 두 축으로 쪼갠다:
  quality 별   clean → extreme.  전처리·해상도 문제인지 판별
  필드 별      금액 / 상호명 / 날짜.  숫자 문제인지 한글 문제인지 판별

측정 항목
  CER          문자 오류율 (편집거리 / 정답 길이)
  정확일치율    완전히 같은 비율. 금액 필드는 부분점수가 무의미하다
  숫자 정확도   숫자만 뽑아 비교. 영수증에서 금액은 1자만 틀려도 문서가 무의미

usage:
    python3 scripts/bench_receipt.py --model Qwen3.5-4B --port 15030
    python3 scripts/bench_receipt.py --model Qwen3.5-4B --port 15030 --limit 100
"""
import argparse
import base64
import json
import re
import time
import urllib.request
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path("/workspace")
TEST = ROOT / "receipt_data/unified/test.jsonl"
OUT = ROOT / "receipt_data/bench"

NUM = re.compile(r"\d")


def cer(ref, hyp):
    ref, hyp = ref.strip(), hyp.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    sm = SequenceMatcher(None, ref, hyp, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes()
               if t != "equal") / max(len(ref), len(hyp))


def digits(s):
    return "".join(NUM.findall(s))


def call(url, model, b64, prompt, timeout=300):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            {"type": "text", "text": prompt}]}],
        "max_tokens": 2048, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    t = d["choices"][0]["message"]["content"]
    if "</think>" in t:
        t = t.split("</think>")[-1]
    return re.sub(r"^```\w*\n?|\n?```$", "", t.strip()).strip()


def agg(rows):
    """CER / 정확일치 / 숫자정확도 집계."""
    if not rows:
        return None
    n = len(rows)
    exact = sum(1 for r in rows if r["gt"].strip() == r["pred"].strip())
    dig = [r for r in rows if digits(r["gt"])]
    dig_ok = sum(1 for r in dig if digits(r["gt"]) == digits(r["pred"]))
    return {
        "n": n,
        "cer": sum(r["cer"] for r in rows) / n,
        "exact": exact / n,
        "digit_n": len(dig),
        "digit_acc": (dig_ok / len(dig)) if dig else None,
    }


def show(title, table):
    print(f"\n{title}")
    print(f"{'구분':<26}{'건수':>6}{'CER':>9}{'정확일치':>10}{'숫자정확':>10}")
    print("-" * 62)
    for k, m in table:
        if not m:
            continue
        da = f"{m['digit_acc']:.1%}" if m["digit_acc"] is not None else "—"
        print(f"{str(k)[:25]:<26}{m['n']:>6}{m['cer']:>8.2%}{m['exact']:>10.1%}{da:>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3.5-4B")
    ap.add_argument("--port", type=int, default=15030)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=None, help="결과 파일 이름")
    a = ap.parse_args()
    url = f"http://localhost:{a.port}/v1/chat/completions"

    rows = [json.loads(l) for l in TEST.open(encoding="utf-8")]
    if a.limit:
        rows = rows[:a.limit]
    print(f"{a.model} @ :{a.port}  테스트 {len(rows)}건")

    res, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        img = ROOT / r["images"][0]
        prompt = r["messages"][0]["content"].replace("<image>", "")
        gt = r["messages"][1]["content"]
        meta = r.get("meta", {})
        try:
            b64 = base64.b64encode(img.read_bytes()).decode()
            pred = call(url, a.model, b64, prompt)
        except Exception as exc:
            pred = f"__ERROR__ {exc}"
        res.append({"id": r["images"][0], "gt": gt, "pred": pred,
                    "cer": cer(gt, pred),
                    "task": "full_md" if "마크다운" in prompt else "field",
                    **meta})
        if i % 50 == 0 or i == len(rows):
            el = time.time() - t0
            print(f"  {i}/{len(rows)}  ({el:.0f}s, 남은 {el/i*(len(rows)-i):.0f}s)",
                  flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.tag or a.model.replace("/", "_")
    (OUT / f"{tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*62}\n{a.model}  베이스라인\n{'='*62}")
    show("전체", [("전체", agg(res))])
    show("태스크별", [(k, agg([r for r in res if r["task"] == k]))
                   for k in ("field", "full_md")])

    q = defaultdict(list)
    for r in res:
        if r.get("quality"):
            q[r["quality"]].append(r)
    order = ["clean", "light", "sharp_photo", "screenshot", "heavy", "extreme"]
    show("열화 단계별 (full_md)",
         [(k, agg(q[k])) for k in order if k in q])

    f = defaultdict(list)
    for r in res:
        if r.get("field"):
            f[r["field"]].append(r)
    show("필드별 (field)",
         sorted(((k, agg(v)) for k, v in f.items()),
                key=lambda x: -x[1]["cer"]))

    print(f"\n상세 -> {OUT}/{tag}.json")


if __name__ == "__main__":
    main()
