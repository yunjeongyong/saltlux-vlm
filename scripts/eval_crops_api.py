"""
크롭 평가를 OpenAI 호환 API 로 돌린다 (LUXIA CHAT API 등 사내 서빙 모델용).

eval_crops.py 와 같은 manifest / 같은 프롬프트 / 같은 지표 코드를 쓴다.
다른 것은 추론 경로뿐이다 — 로컬 가중치 대신 HTTP 호출이다.
그래서 두 결과 JSON 은 그대로 나란히 비교할 수 있다.

  text  -> OCR 프롬프트   -> CER  (낮을수록 좋음)
  table -> TABLE 프롬프트 -> TEDS (높을수록 좋음)

서빙 모델은 기본이 사고 ON 이라 전사 대신 추론 과정을 뱉는다. 학습·로컬 평가와
조건을 맞추려면 반드시 꺼야 한다 (`chat_template_kwargs.enable_thinking=False`).

usage:
    python3 scripts/eval_crops_api.py --endpoint http://172.16.100.242:60100 --limit 20
    python3 scripts/eval_crops_api.py --endpoint http://172.16.100.242:60100 \
        --name luxia-document-parsing-high --out receipt_data/review/eval_crops_luxia.json
"""
import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_crops import load_rows, score, show, summarize   # noqa: E402
from prompts_receipt import PROMPT_VERSION, PROMPTS        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def encode(path, max_px):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w * h > max_px:                       # 로컬 평가와 같은 축소 규칙
        s = (max_px / (w * h)) ** 0.5
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call(endpoint, b64, prompt, max_new, timeout, retries=3):
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt}]}],
        "temperature": 0,
        "max_completion_tokens": max_new,
        # 서빙 기본값이 사고 ON 이다. 끄지 않으면 전사 대신 추론 과정이 나와
        # CER 이 실제 성능과 무관하게 무너진다.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    last = None
    for i in range(retries):
        t0 = time.time()
        try:
            req = urllib.request.Request(
                f"{endpoint.rstrip('/')}/v1/chat/completions", data=data,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                res = json.load(r)
            return res["choices"][0]["message"]["content"].strip(), time.time() - t0
        except Exception as e:                       # 일시적 과부하는 재시도
            last = f"{type(e).__name__}: {e}"[:200]
            time.sleep(2 * (i + 1))
    raise RuntimeError(last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="예: http://172.16.100.242:60100")
    ap.add_argument("--name", default=None, help="결과에 쓸 이름 (기본 /health 의 model_id)")
    ap.add_argument("--manifest",
                    default=str(ROOT / "receipt_data/eval_crops/manifest.jsonl"))
    ap.add_argument("--tasks", nargs="+", default=["text", "table"])
    ap.add_argument("--limit", type=int, default=0, help="task 별 상한 (0=전량)")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--max-px", type=int, default=4_000_000)
    ap.add_argument("--workers", type=int, default=4, help="동시 호출 수")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out",
                    default=str(ROOT / "receipt_data/review/eval_crops_api.json"))
    args = ap.parse_args()

    name = args.name
    if not name:
        try:
            with urllib.request.urlopen(f"{args.endpoint.rstrip('/')}/health",
                                        timeout=15) as r:
                name = json.load(r).get("model_id") or "api"
        except Exception:
            name = "api"

    rows = load_rows(args.manifest, args.tasks, args.limit)
    base_dir = Path(args.manifest).parent
    n_doc = len({r["doc_id"] for r in rows})
    print(f"평가 대상 {len(rows):,}건 (영수증 {n_doc}장) "
          f"— text {sum(1 for r in rows if r['task']=='text')} / "
          f"table {sum(1 for r in rows if r['task']=='table')}")
    print(f"프롬프트 버전 {PROMPT_VERSION}")
    print(f"엔드포인트 {args.endpoint}  모델 {name}  동시 {args.workers}", flush=True)

    t0 = time.time()
    done = [0]

    def work(r):
        try:
            b64 = encode(base_dir / r["image"], args.max_px)
            pred, dt = call(args.endpoint, b64, PROMPTS[r["task"]],
                            args.max_new, args.timeout)
        except Exception as e:                # 한 건 실패로 전체를 버리지 않는다
            pred, dt = "", 0.0
            r.setdefault("errors", {})[name] = f"{type(e).__name__}: {e}"[:200]
        r[name] = {"pred": pred, "sec": round(dt, 1),
                   **score(r["task"], r["gt"], pred)}
        done[0] += 1
        if done[0] % 25 == 0 or done[0] == len(rows):
            el = time.time() - t0
            print(f"    {done[0]}/{len(rows)}  {el/60:.1f}분 경과 "
                  f"(잔여 {(el/done[0]*(len(rows)-done[0]))/60:.1f}분)", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, rows))

    n_err = sum(1 for r in rows if r.get("errors", {}).get(name))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result = {"prompt_version": PROMPT_VERSION, "model": name,
              "endpoint": args.endpoint, "adapters": [], "n": len(rows),
              "n_error": n_err,
              "summary": {name: summarize(rows, name)}, "rows": rows}
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1),
                              encoding="utf-8")

    print("\n" + "=" * 62)
    print("결과 (CER 낮을수록 좋음 / TEDS·정확도·AVG 높을수록 좋음)")
    show(name, result["summary"][name])
    if n_err:
        print(f"\n  실패 {n_err}건 — 점수에서 빈 출력으로 처리됐다")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
