"""
clean_v1 이미지 3,774장을 Document Studio document-parse에 태워 페이지 JSON을 뽑는다.

파이프라인 문서의 1단계(`1_parse.py` → `1_output_json/`)에 해당한다.
처리 단위는 페이지 이미지 1장이고, 출력도 이미지 1장 ↔ JSON 1개로 맞춘다.

  receipt_data/clean_v1/gt/{doc_id}.json      (기존 정답)
  receipt_data/clean_v1/parse/{doc_id}.json   (이 스크립트 산출물)

이미 처리된 항목은 건너뛴다. 중단 후 같은 명령을 다시 실행하면 남은 것부터 이어서 간다.

API (프론트엔드 번들에서 확인):
  POST {base}/api/v1/public/document-parser/parse   multipart file=@... , X-API-KEY 헤더
  응답은 동기. doc_id / page_count / result_json / result_md / result_html.

usage:
    # 연결·응답 스키마 확인 (3장만)
    python3 scripts/parse_clean_v1.py --api-key-file /workspace/.docstudio_key --limit 3

    # 전량
    python3 scripts/parse_clean_v1.py --api-key-file /workspace/.docstudio_key --workers 4
"""
import argparse
import json
import mimetypes
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path("/workspace")
DEFAULT_BASE = "https://document-studio.luxialab.com"
PARSE_PATH = "/api/v1/public/document-parser/parse"

# 재시도할 상태코드. 4xx는 요청 자체가 틀린 거라 재시도해도 같다.
RETRY_STATUS = {429, 500, 502, 503, 504}


def load_targets(manifest, limit):
    """manifest.jsonl -> [(doc_id, 이미지 절대경로), ...]"""
    out = []
    for line in manifest.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.append((d["doc_id"], ROOT / d["image"], d))
    if limit:
        out = out[:limit]
    return out


def already_done(path):
    """산출물이 있고 JSON으로 읽히면 완료로 본다. 중간에 끊겨 깨진 파일은 다시 돈다."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False


def parse_one(session, url, api_key, img_path, options, timeout):
    mime = mimetypes.guess_type(img_path.name)[0] or "application/octet-stream"
    with img_path.open("rb") as fh:
        files = {"file": (img_path.name, fh, mime)}
        data = {k: v for k, v in options.items()}
        r = session.post(
            url,
            headers={"X-API-KEY": api_key},
            files=files,
            data=data or None,
            timeout=timeout,
        )
    return r


def worker(doc_id, img_path, meta, args, out_dir, session, url, options, counters, lock):
    out_path = out_dir / f"{doc_id}.json"
    if already_done(out_path):
        with lock:
            counters["skip"] += 1
        return None

    if not img_path.exists():
        return {"doc_id": doc_id, "error": "image_not_found", "image": str(img_path)}

    delay = args.backoff
    last = None
    for attempt in range(1, args.retries + 1):
        try:
            r = parse_one(session, url, args.api_key, img_path, options, args.timeout)
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(delay)
            delay *= 2
            continue

        if r.status_code == 200:
            try:
                body = r.json()
            except ValueError:
                last = f"200 but non-JSON body ({r.text[:200]!r})"
                break
            # 원본 응답을 그대로 보존하고, 대조에 필요한 메타만 얹는다.
            body["_source_doc_id"] = doc_id
            body["_source_image"] = meta["image"]
            body["_source_width"] = meta.get("width")
            body["_source_height"] = meta.get("height")
            tmp = out_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
            tmp.replace(out_path)  # 원자적 교체 — 중단돼도 반쪽 파일이 안 남는다
            with lock:
                counters["ok"] += 1
            return None

        last = f"HTTP {r.status_code}: {r.text[:200]}"
        if r.status_code not in RETRY_STATUS:
            break
        # 429면 서버가 알려준 대기시간을 우선한다
        wait = r.headers.get("Retry-After")
        time.sleep(float(wait) if wait and wait.isdigit() else delay)
        delay *= 2

    with lock:
        counters["fail"] += 1
    return {"doc_id": doc_id, "error": last, "image": meta["image"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="receipt_data/clean_v1/manifest.jsonl")
    ap.add_argument("--out", default="receipt_data/clean_v1/parse")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--api-key", default=os.environ.get("DOCSTUDIO_API_KEY", ""))
    ap.add_argument("--api-key-file", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=2.0, help="첫 재시도 대기(초), 이후 2배씩")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N장만 (스모크 테스트용)")
    ap.add_argument("--no-chart", action="store_true", help="chartRecognition=false (영수증엔 차트가 없다)")
    ap.add_argument("--redo-failed", action="store_true", help="실패 로그에 있는 건도 다시 시도")
    args = ap.parse_args()

    if args.api_key_file:
        args.api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    if not args.api_key:
        sys.exit("API 키가 없다. --api-key / --api-key-file / DOCSTUDIO_API_KEY 중 하나로 넘겨라.")

    manifest = ROOT / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_log = out_dir.parent / "parse_failed.jsonl"
    if args.redo_failed and fail_log.exists():
        fail_log.unlink()

    targets = load_targets(manifest, args.limit)
    todo = [t for t in targets if not already_done(out_dir / f"{t[0]}.json")]
    print(f"대상 {len(targets)}장 / 미처리 {len(todo)}장 / 완료 {len(targets) - len(todo)}장")
    if not todo:
        print("남은 게 없다.")
        return

    options = {}
    if args.no_chart:
        options["chartRecognition"] = "false"

    url = args.base.rstrip("/") + PARSE_PATH
    counters = {"ok": 0, "fail": 0, "skip": 0}
    lock = threading.Lock()
    t0 = time.time()

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=args.workers * 2)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(worker, doc_id, img, meta, args, out_dir, session, url, options, counters, lock): doc_id
            for doc_id, img, meta in todo
        }
        for i, fut in enumerate(as_completed(futs), 1):
            err = fut.result()
            if err:
                failures.append(err)
            if i % 25 == 0 or i == len(futs):
                el = time.time() - t0
                rate = i / el if el else 0
                eta = (len(futs) - i) / rate if rate else 0
                print(
                    f"[{i}/{len(futs)}] ok={counters['ok']} fail={counters['fail']} "
                    f"{rate:.2f}장/s ETA {eta/60:.1f}분",
                    flush=True,
                )

    if failures:
        with fail_log.open("a", encoding="utf-8") as fh:
            for f in failures:
                fh.write(json.dumps(f, ensure_ascii=False) + "\n")
        print(f"\n실패 {len(failures)}건 → {fail_log}")
        for f in failures[:5]:
            print(f"  {f['doc_id']}: {f['error']}")

    print(f"\n완료: ok={counters['ok']} fail={counters['fail']} / {(time.time()-t0)/60:.1f}분")
    print(f"산출물: {out_dir}  ({len(list(out_dir.glob('*.json')))}개)")


if __name__ == "__main__":
    main()
