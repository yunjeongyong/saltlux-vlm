"""
구글에서 영수증 사진을 모은다 — Custom Search JSON API 사용.

처음엔 icrawler(GoogleImageCrawler)로 짰는데 구글이 검색 결과를 초기 HTML에
안 내려주도록 바꿔서(tbm=isch -> udm=2, 결과는 클라이언트 렌더링) 정적 파싱이
불가능해졌다. 헤드리스 브라우저로 렌더링하면 "unusual traffic"으로 막힌다.
그래서 공식 API가 유일하게 남은 경로다.

  GET https://www.googleapis.com/customsearch/v1
      ?key=..&cx=..&q=..&searchType=image&num=10&start=1

쿼터 단위를 헷갈리지 말 것:
  - 요청 1회 = 1 쿼리 = 이미지 최대 10장
  - 무료 100 쿼리/일 = 하루 1,000장
  - start 는 91까지만 → 검색어 하나당 100장이 천장. 목표가 크면 검색어를 늘려야 한다.

수집 후 한 번에 거른다: 깨진 것, 너무 작은 것, 가로로 넓은 것(영수증은 세로로
길다), pHash 중복. 출처 URL을 manifest에 남긴다 — 라이선스·PII 추적에 필요하다.

usage:
    python3 scripts/crawl_receipts.py --target 1000 \
        --api-key-file /workspace/.gcse_key --cx <CSE_ID>

    # 쿼터만 먼저 계산 (요청 안 보냄)
    python3 scripts/crawl_receipts.py --target 1000 --dry-run
"""
import argparse
import json
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

ROOT = Path("/workspace/receipt_data/crawl_google")
ENDPOINT = "https://www.googleapis.com/customsearch/v1"
PER_REQ = 10          # API 상한
MAX_START = 91        # start 상한 → 검색어당 100장

# 검색어 하나당 100장이 천장이라, 목표가 크면 개수로 벌어야 한다.
# 업종을 흩어놔야 같은 이미지가 덜 겹친다.
TERMS = [
    "영수증",
    "영수증 사진",
    "편의점 영수증",
    "마트 영수증",
    "카페 영수증",
    "식당 영수증",
    "신용카드 영수증",
    "현금영수증",
    "POS 영수증",
    "주유소 영수증",
    "약국 영수증",
    "배달 영수증",
    "커피숍 영수증",
    "택시 영수증",
    "korean receipt",
    "receipt photo",
    "백화점 영수증",
    "병원 영수증",
    "온라인 주문 영수증",
    "간이영수증",
    "세금계산서 영수증",
    "지출증빙 영수증",
    "영수증 인증",
    "영수증 리뷰",
    "마트 계산서",
    "카드 전표",
    "receipt scan",
    "grocery receipt",
]


def search_term(session, key, cx, term, want, rights, safe, pause):
    """한 검색어에서 최대 100장. (아이템 리스트, 사용한 쿼리 수)"""
    items, used = [], 0
    for start in range(1, MAX_START + 1, PER_REQ):
        if len(items) >= want:
            break
        params = {
            "key": key, "cx": cx, "q": term,
            "searchType": "image", "num": PER_REQ, "start": start,
            "imgType": "photo", "safe": safe,
        }
        if rights:
            params["rights"] = rights
        try:
            r = session.get(ENDPOINT, params=params, timeout=30)
            used += 1
        except requests.RequestException as e:
            print(f"    [{term}] start={start} 통신 실패: {e}", flush=True)
            break

        if r.status_code == 429:
            print(f"    [{term}] 429 — 일일 쿼터 소진. 여기서 중단한다.", flush=True)
            return items, used, True
        if r.status_code != 200:
            # 403은 대개 API 미사용설정/키 권한, 400은 cx 오류다. 메시지를 그대로 보여준다.
            msg = r.json().get("error", {}).get("message", r.text[:200]) if r.text else ""
            print(f"    [{term}] HTTP {r.status_code}: {msg}", flush=True)
            break

        got = r.json().get("items") or []
        if not got:
            break  # 더 이상 결과 없음
        for it in got:
            img = it.get("image", {})
            items.append({
                "term": term,
                "url": it.get("link"),
                "mime": it.get("mime"),
                "width": img.get("width"),
                "height": img.get("height"),
                "context": img.get("contextLink"),
                "title": it.get("title"),
            })
        time.sleep(pause)
    return items, used, False


def collect_browser(terms, per_term, scrolls, pause, headless=True):
    """실제 크롬으로 구글 이미지를 렌더링해 썸네일 URL을 모은다.

    구글은 결과를 초기 HTML에 안 넣으므로 정적 파싱이 불가능하다. 브라우저로
    렌더링하면 img[src] 에 encrypted-tbn*.gstatic.com 썸네일이 들어온다.

    주의 — 여기서 얻는 건 원본이 아니라 썸네일이다. 실측 평균 332x349 이고
    46x46 짜리 UI 아이콘이 섞여 들어온다. 해상도 필터는 뒤에서 건다.
    원본 화질이 필요하면 썸네일을 하나씩 클릭해 우측 패널 URL을 긁어야 하는데,
    요청 수가 수십 배로 늘고 그만큼 차단 위험이 커진다.

    요청을 몰아치면 429 / "unusual traffic" 을 맞는다. 검색어 사이에 쉰다.
    """
    from playwright.sync_api import sync_playwright
    import urllib.parse

    UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/151.0.0.0 Safari/537.36")
    items, seen = [], set()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        ctx = browser.new_context(
            locale="ko-KR", timezone_id="Asia/Seoul",
            viewport={"width": 1440, "height": 900}, user_agent=UA,
        )
        page = ctx.new_page()
        for term in terms:
            url = "https://www.google.com/search?q=" + urllib.parse.quote(term) + "&udm=2"
            try:
                page.goto(url, timeout=60000)
                page.wait_for_timeout(3500)
            except Exception as e:
                print(f"  [{term}] 페이지 로드 실패: {type(e).__name__}", flush=True)
                continue

            # 진짜 차단은 /sorry/ 리다이렉트나 캡차 폼으로 나타난다.
            # HTML 안의 'recaptcha' 문자열은 평범한 스크립트에도 들어있어 오탐이다.
            if "/sorry/" in page.url or page.locator(
                'form#captcha-form, iframe[src*="recaptcha"]'
            ).count():
                print(f"  [{term}] 캡차/차단 감지 — 중단한다.", flush=True)
                break

            for _ in range(scrolls):
                page.mouse.wheel(0, 4000)
                page.wait_for_timeout(int(pause * 1000))

            srcs = page.eval_on_selector_all(
                "img", "els=>els.map(e=>e.src||'').filter(s=>s.startsWith('http'))"
            )
            got = 0
            for s in srcs:
                if "gstatic.com" not in s or s in seen:
                    continue
                seen.add(s)
                items.append({"term": term, "url": s, "mime": None,
                              "width": None, "height": None,
                              "context": None, "title": None})
                got += 1
            print(f"  [{term}] +{got}장 (누적 {len(items)})", flush=True)
            if len(items) >= per_term * len(terms):
                break
            page.wait_for_timeout(2000)  # 검색어 간 간격
        browser.close()
    return items


def download(items, raw_dir, threads, timeout):
    raw_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    stats = Counter()
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )

    def one(i, it):
        url = it["url"]
        if not url:
            return None
        ext = Path(url.split("?")[0]).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            ext = ".jpg"
        dst = raw_dir / f"raw_{i:05d}{ext}"
        if dst.exists() and dst.stat().st_size > 0:
            with lock:
                stats["skip"] += 1
            return None
        try:
            r = session.get(url, timeout=timeout, stream=True)
            if r.status_code != 200:
                with lock:
                    stats["http_fail"] += 1
                return None
            dst.write_bytes(r.content)
        except requests.RequestException:
            with lock:
                stats["net_fail"] += 1
            return None
        with lock:
            stats["ok"] += 1
        return {"file": dst.name, **it}

    out = []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(one, i, it) for i, it in enumerate(items, 1)]
        for n, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            if rec:
                out.append(rec)
            if n % 100 == 0:
                print(f"    받는 중 {n}/{len(items)} (성공 {stats['ok']})", flush=True)
    return out, stats


def filter_and_dedupe(downloaded, raw_dir, out_dir, min_side, max_wh_ratio):
    """깨진 것·작은 것·가로로 넓은 것을 빼고, pHash로 중복을 없앤다."""
    import imagehash

    out_dir.mkdir(parents=True, exist_ok=True)
    seen, records, stats = {}, [], Counter()

    for rec in downloaded:
        p = raw_dir / rec["file"]
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im = im.convert("RGB")
                w, h = im.size
                if min(w, h) < min_side:
                    stats["too_small"] += 1
                    continue
                # 영수증은 세로로 길다. 가로로 지나치게 넓으면 배너·비교표일 확률이 높다.
                if w / h > max_wh_ratio:
                    stats["too_wide"] += 1
                    continue
                ph = str(imagehash.phash(im))
        except Exception:
            stats["broken"] += 1
            continue

        if ph in seen:
            stats["dup"] += 1
            continue
        seen[ph] = p.name

        dst = out_dir / f"gcse_{len(records) + 1:05d}{p.suffix.lower()}"
        shutil.copy2(p, dst)
        records.append({
            "file": dst.name, "width": w, "height": h, "phash": ph,
            "term": rec["term"], "src_url": rec["url"], "context": rec["context"],
            "title": rec.get("title"),
        })
        stats["kept"] += 1

    return records, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1000, help="필터 통과 목표 장수")
    ap.add_argument("--overfetch", type=float, default=1.4,
                    help="중복·불량으로 빠질 걸 감안해 더 받는 배수")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--api-key-file", default="")
    ap.add_argument("--cx", default="", help="Programmable Search Engine ID")
    ap.add_argument("--rights", default="",
                    help="사용권 필터 예: cc_publicdomain|cc_attribute|cc_sharealike")
    ap.add_argument("--safe", default="off", choices=["off", "active"])
    ap.add_argument("--pause", type=float, default=0.3, help="API 요청 간 간격(초)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--min-side", type=int, default=200)
    ap.add_argument("--max-wh-ratio", type=float, default=1.6)
    ap.add_argument("--out", default=str(ROOT))
    ap.add_argument("--backend", default="browser", choices=["browser", "cse"],
                    help="browser=실제 크롬 렌더링(키 불필요, 썸네일), cse=공식 API(키 필요)")
    ap.add_argument("--scrolls", type=int, default=8, help="browser: 검색어당 스크롤 횟수")
    ap.add_argument("--dry-run", action="store_true", help="쿼터만 계산하고 종료")
    args = ap.parse_args()

    if args.backend == "browser":
        root = Path(args.out)
        raw_dir, out_dir = root / "raw", root / "images"
        root.mkdir(parents=True, exist_ok=True)
        raw_need = int(args.target * args.overfetch)
        per_term = -(-raw_need // len(TERMS))
        print(f"브라우저 백엔드 — 목표 {args.target}장, 원본 {raw_need}장 수집 "
              f"(검색어 {len(TERMS)}개)")
        items = collect_browser(TERMS, per_term, args.scrolls, args.pause)
        if not items:
            sys.exit("한 장도 못 모았다. 차단됐을 가능성이 높다.")
        print(f"\n썸네일 URL {len(items)}건")
        downloaded, dstats = download(items, raw_dir, args.threads, args.timeout)
        print(f"다운로드: 성공 {dstats['ok']} / HTTP실패 {dstats['http_fail']} / "
              f"통신실패 {dstats['net_fail']} / 기존 {dstats['skip']}")
        records, fstats = filter_and_dedupe(
            downloaded, raw_dir, out_dir, args.min_side, args.max_wh_ratio
        )
        meta = root / "crawl_manifest.jsonl"
        with meta.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n필터: 유지 {fstats['kept']} / 중복 {fstats['dup']} / "
              f"작음 {fstats['too_small']} / 가로과다 {fstats['too_wide']} / 깨짐 {fstats['broken']}")
        print(f"최종: {out_dir} ({fstats['kept']}장)\n메타: {meta}")
        if fstats["kept"] < args.target:
            print(f"\n목표에 {args.target - fstats['kept']}장 부족. "
                  f"--scrolls 를 올리거나 TERMS 에 검색어를 추가해라.")
        return

    raw_need = int(args.target * args.overfetch)
    per_term = min(100, -(-raw_need // len(TERMS)))
    est_queries = sum(min(PER_REQ, 1) for _ in range(0))  # placeholder 제거용
    est_queries = len(TERMS) * -(-per_term // PER_REQ)

    print(f"목표 {args.target}장 → 원본 {raw_need}장 수집 (overfetch {args.overfetch}x)")
    print(f"검색어 {len(TERMS)}개 × 검색어당 최대 {per_term}장 = 약 {est_queries} 쿼리")
    print(f"무료 한도 100 쿼리/일 대비: {'초과 — 초과분 $%.2f' % ((est_queries-100)/1000*5) if est_queries > 100 else '이내'}")
    if args.dry_run:
        return

    if args.api_key_file:
        args.api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    if not args.api_key or not args.cx:
        sys.exit("API 키와 CSE ID(--cx)가 둘 다 필요하다.")

    root = Path(args.out)
    raw_dir, out_dir = root / "raw", root / "images"
    root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    all_items, used, quota_hit = [], 0, False
    print("\n수집 시작")
    for term in TERMS:
        if len(all_items) >= raw_need or quota_hit:
            break
        want = min(per_term, raw_need - len(all_items))
        items, q, quota_hit = search_term(
            session, args.api_key, args.cx, term, want, args.rights, args.safe, args.pause
        )
        used += q
        all_items.extend(items)
        print(f"  [{term}] +{len(items)}장 (누적 {len(all_items)}, 쿼리 {used})", flush=True)

    if not all_items:
        sys.exit("한 장도 못 받았다. 키·cx·API 사용설정을 확인해라.")

    # 같은 URL이 검색어 간에 겹친다. 내려받기 전에 먼저 줄인다.
    uniq, seen_url = [], set()
    for it in all_items:
        if it["url"] and it["url"] not in seen_url:
            seen_url.add(it["url"])
            uniq.append(it)
    print(f"\nURL 수집 {len(all_items)}건 → 중복 제거 후 {len(uniq)}건 (쿼리 {used}회 사용)")

    downloaded, dstats = download(uniq, raw_dir, args.threads, args.timeout)
    print(f"다운로드: 성공 {dstats['ok']} / HTTP실패 {dstats['http_fail']} / "
          f"통신실패 {dstats['net_fail']} / 기존 {dstats['skip']}")

    records, fstats = filter_and_dedupe(
        downloaded, raw_dir, out_dir, args.min_side, args.max_wh_ratio
    )

    meta = root / "crawl_manifest.jsonl"
    with meta.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n필터: 유지 {fstats['kept']} / 중복 {fstats['dup']} / 작음 {fstats['too_small']} / "
          f"가로과다 {fstats['too_wide']} / 깨짐 {fstats['broken']}")
    print(f"최종: {out_dir} ({fstats['kept']}장)")
    print(f"메타: {meta}  (검색어·원본URL·출처페이지 포함)")
    if fstats["kept"] < args.target:
        print(f"\n목표 {args.target}장에 {args.target - fstats['kept']}장 부족. "
              f"--overfetch 를 올리거나 TERMS 에 검색어를 추가해라.")


if __name__ == "__main__":
    main()
