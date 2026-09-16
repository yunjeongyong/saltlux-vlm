import asyncio
import json
import mimetypes
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "test_output_alt" / "parsing"
DOC_TYPES = ["bill", "invoice", "quotation", "receipt"]
CONCURRENCY = 1

BASE_URL = "http://192.168.250.52:12000"
SUBMIT_URL = f"{BASE_URL}/v1/jobs/submit"
POLL_INTERVAL_S = 2


def find_images(doc_type: str) -> list[Path]:
    images_dir = DATA_DIR / doc_type / "images"
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    files = []
    for ext in exts:
        files.extend(images_dir.glob(ext))
    return sorted(files)


async def parse_doc(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    doc_type: str,
    img_fp: Path,
) -> str:
    out_json = OUT_DIR / doc_type / f"{img_fp.stem}.json"
    if out_json.exists():
        return f"[skip] {doc_type}/{img_fp.stem}: already parsed"

    async with sem:
        files = {
            "file": (
                img_fp.name,
                img_fp.read_bytes(),
                mimetypes.guess_type(img_fp)[0] or "application/octet-stream",
            )
        }
        data = {
            'modelQuality': 'high'
        }
        try:
            resp = await client.post(SUBMIT_URL, data=data, files=files)
            resp.raise_for_status()
            job = resp.json()
            job_id = job["job_id"]

            while True:
                await asyncio.sleep(POLL_INTERVAL_S)
                resp = await client.get(f"{BASE_URL}/v1/jobs/{job_id}")
                resp.raise_for_status()
                status = resp.json()
                job_status = status.get("job_status")
                if job_status == "completed":
                    break
                if job_status == "failed":
                    print(status)
                    return f"[error] {doc_type}/{img_fp.stem}: {status.get('error', 'job failed')}"

            resp = await client.get(f"{BASE_URL}/v1/results/{job_id}")
            resp.raise_for_status()
            result = resp.json()
        except httpx.HTTPStatusError as e:
            return f"[error] {doc_type}/{img_fp.stem}: HTTP {e.response.status_code} - {e.response.text[:200]}"
        except httpx.RequestError as e:
            return f"[error] {doc_type}/{img_fp.stem}: request failed - {e}"

    if result.get("status") != 200:
        return f"[error] {doc_type}/{img_fp.stem}: {result.get('error', 'unknown error')}"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    out_json.with_suffix(".md").write_text(result["result"]["md"], encoding="utf-8")
    return f"[done] {doc_type}/{img_fp.stem}"


async def run() -> None:
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(timeout=300) as client:
        tasks = [
            parse_doc(client, sem, doc_type, img_fp)
            for doc_type in DOC_TYPES
            for img_fp in find_images(doc_type)
        ]
        for coro in asyncio.as_completed(tasks):
            print(await coro)


if __name__ == "__main__":
    asyncio.run(run())
