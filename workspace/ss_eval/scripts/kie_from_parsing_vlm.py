"""
Key-information-extraction labeling from already-parsed documents. For each
parsed doc in ./test_output/parsing/<type>/<name>.json (produced by
document_parsing.py) this calls a GPT text model — no image, just the parsed
markdown/html — and writes a schema-matched JSON to
./test_output/kie/<type>/<name>.json, shaped as:

    {"General": {...overall doc fields...}, "Table": [...itemized rows...]}

Prompts and per-type field schemas are reused from kie_label.py.

Usage:
    uv run python scripts/kie_from_parsing.py --api-url https://api.openai.com/v1 --api-model gpt-5.6 --api-key sk-...
    uv run python scripts/kie_from_parsing.py --api-url http://localhost:8000/v1 --api-model my-model --api-key "" --pilot
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import openai
from openai import AsyncOpenAI

from kie_label import (
    CONCURRENCY,
    DOC_TYPES,
    PROMPTS,
    SCHEMAS,
    UsageTracker,
)

ROOT = Path(__file__).resolve().parent.parent
PARSING_DIR = ROOT / "test_output" / "parsing"      # --parsing-dir 로 덮어씀
KIE_OUTPUT_DIR = ROOT / "test_output" / "kie"       # --kie-dir 로 덮어씀
RUNS_DIR = ROOT / "runs"


@dataclass
class Doc:
    doc_type: str
    name: str
    parsed_path: Path
    out_path: Path


def find_docs(doc_type: str) -> list[Doc]:
    parsed_dir = PARSING_DIR / doc_type
    out_dir = KIE_OUTPUT_DIR / doc_type
    docs = []
    for md_path in sorted(parsed_dir.glob("*.md")):
        docs.append(
            Doc(
                doc_type=doc_type,
                name=md_path.stem,
                parsed_path=md_path,
                out_path=out_dir / f"{md_path.stem}.json",
            )
        )
    return docs


async def label_doc(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    doc: Doc,
    model: str,
    effort: str,
    usage: UsageTracker,
) -> str:
    async with sem:
        context_text = doc.parsed_path.read_text(encoding="utf-8")
        if not context_text:
            return f"[skip] {doc.doc_type}/{doc.name}: no parsed text found"

        text = PROMPTS[doc.doc_type]
        text += (
            "\n\nParsed page text/table (may contain minor OCR errors):\n\n"
            + context_text
        )

        extra_body = {} if effort == "none" else {"reasoning_effort": effort}
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": text}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "kie_extraction",
                        "strict": True,
                        "schema": SCHEMAS[doc.doc_type],
                    },
                },
                extra_body=extra_body,
            )
        except openai.APITimeoutError:
            return f"[error] {doc.doc_type}/{doc.name}: request timed out"
        except openai.APIError as e:
            return f"[error] {doc.doc_type}/{doc.name}: API error - {e}"

        if resp.usage:
            await usage.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)

        try:
            parsed = json.loads(resp.choices[0].message.content)
        except (json.JSONDecodeError, TypeError, IndexError):
            return f"[error] {doc.doc_type}/{doc.name}: could not parse model output as JSON"

        doc.out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.out_path.write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n_rows = len(parsed.get("Table", []))
        return f"[done] {doc.doc_type}/{doc.name}: {n_rows} table row(s)"


async def run(
    doc_types: list[str],
    pilot: bool,
    skip_existing: bool,
    api_url: str,
    api_model: str,
    api_key: str,
    effort: str,
) -> None:
    client = AsyncOpenAI(base_url=api_url, api_key=api_key or "not-needed", timeout=600)
    model = api_model
    sem = asyncio.Semaphore(CONCURRENCY)
    usage = UsageTracker(model=model)
    tasks = []
    for dt in doc_types:
        docs = find_docs(dt)
        if pilot:
            docs = docs[:2]
        if skip_existing:
            docs = [d for d in docs if not d.out_path.exists()]
        for d in docs:
            tasks.append(label_doc(client, sem, d, model, effort, usage))

    print(f"Processing {len(tasks)} documents with {model} (reasoning effort={effort})...")
    doc_results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(result)
        doc_results.append(result)

    print(usage.summary())

    RUNS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "task": "kie_from_parsing",
        "timestamp_utc": ts,
        "doc_types": doc_types,
        "pilot": pilot,
        "num_documents": len(tasks),
        **usage.as_dict(),
        "results": doc_results,
    }
    report_path = RUNS_DIR / f"{ts}_kie_from_parsing.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Run report written to {report_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Only 2 docs per type")
    parser.add_argument("--types", nargs="+", choices=DOC_TYPES, default=DOC_TYPES)
    parser.add_argument("--parsing-dir", help="파싱 .md 가 있는 디렉터리 (기본 test_output/parsing)")
    parser.add_argument("--kie-dir", help="KIE json 출력 디렉터리 (기본 test_output/kie)")
    parser.add_argument("--api-url", required=True, help="OpenAI-compatible base URL")
    parser.add_argument(
        "--api-key", required=True, help="API key; pass an empty string if none is needed"
    )
    parser.add_argument("--api-model", default="luxia", help="model name to request")
    parser.add_argument(
        "--effort",
        default="low",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess docs even if output already exists",
    )
    args = parser.parse_args()
    global PARSING_DIR, KIE_OUTPUT_DIR
    if args.parsing_dir: PARSING_DIR = Path(args.parsing_dir)
    if args.kie_dir: KIE_OUTPUT_DIR = Path(args.kie_dir)

    asyncio.run(
        run(
            doc_types=args.types,
            pilot=args.pilot,
            skip_existing=not args.no_skip_existing,
            api_url=args.api_url,
            api_model=args.api_model,
            api_key=args.api_key,
            effort=args.effort,
        )
    )


if __name__ == "__main__":
    main()
