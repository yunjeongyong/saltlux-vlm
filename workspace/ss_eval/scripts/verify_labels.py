"""
Verify/refine document layout-parse labels (./data/<type>/jsons/*.json)
against their source images (./data/<type>/images/*) using a GPT vision
model, writing corrected labels to ./output/<type>/jsons/*.json with the
same JSON schema as the input.

For each page this sends the full image plus every current
parsing_res_list block (label/bbox/content) in one batched call, and asks
the model to return a corrected block list: existing blocks may be kept,
content/bbox-edited, or removed as false positives; missed real content
may be added as new blocks. Table blocks are corrected for TEDS-relevant
structure/content only (row/col/span/text) — cell styling is ignored.
elements[].markdown/html and result.md/result.html are then regenerated
deterministically from the corrected block list to stay consistent.
layout_det_res (raw detector boxes) is left untouched.

Usage (OPENAI_API_KEY must be in .env):
    uv run --env-file .env python scripts/verify_labels.py --pilot
    uv run --env-file .env python scripts/verify_labels.py
    uv run --env-file .env python scripts/verify_labels.py --types bill
    uv run --env-file .env python scripts/verify_labels.py --model gpt-4.1
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import markdown as md
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
DOC_TYPES = ["bill", "invoice", "quotation"]
DEFAULT_MODEL = "gpt-5.6"
CONCURRENCY = 5

# USD per 1M tokens. Checked against platform.openai.com/docs/pricing on
# 2026-08-22 — re-verify before a large/expensive run, prices change.
PRICING = {
    "gpt-5.6": {"input": 1.25, "output": 10.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

KNOWN_LABELS = [
    "doc_title",
    "paragraph_title",
    "figure_title",
    "text",
    "table",
    "header",
    "footer",
    "header_image",
    "footer_image",
    "image",
    "seal",
    "number",
    "vision_footnote",
    "chart",
    "inline_formula",
]

SYSTEM_PROMPT = (
    "You are verifying an automated document-layout-parsing result against the "
    "original document page image. You will be given the full page image and "
    "the current list of parsed blocks (label, bounding box [x0,y0,x1,y1] in "
    "pixels, and extracted content).\n\n"
    "For each existing block, decide:\n"
    "- keep: content, label and bbox are already correct.\n"
    "- edit: content/label/bbox need correction (wrong/garbled text, wrong "
    "label, misplaced or oversized/undersized bbox). For 'table' blocks, "
    "content is an HTML table — only fix TEDS-relevant structure: row/column "
    "layout, colspan/rowspan, and cell text content. Do not worry about "
    "inline style attributes or formatting cosmetics.\n"
    "- remove: this block is a false positive — the bbox region does not "
    "correspond to real, distinct document content (e.g. stray detection "
    "over blank space, or a duplicate of another block).\n\n"
    "Also scan the full image for real content (text, table, title, image, "
    "seal, etc.) that has NO corresponding block at all, and add it as a new "
    "block with action 'add' (ref_id null), a tight bbox, correct label, and "
    "its content.\n\n"
    "Every block's label must be exactly one of the known labels provided. "
    "bbox values are pixel coordinates within the given image dimensions. "
    "Return keep/edit/remove for every existing block (by its ref_id) plus "
    "add entries for anything missing. Do not invent content that is not "
    "visible in the image."
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ref_id": {
                        "type": ["integer", "null"],
                        "description": "original block_id for keep/edit/remove; null for add",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["keep", "edit", "remove", "add"],
                    },
                    "label": {"type": "string", "enum": KNOWN_LABELS},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "content": {"type": "string"},
                },
                "required": ["ref_id", "action", "label", "bbox", "content"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["blocks"],
    "additionalProperties": False,
}


@dataclass
class Doc:
    doc_type: str
    name: str
    json_path: Path
    image_path: Path
    out_path: Path


@dataclass
class UsageTracker:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add(self, input_tokens: int, output_tokens: int) -> None:
        async with self.lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.calls += 1

    def cost(self) -> float:
        price = PRICING.get(self.model)
        if not price:
            return float("nan")
        return (
            self.input_tokens / 1_000_000 * price["input"]
            + self.output_tokens / 1_000_000 * price["output"]
        )

    def summary(self) -> str:
        cost = self.cost()
        cost_str = f"${cost:.4f}" if cost == cost else "unknown (no pricing entry)"
        return (
            f"\n--- usage summary ({self.model}) ---\n"
            f"calls: {self.calls}\n"
            f"input tokens:  {self.input_tokens:,}\n"
            f"output tokens: {self.output_tokens:,}\n"
            f"estimated cost: {cost_str}"
        )

    def as_dict(self) -> dict:
        cost = self.cost()
        return {
            "model": self.model,
            "num_requests": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": None if cost != cost else round(cost, 4),
        }


def find_docs(doc_type: str) -> list[Doc]:
    jsons_dir = DATA_DIR / doc_type / "jsons"
    images_dir = DATA_DIR / doc_type / "images"
    out_dir = OUTPUT_DIR / doc_type / "jsons"
    docs = []
    for jpath in sorted(jsons_dir.glob("*.json")):
        name = jpath.stem
        candidates = list(images_dir.glob(f"{name}.*"))
        if not candidates:
            print(f"  [skip] no image for {jpath}")
            continue
        docs.append(
            Doc(
                doc_type=doc_type,
                name=name,
                json_path=jpath,
                image_path=candidates[0],
                out_path=out_dir / f"{name}.json",
            )
        )
    return docs


def image_to_data_url(path: Path) -> str:
    ext = path.suffix.lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    x0, y0, x1, y1 = bbox
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def render_markdown(blocks: list[dict]) -> str:
    parts = []
    for b in blocks:
        content = b.get("block_content") or ""
        if b["block_label"] == "doc_title":
            content = f"# {content}"
        parts.append(content)
    return "\n\n".join(p for p in parts if p != "")


def render_full_html(old_full_html: str, body_html: str) -> str:
    if "<body>" in old_full_html and "</body>" in old_full_html:
        return re.sub(
            r"<body>.*</body>",
            lambda _: f"<body>\n{body_html}\n</body>",
            old_full_html,
            flags=re.DOTALL,
        )
    return old_full_html


def reconcile_blocks(
    original_blocks: list[dict], gpt_blocks: list[dict]
) -> tuple[list[dict], dict[str, int]]:
    by_id = {b["block_id"]: b for b in original_blocks}
    counts = {"kept": 0, "edited": 0, "removed": 0, "added": 0, "warnings": 0}
    result = []

    for item in gpt_blocks:
        action = item["action"]
        if action == "remove":
            counts["removed"] += 1
            continue

        if action in ("keep", "edit"):
            orig = by_id.get(item["ref_id"])
            if orig is None:
                counts["warnings"] += 1
                action = "add"
            else:
                changed = (
                    item["content"] != orig.get("block_content")
                    or item["label"] != orig.get("block_label")
                    or list(item["bbox"]) != list(orig.get("block_bbox") or [])
                )
                newb = dict(orig)
                newb["block_label"] = item["label"]
                newb["block_content"] = item["content"]
                newb["block_bbox"] = item["bbox"]
                newb["block_polygon_points"] = bbox_to_polygon(item["bbox"])
                counts["edited" if changed else "kept"] += 1
                result.append(newb)
                continue

        if action == "add":
            newb = {
                "block_label": item["label"],
                "block_content": item["content"],
                "block_bbox": item["bbox"],
                "block_id": None,
                "block_order": None,
                "layout_score": None,
                "avg_logprob": None,
                "group_id": None,
                "block_polygon_points": bbox_to_polygon(item["bbox"]),
            }
            counts["added"] += 1
            result.append(newb)

    result.sort(key=lambda b: (b["block_bbox"][1], b["block_bbox"][0]))
    for i, b in enumerate(result):
        b["block_id"] = i
        b["block_order"] = i + 1

    return result, counts


async def verify_doc(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    doc: Doc,
    model: str,
    effort: str,
    usage: UsageTracker,
) -> str:
    async with sem:
        raw = doc.json_path.read_text(encoding="utf-8")
        doc_json = json.loads(raw)
        elements = doc_json.get("result", {}).get("elements", [])
        if not elements:
            return f"[skip] {doc.doc_type}/{doc.name}: no elements found"

        image_url = image_to_data_url(doc.image_path)
        total_counts = {"kept": 0, "edited": 0, "removed": 0, "added": 0, "warnings": 0}

        for el in elements:
            j = el.get("json", {})
            original_blocks = j.get("parsing_res_list", [])
            width = j.get("width")
            height = j.get("height")

            current_blocks_payload = [
                {
                    "ref_id": b["block_id"],
                    "label": b["block_label"],
                    "bbox": b["block_bbox"],
                    "content": b["block_content"],
                }
                for b in original_blocks
            ]

            resp = await client.responses.create(
                model=model,
                reasoning={"effort": effort},
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": image_url},
                            {
                                "type": "input_text",
                                "text": (
                                    f"Page image size: {width}x{height} px.\n"
                                    f"Known labels: {', '.join(KNOWN_LABELS)}.\n\n"
                                    "Current blocks (JSON):\n"
                                    + json.dumps(current_blocks_payload, ensure_ascii=False)
                                ),
                            },
                        ],
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "block_review",
                        "strict": True,
                        "schema": RESPONSE_SCHEMA,
                    }
                },
            )

            if resp.usage:
                await usage.add(resp.usage.input_tokens, resp.usage.output_tokens)

            try:
                parsed = json.loads(resp.output_text)
            except (json.JSONDecodeError, TypeError):
                return f"[error] {doc.doc_type}/{doc.name}: could not parse model output as JSON"

            new_blocks, counts = reconcile_blocks(original_blocks, parsed["blocks"])
            for k in total_counts:
                total_counts[k] += counts[k]

            j["parsing_res_list"] = new_blocks
            markdown_text = render_markdown(new_blocks)
            body_html = md.markdown(markdown_text)
            el["markdown"] = markdown_text
            el["html"] = body_html

        # Regenerate page-level mirrors from the (last/only) element's markdown.
        combined_markdown = "\n\n".join(el["markdown"] for el in elements)
        combined_body_html = "\n".join(el["html"] for el in elements)
        doc_json["result"]["md"] = combined_markdown
        old_full_html = doc_json["result"].get("html", "")
        if isinstance(old_full_html, str):
            doc_json["result"]["html"] = render_full_html(old_full_html, combined_body_html)

        doc.out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.out_path.write_text(
            json.dumps(doc_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        warn = f", {total_counts['warnings']} warnings" if total_counts["warnings"] else ""
        return (
            f"[done] {doc.doc_type}/{doc.name}: "
            f"kept={total_counts['kept']} edited={total_counts['edited']} "
            f"removed={total_counts['removed']} added={total_counts['added']}{warn}"
        )


async def run(
    doc_types: list[str],
    pilot: bool,
    skip_existing: bool,
    model: str,
    effort: str,
) -> None:
    client = AsyncOpenAI()
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
            tasks.append(verify_doc(client, sem, d, model, effort, usage))

    print(f"Processing {len(tasks)} documents with {model} (reasoning effort={effort})...")
    doc_results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(result)
        doc_results.append(result)

    print(usage.summary())

    runs_dir = ROOT / "runs"
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "timestamp_utc": ts,
        "doc_types": doc_types,
        "pilot": pilot,
        "num_documents": len(tasks),
        **usage.as_dict(),
        "results": doc_results,
    }
    report_path = runs_dir / f"{ts}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Run report written to {report_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Only 2 docs per type")
    parser.add_argument("--types", nargs="+", choices=DOC_TYPES, default=DOC_TYPES)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--effort",
        default="low",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        help="reasoning effort for gpt-5.x models",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess docs even if output already exists",
    )
    args = parser.parse_args()

    asyncio.run(
        run(
            doc_types=args.types,
            pilot=args.pilot,
            skip_existing=not args.no_skip_existing,
            model=args.model,
            effort=args.effort,
        )
    )


if __name__ == "__main__":
    main()
