"""
Key-information-extraction labeling. For each image in ./data/<type>/images
this calls a GPT vision model and writes a schema-matched JSON to
./kie_output/<type>/<name>.json, shaped as:

    {"General": {...overall doc fields...}, "Table": [...itemized rows...]}

Per-type field schemas live in SCHEMAS below. If a verified label
(./output/<type>/jsons/<name>.json, from verify_labels.py) or the raw
label (./data/<type>/jsons/<name>.json) exists, its extracted text is
passed alongside the image as extra context to improve accuracy.

Usage (OPENAI_API_KEY must be in .env):
    uv run --env-file .env python scripts/kie_label.py --pilot
    uv run --env-file .env python scripts/kie_label.py
    uv run --env-file .env python scripts/kie_label.py --types bill
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
VERIFIED_DIR = ROOT / "output"
KIE_OUTPUT_DIR = ROOT / "kie_output"
RUNS_DIR = ROOT / "runs"
DOC_TYPES = ["bill", "invoice", "quotation", "receipt"]
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

NULL_STR = ["string", "null"]

BILL_SCHEMA = {
    "type": "object",
    "properties": {
        "General": {
            "type": "object",
            "properties": {
                "bl_number": {"type": NULL_STR},
                "booking_number": {"type": NULL_STR},
                "export_references": {"type": NULL_STR},
                "shipper_name": {"type": NULL_STR},
                "shipper_address": {"type": NULL_STR},
                "consignee_name": {"type": NULL_STR},
                "consignee_address": {"type": NULL_STR},
                "notify_party_name": {"type": NULL_STR},
                "notify_party_address": {"type": NULL_STR},
                "forwarding_agent": {"type": NULL_STR},
                "country_of_origin": {"type": NULL_STR},
                "vessel_name": {"type": NULL_STR},
                "port_of_loading": {"type": NULL_STR},
                "port_of_discharge": {"type": NULL_STR},
                "place_of_initial_receipt": {"type": NULL_STR},
                "place_of_delivery": {"type": NULL_STR},
                "mode_of_initial_carriage": {"type": NULL_STR},
                "type_of_movement": {"type": NULL_STR},
                "freight_payable_at": {"type": NULL_STR},
                "total_number_of_packages": {"type": NULL_STR},
                "declared_value": {"type": ["number", "null"]},
                "currency": {"type": NULL_STR},
                "insurance_requested": {"type": ["boolean", "null"]},
                "insurance_amount": {"type": ["number", "null"]},
                "carrier_name": {"type": NULL_STR},
                "date_issued": {"type": NULL_STR},
            },
            "required": [
                "bl_number", "booking_number", "export_references",
                "shipper_name", "shipper_address", "consignee_name",
                "consignee_address", "notify_party_name", "notify_party_address",
                "forwarding_agent", "country_of_origin", "vessel_name",
                "port_of_loading", "port_of_discharge", "place_of_initial_receipt",
                "place_of_delivery", "mode_of_initial_carriage", "type_of_movement",
                "freight_payable_at", "total_number_of_packages", "declared_value",
                "currency", "insurance_requested", "insurance_amount",
                "carrier_name", "date_issued",
            ],
            "additionalProperties": False,
        },
        "Table": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "marks_and_nos": {"type": NULL_STR},
                    "no_of_packages": {"type": NULL_STR},
                    "description": {"type": NULL_STR},
                    "gross_weight": {"type": NULL_STR},
                    "measurement": {"type": NULL_STR},
                },
                "required": [
                    "marks_and_nos", "no_of_packages", "description",
                    "gross_weight", "measurement",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["General", "Table"],
    "additionalProperties": False,
}

INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "General": {
            "type": "object",
            "properties": {
                "document_title": {"type": NULL_STR},
                "issuer": {"type": NULL_STR},
                "recipient": {"type": NULL_STR},
                "notify_party": {"type": NULL_STR},
                "signatory": {"type": NULL_STR},
                "subject_description": {"type": NULL_STR},
                "reference_numbers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "value": {"type": "string"},
                        },
                        "required": ["label", "value"],
                        "additionalProperties": False,
                    },
                },
                "issue_date": {"type": NULL_STR},
                "shipment_date": {"type": NULL_STR},
                "issuer_address": {"type": NULL_STR},
                "recipient_address": {"type": NULL_STR},
                "port_of_loading": {"type": NULL_STR},
                "port_of_discharge": {"type": NULL_STR},
                "shipment_route": {"type": NULL_STR},
                "purpose": {"type": NULL_STR},
                "coverage_or_certification_basis": {"type": NULL_STR},
                "vessel_voyage": {"type": NULL_STR},
                "mode_of_transport": {"type": NULL_STR},
                "credit_terms": {"type": NULL_STR},
                "amount": {"type": ["number", "null"]},
                "currency": {"type": NULL_STR},
            },
            "required": [
                "document_title", "issuer", "recipient", "notify_party",
                "signatory", "subject_description", "reference_numbers",
                "issue_date", "shipment_date", "issuer_address",
                "recipient_address", "port_of_loading", "port_of_discharge",
                "shipment_route", "purpose", "coverage_or_certification_basis",
                "vessel_voyage", "mode_of_transport", "credit_terms",
                "amount", "currency",
            ],
            "additionalProperties": False,
        },
        "Table": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_description": {"type": NULL_STR},
                    "quantity": {"type": NULL_STR},
                    "unit_price": {"type": ["number", "null"]},
                    "amount": {"type": ["number", "null"]},
                },
                "required": ["item_description", "quantity", "unit_price", "amount"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["General", "Table"],
    "additionalProperties": False,
}

QUOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "General": {
            "type": "object",
            "properties": {
                "document_type_label": {"type": NULL_STR},
                "document_number": {"type": NULL_STR},
                "date": {"type": NULL_STR},
                "issuer_name": {"type": NULL_STR},
                "issuer_address": {"type": NULL_STR},
                "recipient_name": {"type": NULL_STR},
                "recipient_address": {"type": NULL_STR},
                "currency": {"type": NULL_STR},
                "subtotal": {"type": ["number", "null"]},
                "tax_price": {"type": ["number", "null"]},
                "total_charged_price": {"type": ["number", "null"]},
                "valid_until": {"type": NULL_STR},
                "payment_terms": {"type": NULL_STR},
            },
            "required": [
                "document_type_label", "document_number", "date",
                "issuer_name", "issuer_address", "recipient_name",
                "recipient_address", "currency", "subtotal", "tax_price",
                "total_charged_price", "valid_until", "payment_terms",
            ],
            "additionalProperties": False,
        },
        "Table": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "product_name": {"type": NULL_STR},
                    "product_quantity": {"type": ["number", "null"]},
                    "product_price": {"type": ["number", "null"]},
                },
                "required": ["product_name", "product_quantity", "product_price"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["General", "Table"],
    "additionalProperties": False,
}

RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "General": {
            "type": "object",
            "properties": {
                "store_name": {"type": NULL_STR},
                "store_address": {"type": NULL_STR},
                "transaction_date": {"type": NULL_STR},
                "transaction_time": {"type": NULL_STR},
                "approval_code": {"type": NULL_STR},
                "tax_price": {"type": ["number", "null"]},
                "total_charged_price": {"type": ["number", "null"]},
                "currency": {"type": NULL_STR},
            },
            "required": [
                "store_name", "store_address", "transaction_date",
                "transaction_time", "approval_code", "tax_price",
                "total_charged_price", "currency",
            ],
            "additionalProperties": False,
        },
        "Table": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "product_name": {"type": NULL_STR},
                    "product_quantity": {"type": ["number", "null"]},
                    "product_price": {"type": ["number", "null"]},
                },
                "required": ["product_name", "product_quantity", "product_price"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["General", "Table"],
    "additionalProperties": False,
}

SCHEMAS = {
    "bill": BILL_SCHEMA,
    "invoice": INVOICE_SCHEMA,
    "quotation": QUOTATION_SCHEMA,
    "receipt": RECEIPT_SCHEMA,
}

PROMPTS = {
    "bill": (
        "Extract key information from this Bill of Lading image into the given "
        "schema.\nGeneral holds document-level shipping fields (parties, ports, "
        "vessel, freight terms, etc).\nTable holds one row per cargo line item "
        "(marks/packages/description/weight/measurement). Use null for any "
        "field not present on the document. Do not invent values.\n"
"""
{
  "General": {
    "bl_number": "HG765007",
    "booking_number": "64183",
    "export_references": "4498.43-3340",
    "shipper_name": "MISAN TRADING CO., LTD.",
    "shipper_address": "684, PANPO-DONG, SOCHO-KU, INCHEON 20757, REP. OF KOREA; TEL:+(38)(78)4852-0620; FAX:+(13)(58)7909-4788",
    "consignee_name": "FRITERION SYSTEMS CO., LTD.",
    "consignee_address": "256 BUSTLETON AVENUE ORRLANDO, OH 56466, UNITED STATES; TEL:+(70)(46)8549-1760; FAX:+(32)(45)1811-6156",
    "notify_party_name": "PIDWAY STAFFING CO., LTD.",
    "notify_party_address": "72 ENOKI-CHO, SHINJUKU-KU, TOKYO, 179-2077, JAPAN; TEL:+(53)(22)9190-9405; FAX:+(60)(35)7194-5240",
    "forwarding_agent": "YAMADA SHOKAI CO., LTD.; 23786-8613-8820",
    "country_of_origin": "TAIWAN",
    "vessel_name": "HERCULES LEADER",
    "port_of_loading": "VENTANAS, CHILE",
    "port_of_discharge": "KANAZAWA, JAPAN",
    "place_of_initial_receipt": "SHUSHI, JAPAN",
    "place_of_delivery": "RYOTSU, JAPAN",
    "mode_of_initial_carriage": "DEQ",
    "type_of_movement": "OCEAN",
    "freight_payable_at": null,
    "total_number_of_packages": "66 PKG",
    "declared_value": 948.19,
    "currency": "USD",
    "insurance_requested": true,
    "insurance_amount": 672.6,
    "carrier_name": "JUGHES ENVIRONMENAL CO., LTD.",
    "date_issued": "2011-07-06"
  },
  "Table": [
    {
      "marks_and_nos": "DLSU1092675; U325853",
      "no_of_packages": "34 PKG",
      "description": "SL240PMC-1470NMTRANSIVER",
      "gross_weight": "207KG",
      "measurement": "577.14 CBM"
    },
    {
      "marks_and_nos": null,
      "no_of_packages": "24 PKG",
      "description": "SEMIPERMEABLE MEMBRANE",
      "gross_weight": "560KG",
      "measurement": "284.81 CBM"
    }
  ]
}
"""
    ),
    "invoice": (
        "This document is one of: Marine Cargo Insurance Policy, Inspection "
        "Certificate, or Certificate of Quality.\nExtract key information into "
        "the given schema.\nGeneral covers who issued/received it, what it "
        "covers, when, where, why, and how (organize your reading around "
        "those questions, but only fill the listed fields). reference_numbers "
        "should capture every printed reference/number on the doc as "
        "{label, value} pairs (e.g. policy no, invoice no, L/C no, contract "
        "no, certificate no, bill of lading no) using the label as printed. "
        "Table holds itemized line items if any are present (rare for these "
        "doc types — return an empty array if none). Use null for absent "
        "General fields. Do not invent values.\n"
"""
{
  "General": {
    "document_title": "MARINE CARGO INSURANCE POLICY",
    "issuer": "International Marine Insurance Co., Ltd.",
    "recipient": "Dlearcom It Solutions Co., Ltd.",
    "notify_party": "Ded Bath&Beyond Co., Ltd.",
    "signatory": "AUTHORIZED SIGNATORY",
    "subject_description": "FILTER ASSY and Shoulder Suspenders",
    "reference_numbers": [
      {
        "label": "Policy No.",
        "value": "733-457-9784"
      },
      {
        "label": "Invoice No.",
        "value": "HG459023"
      },
      {
        "label": "L/C No.",
        "value": "M0149265NS96976"
      },
      {
        "label": "No. of Policy issued.",
        "value": "FOUR"
      }
    ],
    "issue_date": "05-May-2003",
    "shipment_date": "19-Jul-2016",
    "issuer_address": null,
    "recipient_address": null,
    "port_of_loading": "BOL, CROATIA; LARYMNA, GREECE",
    "port_of_discharge": "GYOR, HUNGARY; KUSUBO, JAPAN",
    "shipment_route": "From BOL, CROATIA; at and from LARYMNA, GREECE; arrived at GYOR, HUNGARY; transshipped at SHIROKO, JAPAN; thence to KUSUBO, JAPAN",
    "purpose": "Marine cargo insurance against loss of or damage to the insured goods during transit",
    "coverage_or_certification_basis": "Against all risks, subject to the listed Institute Cargo, War, Strikes, Air Cargo, Classification, Special Replacement, Radioactive Contamination Exclusion and Co-Insurance clauses; excluding rust, oxidation and discoloration; warranted containerized shipment throughout the entire transit.",
    "vessel_voyage": "WESER HIGHWAY, V.160",
    "mode_of_transport": "Marine vessel; containerized shipment",
    "credit_terms": null,
    "amount": 5967.07,
    "currency": "USD"
  },
  "Table": [
    {
      "item_description": "FILTER ASSY",
      "quantity": "99 KGS",
      "unit_price": null,
      "amount": null
    },
    {
      "item_description": "Shoulder Suspenders",
      "quantity": "33 KGS",
      "unit_price": null,
      "amount": null
    }
  ]
}
"""
    ),
    "quotation": (
        "Extract key information from this quotation/estimate/purchase-order "
        "style document into the given schema.\nGeneral holds document-level "
        "fields (parties, dates, totals).\nTable holds one row per priced "
        "line item.\nUse null for any field not present. Do not invent values.\n"
"""
{
  "General": {
    "document_type_label": "Quotation",
    "document_number": "Q00002",
    "date": "23/03/26 11:54",
    "issuer_name": "Caffyns PLC",
    "issuer_address": "46 Lottbridge Drive, Eastbourne, East Sussex BN23 6PJ",
    "recipient_name": "CASH RETAIL",
    "recipient_address": "Lottbridge Drove, Eastbourne, BN23 6PJ",
    "currency": "GBP",
    "subtotal": 585.4,
    "tax_price": 117.08,
    "total_charged_price": 702.48,
    "valid_until": null,
    "payment_terms": "Cash"
  },
  "Table": [
    {
      "product_name": "VOU 31256775 - TRANSMISSION OI",
      "product_quantity": 3,
      "product_price": 89
    },
    {
      "product_name": "VOU 31259380 - TRANSMISSION OI",
      "product_quantity": 1,
      "product_price": 32.8
    },
    ...
  ]
}
"""
    ),
    "receipt": (
        "Extract key information from this store receipt image into the "
        "given schema.\nGeneral holds store/transaction-level fields.\nTable "
        "holds one row per purchased item (product_quantity is the purchased "
        "quantity, kept as a number). Receipts may be in any language (e.g. "
        "Korean) — extract values as printed, using the printed currency "
        "code/symbol for the currency field, inferring the ISO currency code "
        "from language/country context when no symbol is printed (e.g. a "
        "Korean-language receipt implies KRW). Use null for any other field "
        "not present. Do not invent values.\n"
"""
{
  "General": {
    "store_name": "쭈꾸미낙지볶음전문점 대성",
    "store_address": "대구 수성구 달구벌대로 2637 1층",
    "transaction_date": "2019-11-24",
    "transaction_time": null,
    "approval_code": null,
    "tax_price": 14109,
    "total_charged_price": 155200,
    "currency": "KRW"
  },
  "Table": [
    {
      "product_name": "쭈꾸미전세트",
      "product_price": 59400,
      "product_quantity": 6
    },
    {
      "product_name": "쭈꾸미피자세트",
      "product_price": 69300,
      "product_quantity": 7
    },
    {
      "product_name": "소주",
      "product_price": 10500,
      "product_quantity": 3
    },
    {
      "product_name": "맥주",
      "product_price": 16000,
      "product_quantity": 4
    }
  ]
}
"""
    ),
}


@dataclass
class Doc:
    doc_type: str
    name: str
    image_path: Path
    context_path: Path | None
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
    images_dir = DATA_DIR / doc_type / "images"
    out_dir = KIE_OUTPUT_DIR / doc_type
    docs = []
    for ipath in sorted(images_dir.glob("*.*")):
        name = ipath.stem
        verified = VERIFIED_DIR / doc_type / "jsons" / f"{name}.json"
        raw = DATA_DIR / doc_type / "jsons" / f"{name}.json"
        context_path = verified if verified.exists() else (raw if raw.exists() else None)
        docs.append(
            Doc(
                doc_type=doc_type,
                name=name,
                image_path=ipath,
                context_path=context_path,
                out_path=out_dir / f"{name}.json",
            )
        )
    return docs


def image_to_data_url(path: Path) -> str:
    ext = path.suffix.lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def load_context_text(context_path: Path | None) -> str | None:
    if context_path is None:
        return None
    try:
        doc_json = json.loads(context_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    elements = doc_json.get("result", {}).get("elements", [])
    if not elements:
        return None
    return elements[0].get("markdown") or elements[0].get("html")


async def label_doc(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    doc: Doc,
    model: str,
    effort: str,
    usage: UsageTracker,
) -> str:
    async with sem:
        image_url = image_to_data_url(doc.image_path)
        context_text = load_context_text(doc.context_path)

        user_content = [{"type": "input_image", "image_url": image_url}]
        text = PROMPTS[doc.doc_type]
        if context_text:
            text += (
                "\n\nExtracted page text/table (may contain minor OCR errors, "
                "use the image as ground truth when they conflict):\n\n"
                + context_text
            )
        user_content.append({"type": "input_text", "text": text})

        resp = await client.responses.create(
            model=model,
            reasoning={"effort": effort},
            input=[{"role": "user", "content": user_content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "kie_extraction",
                    "strict": True,
                    "schema": SCHEMAS[doc.doc_type],
                }
            },
        )

        if resp.usage:
            await usage.add(resp.usage.input_tokens, resp.usage.output_tokens)

        try:
            parsed = json.loads(resp.output_text)
        except (json.JSONDecodeError, TypeError):
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
        "task": "kie_label",
        "timestamp_utc": ts,
        "doc_types": doc_types,
        "pilot": pilot,
        "num_documents": len(tasks),
        **usage.as_dict(),
        "results": doc_results,
    }
    report_path = RUNS_DIR / f"{ts}_kie.json"
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
