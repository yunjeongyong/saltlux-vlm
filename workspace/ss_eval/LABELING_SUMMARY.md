# Labeling Project Summary

Document layout-parsing + key-information-extraction (KIE) labeling pipeline for 4 document types: **bill** (Bill of Lading), **invoice** (Marine Cargo Insurance Policy / Inspection Certificate / Certificate of Quality — mixed), **quotation** (purchase orders / estimates / quotations, mixed), **receipt** (store receipts, mostly Korean). Source data: `./data/<type>/{images,jsons}`, 100 docs/type except receipt (99 images had labels; a missing `receipt60` image was later added and labeled, bringing receipt to 100/100).

## Pipeline stages

1. **Layout verification** (`scripts/verify_labels.py`) — GPT-5.6 vision, one batched call per page: sends the full image + every current `parsing_res_list` block (label/bbox/content), asks the model to `keep`/`edit`/`remove`/`add` blocks against a fixed label enum. Corrections are reconciled back deterministically (`reconcile_blocks`), block IDs/order renumbered top-to-bottom, `elements[].markdown/html` and `result.md/html` regenerated to stay consistent, `layout_det_res` (raw detector boxes) left untouched. Table corrections are restricted to TEDS-relevant structure/content (rows/cols/spans/text) — cell styling is ignored by design.
   - Scope: `bill`, `invoice`, `quotation` only (receipt excluded).
   - Original full-run output (`./output/`) and its cost report are **no longer present in the repo** (superseded by a later `parsing_dataset/` re-run via `document_parsing.py`/`document_parsing_alt.py` against a different parse API backend — see below). At the time it ran: 300/300 docs, $8.85 (gpt-5.6, ~1.88M input / 650K output tokens).

2. **Alternate layout parsing** (`scripts/document_parsing.py` vs `document_parsing_alt.py`) — a separate A/B track hitting an external layout-parse API two ways:
   - `document_parsing.py`: synchronous single-shot call → `./test_output/parsing/<type>/`.
   - `document_parsing_alt.py`: async job API, `modelQuality: high` → `./test_output_alt/parsing/<type>/`.
   - `parsing_dataset/{bill,invoice,quotation}/jsons/` (300 files, no receipt) is a fresh re-parse of the same source images through this newer API (same `documentId`s as `./data`, but `block_order` now populated) — not a copy, not a GPT correction pass.

3. **Key-information extraction** — two parallel routes:
   - `scripts/kie_label.py`: GPT-5.6 **vision** KIE, per-type `{General, Table}` JSON schema (strict structured outputs), image + best-available context text (verified label → raw label fallback). Full run: 399/400 docs, $4.47 (this run) + ~$0.02 pilot ≈ **$4.6 total** (gpt-5.6, ~1.94M input / 205K output tokens), plus a follow-up $0.006 run for the added `receipt60`. Output was written to `./kie_output/`, **which no longer exists in the repo** — superseded by the `test_output`/`test_output_alt` A/B track below.
   - `scripts/kie_from_parsing.py` / `kie_from_parsing_alt.py`: **text-only** KIE (no image), reusing `kie_label.py`'s prompts/schemas but feeding the parsed markdown into a self-hosted OpenAI-compatible endpoint (`luxia3.5-120b-sft-v1.4`) instead of GPT vision. Base reads `test_output/parsing/` → writes `test_output/kie/`; alt reads `test_output_alt/parsing/` → writes `test_output_alt/kie/`. 400 docs/side.
   - Token totals recovered from surviving run reports (luxia model, **not cost-tracked** — no pricing entry for this self-hosted model, `estimated_cost_usd: null` in every report): ~1.51M input / ~2.22M output tokens across 6 `kie_from_parsing_alt` runs.

4. **Ground truth** — `kie_dataset/{bill,invoice,quotation,receipt}/*.json` (400 files), schema-matched to `kie_label.py`'s per-type schemas. No script currently writes to this directory — it functions as a frozen/curated evaluation ground truth (likely GPT-generated then reviewed via `scripts/label_validator.py`, a Gradio review UI — **note: that UI's hardcoded `KIE_DIR` still points at the deleted `./kie_output/`, so it needs a path fix before it can be used again**).

5. **Evaluation** (`scripts/eval_kie.py`) — field-level scoring of a predictions dir against `kie_dataset/`: numeric fields compared with tolerance, string fields via normalized Levenshtein similarity, table rows matched greedily/order-independently. Outputs `general_accuracy`, `table_accuracy`, `total_accuracy`, and row-count-mismatch counts.

## Latest eval results (`test_output` = base parse, `test_output_alt` = alt/high-quality parse; both 400/400 docs scored)

| pred_dir | general_acc | table_acc | total_acc | row-count mismatches |
|---|---|---|---|---|
| base (`test_output/kie`) | 0.8024 | 0.9003 | 0.8476 | 61 |
| alt (`test_output_alt/kie`) | 0.8159 | 0.9117 | 0.8601 | 55 |

By doc type (total_acc / row-count mismatches):

| type | base | alt |
|---|---|---|
| bill | 0.8747 / 2 | 0.8658 / 0 |
| invoice | 0.7803 / 28 | 0.8006 / 29 |
| quotation | 0.8392 / 23 | 0.8699 / 17 |
| receipt | 0.8975 / 8 | 0.9023 / 9 |

**Alt parsing wins overall** (+1.3pp total accuracy, better on invoice/quotation/receipt, essentially a wash on bill) and produces fewer table row-count mismatches on 3 of 4 types.

## Known gaps / caveats

- **Cost history is incomplete.** The original GPT-5.6 `verify_labels.py` and `kie_label.py` full-run cost reports (`./runs/*.json`) and their output directories (`./output/`, `./kie_output/`) are no longer present in the repo — likely cleaned up after the pipeline moved to the `parsing_dataset` / `test_output*` A/B structure. Figures above for those stages are reconstructed from earlier conversation logs, not re-verifiable from current repo state.
- **Receipt was excluded from `verify_labels.py` and `document_parsing*.py` scope** (`parsing_dataset/` has no receipt subfolder); its KIE ground truth/predictions rely on the raw (unverified) `./data/receipt/jsons` labels.