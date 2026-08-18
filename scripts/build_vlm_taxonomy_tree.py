#!/usr/bin/env python3
"""VLM 데이터셋을 Post-Training(Distill-SFT) 컨벤션의 taxonomy 트리로 재구성한다.

원본은 /data/workspace/VLM/dataset (읽기 전용). 읽기만 하고 산출은 전부
/data/workspace/yjyong/vlm_data 아래에 만든다.

폴더 축은 Post-Training 의 4단계를 VLM 어휘로 채운다.

    Post-Training : {category}/{task_details}/{dataset_name}/{think|no_think}.json
    VLM           : {domain}/{sub_domain}/{dataset_name}/{task_category}.json

domain/sub_domain/task_category 는 팀 스펙(/data/workspace/VLM/taxonomy/taxonomy/
taxonomy.yaml, schema_version 2.0.0)의 controlled vocabulary 를 그대로 쓴다.
새 값을 지어내지 않는다 — 스펙에 없는 세부값은 task.detail(자유 텍스트)로 내린다.

레코드는 두 계열을 합친다.

    Post-Training  state / id / generation_model / evaluation_model / input_tokens
    VLM taxonomy   domain / sub_domain / task{} / source{} / collection_method{} /
                   formatting_method{} / privacy{} / review_status / split / languages

usage:
    python3 scripts/build_vlm_taxonomy_tree.py --dry
    python3 scripts/build_vlm_taxonomy_tree.py
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

SRC = Path("/data/workspace/VLM/dataset")
OUT = Path("/data/workspace/yjyong/vlm_data/Distill-VLM")
MODEL = "Qwen3.6-35B-A3B"          # 01~03 은 모델 의존 단계라 모델명 아래에 둔다
DATASET_VERSION = "0.3.0"
BATCH_DATE = "20260812"
CREATED_AT = "2026-08-12T06:00:00Z"

# ---------------------------------------------------------------------------
# 패키지별 매핑. row 의 task 값에 따라 갈라지는 경우 task_map 으로 분기한다.
# split_by 가 있으면 그 필드 값으로 dataset_name 을 나눈다 (예: 사람검수 vs 합성).
# ---------------------------------------------------------------------------
PKGS = [
    {
        "path": "train/human_annotated/receipt",
        "split_by": "source",
        "dataset_names": {
            "human_annotated": "luxia-receipt",
            "synthetic": "receipts3000-syn",
        },
        "domain": "document",
        "sub_domain": "receipt",
        "task_map": {
            "receipt_markdown": ("markdown_table_generation", "영수증 전체 마크다운", "markdown"),
        },
        "reference": "internal (luxia labeling tool)",
        "per_source": {
            "human_annotated": {
                "source": {"type": "self_collected", "annotations_creator": "expert-generated",
                           "license": {"id": "internal_only"}},
                "collection_method": {"type": "semi_auto",
                                      "generator_model": "document-parse",
                                      "reviewer": "luxia labeling tool (사람 검수)"},
                "formatting_method": {"type": "template_based",
                                      "template_id": "receipt_markdown_v1"},
            },
            "synthetic": {
                "source": {"type": "synthetic", "annotations_creator": "machine-generated",
                           "license": {"id": "internal_only"}},
                "collection_method": {"type": "synthetic_generation"},
                "formatting_method": {"type": "template_based",
                                      "template_id": "receipt_markdown_v1"},
            },
        },
        "privacy": {"contains_pii": True,
                    "pii_types": ["business_registration_no", "phone_number", "address"],
                    "consent": "not_applicable", "anonymized": False},
        "languages": ["ko"],
        "review_status": "approved",
    },
    {
        "path": "train/public/aihub-71299-ocr",
        "dataset_name": "aihub-71299",
        "domain": "document",
        "sub_domain": "ocr",
        "task_map": {
            "page_ocr": ("ocr_text_recognition", "페이지 전문 평문", "free_text"),
        },
        "reference": "https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71299",
        "source": {"type": "public_dataset", "annotations_creator": "found",
                   "license": {"id": "other", "name": "AI Hub 이용약관 (국내 연구·개발 목적)",
                               "url": "https://aihub.or.kr"}},
        "collection_method": {"type": "inherited_from_source"},
        "formatting_method": {"type": "template_based", "template_id": "page_ocr_v1"},
        "privacy": {"contains_pii": False, "consent": "not_applicable", "anonymized": False},
        "languages": ["ko"],
        "review_status": "approved",
    },
    {
        "path": "train/public/kogovdoc-bench",
        "dataset_name": "kogovdoc-bench",
        "domain": "document",
        "sub_domain": "ocr",
        "task_map": {
            "doc_parsing/government": ("ocr_text_recognition", "공공문서 마크다운 파싱", "markdown"),
            "doc_parsing/paper": ("ocr_text_recognition", "논문 마크다운 파싱", "markdown"),
        },
        "reference": "https://huggingface.co/datasets/Wigtn/KoGovDoc-Bench",
        "source": {"type": "public_dataset", "annotations_creator": "found",
                   "license": {"id": "unknown",
                               "name": "HuggingFace 카드에 명시 없음 — 확인 필요"}},
        "collection_method": {"type": "inherited_from_source"},
        "formatting_method": {"type": "template_based", "template_id": "doc_parsing_v1"},
        "privacy": {"contains_pii": False, "consent": "not_applicable", "anonymized": False},
        "languages": ["ko"],
        "review_status": "in_review",
    },
    {
        "path": "train/public/multimodal-retrieval",
        "dataset_name": "multimodal-retrieval",
        "domain": "document",
        "sub_domain": "ocr",
        "task_map": {
            "parsing/전체텍스트": ("ocr_text_recognition", "전체텍스트 파싱", "free_text"),
        },
        "reference": "AI Hub 20.멀티모달 정보검색 데이터",
        "source": {"type": "public_dataset", "annotations_creator": "found",
                   "license": {"id": "other", "name": "AI Hub 이용약관",
                               "url": "https://aihub.or.kr"}},
        "collection_method": {"type": "inherited_from_source"},
        "formatting_method": {"type": "template_based", "template_id": "page_parsing_v1"},
        "privacy": {"contains_pii": False, "consent": "not_applicable", "anonymized": False},
        "languages": ["ko"],
        "review_status": "approved",
    },
    {
        "path": "train/public/pubtabnet-html",
        "dataset_name": "pubtabnet-html",
        "domain": "document",
        "sub_domain": "table",
        "task_map": {
            "html_table": ("table_extraction", "HTML 표 복원", "table"),
        },
        "reference": "https://huggingface.co/datasets/apoidea/pubtabnet-html",
        "source": {"type": "public_dataset", "annotations_creator": "found",
                   "license": {"id": "cc_by", "name": "PubTabNet (IBM) CDLA-Permissive",
                               "url": "https://github.com/ibm-aur-nlp/PubTabNet"}},
        "collection_method": {"type": "inherited_from_source"},
        "formatting_method": {"type": "template_based", "template_id": "html_table_v1"},
        "privacy": {"contains_pii": False, "consent": "not_applicable", "anonymized": False},
        "languages": ["en"],
        "review_status": "approved",
    },
]


def rec_id(dataset_name, doc_id, task):
    h = hashlib.sha1(f"{dataset_name}|{doc_id}|{task}".encode()).hexdigest()[:8]
    return f"{dataset_name}#{h}"


def build_record(row, pkg, dataset_name, cat, detail, out_fmt, seq):
    """원본 row 를 taxonomy 레코드로 감싼다. messages/images 는 그대로 보존한다."""
    meta = pkg.get("per_source", {}).get(row.get("source"), pkg)
    doc_id = row.get("doc_id", f"row{seq}")
    rec = {
        "state": "taxonomy",
        "id": rec_id(dataset_name, doc_id, row.get("task", "")),
        "doc_id": doc_id,
        "dataset_name": dataset_name,
        "reference": pkg["reference"],
        "dataset_version": DATASET_VERSION,
        "batch_id": pkg["batch_id"],
        "domain": pkg["domain"],
        "sub_domain": pkg["sub_domain"],
        "task": {"category": cat, "detail": detail, "output_format": out_fmt},
        "languages": pkg["languages"],
        "source": meta["source"],
        "collection_method": meta["collection_method"],
        "formatting_method": meta["formatting_method"],
        "privacy": pkg["privacy"],
        "created_at": CREATED_AT,
        "review_status": pkg["review_status"],
        "split": "train",
        "num_images": len(row.get("images") or []),
        "num_turns": sum(1 for m in row.get("messages", []) if m.get("role") == "user"),
        "input_tokens": {},
        "generation_model": {"for_completion": None},
        "evaluation_model": {"for_prompt": None, "for_completion": None},
        "images": row.get("images"),
        "messages": row.get("messages"),
    }
    # 패키지 고유 필드는 버리지 않고 extra 로 남긴다 (스키마가 열린 구조라 허용)
    extra = {k: v for k, v in row.items()
             if k not in ("doc_id", "task", "images", "messages", "source")}
    if extra:
        rec["extra"] = extra
    return rec


class Writer:
    """JSON 배열을 스트리밍으로 쓴다 (500k 행을 메모리에 올리지 않기 위해)."""

    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("w", encoding="utf-8")
        self.f.write("[\n")
        self.n = 0

    def write(self, rec):
        if self.n:
            self.f.write(",\n")
        self.f.write(json.dumps(rec, ensure_ascii=False))
        self.n += 1

    def close(self):
        self.f.write("\n]\n")
        self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, help="패키지당 최대 행수 (테스트용)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    tax_root = out / MODEL / "01_taxonomy"

    # 빈 단계 폴더도 미리 만들어 둔다 — 구조 자체가 문서 역할을 한다
    skeleton = [".new", ".seed", ".dump", ".logs", ".tmp", ".src",
                f"{MODEL}/02_preprocessed", f"{MODEL}/03_train",
                f"{MODEL}/config", f"{MODEL}/src", f"{MODEL}/logs", "registry"]

    for i, pkg in enumerate(PKGS, 1):
        pkg["batch_id"] = f"{BATCH_DATE}-{i:02d}"

    registry, stats = [], []
    for pkg in PKGS:
        jl = SRC / pkg["path"] / "train.jsonl"
        writers, counts = {}, {}
        n = 0
        for line in jl.open(encoding="utf-8"):
            if not line.strip():
                continue
            if a.limit and n >= a.limit:
                break
            row = json.loads(line)
            task = row.get("task")
            if task not in pkg["task_map"]:
                raise SystemExit(f"매핑 없는 task: {task} ({pkg['path']})")
            cat, detail, out_fmt = pkg["task_map"][task]

            if "split_by" in pkg:
                dsname = pkg["dataset_names"][row[pkg["split_by"]]]
            else:
                dsname = pkg["dataset_name"]

            key = (dsname, cat)
            counts[key] = counts.get(key, 0) + 1
            if not a.dry:
                if key not in writers:
                    p = tax_root / pkg["domain"] / pkg["sub_domain"] / dsname / f"{cat}.json"
                    writers[key] = Writer(p)
                writers[key].write(build_record(row, pkg, dsname, cat, detail, out_fmt, n))
            n += 1

        for w in writers.values():
            w.close()

        for (dsname, cat), c in sorted(counts.items()):
            rel = f"{pkg['domain']}/{pkg['sub_domain']}/{dsname}/{cat}.json"
            stats.append((rel, c))
            meta = pkg.get("per_source", {})
            m = next(iter(meta.values())) if meta and dsname != pkg.get("dataset_name") else pkg
            if "split_by" in pkg:
                inv = {v: k for k, v in pkg["dataset_names"].items()}
                m = pkg["per_source"][inv[dsname]]
            registry.append({
                "batch_id": pkg["batch_id"], "dataset_version": DATASET_VERSION,
                "doc_id_prefix": None, "dataset_name": dsname,
                "domain": pkg["domain"], "sub_domain": pkg["sub_domain"],
                "task_category": cat,
                "source_type": m["source"]["type"],
                "license_id": m["source"]["license"]["id"],
                "collection_method": m["collection_method"]["type"],
                "formatting_method": m["formatting_method"]["type"],
                "contains_pii": pkg["privacy"]["contains_pii"],
                "sample_count": c, "created_at": CREATED_AT,
                "review_status": pkg["review_status"], "split": "train",
                "path": f"{MODEL}/01_taxonomy/{rel}",
            })

    print(f"{'경로':70s} 행수")
    for rel, c in stats:
        print(f"  {rel:68s} {c:>9,}")
    print(f"  {'합계':68s} {sum(c for _, c in stats):>9,}")

    if a.dry:
        print("\n(--dry: 쓰지 않았습니다)")
        return

    for d in skeleton:
        (out / d).mkdir(parents=True, exist_ok=True)
    reg = out / "registry" / "dataset_registry.jsonl"
    with reg.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_comment": "배치 단위 인덱스. 1줄 = 1개 "
                            "{dataset_name, task_category} 묶음."}, ensure_ascii=False) + "\n")
        for r in registry:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nregistry → {reg} ({len(registry)}줄)")
    print(f"트리     → {tax_root}")


if __name__ == "__main__":
    main()
