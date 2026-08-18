"""
labeled/train 의 JSON 을 val·test 와 같은 규격(API 응답 형태)으로 맞춘다.

지금 labeled/ 안에 두 형태가 섞여 있다.

    train      페이지 JSON 그대로   {input_path, width, parsing_res_list, ...}
    val·test   API 응답 전체       {status, previewFile, result{elements[...]}, documentId}

파이프라인이 한쪽만 가정하면 나머지가 조용히 깨진다. val·test 쪽으로 통일한다.

껍데기를 지어내지 않는다. 84건 모두 파싱 원본 응답이 남아 있으므로
(clean_v1/parse, crawl_google/parse) 그 응답을 가져와 **검수된 블록만 갈아끼운다.**
documentId·markdown·html 같은 필드가 실제 값으로 유지된다.

markdown 은 검수 결과에 맞춰 다시 만든다. 사람이 블록을 고쳤는데 파서가 처음 뽑은
markdown 을 그대로 두면 둘이 어긋나기 때문이다. 조립 규칙은 파서와 같게 맞춘다
(block_order 순 · doc_title=# · paragraph_title=## · table 은 HTML 그대로 ·
 markdown_ignore_labels 제외) — vlm 학습셋을 만드는 make_vlm_taxonomy 규칙과는
다른 규칙이라는 점에 주의.

usage:
    python3 scripts/unify_labeled_format.py --dry
    python3 scripts/unify_labeled_format.py
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path("/data/workspace/yjyong/receipt_data")
TRAIN = ROOT / "labeled/train/json"
PARSE_DIRS = [ROOT / "clean_v1/parse", ROOT / "crawl_google/parse"]

# 파서 기본값. model_settings 에 실제 값이 있으면 그쪽을 쓴다.
DEFAULT_IGNORE = ["number", "header_image", "footer", "footer_image"]
H1 = {"doc_title"}
H2 = {"paragraph_title"}


def find_parse(doc_id):
    for d in PARSE_DIRS:
        p = d / f"{doc_id}.json"
        if p.exists():
            return p
    return None


def parser_markdown(blocks, ignore):
    """파서의 조립 규칙을 따라 markdown 을 만든다."""
    out = []
    for b in sorted(blocks, key=lambda x: (x.get("block_order")
                                           if x.get("block_order") is not None
                                           else x.get("block_id", 0))):
        lab = b.get("block_label") or "text"
        c = (b.get("block_content") or "").strip()
        if not c or lab in ignore:
            continue
        if lab in H1:
            out.append("# " + c.replace("\n", " ").strip())
        elif lab in H2:
            out.append("## " + c.replace("\n", " ").strip())
        else:
            out.append(c)
    return "\n\n".join(out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--backup", default=str(ROOT / "labeled/train/json_pagejson_backup"))
    args = ap.parse_args()

    files = sorted(TRAIN.glob("*.json"))
    print(f"labeled/train {len(files)}건")

    stat = Counter()
    plan = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        if "parsing_res_list" not in d:
            stat["이미 API 규격"] += 1
            continue
        src = find_parse(f.stem)
        if src is None:
            stat["원본 응답 없음"] += 1
            continue
        env = json.loads(src.read_text(encoding="utf-8"))
        els = (env.get("result") or {}).get("elements") or []
        if not els:
            stat["원본에 elements 없음"] += 1
            continue

        # 검수된 페이지 JSON 을 통째로 끼운다. 좌표·라벨·순서가 전부 검수본이 된다.
        els[0]["json"] = d
        ignore = set((d.get("model_settings") or {}).get("markdown_ignore_labels")
                     or DEFAULT_IGNORE)
        md = parser_markdown(d.get("parsing_res_list") or [], ignore)
        els[0]["markdown"] = md
        env["result"]["md"] = md
        # 파싱 스크립트가 붙여둔 내부 메타는 규격 밖이라 뺀다
        for k in ("_source_doc_id", "_source_image", "_source_width", "_source_height"):
            env.pop(k, None)
        env.setdefault("status", 200)
        env.setdefault("previewFile", None)
        env.setdefault("previewFilename", None)
        plan.append((f, env))
        stat["변환"] += 1

    print(f"  {dict(stat)}")
    if not plan:
        print("바꿀 게 없다.")
        return

    if args.dry:
        f, env = plan[0]
        print(f"\n샘플 {f.stem} 변환 후 키: {list(env.keys())}")
        print(f"  result 키: {list(env['result'].keys())}")
        print(f"  elements[0] 키: {list(env['result']['elements'][0].keys())}")
        print(f"  블록 {len(env['result']['elements'][0]['json']['parsing_res_list'])}개")
        print(f"  markdown 앞부분: {env['result']['md'][:120]!r}")
        print("\n(--dry: 저장하지 않았다)")
        return

    bk = Path(args.backup)
    bk.mkdir(parents=True, exist_ok=True)
    for f, env in plan:
        shutil.copy2(f, bk / f.name)                 # 되돌릴 수 있게 원본 보관
        f.write_text(json.dumps(env, ensure_ascii=False), encoding="utf-8")
    print(f"\n변환 {len(plan)}건 · 원본 백업 → {bk}")


if __name__ == "__main__":
    main()
