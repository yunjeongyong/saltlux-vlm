"""
AIHub 71299(대규모 OCR 데이터, 공공) 를 팀 공용 데이터셋 규격으로 포장한다.

규격은 `dataset/train/public/pubtabnet-html` 를 따른다.

    {데이터셋명}/
      data/{id}.jpg      원본 이미지
      data/{id}.json     원본 라벨 (AIHub 배포본 그대로)
      manifest.json      info + schema + data[] 인덱스
      train.jsonl        학습셋 (messages 포맷)

두 가지 판단이 들어가 있다.

  1) 손글씨 제외. typeface 2·3·4 가 손글씨/혼합이다(크롭으로 확인). 한 장에서
     손글씨 비율이 임계값을 넘으면 그 장을 통째로 뺀다. 박스만 걸러내면
     이미지에는 손글씨가 보이는데 정답에는 없어서 모델이 '생략'을 배운다.

  2) 줄 묶기. 라벨은 어절 단위(중앙값 3자)라 그대로 이으면 정답이 부자연스럽다.
     y 가 비슷한 박스를 한 줄로 묶고 x 순으로 이어 붙여 사람이 읽는 형태로 만든다.
     id 순서가 이미 위→아래라(표본 40장 전수 확인) 줄 순서는 그대로 살아난다.

이미지는 하드링크로 건다 — 같은 파일시스템이라 용량이 두 배로 늘지 않고,
복사해 옮길 때는 실제 파일처럼 따라온다.

usage:
    python3 scripts/build_71299_dataset.py --limit 200 --dry
    python3 scripts/build_71299_dataset.py
"""
import argparse
import json
import os
import statistics as st
from collections import Counter
from pathlib import Path

SRC = Path("/data/workspace/yjyong/aihub/71299/extracted")
OUT = Path("/data/workspace/yjyong/aihub/71299/pkg/aihub-71299-ocr")
PROMPT = "이 문서에 적힌 텍스트를 순서대로 옮겨 적어줘."

# 크롭을 잘라 확인한 결과: 1=인쇄체, 2·3=손글씨, 4=인쇄 양식에 손으로 채운 혼합
PRINTED = 1


def line_group(boxes):
    """y 가 겹치는 박스를 한 줄로 묶는다. 반환: [[box, ...], ...] (위→아래)"""
    items = []
    for b in boxes:
        try:
            x1, y1 = min(b["x"]), min(b["y"])
            x2, y2 = max(b["x"]), max(b["y"])
        except (KeyError, ValueError, TypeError):
            continue
        items.append((y1, y2, x1, b))
    if not items:
        return []
    heights = [y2 - y1 for y1, y2, _, _ in items]
    tol = max(8, st.median(heights) * 0.6)

    items.sort(key=lambda t: (t[0], t[2]))
    lines, cur, cur_y = [], [], None
    for y1, y2, x1, b in items:
        if cur_y is None or abs(y1 - cur_y) <= tol:
            cur.append((x1, b))
            cur_y = y1 if cur_y is None else cur_y
        else:
            lines.append(cur)
            cur, cur_y = [(x1, b)], y1
    if cur:
        lines.append(cur)
    return [[b for _, b in sorted(ln, key=lambda t: t[0])] for ln in lines]


def looks_like_table(boxes, bin_px=15, min_col=5, min_cols=2):
    """표 서식인지 좌표만으로 판정한다.

    같은 x 시작점(±bin_px)에 세로로 여러 박스가 줄지어 서면 '열'로 본다.
    열이 둘 이상이면 표로 간주한다. 표본 600장 중 18% 가 걸리며,
    걸린 장을 눈으로 확인하니 등기 서식·조사표·시설물표 같은 실제 표였다.

    이런 장을 빼는 이유: 이 데이터의 정답은 평문이라, 표를 보고 평문을 뱉도록
    가르치게 된다. 영수증 쪽에서 '표 → 마크다운 표' 를 2,400건 가르치는데
    반대 방향 신호가 그보다 많으면 형식이 흔들린다.
    """
    if len(boxes) < 20:
        return False
    xs = []
    for b in boxes:
        try:
            xs.append(round(min(b["x"]) / bin_px))
        except (KeyError, ValueError, TypeError):
            continue
    c = Counter(xs)
    return sum(1 for _, v in c.items() if v >= min_col) >= min_cols


def page_text(boxes):
    out = []
    for ln in line_group(boxes):
        s = " ".join(str(b.get("data", "")).strip() for b in ln if str(b.get("data", "")).strip())
        if s:
            out.append(s)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hand-max", type=float, default=0.10,
                    help="손글씨 박스 비율이 이 값을 넘으면 그 장을 제외")
    ap.add_argument("--min-boxes", type=int, default=5)
    ap.add_argument("--keep-table", action="store_true",
                    help="표 서식 문서도 포함한다 (기본은 제외)")
    # manifest 의 경로는 pubtabnet-html 과 같이 절대경로로 적는다. 다만 이 컨테이너에서는
    # 목적지(/data/workspace/VLM)가 읽기 전용이라 여기서 만들고 옮겨야 하므로,
    # '옮긴 뒤의 경로'를 미리 박아 둔다. 안 그러면 복사 직후 경로가 전부 틀린다.
    ap.add_argument("--dest-root",
                    default="/data/workspace/VLM/dataset/train/public/aihub-71299-ocr",
                    help="manifest 에 기록할 최종 배치 경로")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    labels = {p.stem: p for p in SRC.glob("VL_*/*.json")}
    images = {p.stem: p for p in SRC.glob("VS_*/*.jpg")}
    ids = sorted(set(labels) & set(images))
    if args.limit:
        ids = ids[:args.limit]
    print(f"짝이 맞는 문서 {len(ids):,}건")

    out = Path(args.out)
    data_dir = out / "data"
    if not args.dry:
        data_dir.mkdir(parents=True, exist_ok=True)

    kept, rows, index = 0, [], []
    drop = Counter()
    hand_ratio, nbox, tlen = [], [], []

    for i in ids:
        lp, ip = labels[i], images[i]
        d = json.loads(lp.read_text(encoding="utf-8"))
        boxes = d.get("Bbox") or []
        if len(boxes) < args.min_boxes:
            drop["박스 부족"] += 1
            continue
        hand = sum(1 for b in boxes if b.get("typeface") != PRINTED)
        r = hand / len(boxes)
        hand_ratio.append(r)
        if r > args.hand_max:
            drop[f"손글씨 {int(args.hand_max*100)}% 초과"] += 1
            continue
        if not args.keep_table and looks_like_table(boxes):
            drop["표 서식"] += 1
            continue
        txt = page_text(boxes)
        if not txt.strip():
            drop["텍스트 없음"] += 1
            continue

        kept += 1
        nbox.append(len(boxes))
        tlen.append(len(txt))
        rel_img = f"data/{i}.jpg"
        rel_lab = f"data/{i}.json"
        if not args.dry:
            dst_i, dst_l = out / rel_img, out / rel_lab
            if not dst_i.exists():
                try:
                    os.link(ip, dst_i)          # 같은 파일시스템이면 용량 안 늘어난다
                except OSError:
                    dst_i.write_bytes(ip.read_bytes())
            if not dst_l.exists():
                dst_l.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

        rows.append({
            "doc_id": i,
            "task": "page_ocr",
            # 장별 손글씨 비율을 남겨 둔다. 분포가 0% 아니면 100% 쪽으로 갈리므로
            # 나중에 "완전 인쇄체만" 으로 조이고 싶을 때 다시 만들 필요가 없다.
            "handwriting_ratio": round(r, 4),
            "images": [f"aihub-71299-ocr/{rel_img}"],
            "messages": [
                {"role": "user", "content": f"<image>{PROMPT}"},
                {"role": "assistant", "content": txt},
            ],
        })
        # pubtabnet-html 의 data[] 는 id / image_path / annotation_path 세 키뿐이다.
        # 규격을 그대로 따르고, 장별 부가정보는 train.jsonl 쪽에 남긴다.
        index.append({
            "id": i,
            "image_path": f"{args.dest_root}/{rel_img}",
            "annotation_path": f"{args.dest_root}/{rel_lab}",
        })

    print(f"\n채택 {kept:,}건 / 제외 {sum(drop.values()):,}건  {dict(drop)}")
    if nbox:
        print(f"  박스/장 중앙값 {st.median(nbox):.0f} · 정답 길이 중앙값 {st.median(tlen):.0f}자")
        print(f"  손글씨 비율 중앙값 {100*st.median(hand_ratio):.1f}% "
              f"(제외 임계 {int(args.hand_max*100)}%)")
    if args.dry:
        print("\n(--dry: 저장하지 않았다)")
        if rows:
            r = rows[0]
            print("\n샘플 train.jsonl 1행:")
            print(json.dumps({**r, "messages": [
                r["messages"][0],
                {"role": "assistant", "content": r["messages"][1]["content"][:200] + " …"}]},
                ensure_ascii=False, indent=1))
        return

    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    manifest = {
        "info": {
            "description": "AIHub 71299 대규모 OCR 데이터(공공) — 스캔 공공문서의 어절 단위 OCR 라벨.",
            "version": "1.0",
            "source": "https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71299",
            "date_created": "2026-08-11",
            "note": (
                "AIHub Validation 분할(이미지+라벨 짝이 맞는 유일한 구간)에서 만들었다. "
                "Training 원천 385묶음(133GB)은 내려받았으나 tar 미해제 상태다. "
                f"손글씨(typeface 2·3·4) 비율이 {int(args.hand_max*100)}%를 넘는 장은 제외했다. "
                + ("" if args.keep_table else
                   "표 서식 문서도 제외했다 — 이 데이터의 정답은 평문이라 표를 보고 "
                   "평문을 뱉도록 가르치게 되고, 영수증 쪽 '표→마크다운' 학습과 충돌한다.")
            ),
            "stats": {
                "n_documents": kept,
                "n_excluded": sum(drop.values()),
                "excluded_reason": dict(drop),
                "median_boxes_per_page": st.median(nbox) if nbox else 0,
                "median_target_chars": st.median(tlen) if tlen else 0,
            },
        },
        "schema": {
            "Images": "dict — width/height/dpi/writing_style/year 등 장 속성",
            "Bbox": "list — {id, data, type, typeface, x[4], y[4]} 어절 단위 박스",
            "typeface": "1=인쇄체 / 2·3=손글씨 / 4=인쇄+손글씨 혼합 (크롭 확인으로 판별, 배포 문서에는 미기재)",
            "coordinate": "원본 이미지 픽셀 기준. x/y 는 축별 4개 나열이라 min·max 로 사각형을 만든다",
            "reading_order": "Bbox 의 id 순서가 위→아래 읽기 순서 (표본 40장 전수 확인)",
        },
        "data": index,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n저장: {out}")
    print(f"  data/        {kept*2:,}개 파일 (jpg+json)")
    print(f"  train.jsonl  {len(rows):,}행")
    print(f"  manifest.json")


if __name__ == "__main__":
    main()
