"""
저장된 크롭 실물이 스케일 보정을 거쳐 잘렸는지 검증한다.

코드를 읽어서 "보정 코드가 있다"고 말하는 것과, 실제로 나온 PNG 가 보정된
좌표로 잘렸다고 확인하는 것은 다르다. 여기서는 후자를 한다.

방법 — 크롭 1건마다 두 가지 예상 크기를 계산한다.
    보정함   bbox × (W/pw, H/ph) + 여백  -> 예상 크기 A
    보정안함 bbox 를 원본에 그대로       -> 예상 크기 B
실제 저장된 PNG 크기가 A 와 맞고 B 와 다르면, 보정된 좌표로 잘린 것이다.

덤으로 "보정 안 했다면 얼마나 망가졌을까"를 같이 잰다.
    - 이미지 밖으로 나가는 건수 (좌표가 원본보다 큰 경우)
    - 정답 영역과 실제로 겹치는 비율 (IoU)

usage:
    python3 scripts/verify_crop_scaling.py
    python3 scripts/verify_crop_scaling.py --samples 6   # 비교 이미지도 저장
"""
import argparse
import json
from pathlib import Path

from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_eval_crops import is_transposed          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CROPS = ROOT / "receipt_data/eval_crops"
SRC = ROOT / "receipt_data/labeled"

PAD_RATIO, MIN_PAD = 0.02, 4      # build_eval_crops.py 기본값과 같아야 한다


def page_size(doc_id, split):
    """페이지 JSON 이 적어둔 크기와 실제 이미지 크기를 같이 낸다."""
    jf = SRC / split / "json" / f"{doc_id}.json"
    d = json.loads(jf.read_text(encoding="utf-8"))
    els = (d.get("result") or {}).get("elements") or []
    pj = els[0].get("json") if els else None
    cand = list((SRC / split / "images").glob(f"{doc_id}.*"))
    im = Image.open(cand[0])
    W, H = im.size
    pw, ph = pj["width"], pj["height"]
    # 좌표계가 90도 돌아간 장은 회전 후 크기로 봐야 한다 (build_eval_crops 와 동일).
    rot = is_transposed(W, H, pw, ph)
    if rot:
        W, H = H, W
    return (W, H), (pw, ph), cand[0], rot


def box_of(bbox, W, H, sx=1.0, sy=1.0):
    """build_eval_crops.crop() 과 같은 계산. 여백 주고 이미지 안으로 자른다."""
    x1, y1, x2, y2 = bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy
    px = max(MIN_PAD, (x2 - x1) * PAD_RATIO)
    py = max(MIN_PAD, (y2 - y1) * PAD_RATIO)
    return (max(0, int(x1 - px)), max(0, int(y1 - py)),
            min(W, int(x2 + px)), min(H, int(y2 + py)))


def size_of(box):
    return (box[2] - box[0], box[3] - box[1])


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=str(CROPS),
                    help="검사할 크롭 디렉터리")
    ap.add_argument("--samples", type=int, default=0,
                    help="보정/미보정 비교 이미지를 이 개수만큼 저장")
    ap.add_argument("--out", default=str(ROOT / "receipt_data/review/crop_scale_check"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in (Path(args.crops) / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]

    cache, ok_scaled, ok_raw, mismatch = {}, 0, 0, []
    oob, ious, shrunk = 0, [], 0
    for r in rows:
        key = (r["doc_id"], r["split"])
        if key not in cache:
            cache[key] = page_size(*key)
        (W, H), (pw, ph), _, _rot = cache[key]
        sx, sy = W / pw, H / ph

        box_s = box_of(r["bbox"], W, H, sx, sy)     # 보정함
        box_r = box_of(r["bbox"], W, H)             # 보정 안 함
        actual = tuple(r["size"])                   # 실제 저장된 PNG 크기

        if size_of(box_s) == actual:
            ok_scaled += 1
        elif size_of(box_r) == actual:
            ok_raw += 1
        else:
            mismatch.append((r["crop_id"], actual, size_of(box_s), size_of(box_r)))

        # 보정 안 했다면 얼마나 망가졌을까
        if r["bbox"][2] > W or r["bbox"][3] > H:
            oob += 1
        ious.append(iou(box_s, box_r))
        if size_of(box_r)[0] < 2 or size_of(box_r)[1] < 2:
            shrunk += 1

    n = len(rows)
    print(f"크롭 {n:,}건 · 영수증 {len({r['doc_id'] for r in rows})}장\n")
    print("=== 실제 저장된 PNG 크기가 어느 계산과 맞나 ===")
    print(f"  보정한 좌표와 일치      {ok_scaled:>6,}건  ({ok_scaled/n*100:5.1f}%)")
    print(f"  보정 안 한 좌표와 일치  {ok_raw:>6,}건  ({ok_raw/n*100:5.1f}%)")
    print(f"  둘 다 아님              {len(mismatch):>6,}건  ({len(mismatch)/n*100:5.1f}%)")
    for m in mismatch[:5]:
        print(f"    {m[0]}: 실제 {m[1]} / 보정 {m[2]} / 미보정 {m[3]}")

    print("\n=== 보정을 안 했다면 (피해 규모) ===")
    print(f"  좌표가 이미지 밖으로 나감      {oob:>6,}건  ({oob/n*100:5.1f}%)")
    print(f"  잘림 후 2px 미만으로 소멸      {shrunk:>6,}건  ({shrunk/n*100:5.1f}%)")
    print(f"  정답 영역과 겹침(IoU) 평균     {sum(ious)/n:.3f}")
    print(f"  IoU 0 (전혀 안 겹침)           {sum(1 for v in ious if v == 0):>6,}건  "
          f"({sum(1 for v in ious if v == 0)/n*100:5.1f}%)")

    if args.samples:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        saved = 0
        for r in rows[::max(1, n // (args.samples * 2))]:
            if saved >= args.samples:
                break
            (W, H), (pw, ph), path, rot = cache[(r["doc_id"], r["split"])]
            im = Image.open(path).convert("RGB")
            if rot:
                im = im.transpose(Image.ROTATE_270)
            a = im.crop(box_of(r["bbox"], W, H, W / pw, H / ph))
            # 미보정 박스는 뒤집히거나 0px 로 소멸하는 장이 많다. 그 자체가 증거라
            # 잘라내지 못하면 회색 판으로 대신 표시한다.
            bx = box_of(r["bbox"], W, H)
            if bx[2] - bx[0] >= 2 and bx[3] - bx[1] >= 2:
                b = im.crop(bx)
            else:
                b = Image.new("RGB", (max(a.width, 40), max(a.height, 20)), "gray")
            cv = Image.new("RGB", (max(a.width, b.width),
                                   a.height + b.height + 8), "white")
            cv.paste(a, (0, 0))
            cv.paste(b, (0, a.height + 8))
            cv.save(out / f"{r['crop_id']}_위보정_아래미보정.png")
            saved += 1
        print(f"\n비교 이미지 {saved}건 저장: {out}")

    verdict = ("보정된 좌표로 잘렸다 — 크롭 정합함"
               if ok_scaled == n else "확인 필요")
    print(f"\n판정: {verdict}")


if __name__ == "__main__":
    main()
