"""패널에 쓴 원본 이미지와 출력 원문을 건별로 뽑아낸다.

패널 PNG 는 글자를 그림으로 구워 넣은 것이라 복사·검색이 안 된다. 검토하려면
크롭 원본과 모델 출력 텍스트가 따로 있어야 해서 별도로 내보낸다.

  <출력>/01_이웃줄혼입_val_receipt56_9/
      crop.png              평가에 쓴 크롭 (원본 그대로)
      page.jpg              그 크롭이 나온 영수증 페이지 전체
      gt.txt                정답 (검수 정정본)
      asis_luxia.txt        AS-IS 베이스 모델 출력
      tobe_ckpt1000.txt     TO-BE 학습 모델 출력
      meta.json             bbox·점수·라벨
      panel.png             렌더된 패널 (참고용)
  <출력>/요약.csv

usage: python3 scripts/export_panel_sources.py <패널폴더> <출력폴더>
"""
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "receipt_data"
SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

SCORED = RD / "review/asis_tobe.json"
A = "베이스(luxia)"
B = "학습 ckpt1000"


def page_of(doc_id, split):
    for ext in (".jpg", ".png", ".jpeg"):
        p = RD / f"labeled/{split}/images/{doc_id}{ext}"
        if p.exists():
            return p
    return None


def main():
    rows = {r["crop_id"]: r for r in json.loads(
        SCORED.read_text(encoding="utf-8"))["rows"]}
    panels = sorted(SRC.glob("*.png"))
    if not panels:
        raise SystemExit(f"패널 PNG 가 없다: {SRC}")

    summary = []
    for png in panels:
        parts = png.stem.split("_")
        crop_id = "_".join(parts[3:])          # panel_01_유형_<crop_id>
        rec = rows.get(crop_id)
        if not rec:
            print(f"  ✗ {png.name} — crop_id '{crop_id}' 를 못 찾음")
            continue

        d = OUT / png.stem.replace("panel_", "").replace("asis_", "")
        d.mkdir(exist_ok=True)

        crop = RD / "eval_crops" / rec["image"]
        shutil.copy2(crop, d / "crop.png")
        page = page_of(rec["doc_id"], rec["split"])
        if page:
            shutil.copy2(page, d / f"page{page.suffix}")
        shutil.copy2(png, d / "panel.png")

        (d / "gt.txt").write_text(rec["gt"], encoding="utf-8")
        (d / "asis_luxia.txt").write_text(rec[A]["pred"], encoding="utf-8")
        (d / "tobe_ckpt1000.txt").write_text(rec[B]["pred"], encoding="utf-8")

        m = "TEDS" if rec["task"] == "table" else "CER"
        sa = rec[A]["teds"] if m == "TEDS" else rec[A]["cer"]
        sb = rec[B]["teds"] if m == "TEDS" else rec[B]["cer"]
        meta = {
            "crop_id": crop_id, "doc_id": rec["doc_id"], "split": rec["split"],
            "task": rec["task"], "label": rec["label"],
            "bbox": rec["bbox"], "crop_size": rec["size"],
            "metric": m, "AS-IS": round(sa, 4), "TO-BE": round(sb, 4),
            "AS-IS_model": "luxia-document-parsing-high (사내 서빙 베이스)",
            "TO-BE_model": "Qwen3.6-35B-A3B + LoRA "
                           "(ml/runs/upload-mix-qwen36/checkpoint-1000)",
            "평가셋": "eval_crops 978건 / 90장 — 학습 미사용",
            "정답출처": "검수 정정본 (fix_gt.py 반영)",
        }
        (d / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

        summary.append({
            "폴더": d.name, "crop_id": crop_id, "task": rec["task"],
            "지표": m, "AS-IS": round(sa, 4), "TO-BE": round(sb, 4),
            "정답길이": len(rec["gt"]),
            "AS-IS길이": len(rec[A]["pred"]),
            "TO-BE길이": len(rec[B]["pred"]),
        })
        print(f"  {d.name:44} {m} {sa:.4f} → {sb:.4f}")

    with (OUT / "요약.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"\n{len(summary)}건 → {OUT}")


if __name__ == "__main__":
    main()
