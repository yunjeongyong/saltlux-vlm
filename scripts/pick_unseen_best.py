"""슬롯마다 가장 잘 나온 후보 하나만 남긴다.

표 계열은 TEDS(높을수록), 나머지는 CER(낮을수록)로 고른다. 평가 하네스와 같은
metrics_doc 를 쓰므로 보고서 지표와 같은 자로 잰 값이다.

usage: python3 scripts/pick_unseen_best.py <폴더>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_doc import cer, teds  # noqa: E402

IMG = Path(sys.argv[1])


def measure(r):
    if "table" in r["task"]:
        return "TEDS", teds(r["gt"], r["pred"]), False      # 높을수록 좋음
    return "CER", cer(r["gt"], r["pred"]), True             # 낮을수록 좋음


def main():
    recs = json.loads((IMG / "_pred.json").read_text(encoding="utf-8"))
    for r in recs:
        m, s, lower = measure(r)
        r["metric"], r["score"], r["_lower"] = m, round(s, 4), lower

    slots = {}
    for r in recs:
        slots.setdefault(r["slot"], []).append(r)

    best = []
    for slot, cands in slots.items():
        cands.sort(key=lambda r: (r["score"] if r["_lower"] else -r["score"]))
        win = cands[0]
        # 후보가 여럿이면 무엇을 제쳤는지 남긴다 — 고른 근거가 보여야 한다
        win["candidates"] = [{"file": c["file"], c["metric"]: c["score"]}
                             for c in cands]
        win["file"] = win["file"].split("_c")[0] + ".png" \
            if "_c" in win["file"] else win["file"]
        win.pop("_lower", None)
        best.append(win)
        mark = "" if len(cands) == 1 else f"  ({len(cands)}장 중)"
        print(f"  {slot:24} {win['metric']:5} {win['score']:7.4f}{mark}")

    best.sort(key=lambda r: r["file"])
    (IMG / "_pred_best.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(best)}장 선정 → {IMG/'_pred_best.json'}")


if __name__ == "__main__":
    main()
