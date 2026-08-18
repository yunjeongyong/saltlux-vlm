#!/usr/bin/env python3
"""라벨러가 어느 파일을 먼저 고쳐야 하는지 우선순위 목록을 만든다.

텍스트 오류는 삭제로 해결되지 않고 사람이 교정해야 한다. 다만 전량을 훑는 건
낭비이므로, 파서가 실패한 정도를 점수로 매겨 심한 것부터 배치한다.

점수 (높을수록 급함)
  표 빈칸 비율      셀이 비어 있을수록 품목·금액을 못 읽은 것
  금액 패턴 없음    영수증인데 숫자·금액이 안 잡힘
  블록당 글자 수    지나치게 짧으면 인식 실패

출력
  worklist.csv    파일ID, 점수, 사유, 툴에서 열 링크
  worklist.html   같은 내용 + 이미지 미리보기

사용:
  python3 make_worklist.py --file <택소노미> --images <폴더> --out receipt_data/worklist
"""
import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path

TOOL = "http://172.16.100.242:15074"


def empty_ratio(ans):
    cells = re.findall(r"\|([^|\n]*)", ans)
    if len(cells) < 6:
        return 0.0, 0
    empty = sum(1 for c in cells if not c.strip() or set(c.strip()) <= {"-"})
    return empty / len(cells), len(cells)


def score(ans):
    """0~100. 클수록 교정이 급하다."""
    reasons, s = [], 0.0
    er, ncell = empty_ratio(ans)
    if ncell:
        s += er * 60
        if er > 0.5:
            reasons.append(f"표 빈칸 {er*100:.0f}%")

    # 영수증이면 금액이 있어야 한다
    money = len(re.findall(r"\d{1,3}(?:,\d{3})+|\d+원", ans))
    if money == 0:
        s += 25
        reasons.append("금액 패턴 없음")
    elif money < 3:
        s += 10
        reasons.append(f"금액 {money}개뿐")

    # 날짜가 없으면 거래 정보가 빠진 것
    if not re.search(r"\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{2}[-./]\d{1,2}[-./]\d{1,2}", ans):
        s += 8
        reasons.append("날짜 없음")

    if len(ans) < 200:
        s += 12
        reasons.append(f"정답 {len(ans)}자")

    return min(100.0, s), reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True, help="확장자 없는 출력 경로 앞부분")
    ap.add_argument("--min-score", type=float, default=20.0)
    ap.add_argument("--html-top", type=int, default=120, help="HTML 에 담을 상위 건수")
    a = ap.parse_args()

    rows = json.loads(Path(a.file).read_text(encoding="utf-8"))
    scored = []
    for r in rows:
        ans = r["completion"]["chosen"]
        s, why = score(ans)
        fid = Path(r["prompt"]["image"][0]).stem
        scored.append((s, fid, r["prompt"]["image"][0], why, len(ans)))
    scored.sort(reverse=True)

    todo = [x for x in scored if x[0] >= a.min_score]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["순위", "파일ID", "점수", "사유", "정답길이", "툴 링크"])
        for i, (s, fid, img, why, n) in enumerate(todo, 1):
            w.writerow([i, fid, f"{s:.0f}", " / ".join(why), n, f"{TOOL}/?file={fid}"])

    # 구간별 요약
    band = {"80 이상": 0, "60~79": 0, "40~59": 0, "20~39": 0}
    for s, *_ in scored:
        if s >= 80:
            band["80 이상"] += 1
        elif s >= 60:
            band["60~79"] += 1
        elif s >= 40:
            band["40~59"] += 1
        elif s >= 20:
            band["20~39"] += 1

    print(f"전체 {len(rows)}건 중 교정 필요 {len(todo)}건 (점수 {a.min_score:.0f} 이상)\n")
    print("점수 구간별:")
    for k, v in band.items():
        print(f"   {k:8s} {v:4d}건")
    print(f"   20 미만  {len(rows)-len(todo):4d}건  (교정 불필요)")
    print(f"\nCSV → {csv_path}")

    imgdir = Path(a.images)
    try:
        import base64
        import io

        from PIL import Image
    except ImportError:
        print("(PIL 없음: HTML 생략)")
        return 0

    parts = []
    for i, (s, fid, img, why, n) in enumerate(todo[:a.html_top], 1):
        p = imgdir / img
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        if im.width > 300:
            im = im.resize((300, max(1, int(im.height * 300 / im.width))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=60)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        parts.append(f"""<div class=i><div class=r>{i}</div>
<img src="{uri}"><div><b>{html.escape(fid)}</b> · 점수 {s:.0f} · 정답 {n}자
<div class=w>{html.escape(' / '.join(why))}</div></div></div>""")

    doc = f"""<title>라벨링 우선순위 작업 목록</title>
<style>
:root{{--bg:#fff;--fg:#18181b;--mut:#71717a;--line:#e4e4e7;--card:#fafafa;--hot:#b91c1c}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{
 --bg:#0b0b0e;--fg:#e4e4e7;--mut:#a1a1aa;--line:#27272a;--card:#141418;--hot:#fca5a5}}}}
body{{margin:0;padding:24px;background:var(--bg);color:var(--fg);
 font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
.i{{display:grid;grid-template-columns:44px 300px 1fr;gap:16px;align-items:start;
 padding:12px;border:1px solid var(--line);border-radius:8px;margin-bottom:10px;background:var(--card)}}
.i img{{width:100%;border-radius:4px;border:1px solid var(--line)}}
.r{{font-size:20px;font-weight:700;color:var(--mut)}}
.w{{color:var(--hot);font-size:13px;margin-top:4px}}
h1{{font-size:19px;margin:0 0 4px}} .s{{color:var(--mut);font-size:13px;margin-bottom:20px}}
</style>
<h1>라벨링 우선순위 작업 목록</h1>
<div class="s">전체 {len(rows)}건 중 교정 필요 {len(todo)}건 · 상위 {len(parts)}건 표시 ·
 점수가 높을수록 파서 실패가 심한 것 · 전체 목록은 worklist.csv</div>
{''.join(parts)}"""
    html_path = out.with_suffix(".html")
    html_path.write_text(doc, encoding="utf-8")
    print(f"HTML → {html_path}  ({html_path.stat().st_size/1e6:.1f}MB, 상위 {len(parts)}건)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
