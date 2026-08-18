#!/usr/bin/env python3
"""원본 이미지와 생성된 정답을 나란히 놓은 HTML 을 만든다.

데이터셋이 정상인지는 결국 사람이 이미지를 보고 정답과 대조해야 안다.
validate_vlm_taxonomy.py 가 잡아낸 의심 건을 우선 보여주고, 정상 건도 섞는다.

사용:
  python3 preview_vlm_taxonomy.py --file ... --images ... --out preview.html
"""
import argparse
import base64
import collections
import html
import io
import json
import random
import re
from pathlib import Path

from PIL import Image

random.seed(42)
MAXW = 520          # 임베드할 이미지 최대 폭
QUALITY = 72
MAX_ANS_CHARS = 6000   # 정답이 길면 HTML 이 비대해져 잘라서 보여준다


def flags_for(ans):
    """validate 와 같은 규칙. 의심 사유를 붙인다."""
    f = []
    n = len(ans)
    if n < 30:
        f.append("정답 너무 짧음")
    if n > 8000:
        f.append("정답 너무 긺")
    if "�" in ans:
        f.append("대체문자")
    if "<td>" in ans or "<table>" in ans:
        f.append("HTML 표 변환 실패")
    lines = [l.strip() for l in ans.split("\n") if l.strip()]
    if lines:
        top, c = collections.Counter(lines).most_common(1)[0]
        if c >= 4 and len(top) > 5:
            f.append(f"같은 줄 {c}회 반복")
    cells = re.findall(r"\|([^|\n]*)", ans)
    if len(cells) >= 12:
        empty = sum(1 for c in cells if not c.strip() or set(c.strip()) <= {"-"})
        if empty / len(cells) > 0.6:
            f.append(f"표 빈칸 {empty*100//len(cells)}%")
    return f


# 파일명은 영문으로. 한글 파일명은 셸에서 다루기 번거롭다.
SLUG = {
    "표 빈칸 60% 이상": "empty_table_cells",
    "정답 너무 긺": "answer_too_long",
    "같은 줄 반복": "repeated_lines",
    "정답 너무 짧음": "answer_too_short",
    "대체문자": "replacement_char",
    "HTML 표 변환 실패": "html_table_failed",
}


def norm_reason(s):
    """'표 빈칸 63%', '같은 줄 7회 반복' 처럼 숫자가 섞인 사유를 하나로 묶는다."""
    if s.startswith("표 빈칸"):
        return "표 빈칸 60% 이상"
    if s.startswith("같은 줄"):
        return "같은 줄 반복"
    return s


def embed(path):
    orig = Image.open(path).size
    im = Image.open(path).convert("RGB")
    if im.width > MAXW:
        im = im.resize((MAXW, int(im.height * MAXW / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}", orig


def md_to_html(md):
    """표와 제목만 최소로 렌더링. 나머지는 문단."""
    out, i, lines = [], 0, md.split("\n")
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|") and ln.rstrip().endswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= {"-", ":", " "} and c for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                head, *body = rows
                t = ["<table><thead><tr>"]
                t += [f"<th>{html.escape(c)}</th>" for c in head]
                t.append("</tr></thead><tbody>")
                for r in body:
                    t.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>")
                t.append("</tbody></table>")
                out.append("".join(t))
            continue
        if ln.startswith("## "):
            out.append(f"<h4>{html.escape(ln[3:])}</h4>")
        elif ln.startswith("# "):
            out.append(f"<h3>{html.escape(ln[2:])}</h3>")
        elif ln.strip():
            out.append(f"<p>{html.escape(ln)}</p>")
        i += 1
    return "\n".join(out)


CSS = """
:root{--bg:#fff;--fg:#18181b;--mut:#71717a;--line:#e4e4e7;--card:#fafafa;
      --warn:#b45309;--warnbg:#fef3c7;--ok:#15803d}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
      --bg:#0b0b0e;--fg:#e4e4e7;--mut:#a1a1aa;--line:#27272a;--card:#141418;
      --warn:#fbbf24;--warnbg:#3f2d0a;--ok:#4ade80}}
*{box-sizing:border-box}
body{margin:0;padding:24px;background:var(--bg);color:var(--fg);
     font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--mut);font-size:13px;margin-bottom:24px}
.item{display:grid;grid-template-columns:minmax(0,${IMGW}px) minmax(0,1fr);gap:20px;
      padding:20px;margin-bottom:16px;border:1px solid var(--line);
      border-radius:10px;background:var(--card)}
@media(max-width:820px){.item{grid-template-columns:1fr}}
.item img{width:100%;height:auto;border-radius:6px;border:1px solid var(--line)}
.meta{font-size:12px;color:var(--mut);margin-top:8px;word-break:break-all}
.ans{min-width:0;overflow-x:auto}
.ans table{border-collapse:collapse;font-size:13px;margin:8px 0;width:auto}
.ans th,.ans td{border:1px solid var(--line);padding:4px 8px;text-align:left}
.ans th{background:var(--bg);font-weight:600}
.ans h3{font-size:16px;margin:4px 0 8px}
.ans h4{font-size:14px;margin:8px 0 4px;color:var(--mut)}
.ans p{margin:2px 0}
.flag{display:inline-block;background:var(--warnbg);color:var(--warn);
      font-size:11px;padding:2px 8px;border-radius:99px;margin:0 4px 4px 0}
.clean{color:var(--ok);font-size:11px}
.name{font-weight:600;font-size:13px;margin-bottom:6px}
"""


def render(items, title, subtitle, out):
    parts = []
    for r, f, imgdir in items:
        name = r["prompt"]["image"][0]
        p = imgdir / name
        if not p.exists():
            continue
        uri, (ow, oh) = embed(p)
        badges = ("".join(f'<span class="flag">{html.escape(x)}</span>' for x in f)
                  if f else '<span class="clean">✓ 자동검사 통과</span>')
        ans = r["completion"]["chosen"]
        shown = ans if len(ans) <= MAX_ANS_CHARS else ans[:MAX_ANS_CHARS]
        cut = ("" if len(ans) <= MAX_ANS_CHARS
               else f'<p class="meta">… 이하 {len(ans)-MAX_ANS_CHARS}자 생략</p>')
        parts.append(f"""<div class="item">
<div><img src="{uri}" alt="{html.escape(name)}">
<div class="meta">{html.escape(name)} · 원본 {ow}×{oh} · 정답 {len(ans)}자
 · {html.escape(r['feature']['completion']['chosen']['language'])}</div></div>
<div class="ans"><div class="name">{badges}</div>{md_to_html(shown)}{cut}</div></div>""")

    doc = f"""<title>{html.escape(title)}</title>
<style>{CSS.replace('${IMGW}', str(MAXW))}</style>
<h1>{html.escape(title)}</h1>
<div class="sub">{subtitle}</div>
{''.join(parts)}"""
    Path(out).write_text(doc, encoding="utf-8")
    return len(parts), Path(out).stat().st_size / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True, help="단일 파일 경로, 또는 --split 시 출력 폴더")
    ap.add_argument("--n-flagged", type=int, default=16)
    ap.add_argument("--n-clean", type=int, default=12)
    ap.add_argument("--all", action="store_true", help="의심 건 전부 포함")
    ap.add_argument("--split", action="store_true", help="사유별로 파일을 나눠 생성")
    ap.add_argument("--chunk", type=int, default=50, help="한 파일에 담을 최대 건수")
    ap.add_argument("--maxw", type=int, help="이미지 폭 (기본 520)")
    ap.add_argument("--quality", type=int, help="JPEG 품질 (기본 72)")
    a = ap.parse_args()

    global MAXW, QUALITY
    if a.maxw:
        MAXW = a.maxw
    if a.quality:
        QUALITY = a.quality

    rows = json.loads(Path(a.file).read_text(encoding="utf-8"))
    imgdir = Path(a.images)

    flagged, clean = [], []
    for r in rows:
        f = flags_for(r["completion"]["chosen"])
        (flagged if f else clean).append((r, f, imgdir))

    if a.split:
        outdir = Path(a.out)
        outdir.mkdir(parents=True, exist_ok=True)
        by_reason = collections.defaultdict(list)
        for r, f, d in flagged:
            by_reason[norm_reason(f[0])].append((r, f, d))   # 첫 사유 기준으로 묶는다
        slug = lambda s: SLUG.get(s) or re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_") or "other"
        n_ok = len(clean)
        print(f"의심 {len(flagged)}건을 사유 {len(by_reason)}종으로 분리\n")
        for reason, items in sorted(by_reason.items(), key=lambda x: -len(x[1])):
            base = slug(reason)
            chunks = [items[i:i + a.chunk] for i in range(0, len(items), a.chunk)]
            for ci, part in enumerate(chunks, 1):
                fp = (outdir / f"{base}.html" if len(chunks) == 1
                      else outdir / f"{base}_{ci:02d}.html")
                sub = (f"{len(part)}건 (전체 {len(items)}건 중 {ci}/{len(chunks)}) · "
                       "왼쪽 원본 이미지, 오른쪽 생성된 정답")
                n, mb = render(part, f"의심: {reason}", sub, fp)
                print(f"   {reason:18s} {n:4d}건  {mb:5.1f}MB  → {fp.name}")
        # 정상 표본도 비교용으로 하나 뽑는다
        if clean:
            sample = random.sample(clean, min(a.n_clean, len(clean)))
            fp = outdir / "clean_sample.html"
            n, mb = render(sample, "자동검사 통과 표본",
                           f"{n_ok}건 중 {len(sample)}건 · 비교용", fp)
            print(f"   {'정상 표본':18s} {n:4d}건  {mb:5.1f}MB  → {fp.name}")
        return

    if a.all:
        picked = flagged
    else:
        by_reason = collections.defaultdict(list)
        for r, f, d in flagged:
            by_reason[f[0]].append((r, f, d))
        picked, seen = [], set()
        while len(picked) < a.n_flagged:
            added = False
            for reason in sorted(by_reason):
                pool = [x for x in by_reason[reason] if id(x[0]) not in seen]
                if pool and len(picked) < a.n_flagged:
                    x = random.choice(pool)
                    picked.append(x); seen.add(id(x[0])); added = True
            if not added:
                break
        picked += random.sample(clean, min(a.n_clean, len(clean)))

    n, mb = render(picked, "영수증 데이터셋 육안 검수",
                   f"전체 {len(rows)}건 중 {len(picked)}건 · "
                   f"자동검사 의심 {len(flagged)}건 / 통과 {len(clean)}건 · "
                   "왼쪽 원본 이미지, 오른쪽 생성된 정답", a.out)
    print(f"{n}건 → {a.out}  ({mb:.1f}MB)")
    print(f"  의심 {len(flagged)}건 / 통과 {len(clean)}건")


if __name__ == "__main__":
    main()
