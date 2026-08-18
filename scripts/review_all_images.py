#!/usr/bin/env python3
"""전체 데이터셋을 눈으로 훑으며 제거 대상을 체크하는 HTML 을 만든다.

각 항목에 체크박스가 있고, 고른 파일명을 한 번에 복사할 수 있다.
체크 상태는 브라우저에 저장돼 새로고침해도 남는다.

사용:
  python3 review_all_images.py --file <택소노미> --images <이미지폴더> \
      --out <출력폴더> [--scores mask_scores.json] [--chunk 40]
"""
import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

from PIL import Image

MAXW = 340
QUALITY = 60

CSS = """
:root{--bg:#fff;--fg:#18181b;--mut:#71717a;--line:#e4e4e7;--card:#fafafa;
      --warn:#b45309;--warnbg:#fef3c7;--bad:#b91c1c;--badbg:#fee2e2;--acc:#2563eb}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
      --bg:#0b0b0e;--fg:#e4e4e7;--mut:#a1a1aa;--line:#27272a;--card:#141418;
      --warn:#fbbf24;--warnbg:#3f2d0a;--bad:#fca5a5;--badbg:#450a0a;--acc:#60a5fa}}
*{box-sizing:border-box}
body{margin:0;padding:20px 20px 100px;background:var(--bg);color:var(--fg);
     font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
h1{font-size:19px;margin:0 0 4px}
.sub{color:var(--mut);font-size:13px;margin-bottom:20px}
.nav{font-size:13px;margin-bottom:16px}
.nav a{color:var(--acc);margin-right:10px}
.item{display:grid;grid-template-columns:32px minmax(0,340px) minmax(0,1fr);gap:16px;
      padding:16px;margin-bottom:12px;border:1px solid var(--line);
      border-radius:10px;background:var(--card);scroll-margin-top:20px}
.item.on{border-color:var(--bad);background:var(--badbg)}
@media(max-width:800px){.item{grid-template-columns:32px 1fr}.item .ans{grid-column:1/-1}}
.item img{width:100%;height:auto;border-radius:6px;border:1px solid var(--line)}
.chk{width:22px;height:22px;cursor:pointer;margin-top:4px}
.meta{font-size:12px;color:var(--mut);margin-top:6px;word-break:break-all}
.ans{min-width:0;overflow-x:auto;max-height:420px;overflow-y:auto}
.ans table{border-collapse:collapse;font-size:12px;margin:6px 0}
.ans th,.ans td{border:1px solid var(--line);padding:3px 6px;text-align:left}
.ans h3{font-size:15px;margin:2px 0 6px}.ans h4{font-size:13px;margin:6px 0 2px;color:var(--mut)}
.ans p{margin:1px 0;font-size:13px}
.badge{display:inline-block;font-size:11px;padding:2px 8px;border-radius:99px;
       margin:0 4px 4px 0;background:var(--warnbg);color:var(--warn)}
.bar{position:fixed;left:0;right:0;bottom:0;background:var(--card);
     border-top:1px solid var(--line);padding:10px 20px;display:flex;
     gap:12px;align-items:center;font-size:13px;z-index:9}
.bar button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--line);
            background:var(--bg);color:var(--fg);cursor:pointer}
.bar button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
#out{position:fixed;inset:10% 10%;background:var(--bg);border:1px solid var(--line);
     border-radius:10px;padding:16px;display:none;z-index:10;flex-direction:column}
#out textarea{flex:1;width:100%;font:12px/1.5 ui-monospace,monospace;
     background:var(--card);color:var(--fg);border:1px solid var(--line);
     border-radius:6px;padding:10px;resize:none}
"""

JS = """
const KEY='vlm_review_drop';
const get=()=>new Set(JSON.parse(localStorage.getItem(KEY)||'[]'));
function sync(){
  const s=get();
  document.querySelectorAll('.item').forEach(el=>{
    const on=s.has(el.dataset.name);
    el.querySelector('.chk').checked=on; el.classList.toggle('on',on);
  });
  document.getElementById('cnt').textContent=s.size;
}
function toggle(el){
  const s=get(), n=el.dataset.name;
  s.has(n)?s.delete(n):s.add(n);
  localStorage.setItem(KEY,JSON.stringify([...s])); sync();
}
function show(){
  const s=[...get()].sort();
  document.getElementById('ta').value=s.join('\\n');
  document.getElementById('out').style.display='flex';
}
function copyAll(){
  const ta=document.getElementById('ta'); ta.select();
  document.execCommand('copy');
}
function clearAll(){
  if(confirm('체크를 전부 지웁니다.')){localStorage.removeItem(KEY);sync();}
}
document.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('.item').forEach(el=>{
    el.querySelector('.chk').addEventListener('change',()=>toggle(el));
  });
  sync();
});
"""


def embed(path):
    orig = Image.open(path).size
    im = Image.open(path).convert("RGB")
    if im.width > MAXW:
        im = im.resize((MAXW, max(1, int(im.height * MAXW / im.width))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), orig


def md_to_html(md):
    out, i, lines = [], 0, md.split("\n")
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= {"-", ":", " "} and c for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                head, *body = rows
                t = ["<table><tr>"] + [f"<th>{html.escape(c)}</th>" for c in head] + ["</tr>"]
                for r in body:
                    t.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>")
                t.append("</table>")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores", help="detect_masked_images.py --dump-scores 결과")
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--sort", choices=("name", "res", "mask"), default="name")
    ap.add_argument("--maxw", type=int)
    a = ap.parse_args()

    global MAXW
    if a.maxw:
        MAXW = a.maxw

    rows = json.loads(Path(a.file).read_text(encoding="utf-8"))
    imgdir = Path(a.images)
    scores = json.loads(Path(a.scores).read_text()) if a.scores else {}
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    entries = []
    for r in rows:
        name = r["prompt"]["image"][0]
        p = imgdir / name
        if not p.exists():
            continue
        entries.append((name, r, p))

    if a.sort == "res":
        entries.sort(key=lambda e: Image.open(e[2]).size[0] * Image.open(e[2]).size[1])
    elif a.sort == "mask":
        entries.sort(key=lambda e: -scores.get(e[0], 0))

    chunks = [entries[i:i + a.chunk] for i in range(0, len(entries), a.chunk)]
    names = [f"review_{i:02d}.html" for i in range(1, len(chunks) + 1)]

    for ci, part in enumerate(chunks):
        items = []
        for name, r, p in part:
            uri, (ow, oh) = embed(p)
            ans = r["completion"]["chosen"]
            b = []
            px = ow * oh
            if px < 200_000:
                b.append(f"저해상도 {ow}×{oh}")
            m = scores.get(name, 0)
            if m >= 0.3:
                b.append(f"가림 {m*100:.0f}%")
            if len(ans) < 100:
                b.append(f"정답 {len(ans)}자")
            badges = "".join(f'<span class="badge">{html.escape(x)}</span>' for x in b)
            items.append(f"""<div class="item" id="{html.escape(name)}" data-name="{html.escape(name)}">
<div><input type="checkbox" class="chk"></div>
<div><img src="{uri}" alt="{html.escape(name)}">
<div class="meta"><b>{html.escape(name)}</b><br>{ow}×{oh} · 정답 {len(ans)}자</div></div>
<div class="ans">{badges}{md_to_html(ans[:5000])}</div></div>""")

        nav = " ".join(
            f'<a href="{n}">{i+1}</a>' if i != ci else f"<b>{i+1}</b>"
            for i, n in enumerate(names))
        doc = f"""<title>데이터셋 검수 {ci+1}/{len(chunks)}</title>
<style>{CSS}</style>
<h1>데이터셋 검수 — {ci+1} / {len(chunks)}</h1>
<div class="sub">전체 {len(entries)}건 · 이 페이지 {len(part)}건 ·
 제거할 것에 체크하세요. 체크는 브라우저에 저장되어 페이지를 옮겨도 유지됩니다.</div>
<div class="nav">{nav}</div>
{''.join(items)}
<div class="bar"><b>선택 <span id="cnt">0</span>건</b>
 <button class="primary" onclick="show()">선택 목록 보기</button>
 <button onclick="clearAll()">전체 해제</button>
 <span style="color:var(--mut)">페이지를 옮겨도 선택은 유지됩니다</span></div>
<div id="out"><p><b>제거 목록</b> — 복사해서 알려주세요</p>
 <textarea id="ta"></textarea>
 <p><button class="primary" onclick="copyAll()">복사</button>
    <button onclick="document.getElementById('out').style.display='none'">닫기</button></p></div>
<script>{JS}</script>"""
        fp = outdir / names[ci]
        fp.write_text(doc, encoding="utf-8")
        print(f"  {fp.name}  {len(part)}건  {fp.stat().st_size/1e6:.1f}MB")

    print(f"\n총 {len(entries)}건 → {len(chunks)}개 파일")
    print(f"시작: {outdir/names[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
