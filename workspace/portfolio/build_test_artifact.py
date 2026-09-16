import base64, collections, html, io, json, re
from pathlib import Path
from PIL import Image

ROOT = Path("/data/workspace/yyj/data/vlm_exp34")
D = json.loads((ROOT/"receipt_data/review/all5_1019.json").read_text(encoding="utf-8"))
MS = ["exp_003","exp_004","exp_005","exp_006","exp_007"]
BASE = ROOT/"receipt_data/eval_crops"

PAT=[re.compile(r'\d{3}-\d{2}-\d{5}'), re.compile(r'\(?0\d{1,2}\)?[-\s]?\d{3,4}[-\s]?\d{4}'),
     re.compile(r'\d{4}[-\s]?\d{2,4}\*+|\*{2,}\d{4}|\d{4}-\d{4}-\d{4}'),
     re.compile(r'(시|도)\s*\S*(구|시|군)\s*\S*(로|길|동)')]
RISKY={'doc_title','header','footer'}
def safe(r): return r['label'] not in RISKY and not any(p.search(r['gt']) for p in PAT)

rows=[r for r in D["rows"] if r["split"]=="test" and safe(r) and all(r.get(m) for m in MS)]
rows.sort(key=lambda r:(r["doc_id"], r["crop_id"]))

def b64(p, mw=210):
    im=Image.open(p).convert("RGB")
    if im.width>mw: im=im.resize((mw,max(1,int(im.height*mw/im.width))), Image.LANCZOS)
    b=io.BytesIO(); im.save(b,"PNG",optimize=True)
    return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()

def clean(t):
    """깨진 문자와 제어문자를 걷어낸다. 아티팩트 발행이 U+FFFD 에서 막힌다."""
    import unicodedata
    return "".join(c for c in t
                   if c not in "\ufffd"
                   and (c in "\n\r\t" or unicodedata.category(c)[0] != "C"))


def score(r,m): return r[m]["cer"] if r["task"]=="text" else r[m]["teds"]
def good(r,m):
    s=score(r,m); return s==0 if r["task"]=="text" else s>=0.95
def tag(r,m):
    s=score(r,m)
    return "정확" if good(r,m) else (f"CER {s:.2f}" if r["task"]=="text" else f"TEDS {s:.3f}")

# 모델별 요약
summ=[]
for m in MS:
    t=[r for r in rows if r["task"]=="text"]
    ok=sum(1 for r in t if r[m]["cer"]==0)
    summ.append((m, ok, len(t), sum(1 for r in t if r[m]["cer"]>1)))
mx=max(s[1] for s in summ)
srows="".join(
  f'<div class="sr"><span class="sn">{m}</span>'
  f'<div class="tk"><i style="width:{ok/n*100:.0f}%"></i></div>'
  f'<b>{ok}/{n}</b><span class="sc">{ok/n*100:.1f}% · 폭주 {bl}</span></div>'
  for m,ok,n,bl in summ)

by=collections.OrderedDict()
for r in rows: by.setdefault(r["doc_id"],[]).append(r)

docs=[]
for doc,rs in by.items():
    cards=[]
    for r in rs:
        p=BASE/r["image"]
        pic=f'<img src="{b64(p)}" alt="">' if p.exists() else ""
        lines=[f'<div class="row gt"><span class="nm">정답</span>'
               f'<span class="tx">{html.escape(clean(r["gt"])[:300])}</span><span class="sc2"></span></div>']
        for m in MS:
            lines.append(
              f'<div class="row{"" if good(r,m) else " miss"}"><span class="nm">{m}</span>'
              f'<span class="tx">{html.escape(clean(r[m]["pred"])[:300])}</span>'
              f'<span class="sc2{" ok" if good(r,m) else ""}">{tag(r,m)}</span></div>')
        cards.append(f'<div class="card"><div class="ci">{pic}'
                     f'<div class="cm">{r["size"][0]}×{r["size"][1]}px · {r["label"]}</div></div>'
                     f'<div class="cb">{"".join(lines)}</div></div>')
    # 문서별 모델 성적 — 펼치지 않고도 어디서 갈리는지 보이게
    per = {m: sum(1 for r in rs if good(r, m)) for m in MS}
    best = max(per.values())
    badges = "".join(
        f'<span class="bd{" top" if per[m]==best else ""}">'
        f'{m.split("_")[1]}<em>{per[m]}</em></span>' for m in MS)
    docs.append(f'<details class="doc"><summary><span class="dn">{doc}</span>'
                f'<span class="bds">{badges}</span>'
                f'<em>{len(rs)}건</em></summary>{"".join(cards)}</details>')

T=Path("test_template.html").read_text(encoding="utf-8")
out=(T.replace("{SUMMARY}",srows).replace("{DOCS}","".join(docs))
      .replace("{NDOC}",str(len(by))).replace("{NCROP}",str(len(rows))))
Path("test_report.html").write_text(out,encoding="utf-8")
print("test_report.html", len(out)//1024//1024, "MB", "·", len(by), "문서", len(rows), "건")
