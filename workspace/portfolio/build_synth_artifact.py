import base64, html, io, json, sys
from pathlib import Path
from PIL import Image

ROOT = Path("/data/workspace/yyj/data/vlm_exp34")
D = json.loads((ROOT/"receipt_data/review/synth_hard_5models.json").read_text(encoding="utf-8"))
NM = {"upload-mix-qwen36":"exp_003","exp004-260813":"exp_004","exp005-aug-online":"exp_005",
      "exp006-all":"exp_006","exp007-final":"exp_007"}
MS = list(D["summary"])
SYNTH = {"exp_006","exp_007"}          # 합성 데이터로 학습한 실험

def b64(p, maxw=460):
    im = Image.open(p).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, max(1,int(im.height*maxw/im.width))), Image.LANCZOS)
    b = io.BytesIO(); im.save(b,"PNG")
    return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()

def esc(t):
    return html.escape(t).replace(" ", "&middot;").replace("\n","<br>")

# 요약
summ = []
for m in MS:
    t=[r[m] for r in D["rows"] if r.get(m)]
    ex=sum(1 for x in t if x["cer"]==0)
    cer=sum(x["cer"] for x in t)/len(t)
    n=NM[m]
    summ.append((n, ex, cer, n in SYNTH))
mx = max(s[1] for s in summ)
rows_sum = "".join(
  f'<div class="sr{" hit" if syn else ""}"><span class="sn">{n}'
  f'{"<em>합성 학습</em>" if syn else "<em>미학습</em>"}</span>'
  f'<div class="tk"><i style="width:{ex/24*100:.0f}%"></i></div>'
  f'<b>{ex}/24</b><span class="sc">CER {cer:.3f}</span></div>'
  for n,ex,cer,syn in summ)

# 사례 — 합성 학습 여부가 갈린 건 위주
rows = sorted(D["rows"], key=lambda r: -(r[MS[3]]["cer"] < r[MS[0]]["cer"]))
cards=[]
for r in rows[:10]:
    img = b64(ROOT/"receipt_data/synth_hard_crops"/r["image"])
    def pv(m):
        v = r[m]
        mark = " hit" if NM[m] in SYNTH else ""
        tag = "정확" if v["cer"] == 0 else "CER %.2f" % v["cer"]
        return ('<div class="pv%s"><span class="pn">%s</span><code>%s</code>'
                '<span class="pc">%s</span></div>') % (mark, NM[m], esc(v["pred"][:70]), tag)
    preds = "".join(pv(m) for m in MS)
    cards.append(
      f'<div class="card"><div class="ci"><img src="{img}" alt="영수증 크롭">'
      f'<div class="cm">{r["size"][0]}&times;{r["size"][1]}px &middot; {r["label"]}</div></div>'
      f'<div class="cb"><div class="gt"><span class="pn">정답</span>'
      f'<code>{esc(r["gt"][:70])}</code></div>{preds}</div></div>')

T = Path("synth_template.html").read_text(encoding="utf-8")
out = T.replace("{SUMMARY}", rows_sum).replace("{CARDS}", "".join(cards))
Path("synth_report.html").write_text(out, encoding="utf-8")
print("synth_report.html", len(out)//1024, "KB")
