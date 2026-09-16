import json
M = json.load(open("/tmp/claude-0/-data-workspace-yyj/0a15d8b1-9f64-4619-bbb3-de8cde7a8efb/scratchpad/split_metrics.json"))
EXPS = ["exp_003", "exp_004", "exp_005", "exp_006"]
LABEL = {"exp_003": "4,496행", "exp_004": "11,173행",
         "exp_005": "+온라인 증강", "exp_006": "50,457행"}

def bars(key, fmt, hi_is_good=True, unit=""):
    vals = [M[e][s][key] for e in EXPS for s in ("val", "test")]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1
    # 0 기준이 아니라 값 범위를 쓰면 과장된다 -> 0 기준 유지, 최댓값에 여백
    top = hi * 1.06
    out = []
    for e in EXPS:
        rows = []
        for s, cls in (("val", "v"), ("test", "t")):
            x = M[e][s][key]
            rows.append(
                f'<div class="bar {cls}"><span class="tick">{s}</span>'
                f'<div class="track"><i style="width:{x/top*100:.1f}%"></i></div>'
                f'<b>{fmt(x)}{unit}</b></div>')
        out.append(f'<div class="grp"><div class="gl">{e}<em>{LABEL[e]}</em></div>'
                   f'<div class="gb">{"".join(rows)}</div></div>')
    return "".join(out)

def rank(key, rev=True):
    v = sorted(EXPS, key=lambda e: -M[e]["val"][key] if rev else M[e]["val"][key])
    t = sorted(EXPS, key=lambda e: -M[e]["test"][key] if rev else M[e]["test"][key])
    same = v == t
    def chips(lst):
        return "".join(f'<span class="chip">{i+1}<em>{x}</em></span>' for i, x in enumerate(lst))
    return (f'<div class="rk"><div class="rkr"><span class="rkl">val</span>{chips(v)}</div>'
            f'<div class="rkr"><span class="rkl">test</span>{chips(t)}</div>'
            f'<p class="rkv {"ok" if same else "no"}">{"두 셋의 순위가 같다" if same else "두 셋의 순위가 다르다"}</p></div>')

HTML = open("split_template.html", encoding="utf-8").read()
HTML = (HTML
  .replace("{BARS_EXACT}", bars("exact", lambda x: f"{x:.1f}", unit="%"))
  .replace("{BARS_TEDS}",  bars("teds",  lambda x: f"{x:.4f}"))
  .replace("{BARS_BLOW}",  bars("blow",  lambda x: f"{x:d}", unit="건"))
  .replace("{BARS_MICRO}", bars("micro", lambda x: f"{x:.4f}"))
  .replace("{RANK_EXACT}", rank("exact"))
  .replace("{RANK_TEDS}",  rank("teds")))
open("split_report.html", "w", encoding="utf-8").write(HTML)
print("split_report.html", len(HTML)//1024, "KB")
