"""SS receipt 100건 4자 비교표. Document Studio(base/alt) + exp_004 + exp_009."""
import json, glob, subprocess, sys, os
os.chdir("/data/workspace/yyj/data/ss_eval")

def ds(path, t):
    r = [x for x in json.load(open(path))['results']
         if x['doc_type'] == t and x.get('total_accuracy') is not None]
    g = [x['general_accuracy'] for x in r if x.get('general_accuracy') is not None]
    b = [x['table_accuracy'] for x in r if x.get('table_accuracy') is not None]
    mm = sum(1 for x in r if x.get('table_row_count_pred') != x.get('table_row_count_gt'))
    return dict(n=len(r), total=sum(x['total_accuracy'] for x in r)/len(r),
                gen=sum(g)/len(g) if g else 0, tab=sum(b)/len(b) if b else 0, mism=mm)

def ours(pred_dir, t):
    if not glob.glob(f"{pred_dir}/{t}/*.json"):
        return None
    subprocess.run([sys.executable, "scripts/eval_kie.py", "--types", t,
                    "--pred-dir", pred_dir], capture_output=True, text=True)
    rep = sorted(glob.glob("runs/*eval_kie.json"))[-1]
    r = [x for x in json.load(open(rep))['results']
         if x['doc_type'] == t and x.get('total_accuracy') is not None]
    if not r: return None
    g = [x['general_accuracy'] for x in r if x.get('general_accuracy') is not None]
    b = [x['table_accuracy'] for x in r if x.get('table_accuracy') is not None]
    mm = sum(1 for x in r if x.get('table_row_count_pred') != x.get('table_row_count_gt'))
    return dict(n=len(r), total=sum(x['total_accuracy'] for x in r)/len(r),
                gen=sum(g)/len(g) if g else 0, tab=sum(b)/len(b) if b else 0, mism=mm)

T = sys.argv[1] if len(sys.argv) > 1 else "receipt"
rows = [("Document Studio (base)", ds("test_output.json", T)),
        ("Document Studio (alt)",  ds("test_output_alt.json", T)),
        ("exp_004 (현 배포후보)",   ours("vlm_output/exp004_kie", T)),
        ("exp_009 (신규)",          ours("vlm_output/exp009_kie", T))]
print(f"\nSS {T} 100건 · KIE total_accuracy  (동일 엔드포인트 gemma-4-31B-it)\n")
print("%-24s %5s %9s %10s %9s %10s" % ("모델", "n", "total↑", "general↑", "table↑", "행수불일치↓"))
best = max((d['total'] for _, d in rows if d), default=0)
for name, d in rows:
    if not d:
        print("%-24s %5s %9s" % (name, "-", "미완료")); continue
    mark = "  ★" if abs(d['total'] - best) < 1e-9 else ""
    print("%-24s %5d %9.4f %10.4f %9.4f %10d%s"
          % (name, d['n'], d['total'], d['gen'], d['tab'], d['mism'], mark))
ok = [(n, d) for n, d in rows if d]
if len(ok) >= 3:
    b0 = dict(ok)["Document Studio (alt)"]
    print("\nDocument Studio(alt) 대비")
    for n, d in ok:
        if n.startswith("Document"): continue
        print("  %-22s total %+.4f (%+.1f%%) / general %+.4f / table %+.4f"
              % (n, d['total']-b0['total'], 100*(d['total']-b0['total'])/b0['total'],
                 d['gen']-b0['gen'], d['tab']-b0['tab']))
