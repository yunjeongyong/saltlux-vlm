"""SS 400건 평가 결과를 한 장짜리 HTML 로 굽는다 — 어느 문서가 왜 틀렸는지 눈으로 본다.

analyze_low_acc.py 가 숫자로 내는 것을 사람이 이미지와 함께 확인하는 판이다.
문서 한 건마다 원본 이미지 · 우리 모델이 뽑은 마크다운 · KIE 결과 · 정답을
나란히 놓는다. 필드가 왜 틀렸는지는 값을 나란히 봐야 알 수 있다.

입력은 전부 이미 있는 것들이다. 다시 채점하지 않는다.
    runs/<ts>_eval_kie.json        per-doc 점수와 필드별 mismatch
    vlm_output/<TAG>/<type>/*.md   파싱 단계 출력
    vlm_output/<KIE>/<type>/*.json KIE 단계 출력
    kie_dataset/<type>/*.json      정답
    data/<type>/images/*           원본 이미지

이미지는 전부 축소해 data URI 로 박는다. 아티팩트 한 장이 자족해야 해서다.
하위 N 건만 큰 판을 같이 넣는다 — 400장을 다 크게 넣으면 용량을 넘는다.

usage:
    python3 scripts/build_ss_report.py --tag exp010 --kie-dir vlm_output/gm_exp010 \
        --parsing-dir vlm_output/exp010 --out /tmp/ss_exp010.html
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
GT_DIR = ROOT / "kie_dataset"
IMG_DIR = ROOT / "data"

THUMB_W = 200        # 그리드용
DETAIL_W = 760       # 하위 N건 상세용
MD_CHARS = 4000      # 마크다운은 이만큼만 싣는다


def under_root(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


def latest_report_for(kie_dir: str) -> Path:
    """해당 pred-dir 로 채점한 리포트 중 가장 최근 것."""
    best = None
    for p in sorted(glob.glob(str(ROOT / "runs" / "*_eval_kie.json"))):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        if Path(d.get("pred_dir", "")).name == Path(kie_dir).name:
            best = Path(p)
    if best is None:
        raise SystemExit(f"{kie_dir} 로 채점한 리포트를 runs/ 에서 못 찾았다")
    return best


def find_image(doc_type: str, name: str) -> Path | None:
    for ext in ("jpg", "jpeg", "png", "webp", "JPG", "PNG"):
        p = IMG_DIR / doc_type / "images" / f"{name}.{ext}"
        if p.exists():
            return p
    return None


def encode(path: Path, width: int, quality: int) -> str:
    im = Image.open(path)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    if im.width > width:
        im = im.resize((width, max(1, round(im.height * width / im.width))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def strip_row(field: str) -> str:
    return field.split(".", 1)[1] if field.startswith("row") and "." in field else field


def classify(r: dict, md_chars: int | None) -> str:
    """analyze_low_acc.py 와 같은 기준 — 두 곳의 라벨이 갈리면 안 된다."""
    if r["status"] != "ok":
        return "KIE 산출물 없음"
    if md_chars is not None and md_chars <= 120:
        return "파싱 실패"
    gt_rows = r.get("table_row_count_gt") or 0
    pred_rows = r.get("table_row_count_pred") or 0
    if gt_rows and pred_rows == 0:
        return "표 통째 누락"
    if gt_rows != pred_rows:
        return "표 행수 불일치"
    gen, tab = r.get("general_accuracy"), r.get("table_accuracy")
    if gen is not None and tab is not None:
        if gen < tab - 0.15:
            return "머리말 필드 오류"
        if tab < gen - 0.15:
            return "표 값 오류"
    return "전반 저조"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="실험 이름 (표시용)")
    ap.add_argument("--kie-dir", required=True, help="KIE 결과 디렉터리 (vlm_output/gm_...)")
    ap.add_argument("--parsing-dir", required=True, help="파싱 .md 디렉터리")
    ap.add_argument("--report", type=Path, help="eval_kie 리포트. 생략하면 kie-dir 로 찾는다")
    ap.add_argument("--detail-count", type=int, default=48, help="큰 이미지를 넣을 하위 건수")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    kie_dir = under_root(Path(a.kie_dir))
    parsing_dir = under_root(Path(a.parsing_dir))
    report = under_root(a.report) if a.report else latest_report_for(a.kie_dir)
    rep = json.loads(report.read_text(encoding="utf-8"))

    print(f"리포트  : {report.name}")
    print(f"KIE     : {kie_dir}")
    print(f"파싱    : {parsing_dir}")

    docs = []
    for r in rep["results"]:
        dt, name = r["doc_type"], r["name"]

        md_path = parsing_dir / dt / f"{name}.md"
        md = md_path.read_text(encoding="utf-8", errors="replace") if md_path.exists() else ""
        md_len = len(md.strip())

        gt_path = GT_DIR / dt / f"{name}.json"
        gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}
        pr_path = kie_dir / dt / f"{name}.json"
        pred = json.loads(pr_path.read_text(encoding="utf-8")) if pr_path.exists() else {}

        # 필드별 점수 — 리포트의 mismatch 를 키로 붙인다. 없으면 맞은 것이다.
        gscore = {m["field"]: m["score"] for m in (r.get("general_mismatches") or [])}

        general = []
        for k, gv in (gt.get("General") or {}).items():
            pv = (pred.get("General") or {}).get(k)
            general.append({"k": k, "gt": gv, "pred": pv, "s": gscore.get(k, 1.0)})

        docs.append({
            "type": dt,
            "name": name,
            "total": r.get("total_accuracy"),
            "gen": r.get("general_accuracy"),
            "tab": r.get("table_accuracy"),
            "rp": r.get("table_row_count_pred"),
            "rg": r.get("table_row_count_gt"),
            "cause": classify(r, md_len if md_path.exists() or parsing_dir.exists() else None),
            "md": md[:MD_CHARS],
            "md_len": md_len,
            "general": general,
            "gt_table": (gt.get("Table") or [])[:40],
            "pred_table": (pred.get("Table") or [])[:40],
        })

    scored = sorted([d for d in docs if d["total"] is not None], key=lambda d: d["total"])
    worst = {(d["type"], d["name"]) for d in scored[:a.detail_count]}

    # ── 이미지
    print(f"이미지 인코딩 {len(docs)}건 (상세 {len(worst)}건)...")
    for d in docs:
        p = find_image(d["type"], d["name"])
        if p is None:
            d["thumb"] = d["big"] = ""
            continue
        d["thumb"] = encode(p, THUMB_W, 55)
        d["big"] = encode(p, DETAIL_W, 72) if (d["type"], d["name"]) in worst else ""

    by_type = {}
    for dt in rep["doc_types"]:
        sub = [d for d in docs if d["type"] == dt and d["total"] is not None]
        s = rep["by_type"].get(dt, {})
        by_type[dt] = {
            "n": len(sub),
            "total": s.get("total_accuracy"),
            "gen": s.get("general_accuracy"),
            "tab": s.get("table_accuracy"),
            "mism": s.get("table_row_count_mismatches"),
        }

    causes = defaultdict(int)
    for d in scored:
        if d["total"] < 0.70:
            causes[d["cause"]] += 1

    # 필드별 실패 집계
    field_fail = defaultdict(int)
    for r in rep["results"]:
        if r["status"] != "ok":
            continue
        for key, pre in (("general_mismatches", "General"), ("table_mismatches", "Table")):
            for m in r.get(key) or []:
                field_fail[f"{pre}.{strip_row(m['field'])}"] += 1

    payload = {
        "tag": a.tag,
        "report": report.name,
        "overall": rep["overall"],
        "by_type": by_type,
        "causes": dict(sorted(causes.items(), key=lambda kv: -kv[1])),
        "fields": dict(sorted(field_fail.items(), key=lambda kv: -kv[1])[:30]),
        "docs": docs,
    }

    html = TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    out = under_root(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"저장: {out}  ({out.stat().st_size/1e6:.1f} MB)")


TEMPLATE = r"""<title>SS 400건 파싱 검수</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{
  --paper:#F6F7F9; --card:#FFFFFF; --sunk:#EDEFF3;
  --ink:#141A22; --ink-2:#4A5563; --ink-3:#7C8797;
  --line:#DFE3EA; --line-2:#C9D0DA;
  --accent:#2B4ACB; --accent-soft:#E7EBFB;
  --good:#1F7A55; --good-bg:#E3F3EC;
  --warn:#9A6A05; --warn-bg:#FBF0D8;
  --bad:#B32741;  --bad-bg:#FBE5EA;
  --shadow:0 1px 2px rgba(20,26,34,.06),0 8px 24px -12px rgba(20,26,34,.18);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0E131A; --card:#161D26; --sunk:#1D2530;
    --ink:#E8ECF2; --ink-2:#A3AEBD; --ink-3:#6E7A8A;
    --line:#26303C; --line-2:#354251;
    --accent:#7A93F5; --accent-soft:#1C2748;
    --good:#5CC79B; --good-bg:#12301F;
    --warn:#E0B457; --warn-bg:#332711;
    --bad:#F08099;  --bad-bg:#3A1622;
    --shadow:0 1px 2px rgba(0,0,0,.5),0 10px 28px -14px rgba(0,0,0,.8);
  }
}
:root[data-theme="dark"]{
  --paper:#0E131A; --card:#161D26; --sunk:#1D2530;
  --ink:#E8ECF2; --ink-2:#A3AEBD; --ink-3:#6E7A8A;
  --line:#26303C; --line-2:#354251;
  --accent:#7A93F5; --accent-soft:#1C2748;
  --good:#5CC79B; --good-bg:#12301F;
  --warn:#E0B457; --warn-bg:#332711;
  --bad:#F08099;  --bad-bg:#3A1622;
  --shadow:0 1px 2px rgba(0,0,0,.5),0 10px 28px -14px rgba(0,0,0,.8);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:"IBM Plex Sans KR",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:14px; line-height:1.55; -webkit-font-smoothing:antialiased;
}
.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
h1,h2,h3{margin:0;text-wrap:balance}

header{
  padding:22px 26px 18px; border-bottom:1px solid var(--line); background:var(--card);
  display:flex; flex-wrap:wrap; align-items:flex-end; gap:8px 28px;
}
.brand h1{font-size:19px;font-weight:600;letter-spacing:-.01em}
.brand p{margin:2px 0 0;color:var(--ink-3);font-size:12px}
.brand .tag{
  font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:600;
  color:var(--accent);background:var(--accent-soft);
  padding:2px 7px;border-radius:4px;margin-left:6px;letter-spacing:.02em;
}
.kpis{display:flex;gap:26px;margin-left:auto;flex-wrap:wrap}
.kpi{display:flex;flex-direction:column;gap:1px}
.kpi .v{font-family:"IBM Plex Mono",monospace;font-size:22px;font-weight:600;letter-spacing:-.02em;line-height:1.1}
.kpi .l{font-size:10.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.07em}

main{padding:20px 26px 40px;max-width:1680px;margin:0 auto}
.panels{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(0,1fr);gap:16px;margin-bottom:22px}
@media(max-width:1080px){.panels{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:15px 17px;box-shadow:var(--shadow)}
.panel h2{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-3);margin-bottom:12px}

table.t{width:100%;border-collapse:collapse}
table.t th{font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink-3);font-weight:600;text-align:right;padding:0 0 6px}
table.t th:first-child{text-align:left}
table.t td{padding:5px 0;border-top:1px solid var(--line);text-align:right;font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
table.t td:first-child{text-align:left;font-family:"IBM Plex Sans KR",sans-serif}

.bar{height:5px;border-radius:3px;background:var(--sunk);overflow:hidden;margin-top:5px}
.bar>i{display:block;height:100%;background:var(--accent);border-radius:3px}

.cause{display:flex;align-items:center;gap:9px;padding:5px 0;border-top:1px solid var(--line)}
.cause:first-of-type{border-top:0}
.cause .n{font-family:"IBM Plex Mono",monospace;font-weight:600;min-width:26px;text-align:right}
.cause .t{flex:1;font-size:13px}
.cause .g{height:5px;border-radius:3px;background:var(--bad);opacity:.75}

.flds{max-height:236px;overflow-y:auto;margin:-3px -4px -3px 0;padding-right:4px}
.fld{display:flex;align-items:baseline;gap:9px;padding:3.5px 0;border-top:1px solid var(--line);font-size:12.5px}
.fld:first-child{border-top:0}
.fld .n{font-family:"IBM Plex Mono",monospace;font-weight:600;min-width:32px;text-align:right;color:var(--bad)}
.fld .k{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--ink-2);word-break:break-all}

.toolbar{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-bottom:14px}
.chip{
  font-size:12px;font-weight:500;padding:5px 11px;border-radius:20px;cursor:pointer;
  border:1px solid var(--line-2);background:var(--card);color:var(--ink-2);
  font-family:inherit;transition:.12s;
}
.chip:hover{border-color:var(--accent);color:var(--accent)}
.chip[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.chip:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.spacer{flex:1}
.count{font-size:12px;color:var(--ink-3);font-family:"IBM Plex Mono",monospace}

.split{display:grid;grid-template-columns:minmax(320px,470px) minmax(0,1fr);gap:18px;align-items:start}
@media(max-width:1000px){.split{grid-template-columns:1fr}}

.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(84px,1fr));gap:7px;
  max-height:76vh;overflow-y:auto;padding:3px;background:var(--sunk);
  border:1px solid var(--line);border-radius:8px}
.cell{
  position:relative;border:0;padding:0;cursor:pointer;background:var(--card);
  border-radius:5px;overflow:hidden;aspect-ratio:3/4;
  box-shadow:0 0 0 1px var(--line);transition:.12s;
}
.cell:hover{box-shadow:0 0 0 2px var(--accent);transform:translateY(-1px)}
.cell[aria-current="true"]{box-shadow:0 0 0 2.5px var(--accent)}
.cell:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.cell img{width:100%;height:100%;object-fit:cover;object-position:top;display:block}
.cell .sc{
  position:absolute;left:0;right:0;bottom:0;padding:2px 4px;
  font-family:"IBM Plex Mono",monospace;font-size:10.5px;font-weight:600;
  color:#fff;background:linear-gradient(transparent,rgba(0,0,0,.82) 55%);
  text-align:right;letter-spacing:-.01em;
}
.cell .stripe{position:absolute;top:0;left:0;width:100%;height:3px}
.cell .nm{position:absolute;top:3px;left:4px;font-family:"IBM Plex Mono",monospace;
  font-size:9.5px;color:#fff;text-shadow:0 1px 3px rgba(0,0,0,.9)}

.detail{background:var(--card);border:1px solid var(--line);border-radius:8px;box-shadow:var(--shadow);overflow:hidden}
.dhead{padding:14px 18px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:10px 18px;align-items:baseline}
.dhead .nm{font-family:"IBM Plex Mono",monospace;font-size:15px;font-weight:600}
.pill{font-size:11px;font-weight:600;padding:2.5px 9px;border-radius:20px;letter-spacing:.01em}
.p-bad{background:var(--bad-bg);color:var(--bad)}
.p-warn{background:var(--warn-bg);color:var(--warn)}
.p-good{background:var(--good-bg);color:var(--good)}
.dmetrics{margin-left:auto;display:flex;gap:16px}
.dmetrics div{text-align:right}
.dmetrics .v{font-family:"IBM Plex Mono",monospace;font-size:15px;font-weight:600}
.dmetrics .l{font-size:10px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.06em}

.dbody{display:grid;grid-template-columns:minmax(220px,300px) minmax(0,1fr);gap:0}
@media(max-width:840px){.dbody{grid-template-columns:1fr}}
.dimg{padding:16px;border-right:1px solid var(--line);background:var(--sunk)}
.dimg img{width:100%;border-radius:5px;border:1px solid var(--line-2);display:block;background:#fff}
.dimg p{margin:9px 0 0;font-size:11px;color:var(--ink-3);font-family:"IBM Plex Mono",monospace}
.dcols{padding:16px 18px;min-width:0}

.sec{font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-3);margin:0 0 8px}
.sec+.sec{margin-top:18px}

table.diff{width:100%;border-collapse:collapse;font-size:12.5px;table-layout:fixed}
table.diff th{font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink-3);
  text-align:left;font-weight:600;padding:0 8px 5px 0;border-bottom:1px solid var(--line-2)}
table.diff td{padding:5px 8px 5px 0;border-bottom:1px solid var(--line);vertical-align:top;
  word-break:break-word;overflow-wrap:anywhere}
table.diff .k{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--ink-2);width:27%}
table.diff .v{font-family:"IBM Plex Mono",monospace;font-size:11.5px}
tr.miss{background:var(--bad-bg)}
tr.miss .v.p{color:var(--bad);font-weight:500}
tr.part{background:var(--warn-bg)}
.sc-b{font-family:"IBM Plex Mono",monospace;font-size:10.5px;color:var(--ink-3);text-align:right;width:44px}

.rows{overflow-x:auto;border:1px solid var(--line);border-radius:6px}
table.rw{width:100%;border-collapse:collapse;font-size:11.5px;font-family:"IBM Plex Mono",monospace;white-space:nowrap}
table.rw th{background:var(--sunk);padding:5px 9px;text-align:left;font-weight:600;color:var(--ink-2);
  font-size:10px;text-transform:uppercase;letter-spacing:.04em;border-bottom:1px solid var(--line)}
table.rw td{padding:4px 9px;border-bottom:1px solid var(--line)}
table.rw tbody tr:last-child td{border-bottom:0}
.side{font-size:9.5px;font-weight:600;letter-spacing:.05em;padding:1px 5px;border-radius:3px}
.s-gt{background:var(--good-bg);color:var(--good)}
.s-pr{background:var(--accent-soft);color:var(--accent)}

pre.md{margin:0;padding:12px 13px;background:var(--sunk);border:1px solid var(--line);border-radius:6px;
  font-family:"IBM Plex Mono",monospace;font-size:11.5px;line-height:1.5;
  max-height:300px;overflow:auto;white-space:pre-wrap;word-break:break-word;color:var(--ink-2)}
.empty{padding:26px;text-align:center;color:var(--ink-3);font-size:13px}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>

<header>
  <div class="brand">
    <h1>SS 400건 파싱 검수<span class="tag" id="tag"></span></h1>
    <p id="sub"></p>
  </div>
  <div class="kpis" id="kpis"></div>
</header>

<main>
  <div class="panels">
    <div class="panel">
      <h2>문서 유형별 정확도</h2>
      <table class="t"><thead><tr><th>유형</th><th>건수</th><th>general</th><th>table</th><th>total</th></tr></thead>
      <tbody id="tbType"></tbody></table>
    </div>
    <div class="panel">
      <h2>원인 분포 · total &lt; 70%</h2>
      <div id="causeList"></div>
    </div>
    <div class="panel">
      <h2>필드별 실패 건수 · 상위 30</h2>
      <div class="flds" id="fldList"></div>
    </div>
  </div>

  <div class="toolbar" id="bar"></div>

  <div class="split">
    <div class="grid" id="grid"></div>
    <div class="detail" id="detail"></div>
  </div>
</main>

<script>
const D = __DATA__;
const pct = v => v == null ? "—" : (v*100).toFixed(1) + "%";
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const val = v => v === null || v === undefined ? "∅" : (typeof v === "object" ? JSON.stringify(v) : String(v));
const tone = v => v == null ? "bad" : v >= 0.9 ? "good" : v >= 0.7 ? "warn" : "bad";
const hue = v => v == null ? "var(--bad)" : v >= 0.9 ? "var(--good)" : v >= 0.7 ? "var(--warn)" : "var(--bad)";

document.getElementById("tag").textContent = D.tag;
document.getElementById("sub").textContent =
  `${D.overall.num_docs}건 · 동일 KIE 추출기 · ${D.report}`;

document.getElementById("kpis").innerHTML = [
  ["total", D.overall.total_accuracy], ["general", D.overall.general_accuracy],
  ["table", D.overall.table_accuracy],
].map(([l, v]) => `<div class="kpi"><span class="v">${pct(v)}</span><span class="l">${l}</span></div>`).join("")
 + `<div class="kpi"><span class="v">${D.overall.table_row_count_mismatches}</span><span class="l">행수 불일치</span></div>`;

document.getElementById("tbType").innerHTML = Object.entries(D.by_type).map(([t, s]) =>
  `<tr><td>${t}</td><td>${s.n}</td><td>${pct(s.gen)}</td><td>${pct(s.tab)}</td>
   <td style="color:${hue(s.total)};font-weight:600">${pct(s.total)}</td></tr>`).join("");

const causeMax = Math.max(1, ...Object.values(D.causes));
document.getElementById("causeList").innerHTML = Object.keys(D.causes).length
  ? Object.entries(D.causes).map(([c, n]) =>
      `<div class="cause"><span class="n">${n}</span><span class="t">${esc(c)}</span>
       <span class="g" style="width:${Math.round(n/causeMax*94)}px"></span></div>`).join("")
  : `<div class="empty">70% 미만이 없다</div>`;

const fldMax = Math.max(1, ...Object.values(D.fields));
document.getElementById("fldList").innerHTML = Object.entries(D.fields).map(([f, n]) =>
  `<div class="fld"><span class="n">${n}</span><span class="k">${esc(f)}</span>
   <span style="flex:0 0 ${Math.round(n/fldMax*70)}px;height:4px;border-radius:2px;background:var(--bad);opacity:.6"></span></div>`).join("");

// ── 필터
const TYPES = Object.keys(D.by_type);
let fType = "all", fCause = "all";
const causeKeys = ["all", ...Object.keys(D.causes)];
document.getElementById("bar").innerHTML =
  `<span class="count" style="margin-right:2px">유형</span>` +
  ["all", ...TYPES].map(t => `<button class="chip" data-k="type" data-v="${t}" aria-pressed="${t==="all"}">${t === "all" ? "전체" : t}</button>`).join("") +
  `<span class="count" style="margin:0 2px 0 12px">원인</span>` +
  causeKeys.map(c => `<button class="chip" data-k="cause" data-v="${esc(c)}" aria-pressed="${c==="all"}">${c === "all" ? "전체" : esc(c)}</button>`).join("") +
  `<span class="spacer"></span><span class="count" id="cnt"></span>`;

document.getElementById("bar").addEventListener("click", e => {
  const b = e.target.closest(".chip"); if (!b) return;
  const k = b.dataset.k;
  if (k === "type") fType = b.dataset.v; else fCause = b.dataset.v;
  document.querySelectorAll(`.chip[data-k="${k}"]`).forEach(x =>
    x.setAttribute("aria-pressed", String(x.dataset.v === b.dataset.v)));
  render();
});

const sorted = D.docs.slice().sort((a, b) =>
  (a.total ?? -1) - (b.total ?? -1) || a.type.localeCompare(b.type) || a.name.localeCompare(b.name));
let cur = sorted[0];

function visible() {
  return sorted.filter(d =>
    (fType === "all" || d.type === fType) && (fCause === "all" || d.cause === fCause));
}

function render() {
  const vs = visible();
  document.getElementById("cnt").textContent = `${vs.length} / ${sorted.length}건`;
  document.getElementById("grid").innerHTML = vs.map(d => `
    <button class="cell" data-id="${d.type}/${d.name}" aria-current="${cur && cur.type===d.type && cur.name===d.name}"
            title="${d.type}/${d.name} · ${pct(d.total)} · ${esc(d.cause)}">
      <span class="stripe" style="background:${hue(d.total)}"></span>
      ${d.thumb ? `<img src="${d.thumb}" alt="${d.type} ${d.name}" loading="lazy">` : ``}
      <span class="nm">${esc(d.name)}</span>
      <span class="sc">${pct(d.total)}</span>
    </button>`).join("") || `<div class="empty" style="grid-column:1/-1">해당하는 문서가 없다</div>`;
  if (vs.length && !vs.some(d => cur && d.type === cur.type && d.name === cur.name)) select(vs[0]);
  else detail();
}

document.getElementById("grid").addEventListener("click", e => {
  const b = e.target.closest(".cell"); if (!b) return;
  const [t, n] = b.dataset.id.split("/");
  select(sorted.find(d => d.type === t && d.name === n));
});

function select(d) { cur = d; render(); }

function rowsTable(d) {
  const keys = [...new Set([...(d.gt_table[0] ? Object.keys(d.gt_table[0]) : []),
                            ...(d.pred_table[0] ? Object.keys(d.pred_table[0]) : [])])];
  if (!keys.length) return `<div class="empty">표가 없는 문서다</div>`;
  const line = (r, side, i) => `<tr>
      <td><span class="side ${side === "정답" ? "s-gt" : "s-pr"}">${side}</span> ${i + 1}</td>
      ${keys.map(k => `<td>${esc(val(r ? r[k] : null))}</td>`).join("")}</tr>`;
  const n = Math.max(d.gt_table.length, d.pred_table.length);
  let body = "";
  for (let i = 0; i < n; i++) {
    body += line(d.gt_table[i], "정답", i);
    body += line(d.pred_table[i], "예측", i);
  }
  return `<div class="rows"><table class="rw"><thead><tr><th>행</th>${keys.map(k => `<th>${esc(k)}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function detail() {
  const d = cur, el = document.getElementById("detail");
  if (!d) { el.innerHTML = `<div class="empty">문서를 고르면 여기에 나온다</div>`; return; }
  const gRows = d.general.map(f => {
    const cls = f.s >= 1 ? "" : f.s >= 0.6 ? "part" : "miss";
    return `<tr class="${cls}"><td class="k">${esc(f.k)}</td>
      <td class="v">${esc(val(f.gt))}</td>
      <td class="v p">${esc(val(f.pred))}</td>
      <td class="sc-b">${f.s >= 1 ? "" : f.s.toFixed(2)}</td></tr>`;
  }).join("");

  el.innerHTML = `
    <div class="dhead">
      <span class="nm">${esc(d.type)}/${esc(d.name)}</span>
      <span class="pill p-${tone(d.total)}">${esc(d.cause)}</span>
      <div class="dmetrics">
        <div><div class="v" style="color:${hue(d.total)}">${pct(d.total)}</div><div class="l">total</div></div>
        <div><div class="v">${pct(d.gen)}</div><div class="l">general</div></div>
        <div><div class="v">${pct(d.tab)}</div><div class="l">table</div></div>
        <div><div class="v">${d.rp ?? "—"}/${d.rg ?? "—"}</div><div class="l">행수</div></div>
      </div>
    </div>
    <div class="dbody">
      <div class="dimg">
        ${(d.big || d.thumb) ? `<img src="${d.big || d.thumb}" alt="${esc(d.name)} 원본">` : `<div class="empty">이미지 없음</div>`}
        <p>파싱 출력 ${d.md_len.toLocaleString()}자${d.big ? "" : " · 축소본"}</p>
      </div>
      <div class="dcols">
        <p class="sec">머리말 필드 — 정답 대 예측</p>
        ${d.general.length
          ? `<table class="diff"><thead><tr><th>필드</th><th>정답</th><th>예측</th><th></th></tr></thead><tbody>${gRows}</tbody></table>`
          : `<div class="empty">머리말 필드가 없다</div>`}
        <p class="sec">표 — 행 단위 대조</p>
        ${rowsTable(d)}
        <p class="sec">파싱 단계 출력 (마크다운)</p>
        <pre class="md">${esc(d.md) || "(비어 있다)"}</pre>
      </div>
    </div>`;
}

render();
</script>
"""


if __name__ == "__main__":
    main()
