"""
페이지 원본에 bbox 를 얹어 크롭 위치가 글자에 맞는지 눈으로 확인한다.

보정 좌표(초록)와 미보정 좌표(빨강)를 같은 이미지에 겹쳐 그린다.
초록이 글자 위에 얹히고 빨강이 엉뚱한 데 있으면 보정이 맞다는 뜻이다.

usage:
    python3 scripts/viz_crop_overlay.py                       # 기본 4장
    python3 scripts/viz_crop_overlay.py --docs receipt63      # 특정 장
    python3 scripts/viz_crop_overlay.py --docs receipt63 --labels-only
"""
import argparse
import ast
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_eval_crops import is_transposed          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "receipt_data/labeled"
MANIFEST = ROOT / "receipt_data/eval_crops/manifest.jsonl"
SKIP = {"image", "seal", "header_image", "footer_image"}


def eval_docs():
    """평가에 실제로 쓰인 페이지만. manifest 가 정답 — labeled 폴더에는
    제외된 장(receipt19)도 남아 있다."""
    rows = [json.loads(l) for l in MANIFEST.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    return sorted({r["doc_id"] for r in rows}), len(rows)


def find(doc_id):
    for split in ("test", "val"):
        jf = SRC / split / "json" / f"{doc_id}.json"
        if jf.exists():
            img = next((SRC / split / "images").glob(f"{doc_id}.*"))
            return split, jf, img
    raise SystemExit(f"못 찾음: {doc_id}")


def as_bbox(v):
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return None
    return [float(x) for x in v] if isinstance(v, (list, tuple)) and len(v) == 4 else None


INDEX_CSS = """
:root{color-scheme:light;--ground:#f1f4f2;--surface:#fbfcfb;--surface-2:#f4f6f4;
--ink:#14171a;--ink-2:#535a5e;--muted:#7d8487;--hairline:#e0e4e1;--rule:#c8cdc9;
--ok:#0f9d58;--bad:#d03b3b;
--sans:system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",sans-serif;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,"D2Coding",monospace;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--ground:#0d0f0e;--surface:#191b1a;--surface-2:#202322;--ink:#fff;--ink-2:#c1c6c2;
--muted:#8b918d;--hairline:#2b2f2c;--rule:#3b403c;--ok:#3fbf95;--bad:#f06a6a;}}
:root[data-theme="dark"]{color-scheme:dark;--ground:#0d0f0e;--surface:#191b1a;
--surface-2:#202322;--ink:#fff;--ink-2:#c1c6c2;--muted:#8b918d;--hairline:#2b2f2c;
--rule:#3b403c;--ok:#3fbf95;--bad:#f06a6a;}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
font-size:14px;line-height:1.55}
.wrap{max-width:1500px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--muted);font-size:13px;margin-bottom:20px}
.key{background:var(--surface);border:1px solid var(--hairline);border-radius:10px;
padding:12px 16px;margin-bottom:20px;font-size:13px}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;
vertical-align:middle;margin-right:5px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
.cell{background:var(--surface);border:1px solid var(--hairline);border-radius:10px;
overflow:hidden}
.cell img{width:100%;display:block;background:#fff}
.meta{padding:8px 11px;font-size:11.5px;border-top:1px solid var(--hairline)}
.doc{font-family:var(--mono);font-weight:600;font-size:12.5px}
.dim{color:var(--muted);font-family:var(--mono);font-size:11px}
.warn{color:var(--bad)}
"""


def index_html(cards, n_crop):
    P = [f"<title>크롭 위치 검증 — 평가 {len(cards)}장</title><style>{INDEX_CSS}</style>",
         '<div class="wrap">', "<h1>크롭 위치 검증 — 평가에 쓰인 전체 페이지</h1>",
         f'<div class="sub">영수증 {len(cards)}장 · 크롭 {n_crop:,}건 · '
         "각 페이지에 크롭 좌표를 겹쳐 그림</div>",
         '<div class="key">'
         '<span class="sw" style="background:#00be5a"></span>'
         "<b>초록 = 보정 좌표</b> — 실제 크롭에 쓰인 것 "
         "(<code>bbox × W/pw, H/ph</code>) &nbsp;·&nbsp; "
         '<span class="sw" style="background:#e13c3c"></span>'
         "<b>빨강 = 미보정 좌표</b> — 대조군, 평가에는 안 쓰임 "
         "(<code>bbox</code> 그대로)<br>"
         "초록이 글자 위에 얹혀 있으면 크롭이 정상이다. 빨강이 적게 보이는 장은 "
         "미보정 박스가 이미지 밖으로 완전히 벗어나 그릴 수조차 없었다는 뜻이다."
         "</div>", '<div class="grid">']
    for c in cards:
        P.append(
            f'<div class="cell"><img src="{c["src"]}" alt="{html.escape(c["doc"])}">'
            f'<div class="meta"><span class="doc">{html.escape(c["doc"])}</span> '
            f'<span class="dim">· {c["split"]}</span><br>'
            + ('<span class="warn">⟳ 회전 보정됨 (좌표계 90° 어긋남)</span><br>'
               if c.get("rot") else "")
            + f'<span class="dim">이미지 {c["W"]}×{c["H"]} / json {c["pw"]}×{c["ph"]} '
            f'· 배율 {c["sx"]:.3f},{c["sy"]:.3f}</span><br>'
            f'<span class="dim">초록 {c["n_ok"]}개 · 빨강 {c["n_bad"]}개'
            + (f' <span class="warn">(미보정 {c["n_ok"]-c["n_bad"]}개는 화면 밖)</span>'
               if c["n_bad"] < c["n_ok"] else "")
            + "</span></div></div>")
    P += ["</div></div>"]
    return "\n".join(P)


def card_src(canvas, width=560, quality=74):
    im = canvas
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*", default=["receipt63", "receipt01",
                                                  "receipt04", "receipt05"])
    ap.add_argument("--all", action="store_true",
                    help="평가에 쓰인 페이지 전량 (manifest 기준)")
    ap.add_argument("--index", action="store_true",
                    help="한 페이지에서 훑어볼 HTML 인덱스도 만든다")
    ap.add_argument("--out", default=str(ROOT / "receipt_data/review/crop_overlay"))
    ap.add_argument("--width", type=int, default=1100, help="출력 폭 (작은 원본은 확대)")
    args = ap.parse_args()

    n_crop = 0
    if args.all:
        args.docs, n_crop = eval_docs()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cards = []

    for doc in args.docs:
        split, jf, imgp = find(doc)
        im = Image.open(imgp).convert("RGB")
        W, H = im.size
        pj = json.loads(jf.read_text(encoding="utf-8"))["result"]["elements"][0]["json"]
        pw, ph = pj["width"], pj["height"]
        # 좌표계가 90도 돌아간 장은 회전시킨 뒤 그려야 실제 크롭과 같아진다.
        rotated = is_transposed(W, H, pw, ph)
        if rotated:
            im = im.transpose(Image.ROTATE_270)
            W, H = im.size
        sx, sy = W / pw, H / ph

        # 원본이 작으면 선이 뭉개진다. 키워서 그린다.
        k = max(1.0, args.width / W)
        canvas = im.resize((int(W * k), int(H * k)), Image.LANCZOS)
        d = ImageDraw.Draw(canvas)

        n_ok = n_bad = 0
        for b in pj.get("parsing_res_list", []):
            if b.get("block_label") in SKIP:
                continue
            bb = as_bbox(b.get("block_bbox"))
            if not bb:
                continue
            # 보정 — 이게 실제로 크롭에 쓰인 좌표
            d.rectangle([bb[0] * sx * k, bb[1] * sy * k,
                         bb[2] * sx * k, bb[3] * sy * k],
                        outline=(0, 190, 90), width=max(2, int(3 * k)))
            n_ok += 1
            # 미보정 — json 좌표를 그대로 원본에 얹은 경우
            if bb[0] < W and bb[1] < H:
                d.rectangle([bb[0] * k, bb[1] * k,
                             min(bb[2], W) * k, min(bb[3], H) * k],
                            outline=(225, 60, 60), width=max(2, int(3 * k)))
                n_bad += 1

        p = out / f"{doc}_overlay.png"
        canvas.save(p)
        if args.index:
            cards.append({"doc": doc, "split": split, "W": W, "H": H,
                          "pw": pw, "ph": ph, "sx": sx, "sy": sy,
                          "n_ok": n_ok, "n_bad": n_bad, "rot": rotated,
                          "src": card_src(canvas)})
        if not args.all:
            print(f"{doc:<12} 이미지 {W}x{H} / json {pw}x{ph} / 배율 {sx:.3f},{sy:.3f} "
                  f"— 초록 {n_ok}개, 빨강 {n_bad}개 -> {p.name}")

    if args.all:
        print(f"페이지 {len(args.docs)}장 오버레이 생성 (크롭 {n_crop:,}건의 출처)")
    print(f"\n초록 = 보정 좌표(실제 크롭에 쓰임) / 빨강 = 미보정 좌표")
    print(f"저장: {out}")

    if args.index:
        idx = out.parent / "crop_overlay_index.html"
        idx.write_text(index_html(cards, n_crop or len(cards)), encoding="utf-8")
        print(f"인덱스: {idx}  ({idx.stat().st_size/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
