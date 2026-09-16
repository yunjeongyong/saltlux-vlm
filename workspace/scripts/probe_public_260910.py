#!/usr/bin/env python3
"""수집한 공개 데이터의 스키마를 훑는다 — exp016 변환기를 쓰기 위한 사전 조사.

각 데이터셋 디렉터리에서
  - 파일 구성(확장자별 개수)
  - parquet 이면 컬럼명·dtype·첫 행 요약
  - jsonl/json 이면 키
  - 이미지 개수
을 뽑아 _schema_probe.json 으로 남긴다.

usage:
  python3 scripts/probe_public_260910.py                # 전체
  python3 scripts/probe_public_260910.py korean/ko_vdr_train tbl_semtabnet
"""
import collections, json, os, sys

ROOT = "/data/workspace/yyj/data/external/public_260910"
IMG = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def probe_parquet(fp, limit_cols=40):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {"error": "pyarrow 없음"}
    try:
        f = pq.ParquetFile(fp)
        sch = f.schema_arrow
        out = {"rows": f.metadata.num_rows,
               "cols": [{"name": n, "type": str(sch.field(n).type)[:60]}
                        for n in sch.names[:limit_cols]]}
        # 첫 행 요약
        b = next(f.iter_batches(batch_size=1))
        row = {}
        for n in sch.names[:limit_cols]:
            v = b.column(n)[0].as_py()
            if isinstance(v, (bytes, bytearray)):
                row[n] = f"<bytes {len(v)}>"
            elif isinstance(v, dict):
                row[n] = {k: (f"<bytes {len(x)}>" if isinstance(x, (bytes, bytearray))
                              else str(x)[:120]) for k, x in list(v.items())[:6]}
            else:
                row[n] = str(v)[:300]
        out["first_row"] = row
        return out
    except Exception as e:
        return {"error": repr(e)[:200]}


def probe_jsonl(fp, n=1):
    try:
        with open(fp, encoding="utf-8") as f:
            first = f.readline()
        r = json.loads(first)
        return {"keys": list(r)[:40],
                "first_row": {k: str(v)[:200] for k, v in list(r.items())[:20]}}
    except Exception as e:
        return {"error": repr(e)[:200]}


def probe_dir(d):
    ext = collections.Counter()
    parquets, jsonls, nimg, nbytes = [], [], 0, 0
    for r, dirs, fs in os.walk(d):
        dirs[:] = [x for x in dirs if x not in (".cache", ".git")]
        for f in fs:
            fp = os.path.join(r, f)
            e = os.path.splitext(f)[1].lower()
            ext[e] += 1
            try: nbytes += os.path.getsize(fp)
            except OSError: pass
            if e in IMG: nimg += 1
            elif e == ".parquet" and len(parquets) < 1: parquets.append(fp)
            elif e in (".jsonl", ".json") and len(jsonls) < 3 and not f.startswith("_"):
                jsonls.append(fp)
    rec = {"files": sum(ext.values()), "bytes": nbytes, "images": nimg,
           "ext": dict(ext.most_common(10))}
    if parquets:
        rec["parquet"] = {"file": os.path.relpath(parquets[0], d), **probe_parquet(parquets[0])}
    for j in jsonls:
        rec.setdefault("json", {})[os.path.relpath(j, d)] = probe_jsonl(j)
    return rec


def main():
    targets = sys.argv[1:]
    if not targets:
        targets = sorted(x for x in os.listdir(ROOT)
                         if os.path.isdir(os.path.join(ROOT, x)) and not x.startswith("_"))
        targets += ["korean/" + x for x in sorted(os.listdir(os.path.join(ROOT, "korean")))
                    if os.path.isdir(os.path.join(ROOT, "korean", x))]
        targets = [t for t in targets if t != "korean"]
    out = {}
    for t in targets:
        d = os.path.join(ROOT, t)
        if not os.path.isdir(d):
            print(f"[SKIP] {t} — 없음"); continue
        print(f"[PROBE] {t} ...", flush=True)
        out[t] = probe_dir(d)
        r = out[t]
        print(f"         {r['files']:6d} files  {r['bytes']/1e9:7.2f}GB  img={r['images']:6d}  {list(r['ext'])[:5]}")
    p = os.path.join(ROOT, "_schema_probe.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"\n저장: {p}")


if __name__ == "__main__":
    main()
