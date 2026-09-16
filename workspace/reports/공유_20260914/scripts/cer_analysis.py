"""CER 정체 원인 분석 — 자체 평가셋 921 텍스트 크롭, base~exp_011 크롭 단위 예측 대조.

질문:
 1. exp_004 이후 CER 이 왜 0.106 아래로 안 내려가나 — 평균을 만드는 게 소수 꼬리인가, 전반인가
 2. 모든 실험에서 계속 틀리는 크롭은 무엇이고, 어떤 유형인가 (오독 / 누락 / 형식 / GT 의심)
 3. 실험 간 개선·악화가 상쇄되는가 (churn)
"""
import json, re, sys, collections, statistics
from pathlib import Path
import Levenshtein as L

R = Path("/data/workspace/yyj/data/vlm_exp34/receipt_data/review")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)

def load(f, key):
    d = json.load(open(R / f))
    return {r["crop_id"]: (r, r[key]["pred"]) for r in d["rows"] if r["task"] == "text"}

srcs = {
    "base":    load("base_qwen36_full1019.json", None) if False else None,
}
# 키 이름이 파일마다 다르다 — 첫 행에서 pred 를 가진 dict 키를 찾는다
def load_auto(f, want=None):
    d = json.load(open(R / f)); rows = d["rows"]
    keys = [k for k, v in rows[0].items() if isinstance(v, dict) and "pred" in v]
    out = {}
    for k in keys:
        if want and k not in want: continue
        out[k] = {r["crop_id"]: (r, r[k]["pred"]) for r in rows if r["task"] == "text"}
    return out

M = {}
M.update({"base": list(load_auto("base_qwen36_full1019.json").values())[0]})
a5 = load_auto("all5_1019.json"); M.update({k: a5[k] for k in ["exp_004", "exp_005", "exp_006", "exp_007"]})
M["exp_008"] = list(load_auto("exp008_full1019.json").values())[0]
M["exp_009"] = list(load_auto("exp009_24646_full1019.json").values())[0]
M["exp_010"] = list(load_auto("exp010_full1019.json").values())[0]
M["exp_011"] = list(load_auto("exp011_full1019.json").values())[0]
order = ["base", "exp_004", "exp_005", "exp_006", "exp_007", "exp_008", "exp_009", "exp_010", "exp_011"]
ids = sorted(set.intersection(*(set(M[k]) for k in order)))
print("공통 텍스트 크롭", len(ids))

def cer(gt, pred):
    gt = gt or ""; pred = pred or ""
    m = max(len(gt), len(pred))
    return 0.0 if m == 0 else L.distance(gt, pred) / m

def norm(s):
    s = re.sub(r"\s+", "", s or "")
    s = re.sub(r"[^\w가-힣]", "", s)   # 기호·구두점 제거
    return s

C = {k: {i: cer(M[k][i][0]["gt"], M[k][i][1]) for i in ids} for k in order}
meta = {i: M["exp_011"][i][0] for i in ids}

# ── 1. 분포: 평균 vs 중앙값, 구간별 비중
dist = {}
for k in order:
    v = [C[k][i] for i in ids]
    dist[k] = dict(mean=statistics.mean(v), median=statistics.median(v),
                   zero=sum(1 for x in v if x == 0), le01=sum(1 for x in v if 0 < x <= 0.1),
                   le03=sum(1 for x in v if 0.1 < x <= 0.3), gt03=sum(1 for x in v if x > 0.3),
                   tail_share=sum(x for x in v if x > 0.3) / sum(v))
    print(f"{k:8s} mean {dist[k]['mean']:.4f} median {dist[k]['median']:.4f} | =0 {dist[k]['zero']:3d} | ≤0.1 {dist[k]['le01']:3d} | ≤0.3 {dist[k]['le03']:3d} | >0.3 {dist[k]['gt03']:3d} | >0.3 이 평균에 기여 {dist[k]['tail_share']*100:.0f}%")

# ── 2. 지속 실패 크롭 (exp_004 · 009 · 010 · 011 전부 CER > 0.3)
late = ["exp_004", "exp_009", "exp_010", "exp_011"]
persist = [i for i in ids if all(C[k][i] > 0.3 for k in late)]
ever_ok = [i for i in ids if any(C[k][i] == 0 for k in late)]
print(f"\n지속 실패(4실험 모두 CER>0.3): {len(persist)} / {len(ids)}  — 이들이 exp_011 평균에 기여: {sum(C['exp_011'][i] for i in persist)/sum(C['exp_011'].values())*100:.0f}%")

def classify(i):
    r = meta[i]; gt = r["gt"]; preds = {k: M[k][i][1] for k in late}
    # (a) 모든 후기 모델이 같은 답을 내는데 GT 와 다름 → GT 의심
    ps = set(norm(p) for p in preds.values())
    if len(ps) == 1 and norm(gt) != list(ps)[0] and L.distance(norm(gt), list(ps)[0]) <= max(2, len(norm(gt)) // 4):
        return "GT 의심(모델 4개 일치·GT만 다름)"
    p = preds["exp_011"]
    # (b) 형식 차이 — 공백·기호 제거하면 거의 같음
    if norm(gt) and cer(norm(gt), norm(p)) <= 0.1:
        return "형식 차이(공백·기호·줄바꿈)"
    # (c) 반복/폭주
    if len(p) > 3 * max(len(gt), 8):
        return "반복·폭주"
    # (d) 누락/추가 (길이 차 큼)
    if len(gt) and (len(p) < 0.6 * len(gt)):
        return "누락(출력이 짧음)"
    if len(gt) and (len(p) > 1.6 * len(gt)):
        return "추가(없는 내용 생성)"
    # (e) 숫자 오독 vs 한글 오독
    gd = re.findall(r"\d", gt); pd = re.findall(r"\d", p)
    gk = re.findall(r"[가-힣]", gt); pk = re.findall(r"[가-힣]", p)
    dn = L.distance("".join(gd), "".join(pd)); dk = L.distance("".join(gk), "".join(pk))
    if dn >= dk and gd:
        return "숫자 오독"
    if gk:
        return "한글 오독"
    return "영문·기타 오독"

cat = collections.Counter(classify(i) for i in persist)
print("\n지속 실패 유형:")
for k, v in cat.most_common(): print(f"  {k:32s} {v:3d}")

# 크기·길이와의 관계
def hbin(h): return "≤25px" if h <= 25 else ("26~40px" if h <= 40 else ("41~60px" if h <= 60 else ">60px"))
byh = collections.defaultdict(list)
for i in ids: byh[hbin(meta[i]["size"][1])].append(C["exp_011"][i])
print("\n크롭 높이별 exp_011 CER:")
for k in ["≤25px", "26~40px", "41~60px", ">60px"]:
    v = byh[k]; print(f"  {k:8s} n={len(v):3d} mean {statistics.mean(v):.3f} median {statistics.median(v):.3f}  >0.3: {sum(1 for x in v if x>0.3)}")
def lbin(n): return "≤5자" if n <= 5 else ("6~15자" if n <= 15 else ("16~30자" if n <= 30 else ">30자"))
byl = collections.defaultdict(list)
for i in ids: byl[lbin(len(meta[i]["gt"]))].append(C["exp_011"][i])
print("\nGT 길이별 exp_011 CER:")
for k in ["≤5자", "6~15자", "16~30자", ">30자"]:
    v = byl[k]; print(f"  {k:8s} n={len(v):3d} mean {statistics.mean(v):.3f} median {statistics.median(v):.3f}  >0.3: {sum(1 for x in v if x>0.3)}")

# ── 3. churn: exp_004 → exp_011 크롭 단위 변화
imp = [i for i in ids if C["exp_011"][i] < C["exp_004"][i] - 0.05]
wor = [i for i in ids if C["exp_011"][i] > C["exp_004"][i] + 0.05]
same = len(ids) - len(imp) - len(wor)
print(f"\nexp_004 → exp_011: 개선 {len(imp)} · 악화 {len(wor)} · 변화없음 {same}")
print(f"  개선분 합 {sum(C['exp_004'][i]-C['exp_011'][i] for i in imp):.1f} · 악화분 합 {sum(C['exp_011'][i]-C['exp_004'][i] for i in wor):.1f}")

# ── 4. 상한 추정: 형식 차이·GT 의심을 제외하면 CER 이 얼마인가
excl = set(i for i in persist if classify(i).startswith(("GT", "형식")))
# 전체 크롭에 대해 형식 정규화 CER
cn = {i: cer(norm(meta[i]["gt"]), norm(M["exp_011"][i][1])) for i in ids}
print(f"\nexp_011 CER 원본 {statistics.mean(C['exp_011'].values()):.4f} → 공백·기호 정규화 후 {statistics.mean(cn.values()):.4f}")

# 저장
json.dump({"dist": dist, "persist": [dict(crop_id=i, label=meta[i]["label"], size=meta[i]["size"], gt=meta[i]["gt"], type=classify(i),
                                         preds={k: M[k][i][1] for k in late}, cer={k: C[k][i] for k in late}) for i in persist],
           "cat": cat, "byh": {k: (len(v), statistics.mean(v)) for k, v in byh.items()}, "byl": {k: (len(v), statistics.mean(v)) for k, v in byl.items()},
           "churn": dict(improved=len(imp), worsened=len(wor), same=same), "norm_cer_exp011": statistics.mean(cn.values())},
          open(OUT / "cer_analysis.json", "w"), ensure_ascii=False, indent=1)
# 예시 출력
print("\n지속 실패 예시 (유형별 2건):")
seen = collections.Counter()
for i in persist:
    t = classify(i)
    if seen[t] >= 2: continue
    seen[t] += 1
    print(f"  [{t}] {i} label={meta[i]['label']} size={meta[i]['size']}\n     GT : {meta[i]['gt'][:70]!r}\n     011: {M['exp_011'][i][1][:70]!r}\n     004: {M['exp_004'][i][1][:70]!r}")

# ══════════════ 5. 수작업 분류 (76건 전수 육안 확인, 2026-09-14) 반영한 반사실 CER
GT_BAD = {"test_receipt01_2","test_receipt05_10","test_receipt13_2","test_receipt13_4","test_receipt13_6","test_receipt16_2","test_receipt25_7",
          "test_receipt29_0","test_receipt35_3","test_receipt59_0","test_receipt70_4","test_receipt87_2","test_receipt88_16","test_receipt96_1",
          "val_receipt10_13","val_receipt10_17","val_receipt30_2","val_receipt30_3","val_receipt32_4","val_receipt39_3",
          "val_receipt40_5","val_receipt42_5","val_receipt45_1","val_receipt69_16","val_receipt79_6","val_receipt94_6"}
FORMAT = {"val_receipt30_0"}
ROT_DOCS = {"receipt34", "receipt67"}          # 90° 회전된 원본 2장 (val)
MODEL_ERR = {"test_receipt03_0","test_receipt14_0","test_receipt14_1","test_receipt65_9","test_receipt99_1","test_receipt99_3","val_receipt30_1","val_receipt39_9","val_receipt58_6"}
rot = [i for i in ids if meta[i]["doc_id"] in ROT_DOCS]
print(f"\n회전 문서 2장의 텍스트 크롭 수: {len(rot)}  (지속실패 76 중 {sum(1 for i in persist if meta[i]['doc_id'] in ROT_DOCS)})")
manual = collections.Counter("회전 이미지(원본 2장)" if meta[i]["doc_id"] in ROT_DOCS else "평가셋 GT/bbox 오류" if i in GT_BAD else "형식 차이" if i in FORMAT else "모델 오독·누락(흐림·잘림 포함)" for i in persist)
print("수작업 분류:", dict(manual))

def mean_cer(k, excl=set(), fix=set()):
    v = [0.0 if i in fix else C[k][i] for i in ids if i not in excl]
    return statistics.mean(v), len(v)
print("\n반사실 CER (텍스트 921 크롭):")
print(f"{'실험':8s} {'원본':>7s} {'회전2장 제외':>12s} {'+GT오류 수정':>12s} {'+형식 정규화':>12s}")
cf = {}
for k in ["exp_004", "exp_009", "exp_010", "exp_011"]:
    a, _ = mean_cer(k); b, n = mean_cer(k, excl=set(rot)); c, _ = mean_cer(k, excl=set(rot), fix=GT_BAD)
    v = [0.0 if i in GT_BAD else cer(norm(meta[i]["gt"]), norm(M[k][i][1])) for i in ids if i not in set(rot)]
    d = statistics.mean(v)
    cf[k] = (a, b, c, d); print(f"{k:8s} {a:7.4f} {b:12.4f} {c:12.4f} {d:12.4f}   (n={n})")

# ══════════════ 6. exp_011 전체 오류(CER>0) 유형 분포 — 지속 실패뿐 아니라 전부
allerr = [i for i in ids if C["exp_011"][i] > 0]
def typ_all(i):
    if meta[i]["doc_id"] in ROT_DOCS: return "회전 이미지"
    if i in GT_BAD: return "평가셋 GT/bbox 오류(확인됨)"
    gt = meta[i]["gt"]; p = M["exp_011"][i][1]
    if norm(gt) == norm(p): return "형식 차이(공백·기호·줄바꿈만 다름)"
    if norm(gt) and cer(norm(gt), norm(p)) <= 0.1: return "형식 차이 + 1~2자 오독"
    return classify(i).split("(")[0] if not classify(i).startswith("GT") else "GT 의심(모델 일치)"
ta = collections.Counter(typ_all(i) for i in allerr)
contrib = collections.defaultdict(float)
for i in allerr: contrib[typ_all(i)] += C["exp_011"][i]
tot = sum(C["exp_011"].values())
print(f"\nexp_011 오류 크롭 {len(allerr)}건 유형별 — 건수 / CER 합 기여율:")
for k, v in ta.most_common(): print(f"  {k:36s} {v:4d}건  {contrib[k]/tot*100:5.1f}%")
json.dump({"manual": dict(manual), "counterfactual": cf, "all_err_types": {k: (v, contrib[k]/tot) for k, v in ta.items()}, "n_rot": len(rot)},
          open(OUT / "cer_counterfactual.json", "w"), ensure_ascii=False, indent=1)
