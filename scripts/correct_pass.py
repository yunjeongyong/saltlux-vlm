"""
LLM 보정 패스 실험.

가설: 파서는 픽셀 전사에 강하고(숫자 161셀 100%), LLM은 문맥 판단에 강하다(한글 용어).
     그러면 파서 출력의 한글 계정명만 LLM으로 교정하면, 숫자를 건드리지 않고
     한글 오류를 줄일 수 있다.

측정: GT 대비 (a) 고친 건수 (b) 멀쩡한 걸 망가뜨린 건수.
     (b)가 크면 이 접근은 실패다. 반드시 양쪽을 같이 본다.

이미지를 주지 않는다. 언어 지식만으로 고쳐지는지가 실험의 요점.
"""
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "Table Sample.json"
GT = ROOT / "eval_dataset/annotations/doc001_saltlux_bs_page000.json"
URL = "http://localhost:15010/v1/chat/completions"
MODEL = "gemma-4-31B-it"

PROMPT = """다음은 한국 기업 재무상태표에서 OCR로 추출한 계정과목명 목록입니다.
OCR 오인식으로 잘못 표기된 항목이 있을 수 있습니다.

규칙:
1. 한국 기업회계기준(K-IFRS)의 표준 계정과목명과 대조하여, 실재하지 않는 계정명만 고치세요.
2. 실재하는 정상 계정명은 절대 바꾸지 마세요.
3. 확신이 없으면 원문 그대로 두세요.
4. 순서와 개수를 그대로 유지하세요.

출력 형식: 각 줄에 `원문<TAB>교정문` 만 출력. 설명 금지.

목록:
{items}"""


def call(prompt, timeout=600):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main():
    doc = json.loads(DOC.read_text(encoding="utf-8"))
    html = next(p["block_content"] for p in doc["parsing_res_list"]
                if p["block_label"] == "table")
    rows = [re.sub(r"<[^>]+>", "", c).strip()
            for tr in re.findall(r"<tr>(.*?)</tr>", html, re.S)
            for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)[:1]]
    names = [r for r in rows if r and not re.match(r"^제\s*\d+\s*기", r)]

    gt = json.loads(GT.read_text(encoding="utf-8"))
    gt_names = [r["계정과목"] for r in gt["ground_truth_table"]["rows"]]
    assert len(names) == len(gt_names), f"길이 불일치 {len(names)} vs {len(gt_names)}"

    print(f"계정명 {len(names)}개 -> {MODEL} 보정 요청 중...", flush=True)
    out = call(PROMPT.format(items="\n".join(names)))

    fixed = {}
    for line in out.splitlines():
        if "\t" in line:
            a, b = line.split("\t", 1)
            fixed[a.strip()] = b.strip()
        elif "->" in line:
            a, b = line.split("->", 1)
            fixed[a.strip()] = b.strip()

    repaired, broken, missed, untouched = [], [], [], 0
    for parsed, truth in zip(names, gt_names):
        corrected = fixed.get(parsed, parsed)
        was_wrong = parsed != truth
        now_right = corrected == truth
        if was_wrong and now_right:
            repaired.append((parsed, corrected))
        elif was_wrong and not now_right:
            missed.append((parsed, corrected, truth))
        elif not was_wrong and not now_right:
            broken.append((parsed, corrected))
        else:
            untouched += 1

    total_err = sum(1 for p, t in zip(names, gt_names) if p != t)
    print(f"\n{'='*66}")
    print(f"원래 오류 {total_err}건 / 정상 {len(names)-total_err}건")
    print(f"{'='*66}")
    print(f"  ✅ 고침      : {len(repaired)}건")
    print(f"  ⚠ 못 고침    : {len(missed)}건")
    print(f"  ❌ 망가뜨림  : {len(broken)}건   <<< 이게 핵심 리스크")
    print(f"  ─ 그대로     : {untouched}건")
    print()
    for a, b in repaired:
        print(f"  ✅ {a}  ->  {b}")
    for a, b, t in missed:
        print(f"  ⚠ {a}  ->  {b}   (정답 {t})")
    for a, b in broken:
        print(f"  ❌ {a}  ->  {b}   (원래 맞았음)")

    net = len(repaired) - len(broken)
    print(f"\n순효과: {net:+d}건  ({'개선' if net > 0 else '악화' if net < 0 else '무변화'})")
    print(f"보정 후 오류: {total_err}건 -> {total_err - len(repaired) + len(broken)}건")

    Path(ROOT / "eval_dataset/reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "eval_dataset/reports/correct_pass.json").write_text(json.dumps({
        "model": MODEL, "total_names": len(names), "original_errors": total_err,
        "repaired": repaired, "missed": missed, "broken": broken,
        "net": net, "raw_output": out,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
