#!/usr/bin/env python3
"""파싱 엔진(VLM 기반)의 환각·반복 루프를 찾아낸다.

kid_IMG00071 에서 드러난 유형:
  ① 글자를 옮기지 않고 이미지를 영어로 '설명'  ("The image displays a …")
  ② 숫자 증가 패턴에 갇혀 무한 생성          ("2019-2020 2020-2021 … 2401-2402")

기존 '같은 줄 반복' 규칙은 줄 단위라 한 줄 안에서 도는 루프를 못 잡는다.
여기서는 줄 안쪽의 토큰 반복과 메타 서술 문구를 본다.

사용:
  python3 detect_hallucination.py --file <택소노미> [--out drop.txt] [--show 5]
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

# 파서가 전사(轉寫) 대신 설명·변명을 시작할 때 나오는 문구
META = [
    "the image displays", "the image shows", "the image appears",
    "appears to be", "without additional context", "for the purpose of this task",
    "it is challenging to determine", "if this is part of", "i cannot",
    "unable to read", "the extracted text from the image",
    "further information would be necessary", "possibly forming a pattern",
]

# 연속 증가 수열 (연도 범위, 일련번호 등)
SEQ = re.compile(r"(?:\d{3,4}\s*[-–]\s*\d{3,4}\s+){6,}")


def token_loop(text, min_rep=8):
    """한 줄 안에서 같은 토큰 형태가 반복되는지. (형태 = 숫자를 #으로 치환)"""
    worst = 0
    for line in text.split("\n"):
        toks = line.split()
        if len(toks) < min_rep:
            continue
        shapes = [re.sub(r"\d+", "#", t) for t in toks]
        c = collections.Counter(shapes).most_common(1)
        if c and c[0][1] > worst:
            worst = c[0][1]
    return worst


def check(ans):
    r = []
    low = ans.lower()

    hits = [m for m in META if m in low]
    if hits:
        r.append(f"메타 서술({hits[0][:24]}…)")

    m = SEQ.search(ans)
    if m:
        n = len(re.findall(r"\d{3,4}\s*[-–]\s*\d{3,4}", m.group(0)))
        r.append(f"증가 수열 {n}회")

    loop = token_loop(ans)
    if loop >= 25:
        r.append(f"토큰 반복 {loop}회")

    # 영어 비중이 비정상적으로 높은 한국 영수증
    kor = len(re.findall(r"[가-힣]", ans))
    eng_words = len(re.findall(r"\b[A-Za-z]{4,}\b", ans))
    if kor > 20 and eng_words > 40:
        r.append(f"영문 단어 {eng_words}개 혼입")

    # 코드펜스가 정답에 남아 있으면 파서 출력이 그대로 샌 것
    if "```" in ans:
        r.append("코드펜스 잔여")

    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", help="검출된 파일명을 저장할 텍스트 경로")
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()

    rows = json.loads(Path(a.file).read_text(encoding="utf-8"))
    hit, cnt = [], collections.Counter()
    for r in rows:
        ans = r["completion"]["chosen"]
        why = check(ans)
        if why:
            hit.append((r["prompt"]["image"][0], len(ans), why))
            for w in why:
                cnt[w.split("(")[0].split(" ")[0]] += 1

    print(f"검사 {len(rows)}건 → 환각·반복 의심 {len(hit)}건\n")
    if cnt:
        print("유형별:")
        for k, v in cnt.most_common():
            print(f"   {k:14s} {v:4d}")
    print()
    for name, n, why in sorted(hit, key=lambda x: -x[1])[:a.show]:
        print(f"   {name:22s} {n:6d}자  {' / '.join(why)}")
    if len(hit) > a.show:
        print(f"   … 외 {len(hit)-a.show}건")

    if a.out and hit:
        Path(a.out).write_text("\n".join(n for n, _, _ in hit), encoding="utf-8")
        print(f"\n목록 → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
