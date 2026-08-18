"""
VLM 문서파싱 LoRA 파인튜닝.

목적은 "좋은 모델"이 아니라 "학습이 도는 파이프라인"이다.
모델은 --model 로 교체 가능 (gemma-4-E2B-it / E4B-it / Qwen3.5-2B / 4B 등).
프로세서가 모델별 chat template 을 처리하므로 아키텍처가 바뀌어도 스크립트는 그대로다.

usage:
    python3 scripts/train_vlm.py --model /workspace/ml/models/gemma-4-E2B-it
    python3 scripts/train_vlm.py --model /workspace/ml/models/Qwen3.5-2B --epochs 3
    python3 scripts/train_vlm.py --model ... --max-steps 5      # 스모크 테스트
"""
import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          Trainer, TrainingArguments)

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# 잘림 집계. dataloader worker 마다 사본이 생기므로 합계는 워커별이고, 목적은
# "잘리고 있다"를 눈에 띄게 하는 것이다. 정확한 전수는 아래 preflight 가 낸다.
_TRUNC = {"n": 0, "dead": 0, "show": 20}


def preflight_len(rows, proc, max_len, max_px, tag):
    """학습 전에 토큰 길이를 미리 재서 잘릴 행을 알려준다.

    이미지 토큰까지 세어야 의미가 있다. 공식은 check_token_len.py 와 같은 것을
    쓰고, 그쪽에서 프로세서 실측과 대조해 검증해 두었다.
    """
    from check_token_len import img_tokens

    ip = proc.image_processor
    patch = getattr(ip, "patch_size", 16)
    merge = getattr(ip, "merge_size", 2)
    size = getattr(ip, "size", {}) or {}
    min_px = size.get("shortest_edge", 65536)
    pmax = size.get("longest_edge", 16_777_216)
    tok = proc.tokenizer

    stat = []
    for r in rows:
        p = ROOT / r["images"][0]
        if not p.exists():
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
        except OSError:
            continue
        it, _ = img_tokens(w, h, max_px, patch, merge, min_px, pmax)
        pt = len(tok(r["messages"][0]["content"].replace("<image>", ""))["input_ids"])
        at = len(tok(r["messages"][1]["content"])["input_ids"])
        stat.append((it + pt + at + 20, it, at, r.get("task", "")))
    if not stat:
        return
    stat.sort(reverse=True)
    n = len(stat)
    over = [s for s in stat if s[0] > max_len]
    p50, p99 = stat[n // 2][0], stat[max(0, int(n * 0.01))][0]
    print(f"   [{tag}] 토큰 p50 {p50:,} · p99 {p99:,} · 최대 {stat[0][0]:,} "
          f"/ max_len {max_len:,}", flush=True)
    if over:
        lost = sum(s[0] - max_len for s in over)
        print(f"   ⚠ {len(over):,}행 ({100*len(over)/n:.1f}%) 이 max_len 을 넘어 "
              f"정답 뒤쪽이 잘린다 — 버려지는 토큰 ≈ {lost:,}", flush=True)
        for t, it, at, task in over[:5]:
            print(f"       {task:24} 합계 {t:>7,} (이미지 {it:,} / 정답 {at:,})",
                  flush=True)
        print(f"   → max_len 을 {((stat[0][0] + 1023) // 1024) * 1024:,} 이상으로 "
              f"올리거나 --max-px 를 낮춰야 한다", flush=True)
    else:
        print(f"   잘리는 행 없음 (여유 {max_len - stat[0][0]:,} 토큰)", flush=True)


class DocDataset(torch.utils.data.Dataset):
    """(이미지, 마크다운) 쌍. assistant 응답 부분에만 loss 를 건다."""

    def __init__(self, rows, processor, max_len, max_px):
        self.rows, self.proc = rows, processor
        self.max_len, self.max_px = max_len, max_px

    def __len__(self):
        return len(self.rows)

    def _img(self, rel):
        im = Image.open(ROOT / rel).convert("RGB")
        w, h = im.size
        if w * h > self.max_px:                     # 긴 문서 페이지는 축소
            s = (self.max_px / (w * h)) ** 0.5
            im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
        return im

    def __getitem__(self, i):
        r = self.rows[i]
        user, asst = r["messages"][0], r["messages"][1]
        img = self._img(r["images"][0])

        # 프롬프트만 / 프롬프트+정답 두 번 토크나이즈해서 정답 구간만 학습
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user["content"].replace("<image>", "")}]}]
        # Qwen3.5 등은 기본이 사고 ON 이다. 끄지 않으면 학습 타겟(사고 없음)과
        # 프롬프트(사고 시작)가 어긋나, 모델이 '사고를 끄는 법'을 학습하게 된다.
        # 그 효과가 OCR 개선으로 오독되므로 학습·추론 모두 동일하게 끈다.
        try:
            prompt = self.proc.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                enable_thinking=False)
        except TypeError:
            prompt = self.proc.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)
        full = prompt + asst["content"] + (self.proc.tokenizer.eos_token or "")

        enc = self.proc(text=[full], images=[img], return_tensors="pt",
                        truncation=True, max_length=self.max_len)
        n_prompt = len(self.proc(text=[prompt], images=[img],
                                 return_tensors="pt")["input_ids"][0])

        # truncation=True 는 넘치는 만큼을 말없이 버린다. 잘리는 쪽은 정답 뒤쪽이라
        # 모델이 "중간에서 끊는 법"을 배우게 되는데 로그에 아무 흔적이 없다.
        # 실제로 잘린 행은 몇 건이든 눈에 띄게 남긴다.
        if len(enc["input_ids"][0]) >= self.max_len:
            _TRUNC["n"] += 1
            if n_prompt >= self.max_len:
                _TRUNC["dead"] += 1
                if _TRUNC["dead"] <= _TRUNC["show"]:
                    print(f"   ⚠ 프롬프트만으로 max_len({self.max_len}) 도달 — "
                          f"정답이 통째로 잘려 학습 신호가 없다: "
                          f"{r.get('doc_id')} [{r.get('task')}]", flush=True)
            elif _TRUNC["n"] <= _TRUNC["show"]:
                print(f"   ⚠ max_len({self.max_len}) 초과로 정답 뒤쪽 잘림: "
                      f"{r.get('doc_id')} [{r.get('task')}] "
                      f"프롬프트 {n_prompt}토큰", flush=True)

        # 텍스트 키만 배치 축을 벗긴다. Qwen VL 계열의 pixel_values 는
        # (패치수, 1536) 으로 배치 축이 아예 없어서 v[0] 을 걸면 첫 패치 한 줄만
        # 남는다 — 에러 없이 이미지가 사라지므로 키별로 나눠서 다룬다.
        item = {k: (v[0] if k in ("input_ids", "attention_mask") else v)
                for k, v in enc.items()}
        labels = item["input_ids"].clone()
        labels[:n_prompt] = -100                     # 프롬프트 구간 마스킹
        pad = self.proc.tokenizer.pad_token_id
        if pad is not None:
            labels[labels == pad] = -100
        item["labels"] = labels
        return item


def collate(batch, pad_id):
    out, keys = {}, batch[0].keys()
    maxlen = max(len(b["input_ids"]) for b in batch)
    for k in keys:
        if k in ("input_ids", "attention_mask", "labels"):
            fill = {"input_ids": pad_id, "attention_mask": 0, "labels": -100}[k]
            out[k] = torch.stack([
                torch.cat([b[k], torch.full((maxlen - len(b[k]),), fill,
                                            dtype=b[k].dtype)]) for b in batch])
        else:
            # pixel_values (패치수, 1536) / image_grid_thw (1, 3) 은 배치 축이 없다.
            # 이미지마다 패치 수가 달라 stack 이 아니라 0축으로 이어붙여야 한다.
            # (stack 을 먼저 시도하면 batch 1 일 때만 축이 하나 더 붙어 배치
            #  크기에 따라 shape 이 달라진다.)
            out[k] = torch.cat([b[k] for b in batch], dim=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", default=str(ROOT / "eval_dataset/train/train.jsonl"))
    ap.add_argument("--val", default=str(ROOT / "eval_dataset/train/val.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--skip-len-check", action="store_true",
                    help="학습 전 토큰 길이 점검 생략 (기본: 점검함)")
    # 문서 파싱은 글자당 픽셀 수가 정확도를 좌우한다. 축소는 곧 정보 손실이므로
    # 기본값을 크게 잡아 사실상 원본을 그대로 넣는다. OOM 날 때만 낮춘다.
    # (Qwen3.5는 네이티브 동적 해상도라 크롭 없이 패치 수만 늘어난다)
    ap.add_argument("--max-px", type=int, default=6_000_000)
    ap.add_argument("--lora-r", type=int, default=16)
    # ── 학습 범위 스위치 ──────────────────────────────────────────
    # 파트장 확인 요청 항목. 기본값은 기존 동작(ViT 동결 / aligner 학습 / LoRA)
    # 과 같으므로, 명시하지 않으면 이전 실험과 조건이 바뀌지 않는다.
    ap.add_argument("--tune", choices=["lora", "full"], default="lora",
                    help="lora=어댑터만 / full=전체 파라미터 (VRAM 확인 필수)")
    ap.add_argument("--train-vit", action="store_true",
                    help="vision encoder 본체도 학습 (기본: 동결)")
    ap.add_argument("--freeze-aligner", action="store_true",
                    help="aligner(merger/projector) 동결 (기본: 학습)")
    ap.add_argument("--freeze-projector", dest="freeze_aligner",
                    action="store_true", help="--freeze-aligner 의 옛 이름")
    # auto 는 빈 GPU 를 찾아 나눠 싣는다. cuda:0 고정이면 그 카드가 이미
    # 남의 프로세스로 차 있을 때 CPU 오프로드로 떨어져 수십 배 느려지거나 OOM 난다.
    # 특정 카드만 쓰려면 auto 를 두고 CUDA_VISIBLE_DEVICES 로 범위를 좁힌다.
    ap.add_argument("--device-map", default="auto",
                    help="auto / cuda:0 / balanced — 35B급은 auto 필요")
    # auto 는 층을 고르게 나눠 싣지만, 손실 계산(logits.float())은 lm_head 가 있는
    # 마지막 GPU 한 장에서만 일어난다. 그래서 반반으로 실으면 그 장만 터진다
    # (2장 실험에서 74GB 대 32GB 로 쏠렸다). 마지막 장의 예산을 줄여 층을 앞쪽으로
    # 밀어내면 logits 가 들어갈 자리가 생긴다.
    #   예: --max-memory 0=62GiB,1=38GiB
    ap.add_argument("--max-memory", default=None,
                    help="장치별 상한. 'idx=크기' 를 쉼표로. 마지막 장을 작게 준다")
    # MoE + gradient checkpointing 조합에서 backward 재계산 때 라우팅이 뒤집혀
    # CheckpointError 가 난다(전문가별 토큰 수가 forward 와 달라짐). 그때만 끈다.
    # use_reentrant=True 로 검사만 끄는 우회는 금물 — 틀린 그래디언트가 그대로 흐른다.
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="gradient checkpointing 끄기 (VRAM 더 씀)")
    # eval 은 val 전체를 batch 1 로 돈다. val 이 크면 학습보다 eval 이 더 오래 걸린다
    # (val 3,810건 = eval 1회에 90분). 프로브에서는 끝에 한 번만 보면 된다.
    ap.add_argument("--eval-steps", type=int, default=300)
    ap.add_argument("--save-steps", type=int, default=0,
                    help="N 스텝마다 체크포인트 저장 (0이면 epoch 단위)")
    # exp_003 에서 eval_loss 최저가 step 250 이었는데 save_steps=500 이라 저장조차
    # 안 됐고, 남은 1000/1124 는 둘 다 과적합 구간이었다. 마지막 스텝이 최선이라는
    # 보장이 없으므로 eval_loss 기준으로 best 를 되살려 최종 어댑터로 쓴다.
    # (save_steps 가 eval_steps 의 배수여야 HF 가 best 를 추적할 수 있다)
    ap.add_argument("--load-best", action="store_true",
                    help="eval_loss 최저 체크포인트를 최종 어댑터로 저장")
    ap.add_argument("--save-total-limit", type=int, default=2,
                    help="보관할 체크포인트 수. best 를 쓰려면 넉넉히 둔다")
    # 35B 2epoch 이 100시간이라 중간에 한 번은 끊긴다(컨테이너 재시작·OOM·정전).
    # 체크포인트에는 옵티마이저 상태와 스텝 수가 같이 들어 있으므로, 이어서
    # 돌리면 lr 스케줄과 데이터 순서까지 복원된다. 재시작 후 --resume auto 만
    # 붙이면 out 디렉토리에서 최신 checkpoint-N 을 찾아 그 지점부터 계속한다.
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="auto=out 의 최신 checkpoint / 경로 직접 지정도 가능")
    args = ap.parse_args()

    name = Path(args.model).name
    out = args.out or str(ROOT / f"ml/runs/{name}-lora")

    print(f"[1/5] 프로세서/모델 로드: {name}", flush=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    mm = None
    if args.max_memory:
        mm = {int(k): v for k, v in
              (kv.split("=", 1) for kv in args.max_memory.split(","))}
        print(f"   장치별 상한: {mm}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device_map,
        max_memory=mm, trust_remote_code=True)
    model.config.use_cache = False

    # aligner(비전->언어 연결층). 문서 이미지는 자연사진과 통계가 달라 이 층을
    # 열어주면 도메인 적응이 크게 좋아진다. 반면 vision encoder 본체는 얼려서
    # 시각 표현이 망가지는 것을 막는다 — 그래서 기본값이 서로 다르다.
    ALIGNER_KEYS = ("merger", "projector", "multi_modal_projector", "mm_projector")
    VIT_KEYS = ("visual", "vision_tower", "vision_model")

    def group(n):
        if any(k in n for k in ALIGNER_KEYS):
            return "aligner"
        if any(k in n for k in VIT_KEYS):
            return "vit"
        if n.startswith("mtp.") or ".mtp." in n:
            return "mtp"          # multi-token-prediction 헤드 — 학습에 안 씀
        return "llm"

    mode = ("full-finetuning" if args.tune == "full" else "LoRA")
    print(f"[2/5] 학습 범위: {mode} / "
          f"ViT {'학습' if args.train_vit else '동결'} / "
          f"aligner {'동결' if args.freeze_aligner else '학습'}", flush=True)

    if args.tune == "full":
        for n, p in model.named_parameters():
            g = group(n)
            p.requires_grad = (g == "llm"
                               or (g == "vit" and args.train_vit)
                               or (g == "aligner" and not args.freeze_aligner))
    else:
        # 접미사만으로 고르면 비전 타워의 커스텀 래퍼(gemma-4의 Gemma4ClippableLinear 등)까지
        # 잡혀서 peft가 거부한다. '순수 nn.Linear'만 전체 경로로 지정한다.
        #
        # Qwen3.6-A3B 같은 하이브리드 MoE 는 층의 3/4 이 linear_attn(게이트 델타넷)이라
        # q/k/v/o_proj 만 잡으면 전체 40층 중 10층에만 어댑터가 붙는다. 실제로 붙는
        # 파라미터가 LLM 의 1% 미만이 되어 학습이 안 된다. in_proj_qkv/in_proj_z/
        # out_proj 를 함께 잡아야 30개 linear_attn 층이 학습 대상에 들어온다.
        #
        # 라우팅 전문가(mlp.experts.*)는 전문가 256개가 3D 텐서 하나로 융합돼 있어
        # nn.Linear 가 아니다 → 아래 필터에 자동으로 걸리지 않는다. 라우터(mlp.gate)
        # 도 제외한다. MoE 는 라우터를 건드리면 전문가 분포가 무너진다.
        SUFFIX = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj",
                  "in_proj_qkv", "in_proj_z", "out_proj")
        targets = [n for n, mod in model.named_modules()
                   if isinstance(mod, torch.nn.Linear)
                   and n.split(".")[-1] in SUFFIX
                   and group(n) in (("llm", "vit") if args.train_vit else ("llm",))]
        if not targets:
            raise SystemExit("LoRA 대상 모듈을 못 찾음 — 모델 구조 확인 필요")

        aligner = sorted({n for n, _ in model.named_modules()
                          if n.split(".")[-1] in ALIGNER_KEYS})
        if args.freeze_aligner:
            aligner = []
        print(f"   LoRA 대상 {len(targets)}개 (예: {targets[0]})", flush=True)
        print(f"   aligner    {aligner if aligner else '(없음/미학습)'}", flush=True)

        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM", target_modules=targets,
            # modules_to_save 로 지정하면 peft 가 해당 모듈을 학습 대상으로 유지하고
            # 어댑터와 함께 저장한다. aligner 는 작아서 full-tune 해도 부담이 없다.
            modules_to_save=aligner or None))

        # vision encoder 본체는 명시적으로 동결 (aligner 는 위에서 살려둠)
        frozen = 0
        for n, p in model.named_parameters():
            if group(n) == "vit" and not args.train_vit and "lora_" not in n:
                if p.requires_grad:
                    p.requires_grad = False
                    frozen += 1
        print(f"   vision encoder 동결 파라미터 {frozen}개", flush=True)

    # 어디에 얼마나 붙었는지. "학습이 도는데 성능이 안 오른다"의 대부분은
    # 의도한 층에 어댑터가 안 붙은 경우라, 그룹별로 찍어 눈으로 확인한다.
    tot = tr_n = 0
    per = {}
    for n, p in model.named_parameters():
        g = group(n.replace("base_model.model.", ""))
        tot += p.numel()
        if p.requires_grad:
            tr_n += p.numel()
            per[g] = per.get(g, 0) + p.numel()
    print(f"   학습 파라미터 {tr_n / 1e6:,.1f}M / 전체 {tot / 1e9:,.2f}B "
          f"({tr_n / tot * 100:.3f}%)", flush=True)
    for g, v in sorted(per.items(), key=lambda x: -x[1]):
        print(f"      {g:<8} {v / 1e6:>9,.1f}M", flush=True)
    # AdamW 는 파라미터당 fp32 상태 2개 + fp32 마스터 사본 → 학습 파라미터 × 12B
    print(f"   옵티마이저 예상 VRAM ≈ {tr_n * 12 / 2**30:,.1f}GiB "
          f"(가중치 {tot * 2 / 2**30:,.1f}GiB 별도)", flush=True)

    print("[3/5] 데이터셋", flush=True)
    tr = DocDataset(load_jsonl(args.train), proc, args.max_len, args.max_px)
    va = DocDataset(load_jsonl(args.val), proc, args.max_len, args.max_px)
    print(f"   train {len(tr)} / val {len(va)}", flush=True)
    if not args.skip_len_check:
        preflight_len(tr.rows, proc, args.max_len, args.max_px, "train")
        preflight_len(va.rows, proc, args.max_len, args.max_px, "val")

    pad = proc.tokenizer.pad_token_id or 0
    targs = TrainingArguments(
        output_dir=out, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        bf16=True, logging_steps=1, save_total_limit=args.save_total_limit,
        **({"load_best_model_at_end": True, "metric_for_best_model": "eval_loss",
            "greater_is_better": False} if args.load_best else {}),
        # epoch 단위 저장은 긴 학습에서 위험하다. 35B 2epoch 이 99시간인데
        # 첫 저장까지 50시간이면 그 사이 한 번 죽는 것만으로 전부 날아간다
        # (실제로 48스텝에서 OOM 나고 아무것도 안 남았다).
        **({"save_strategy": "steps", "save_steps": args.save_steps}
           if args.save_steps else {"save_strategy": "epoch"}),
        # eval 없이 학습하면 과적합을 감지할 수 없다. 샘플 수가 적을수록 필수.
        eval_strategy="steps", eval_steps=args.eval_steps,
        per_device_eval_batch_size=1,
        gradient_checkpointing=not args.no_grad_checkpoint,
        # reentrant 방식은 backward 재계산 때 MoE 라우팅이 forward 와 달라져
        # CheckpointError 를 낸다(전문가별 토큰 수가 안 맞는다). 비reentrant 로
        # 켜야 라우팅이 보존된다. 켜지 않을 때는 무시되는 인자다.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[], remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=tr,
                      eval_dataset=va, data_collator=lambda b: collate(b, pad))

    # 재개 지점 결정. 이름의 스텝 수로 고르되(문자열 정렬은 9000>17000 이 된다),
    # 없으면 조용히 처음부터 도는 대신 멈춘다 — 이어달리려던 100시간짜리가
    # 0스텝부터 다시 시작하는 것을 아침에야 발견하는 사고를 막는다.
    resume = args.resume
    if resume == "auto":
        cks = sorted(Path(out).glob("checkpoint-*"),
                     key=lambda p: int(p.name.split("-")[-1]))
        if not cks:
            raise SystemExit(f"--resume auto 인데 체크포인트가 없다: {out}")
        resume = str(cks[-1])
    if resume:
        print(f"   재개: {resume}", flush=True)

    print("[4/5] 학습 시작", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    print("[5/5] 어댑터 저장", flush=True)
    trainer.save_model(out)
    proc.save_pretrained(out)
    print(f"완료 -> {out}")


if __name__ == "__main__":
    main()
