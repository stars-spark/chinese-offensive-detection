"""D5 带 P2 提示词的因果语言模型 LoRA 微调。

每条样本先用 ``llm_zeroshot.build_messages("P2", 正文)`` 加 generation prompt
得到提示串，再拼上答案词元“是/否”；labels 在提示部分全为 -100，只在答案
词元处计损失。训练超参数、精度、梯度检查点、NaN 停止与种子沿用 D4。

推理与 D3 相同：左填充，取最后一个位置的 logits，比较“是”“否”两个词元，
阈值固定 0.5。结果名 ``cold_llmftp_{name}_s{seed}``，只保存 LoRA 权重。

用法：
    python src/finetune_llm_prompt.py --model_path "$QWEN7B" --name qwen7b \
        --seed 0 --grad_ckpt --extra_eval "$EXTRA" --save_preds \
        --save_val_preds --bench
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM

from data import (
    DATASETS,
    clean_text,
    deduplicate_cold_splits,
    eval_set_available,
    load_eval_set,
    load_splits,
    parse_extra_eval,
)
from engine import classification_metrics, seed_everything
from finetune_bert import fine_grained_stats
from finetune_llm import (
    BASE_DTYPES,
    BENCH_BATCH,
    CLASS_NAMES,
    MAX_CHARS,
    TARGET_MODULES,
    NonFiniteLoss,
    amp_context,
    is_trainable_adapter,
    load_tokenizer,
    make_scheduler,
    parameter_counts,
    truncation_stats,
    versions,
)
from llm_zeroshot import (
    build_messages,
    clean_comment,
    derive_name,
    single_token_id,
    yes_prob_from_logits,
)
from result_io import (
    now_iso,
    preds_path,
    result_json_path,
    save_json,
    save_predictions,
    setup_logger,
    val_preds_path,
)

ROOT = Path(__file__).resolve().parents[1]
PROMPT_KEY = "P2"
THRESHOLD = 0.5
MAX_LENGTH = 1024


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="D5 带提示词的 LoRA 微调")
    p.add_argument("--model_path", required=True, help="本地基座模型目录")
    p.add_argument("--name", default="", help="结果名，如 qwen7b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--warmup_frac", type=float, default=0.03)
    p.add_argument("--grad_ckpt", action="store_true",
                   help="开梯度检查点；7B 用")
    p.add_argument("--base_dtype", choices=sorted(BASE_DTYPES), default="float16")
    p.add_argument("--max_len", type=int, default=MAX_LENGTH,
                   help="提示词加答案词元的总长度上限，默认 1024")
    p.add_argument("--extra_eval", default="",
                   help="逗号分隔的评测集名，训练后逐个评测")
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--save_val_preds", action="store_true")
    p.add_argument("--train_limit", type=int, default=None,
                   help="只取训练集前 N 条；默认不限，冒烟用")
    p.add_argument("--test_limit", type=int, default=None,
                   help="只评测测试集前 N 条；默认不限，冒烟用")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default=None, help="默认 results/")
    p.add_argument("--bench", action="store_true", help="评测后测速")
    return p


def validate(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("epochs", "batch_size", "grad_accum", "lora_r", "lora_alpha",
                 "max_len"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name} 必须为正整数")
    if args.lr <= 0:
        raise SystemExit("--lr 必须为正")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise SystemExit("--lora_dropout 必须在 [0, 1) 内")
    if not 0.0 <= args.warmup_frac <= 1.0:
        raise SystemExit("--warmup_frac 必须在 [0, 1] 内")
    if args.max_len < 2:
        raise SystemExit("--max_len 至少为 2")
    for name in ("train_limit", "test_limit"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            raise SystemExit(f"--{name} 必须为正整数或省略")
    args.name = args.name or derive_name(args.model_path)
    return args


def load_base_model(model_path: str, base_dtype: str):
    kwargs = {"local_files_only": True,
              "torch_dtype": BASE_DTYPES[base_dtype]}
    try:
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except TypeError:
        kwargs.pop("torch_dtype")
        kwargs["dtype"] = BASE_DTYPES[base_dtype]
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def wrap_lora(model, r: int, alpha: int, dropout: float):
    """因果语言模型 LoRA：任务类型 CAUSAL_LM，不需要 modules_to_save。"""
    config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=list(TARGET_MODULES), task_type="CAUSAL_LM",
        bias="none",
    )
    wrapped = get_peft_model(model, config)
    for name, param in wrapped.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
        elif is_trainable_adapter(name):
            raise RuntimeError(f"适配器参数被冻结：{name}")
    return wrapped


def encode_example(tokenizer, text: str, label: int, yes_id: int,
                   no_id: int, max_len: int = MAX_LENGTH):
    """返回 (input_ids, labels)：提示全 -100，答案词元处为标签。"""
    prompt = tokenizer.apply_chat_template(
        build_messages(PROMPT_KEY, text),
        tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_id = yes_id if int(label) == 1 else no_id
    if len(prompt_ids) >= max_len:
        prompt_ids = prompt_ids[:max_len - 1]
    input_ids = list(prompt_ids) + [answer_id]
    labels = [-100] * len(prompt_ids) + [answer_id]
    return input_ids, labels


def pad_right_batch(sequences: list[list[int]], label_sequences: list[list[int]],
                    pad_id: int):
    """右侧补齐，返回 input_ids / attention_mask / labels 三个张量。

    labels 的 padding 位置填 -100，与原序列逐位置对齐。
    """
    width = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(sequences), width), dtype=torch.long)
    labels = torch.full((len(sequences), width), -100, dtype=torch.long)
    for row, (seq, lab) in enumerate(zip(sequences, label_sequences)):
        n = len(seq)
        input_ids[row, :n] = torch.tensor(seq, dtype=torch.long)
        attention[row, :n] = 1
        labels[row, :n] = torch.tensor(lab, dtype=torch.long)
    return input_ids, attention, labels


class PromptDataset(Dataset):
    def __init__(self, texts: list[str], labels):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        return {"text": self.texts[i], "label": int(self.labels[i]), "index": i}


def make_collate(tokenizer, yes_id: int, no_id: int, max_len: int):
    pad_id = tokenizer.pad_token_id

    def collate(batch: list[dict]) -> dict:
        sequences, label_sequences = [], []
        for item in batch:
            seq, lab = encode_example(tokenizer, item["text"], item["label"],
                                      yes_id, no_id, max_len)
            sequences.append(seq)
            label_sequences.append(lab)
        input_ids, attention, labels = pad_right_batch(
            sequences, label_sequences, pad_id)
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "index": torch.tensor([b["index"] for b in batch],
                                  dtype=torch.long),
        }
    return collate


def make_loader(texts: list[str], labels, tokenizer, yes_id: int, no_id: int,
                max_len: int, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        PromptDataset(texts, labels), batch_size=batch_size, shuffle=shuffle,
        num_workers=0,
        collate_fn=make_collate(tokenizer, yes_id, no_id, max_len),
    )


@torch.inference_mode()
def score_texts(model, tokenizer, texts: list[str], batch_size: int,
                device: torch.device, base_dtype: str, yes_id: int,
                no_id: int, max_len: int = MAX_LENGTH) -> np.ndarray:
    """按 D3 的方式算“是”的概率：左填充，取最后一个位置的 logits。"""
    model.eval()
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    out = []
    try:
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            chats = [
                tokenizer.apply_chat_template(
                    build_messages(PROMPT_KEY, text),
                    tokenize=False, add_generation_prompt=True,
                )
                for text in chunk
            ]
            enc = tokenizer(chats, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len)
            enc.pop("token_type_ids", None)
            enc = {k: v.to(device) for k, v in enc.items()}
            with amp_context(device, base_dtype):
                logits = model(**enc).logits[:, -1, :]
            out.append(yes_prob_from_logits(logits, yes_id, no_id)
                       .float().cpu().numpy())
    finally:
        tokenizer.padding_side = old_side
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


def evaluate_prompt(model, tokenizer, texts: list[str], labels,
                    batch_size: int, device: torch.device, base_dtype: str,
                    yes_id: int, no_id: int, max_len: int = MAX_LENGTH):
    """返回 (平均交叉熵, 准确率, 预测, 标签, 两类概率矩阵)。"""
    labels = np.asarray(labels)
    probs = score_texts(model, tokenizer, texts, batch_size, device,
                        base_dtype, yes_id, no_id, max_len)
    preds = (probs >= THRESHOLD).astype(np.int64)
    p_true = np.where(labels == 1, probs, 1.0 - probs)
    p_true = np.clip(p_true, 1e-7, 1.0 - 1e-7)
    loss = float(-np.log(p_true).mean()) if len(labels) else 0.0
    acc = float((preds == labels).mean()) if len(labels) else 0.0
    matrix = np.column_stack([1.0 - probs, probs])
    return loss, acc, preds, labels, matrix


def train_model(model, train_loader, val_texts, val_labels, tokenizer, device,
                base_dtype: str, yes_id: int, no_id: int, max_len: int,
                epochs: int, lr: float, grad_accum: int, warmup_frac: float,
                grad_clip: float, logger) -> dict:
    """训练；答案词元损失的 NaN/Inf 立刻抛 NonFiniteLoss。"""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = max(1, epochs * steps_per_epoch)
    scheduler = make_scheduler(optimizer, total_steps, warmup_frac)
    scaler = torch.amp.GradScaler(device.type, enabled=base_dtype == "float16")

    history, step_losses = [], []
    global_step = micro_step = 0
    total_train_sec = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        epoch_loss, epoch_n = 0.0, 0
        window_loss, window_n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for micro, batch in enumerate(train_loader, start=1):
            micro_step += 1
            inputs = {k: v.to(device) for k, v in batch.items()
                      if k in ("input_ids", "attention_mask", "labels")}
            with amp_context(device, base_dtype):
                loss = model(**inputs).loss
            loss_value = float(loss.detach().float().item())
            if not np.isfinite(loss_value):
                raise NonFiniteLoss(global_step + 1, micro_step,
                                    loss_value, history)
            scaler.scale(loss / grad_accum).backward()
            batch_n = int(batch["input_ids"].size(0))
            epoch_loss += loss_value * batch_n
            epoch_n += batch_n
            window_loss += loss_value
            window_n += 1
            at_boundary = (micro % grad_accum == 0) or (micro == len(train_loader))
            if not at_boundary:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if global_step % 50 == 0:
                avg = window_loss / max(window_n, 1)
                step_losses.append({"step": global_step, "loss": avg})
                logger.info("step %d/%d  loss=%.4f  lr=%.2e",
                            global_step, total_steps, avg,
                            scheduler.get_last_lr()[0])
                window_loss, window_n = 0.0, 0
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_sec = time.perf_counter() - start
        total_train_sec += train_sec
        peak_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2
                   if device.type == "cuda" else None)
        val_loss, val_acc, _, _, _ = evaluate_prompt(
            model, tokenizer, val_texts, val_labels, 16, device, base_dtype,
            yes_id, no_id, max_len)
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / max(epoch_n, 1),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "train_time_sec": train_sec,
            "gpu_peak_memory_mb": peak_mb,
        })
        logger.info("epoch %d/%d train_loss=%.4f val_loss=%.4f val_acc=%.4f "
                    "time=%.1fs gpu_peak=%s", epoch, epochs,
                    history[-1]["train_loss"], val_loss, val_acc, train_sec,
                    "—" if peak_mb is None else f"{peak_mb:.0f}MB")
    return {
        "history": history,
        "step_losses": step_losses,
        "steps": global_step,
        "total_steps": total_steps,
        "total_train_sec": total_train_sec,
    }


def predict_texts(model, tokenizer, texts: list[str], batch_size: int,
                  device: torch.device, base_dtype: str, yes_id: int,
                  no_id: int, max_len: int = MAX_LENGTH):
    labels = np.zeros(len(texts), dtype=np.int64)
    _, _, preds, _, probs = evaluate_prompt(
        model, tokenizer, texts, labels, batch_size, device, base_dtype,
        yes_id, no_id, max_len)
    return preds, probs


def run_bench(model, tokenizer, texts: list[str], device: torch.device,
              base_dtype: str, yes_id: int, no_id: int,
              max_len: int = MAX_LENGTH) -> dict:
    """与 llm_zeroshot 相同：batch 16，预热 2 批，前 1000 条计时 3 次。"""
    texts = [clean_comment(t) for t in texts][:1000]
    if not texts:
        return {"n": 0, "ms_per_1k": None, "reps_sec": []}
    warm = min(2 * BENCH_BATCH, len(texts))
    predict_texts(model, tokenizer, texts[:warm], BENCH_BATCH, device,
                  base_dtype, yes_id, no_id, max_len)
    reps = []
    for _ in range(3):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        predict_texts(model, tokenizer, texts, BENCH_BATCH, device,
                      base_dtype, yes_id, no_id, max_len)
        if device.type == "cuda":
            torch.cuda.synchronize()
        reps.append(time.perf_counter() - start)
    median_sec = float(np.median(reps))
    return {
        "n": len(texts),
        "batch_size": BENCH_BATCH,
        "warmup_batches": 2,
        "reps": 3,
        "reps_sec": reps,
        "median_sec": median_sec,
        "ms_per_1k": median_sec / len(texts) * 1_000_000.0,
        "dtype": base_dtype,
        "device": str(device),
    }


def config_dict(args, tokenizer, n_train: int, n_val: int, n_test: int) -> dict:
    return {
        **vars(args),
        "num_classes": 2,
        "vocab_size": tokenizer.vocab_size,
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "amp": args.base_dtype == "float16",
        "prompt": PROMPT_KEY,
        "max_chars": MAX_CHARS,
        "max_length": args.max_len,
        "threshold": THRESHOLD,
        "lora_target_modules": list(TARGET_MODULES),
        "modules_to_save": [],
        "bench_batch_size": BENCH_BATCH,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }


def base_payload(args, config: dict, trainable: int, total: int,
                 history: list, steps: int, total_train_sec: float,
                 removed: dict) -> dict:
    epochs_run = len(history)
    best_val = history[-1]["val_accuracy"] if history else None
    peaks = [h["gpu_peak_memory_mb"] for h in history
             if h["gpu_peak_memory_mb"] is not None]
    return {
        "dataset": "cold",
        "model": "llm_prompt_lora",
        "model_name": args.name,
        "tag": "llmftp",
        "seed": args.seed,
        "run_name": f"cold_llmftp_{args.name}_s{args.seed}",
        "run_at": now_iso(),
        "config": config,
        "vocab_size": config["vocab_size"],
        "num_params": int(total),
        "num_trainable_params": int(trainable),
        "num_total_params": int(total),
        "epochs": args.epochs,
        "stopped_epoch": epochs_run,
        "early_stopped": False,
        "n_train_used": config["n_train"],
        "best_epoch": epochs_run,
        "best_val_accuracy": best_val,
        "history": history,
        "train_step_losses": [],
        "decontaminated": True,
        "decontaminate_removed": removed,
        "aug_csv": None,
        "aug_n_appended": 0,
        "trunc": f"chars_{MAX_CHARS}_prompt_{args.max_len}",
        "chunk_test_accuracy": None,
        "chunk_test_macro_f1": None,
        "chunk_test_per_class_f1": None,
        "chunk_test_n_chunks_mean": None,
        "chunk_test_n_chunks_max": None,
        "train_time_per_epoch_sec": (
            float(np.mean([h["train_time_sec"] for h in history]))
            if history else 0.0),
        "total_train_time_sec": float(total_train_sec),
        "gpu_peak_memory_mb": float(max(peaks)) if peaks else None,
        "status": "ok",
        "base_dtype": args.base_dtype,
        "step_count": int(steps),
        "sec_per_step": (float(total_train_sec) / steps if steps else None),
        "versions": versions(),
        "prompt": PROMPT_KEY,
        "max_length": args.max_len,
        "threshold": THRESHOLD,
    }


def main() -> None:
    args = validate(build_parser().parse_args())
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("要求 CUDA 但当前环境不可用；冒烟请显式传 --device cpu")
    device = torch.device(args.device)
    try:
        extra_names = parse_extra_eval(args.extra_eval)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    run_name = f"cold_llmftp_{args.name}_s{args.seed}"
    logger = setup_logger(ROOT / "logs" / f"{run_name}.log")
    seed_everything(args.seed)
    logger.info("参数: %s", vars(args))

    splits, removed = deduplicate_cold_splits(load_splits("cold"))
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    if args.train_limit is not None:
        train_df = train_df.iloc[:args.train_limit].reset_index(drop=True)
    if args.test_limit is not None:
        test_df = test_df.iloc[:args.test_limit].reset_index(drop=True)
    logger.info("样本数: train=%d val=%d test=%d", len(train_df),
                len(val_df), len(test_df))
    logger.info("跨划分去重剔除: %s", removed)

    tokenizer = load_tokenizer(args.model_path)
    model = load_base_model(args.model_path, args.base_dtype)
    model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.pad_token_id is None:
        raise SystemExit("分词器补齐 pad_token 后 pad_token_id 仍为空")
    model = wrap_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    model.to(device)
    if args.grad_ckpt:
        if hasattr(model, "config"):
            model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        logger.info("已开启梯度检查点与 input require grads")
    trainable, total = parameter_counts(model)
    logger.info("参数量: 可训练=%d 总=%d（%.2f%%）", trainable, total,
                100.0 * trainable / max(total, 1))

    yes_id = single_token_id(tokenizer, "是")
    no_id = single_token_id(tokenizer, "否")
    logger.info("是/否 词元 id: %d / %d", yes_id, no_id)

    raw_texts = {
        "train": train_df["text"].astype(str).tolist(),
        "val": val_df["text"].astype(str).tolist(),
        "test": test_df["text"].astype(str).tolist(),
    }
    texts = {k: [clean_comment(t) for t in raw_texts[k]] for k in raw_texts}
    trunc = {k: truncation_stats(raw_texts[k]) for k in raw_texts}
    logger.info("截断比例（>192 字）: train=%.4f val=%.4f test=%.4f",
                *(trunc[k]["truncation_rate"] for k in ("train", "val", "test")))
    labels = {"train": train_df["label"].to_numpy(),
              "val": val_df["label"].to_numpy(),
              "test": test_df["label"].to_numpy()}
    train_loader = make_loader(texts["train"], labels["train"], tokenizer,
                               yes_id, no_id, args.max_len,
                               args.batch_size, True)
    logger.info("每轮优化步数=%d 有效批=%d",
                math.ceil(len(train_loader) / args.grad_accum),
                args.batch_size * args.grad_accum)

    config = config_dict(args, tokenizer, len(train_df), len(val_df),
                         len(test_df))
    try:
        trained = train_model(
            model, train_loader, texts["val"], labels["val"], tokenizer,
            device, args.base_dtype, yes_id, no_id, args.max_len,
            args.epochs, args.lr, args.grad_accum, args.warmup_frac,
            1.0, logger,
        )
    except NonFiniteLoss as exc:
        payload = base_payload(args, config, trainable, total, exc.history,
                               exc.step - 1, 0.0, removed)
        payload.update({
            "status": "nan_loss",
            "nan_loss_step": int(exc.step),
            "nan_loss_micro_step": int(exc.micro_step),
            "nan_loss_value": None,
        })
        out = result_json_path(args.out_dir, run_name)
        save_json(out, payload)
        logger.error("第 %d 步（micro %d）损失为 %s，按约定停止；结果写入 %s",
                     exc.step, exc.micro_step, exc.loss, out)
        raise SystemExit(1)

    history = trained["history"]
    total_peak = max((h["gpu_peak_memory_mb"] for h in history
                      if h["gpu_peak_memory_mb"] is not None), default=None)
    payload = base_payload(args, config, trainable, total, history,
                           trained["steps"], trained["total_train_sec"],
                           removed)
    payload["train_step_losses"] = trained["step_losses"]
    payload["truncation_rate"] = trunc["train"]["truncation_rate"]
    payload["truncation_rate_by_split"] = {
        k: v["truncation_rate"] for k, v in trunc.items()}

    # 只保存 LoRA；基座仍从 --model_path 读取，不落盘
    ckpt_dir = ROOT / "checkpoints" / f"llmftp_{args.name}_s{args.seed}"
    model.save_pretrained(ckpt_dir)
    logger.info("LoRA 写入 %s（不保存完整基座）", ckpt_dir)

    val_loss, val_acc, val_preds, val_labels, val_probs = evaluate_prompt(
        model, tokenizer, texts["val"], labels["val"], args.batch_size,
        device, args.base_dtype, yes_id, no_id, args.max_len)
    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate_prompt(
        model, tokenizer, texts["test"], labels["test"], args.batch_size,
        device, args.base_dtype, yes_id, no_id, args.max_len)
    metrics = classification_metrics(test_labels, test_preds, CLASS_NAMES)
    logger.info("验证集 loss=%.4f acc=%.4f；测试集 loss=%.4f acc=%.4f 宏F1=%.4f",
                val_loss, val_acc, test_loss, test_acc, metrics["macro_f1"])

    extra_eval, extra_payloads = {}, {}
    for name in extra_names:
        if not eval_set_available(name):
            logger.info("额外评测 %s: 缺文件，跳过", name)
            continue
        try:
            frame = load_eval_set(name)
        except FileNotFoundError as exc:
            logger.info("额外评测 %s: 缺文件（%s），跳过", name, exc)
            continue
        e_texts = [clean_comment(t) for t in frame["text"].astype(str)]
        e_preds, e_probs = predict_texts(
            model, tokenizer, e_texts, args.batch_size, device,
            args.base_dtype, yes_id, no_id, args.max_len)
        e_metrics = classification_metrics(
            frame["label"].to_numpy(), e_preds, CLASS_NAMES)
        extra_eval[name] = {
            "accuracy": e_metrics["accuracy"],
            "macro_f1": e_metrics["macro_f1"],
            "n": int(len(frame)),
            "pos_precision": e_metrics["pos_precision"],
            "pos_recall": e_metrics["pos_recall"],
        }
        extra_payloads[name] = (frame, e_preds, e_probs)
        logger.info("额外评测 %s: n=%d acc=%.4f macro_f1=%.4f",
                    name, len(frame), e_metrics["accuracy"],
                    e_metrics["macro_f1"])

    payload.update({
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_per_class_f1": metrics["per_class_f1"],
        "test_pos_precision": metrics["pos_precision"],
        "test_pos_recall": metrics["pos_recall"],
        "test_per_class_precision": metrics["per_class_precision"],
        "test_per_class_recall": metrics["per_class_recall"],
        "test_fine_grained": fine_grained_stats(test_df, test_preds),
        "extra_eval": extra_eval,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "test_loss": test_loss,
        "gpu_peak_memory_mb": (payload["gpu_peak_memory_mb"]
                               if total_peak is not None else None),
    })
    if history:
        history[-1]["val_loss"] = val_loss
        history[-1]["val_accuracy"] = val_acc
    if args.bench:
        bench = run_bench(model, tokenizer, test_df["text"].tolist(), device,
                          args.base_dtype, yes_id, no_id, args.max_len)
        payload["bench"] = bench
        payload["ms_per_1k"] = bench["ms_per_1k"]
        logger.info("测速: %s ms/千条（%d 条）",
                    "缺" if bench["ms_per_1k"] is None
                    else f"{bench['ms_per_1k']:.2f}", bench["n"])

    out = result_json_path(args.out_dir, run_name)
    save_json(out, payload)
    if args.save_preds:
        pp = preds_path(args.out_dir, run_name)
        save_predictions(pp, np.arange(len(test_labels)), test_labels,
                         test_preds, test_probs, CLASS_NAMES)
        logger.info("逐条预测写入 %s", pp)
        for name, (frame, e_preds, e_probs) in extra_payloads.items():
            ep = preds_path(args.out_dir, f"{run_name}__{name}")
            save_predictions(
                ep, np.arange(len(frame)), frame["label"].to_numpy(),
                e_preds, e_probs, CLASS_NAMES,
                pair_ids=(frame["pair_id"].to_numpy()
                          if "pair_id" in frame else None),
            )
            logger.info("额外评测预测写入 %s", ep)
    if args.save_val_preds:
        vp = val_preds_path(args.out_dir, run_name)
        save_predictions(vp, np.arange(len(val_labels)), val_labels,
                         val_preds, val_probs, CLASS_NAMES)
        logger.info("验证集逐条预测写入 %s", vp)
    logger.info("结果写入 %s", out)


if __name__ == "__main__":
    main()




