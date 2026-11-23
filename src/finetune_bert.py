"""DistilBERT 微调入口（阶段 2 实验 F，流程中的“预训练模型”分支）。

和 train.py 的区别只在模型与优化器：数据清洗、验证集划分、分层抽样、
早停选轮、测试集只评估一次，全部复用同一套代码。DistilBERT 用自己
的分词器（WordPiece），截断长度 IMDB 256、AG News 64。

设置：AdamW，学习率 2e-5，批大小 32，最多 3 轮，
预热 10%，按验证准确率选最佳轮，cuda 上开混合精度。这里不做早停，
因为最多只有 3 轮；``--patience`` 不提供给本脚本，避免和 3 轮上限混淆。

模型只从本地 ``models/distilbert-base-uncased`` 加载，并设
``local_files_only=True``，不会联网。结果 JSON 字段与 train.py 尽量一致，
方便 plot_stage2.py 统一读取。
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import (
    DATASETS,
    append_aug,
    decontaminate as decontaminate_frame,
    deduplicate_cold_splits,
    eval_set_available,
    load_eval_set,
    load_splits,
    parse_extra_eval,
    stratified_sample_n,
    subsample,
)
from engine import classification_metrics, count_parameters, seed_everything
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="微调 DistilBERT 做文本分类")
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_n", type=int, default=None,
                   help="按标签分层抽取的绝对条数，抽样种子等于 --seed")
    p.add_argument("--train_frac", type=float, default=None,
                   help="训练集使用比例，默认 1.0；与 --train_n 不能同时给出")
    p.add_argument("--max_len", type=int, default=None, help="默认 IMDB 256 / AG News 64")
    p.add_argument("--epochs", type=int, default=3, help="最多 3 轮")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_frac", type=float, default=0.10)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", default="cuda", help="cuda 或 cpu；cuda 不可用会报错退出")
    p.add_argument("--tag", default="bert")
    p.add_argument("--save_preds", action="store_true",
                   help="把测试集逐条预测写到结果目录的 preds/ 下")
    p.add_argument("--save_val_preds", action="store_true",
                   help="用最佳轮次模型预测验证集，写到 preds/<run_name>__val.parquet")
    p.add_argument("--out_dir", default=None,
                   help="结果目录，默认 results/；冒烟用 results/smoke2")
    p.add_argument("--run_suffix", default="",
                   help="文件名后缀，同 tag 下区分 n 等设置")
    p.add_argument("--trunc", choices=("head", "tail", "headtail"), default="head",
                   help="子词截断方式；默认 head 与阶段 2 相同")
    p.add_argument("--head_tokens", type=int, default=128,
                   help="headtail 时保留的开头子词数（不含特殊符号），默认 128")
    p.add_argument("--chunk_eval", action="store_true",
                   help="测试时把整条评论切段、各段概率取平均；训练过程不变")
    p.add_argument("--test_limit", type=int, default=None,
                   help="只评测测试集前 N 条；默认不限，冒烟用")
    p.add_argument("--save_model", action="store_true",
                   help="把最佳轮次的模型目录存到 checkpoints/<run_name>/")
    p.add_argument("--decontaminate", dest="decontaminate", action="store_true",
                   default=None, help="剔除污染/重复；COLD 默认开启，英文默认关闭")
    p.add_argument("--no_decontaminate", dest="decontaminate", action="store_false",
                   help="显式关闭去污染（COLD 也关）")
    p.add_argument("--extra_eval", default="",
                   help="逗号分隔的评测集名，训练后逐个评测并写入 extra_eval")
    p.add_argument("--model_path", default=None,
                   help="本地模型目录，默认 models/distilbert-base-uncased")
    p.add_argument("--aug_csv", default="",
                   help="D1 增强 CSV；默认空，不追加。给了只在去重后加训练集")
    return p


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    ds = DATASETS[args.dataset]
    args.max_len = ds["max_len"] if args.max_len is None else args.max_len
    if args.train_n is not None and args.train_frac is not None:
        raise SystemExit("--train_n 与 --train_frac 不能同时给出")
    args.train_frac = 1.0 if args.train_frac is None else args.train_frac
    if not 0.0 < args.train_frac <= 1.0:
        raise SystemExit("--train_frac 必须在 (0, 1] 内")
    if args.train_n is not None and args.train_n <= 0:
        raise SystemExit("--train_n 必须为正整数")
    if args.max_len < 3:
        raise SystemExit("--max_len 至少为 3（要容纳 [CLS] 与 [SEP]）")
    if args.head_tokens < 0:
        raise SystemExit("--head_tokens 必须 >= 0")
    if args.trunc == "headtail" and args.head_tokens > args.max_len - 2:
        raise SystemExit("headtail 的 --head_tokens 不能超过 max_len-2")
    if args.test_limit is not None and args.test_limit <= 0:
        raise SystemExit("--test_limit 必须为正整数或省略")
    if args.decontaminate is None:
        args.decontaminate = args.dataset == "cold"
    if args.model_path is None:
        args.model_path = str(ROOT / "models" / "distilbert-base-uncased")
    return args


def truncate_ids(ids: list[int], max_len: int, trunc: str = "head",
                 head_tokens: int = 128) -> list[int]:
    """按子词截断，不含 [CLS]/[SEP]；max_len 是含两个特殊符号的总长度。

    head：保留前 max_len-2 个；tail：保留后 max_len-2 个；
    headtail：前 head_tokens 个 + 后 max_len-2-head_tokens 个，
    对应 Sun 等 2019 的“开头加结尾”做法（512 时是前 128 + 后 382）。
    """
    budget = max_len - 2
    if budget <= 0:
        raise ValueError("max_len 至少为 3")
    ids = list(ids)
    if len(ids) <= budget:
        return ids
    if trunc == "head":
        return ids[:budget]
    if trunc == "tail":
        return ids[-budget:]
    if trunc != "headtail":
        raise ValueError(f"未知截断方式 {trunc!r}")
    if not 0 <= head_tokens <= budget:
        raise ValueError("headtail 的 head_tokens 必须在 [0, max_len-2] 内")
    tail_tokens = budget - head_tokens
    return ids[:head_tokens] + (ids[-tail_tokens:] if tail_tokens else [])


def chunk_spans(n_tokens: int, chunk_size: int, overlap: int = 128) -> list[tuple[int, int]]:
    """把长度为 n_tokens 的序列切成若干段，返回每段的 [start, end)。

    相邻段重叠 ``overlap`` 个子词（max_len=512 时每段 510、步长 382）。
    最后一段截到序列末尾，所以每段长度不超过 chunk_size，所有位置至少
    被一段覆盖。序列为空时给一个空段，保证模型仍能拿到 [CLS][SEP]。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正")
    if n_tokens <= 0:
        return [(0, 0)]
    if n_tokens <= chunk_size:
        return [(0, n_tokens)]
    # max_len 很小时 overlap 可能不小于段长。此时取段长的一半，
    # 避免步长变成 1 把一条评论切成上千段（J 的 512 不会走到这里）。
    overlap = max(overlap, 0)
    if overlap >= chunk_size:
        overlap = chunk_size // 2
    stride = chunk_size - overlap
    spans = []
    start = 0
    while True:
        end = min(start + chunk_size, n_tokens)
        spans.append((start, end))
        if end >= n_tokens:
            break
        start += stride
    return spans


def encode_split(tokenizer, texts: list[str], max_len: int,
                 trunc: str = "head", head_tokens: int = 128
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把文本编码成定长 id / attention mask，并返回截断前的子词数。

    不交给 tokenizer 的 truncation 参数，而是自己按 trunc 方式截断：
    这样 head/tail/headtail 共用同一套特殊符号和 padding 规则。
    raw_lengths 是未截断、不含特殊符号的 WordPiece 数，用来算截断比例。
    """
    raw = tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]
    raw_lengths = np.array([len(x) for x in raw], dtype=np.int64)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    n = len(texts)
    ids = np.full((n, max_len), pad_id, dtype=np.int64)
    mask = np.zeros((n, max_len), dtype=np.int64)
    for i, seq in enumerate(raw):
        body = truncate_ids(seq, max_len, trunc, head_tokens)
        row = [cls_id] + body + [sep_id]
        ids[i, :len(row)] = row
        mask[i, :len(row)] = 1
    return ids, mask, raw_lengths


def make_loader(ids, mask, labels, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def make_scheduler(optimizer, total_steps: int, warmup_frac: float):
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    grad_clip: float = 1.0):
    """训练一轮，返回 (平均损失, 秒, 本轮显存峰值 MB)。

    混合精度只在 cuda 上开：CPU 上 half 反而更慢，而且部分算子没有实现。
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    on_cuda = device.type == "cuda"
    if on_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    total_loss, n = 0.0, 0
    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=on_cuda):
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            # 先 unscale 再裁剪，否则梯度被放大，裁剪阈值失去意义
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    if on_cuda:
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        peak_mb = 0.0
    return total_loss / max(n, 1), time.perf_counter() - start, peak_mb


@torch.no_grad()
def evaluate(model, loader, device):
    """返回 (平均损失, 准确率, 预测, 标签, softmax 概率)。"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    preds, labels, probs = [], [], []
    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        total_loss += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(y.cpu())
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    probs = torch.cat(probs).numpy()
    acc = float((preds == labels).mean()) if len(labels) else 0.0
    return total_loss / max(n, 1), acc, preds, labels, probs


def evaluate_extra_set(model, tokenizer, name: str, max_len: int, trunc: str,
                      head_tokens: int, device, batch_size: int,
                      class_names: list[str]):
    """在额外评测集上按同一截断方式跑一次，返回 (表, 预测, 概率, 指标)。"""
    frame = load_eval_set(name)
    ids, mask, _ = encode_split(
        tokenizer, frame["text"].tolist(), max_len, trunc, head_tokens
    )
    loader = make_loader(ids, mask, frame["label"].to_numpy(), batch_size, False)
    _, _, preds, labels, probs = evaluate(model, loader, device)
    metrics = classification_metrics(labels, preds, class_names)
    return frame, preds, probs, metrics


def fine_grained_stats(frame, preds) -> dict | None:
    """COLD 测试集按 fine-grained-label 统计每类被判为攻击性的比例。"""
    if "fine-grained-label" not in frame.columns:
        return None
    labels = frame["fine-grained-label"].to_numpy()
    preds = np.asarray(preds)
    out = {}
    for label in sorted(set(int(x) for x in labels)):
        mask = labels == label
        n = int(mask.sum())
        n_off = int((preds[mask] == 1).sum())
        out[str(label)] = {
            "n": n,
            "n_pred_offensive": n_off,
            "rate_pred_offensive": float(n_off / n) if n else 0.0,
        }
    return out


@torch.no_grad()
def predict_chunks(model, tokenizer, texts: list[str], max_len: int,
                   batch_size: int, device):
    """分段评测：整条评论切段，各段 softmax 概率取平均。

    返回 (预测, 平均概率, 每条评论的段数)。每段都单独加 [CLS]/[SEP]，
    段内最多 max_len-2 个子词；不训练，只用于测试时汇总。
    """
    raw = tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]
    chunk_size = max_len - 2
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    rows, owners, counts = [], [], []
    for i, seq in enumerate(raw):
        spans = chunk_spans(len(seq), chunk_size, overlap=128)
        counts.append(len(spans))
        for start, end in spans:
            row = [cls_id] + list(seq[start:end]) + [sep_id]
            padded = np.full(max_len, pad_id, dtype=np.int64)
            mask = np.zeros(max_len, dtype=np.int64)
            padded[:len(row)] = row
            mask[:len(row)] = 1
            rows.append((padded, mask))
            owners.append(i)
    ids = torch.tensor(np.stack([r[0] for r in rows]), dtype=torch.long)
    masks = torch.tensor(np.stack([r[1] for r in rows]), dtype=torch.long)
    loader = DataLoader(TensorDataset(ids, masks), batch_size=batch_size, shuffle=False)
    num_classes = model.config.num_labels
    sums = np.zeros((len(texts), num_classes), dtype=np.float64)
    cursor = 0
    model.eval()
    for batch_ids, batch_mask in loader:
        batch_ids, batch_mask = batch_ids.to(device), batch_mask.to(device)
        logits = model(input_ids=batch_ids, attention_mask=batch_mask).logits
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        owner = np.asarray(owners[cursor:cursor + len(probs)], dtype=np.int64)
        np.add.at(sums, owner, probs)
        cursor += len(probs)
    counts_arr = np.asarray(counts, dtype=np.int64)
    probs = sums / counts_arr[:, None]
    return probs.argmax(axis=1), probs, counts_arr

def main() -> None:
    args = resolve_defaults(build_parser().parse_args())
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "要求使用 CUDA，但当前环境找不到可用显卡。"
            "正式运行请在 GPU 机器上跑；冒烟请显式传 --device cpu。"
        )
    device = torch.device(args.device)
    try:
        extra_names = parse_extra_eval(args.extra_eval)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    suffix = f"_{args.run_suffix}" if args.run_suffix else ""
    if args.dataset == "cold":
        # 中文实验文件名：cold_bert_rbt3_s0 / cold_bert_roberta_s0
        run_name = f"{args.dataset}_bert{suffix}_s{args.seed}"
        model_label = "bert"
    else:
        run_name = f"{args.dataset}_distilbert_{args.tag}{suffix}_s{args.seed}"
        model_label = "distilbert"
    logger = setup_logger(ROOT / "logs" / f"{run_name}.log")
    seed_everything(args.seed)
    logger.info("参数: %s", vars(args))

    splits = load_splits(args.dataset)
    removed = {"train": 0, "val": 0}
    if args.decontaminate and args.dataset == "cold":
        splits, removed = deduplicate_cold_splits(splits)
    train_df = splits["train"]
    if args.train_n is not None:
        train_df = stratified_sample_n(train_df, args.train_n, args.seed)
    elif args.train_frac < 1.0:
        train_df = subsample(train_df, args.train_frac, args.seed)
    # D1：去重/抽样之后追加增强行，只动训练集；默认为空时逐条不变
    train_df, n_aug = append_aug(train_df, args.aug_csv)
    if n_aug:
        logger.info("追加增强 %d 条：%s", n_aug, args.aug_csv)
    val_df = splits["val"]
    if args.decontaminate and args.dataset != "cold":
        train_df, removed["train"] = decontaminate_frame(train_df)
        val_df, removed["val"] = decontaminate_frame(val_df)
    test_df = splits["test"]
    if args.test_limit is not None:
        # 只给 CPU 冒烟用：正式运行不传，测试集仍是完整 25000 条
        test_df = test_df.iloc[:args.test_limit].reset_index(drop=True)
    frames = {"train": train_df, "val": val_df, "test": test_df}
    logger.info("样本数: train=%d val=%d test=%d",
                *(len(frames[k]) for k in ("train", "val", "test")))
    if args.decontaminate:
        logger.info("去污染剔除: train=%d val=%d", removed["train"], removed["val"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    num_classes = DATASETS[args.dataset]["num_classes"]
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=num_classes, local_files_only=True,
    )
    model.to(device)
    num_params = count_parameters(model)
    logger.info("模型参数量=%d", num_params)

    arrays, trunc_stats = {}, {}
    for split in ("train", "val", "test"):
        ids, mask, raw_lengths = encode_split(
            tokenizer, frames[split]["text"].tolist(), args.max_len,
            args.trunc, args.head_tokens,
        )
        arrays[split] = {
            "ids": ids,
            "mask": mask,
            "labels": frames[split]["label"].to_numpy(),
        }
        trunc_stats[split] = {
            "n_samples": len(frames[split]),
            "n_truncated": int((raw_lengths > args.max_len).sum()),
            "truncation_rate": float((raw_lengths > args.max_len).mean()) if len(raw_lengths) else 0.0,
        }
    logger.info("截断比例: train=%.4f val=%.4f test=%.4f",
                *(trunc_stats[k]["truncation_rate"] for k in ("train", "val", "test")))

    loaders = {
        "train": make_loader(arrays["train"]["ids"], arrays["train"]["mask"],
                             arrays["train"]["labels"], args.batch_size, True),
        "val": make_loader(arrays["val"]["ids"], arrays["val"]["mask"],
                           arrays["val"]["labels"], args.batch_size, False),
        "test": make_loader(arrays["test"]["ids"], arrays["test"]["mask"],
                            arrays["test"]["labels"], args.batch_size, False),
    }

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = max(1, args.epochs * len(loaders["train"]))
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_frac)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_state, best_val, best_epoch = None, -1.0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, sec, peak_mb = train_one_epoch(
            model, loaders["train"], optimizer, scheduler, scaler, device, args.grad_clip
        )
        val_loss, val_acc, _, _, _ = evaluate(model, loaders["val"], device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "train_time_sec": sec,
            "gpu_peak_memory_mb": peak_mb if device.type == "cuda" else None,
        })
        logger.info("epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  "
                    "time=%.1fs  gpu_peak=%.0fMB",
                    epoch, args.epochs, train_loss, val_loss, val_acc, sec, peak_mb)
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
    if best_state is not None:
        model.load_state_dict(best_state)

    # 验证集预测必须用最终选定的最佳轮次参数，和测试集评估保持同一份模型。
    val_preds = val_probs = val_labels = None
    if args.save_val_preds:
        _, _, val_preds, val_labels, val_probs = evaluate(model, loaders["val"], device)

    _, _, test_preds, test_labels, test_probs = evaluate(model, loaders["test"], device)
    class_names = DATASETS[args.dataset]["class_names"]
    metrics = classification_metrics(test_labels, test_preds, class_names)
    fine_grained = fine_grained_stats(frames["test"], test_preds)

    # 分段评测不改训练：拿同一份最佳轮次模型，测试时把整条评论切段取平均
    chunk_preds = chunk_probs = chunk_counts = chunk_metrics = None
    if args.chunk_eval:
        chunk_preds, chunk_probs, chunk_counts = predict_chunks(
            model, tokenizer, frames["test"]["text"].tolist(), args.max_len,
            args.batch_size, device,
        )
        chunk_metrics = classification_metrics(test_labels, chunk_preds, class_names)
        logger.info("分段评测: acc=%.4f 平均段数=%.2f", chunk_metrics["accuracy"],
                    float(chunk_counts.mean()))

    extra_eval, extra_payloads = {}, {}
    for name in extra_names:
        if not eval_set_available(name):
            logger.info("额外评测 %s: 缺文件，跳过", name)
            continue
        try:
            frame, e_preds, e_probs, e_metrics = evaluate_extra_set(
                model, tokenizer, name, args.max_len, args.trunc, args.head_tokens,
                device, args.batch_size, class_names,
            )
        except FileNotFoundError as exc:
            logger.info("额外评测 %s: 缺文件（%s），跳过", name, exc)
            continue
        extra_eval[name] = {
            "accuracy": e_metrics["accuracy"],
            "macro_f1": e_metrics["macro_f1"],
            "n": len(frame),
        }
        if args.dataset == "cold" and "pos_precision" in e_metrics:
            extra_eval[name]["pos_precision"] = e_metrics["pos_precision"]
            extra_eval[name]["pos_recall"] = e_metrics["pos_recall"]
        extra_payloads[name] = (frame, e_preds, e_probs)
        logger.info("额外评测 %s: n=%d acc=%.4f macro_f1=%.4f",
                    name, len(frame), e_metrics["accuracy"], e_metrics["macro_f1"])

    if args.save_model:
        # 目录名就是结果同名，以后可以直接 from_pretrained 这个目录
        ckpt_dir = ROOT / "checkpoints" / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        logger.info("最佳轮次模型目录写入 %s", ckpt_dir)

    epoch_times = [h["train_time_sec"] for h in history]
    peaks = [h["gpu_peak_memory_mb"] for h in history if h["gpu_peak_memory_mb"] is not None]
    payload = {
        "dataset": args.dataset,
        "model": model_label,
        "tag": args.tag,
        "seed": args.seed,
        "run_name": run_name,
        "run_at": now_iso(),
        "config": {
            **vars(args),
            "num_classes": num_classes,
            "vocab_size": tokenizer.vocab_size,
            "optimizer": "AdamW",
            "amp": device.type == "cuda",
            "n_train": len(train_df),
            "aug_csv": args.aug_csv or None,
            "aug_n_appended": n_aug,
            "n_val": len(val_df),
            "n_test": len(test_df),
        },
        "vocab_size": tokenizer.vocab_size,
        "num_params": num_params,
        "truncation_rate": trunc_stats["train"]["truncation_rate"],
        "truncation_rate_by_split": {
            k: v["truncation_rate"] for k, v in trunc_stats.items()
        },
        "epochs": args.epochs,
        "stopped_epoch": len(history),
        "early_stopped": False,
        "n_train_used": len(train_df),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "history": history,
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_per_class_f1": metrics["per_class_f1"],
        "decontaminated": bool(args.decontaminate),
        "decontaminate_removed": removed,
        "aug_csv": args.aug_csv or None,
        "aug_n_appended": n_aug,
        "extra_eval": extra_eval,
        "trunc": args.trunc,
        "chunk_test_accuracy": chunk_metrics["accuracy"] if chunk_metrics else None,
        "chunk_test_macro_f1": chunk_metrics["macro_f1"] if chunk_metrics else None,
        "chunk_test_per_class_f1": chunk_metrics["per_class_f1"] if chunk_metrics else None,
        "chunk_test_n_chunks_mean": float(chunk_counts.mean()) if chunk_counts is not None else None,
        "chunk_test_n_chunks_max": int(chunk_counts.max()) if chunk_counts is not None else None,
        "train_time_per_epoch_sec": float(np.mean(epoch_times)) if epoch_times else 0.0,
        "total_train_time_sec": float(np.sum(epoch_times)),
        "gpu_peak_memory_mb": float(max(peaks)) if peaks else None,
    }
    if args.dataset == "cold":
        payload["test_pos_precision"] = metrics["pos_precision"]
        payload["test_pos_recall"] = metrics["pos_recall"]
        payload["test_per_class_precision"] = metrics["per_class_precision"]
        payload["test_per_class_recall"] = metrics["per_class_recall"]
        payload["test_fine_grained"] = fine_grained
    out = result_json_path(args.out_dir, run_name)
    save_json(out, payload)
    if args.save_preds:
        pp = preds_path(args.out_dir, run_name)
        save_predictions(pp, np.arange(len(test_labels)), test_labels,
                         test_preds, test_probs, class_names)
        logger.info("逐条预测写入 %s", pp)
        if chunk_preds is not None:
            cp = preds_path(args.out_dir, f"{run_name}__chunk")
            save_predictions(cp, np.arange(len(test_labels)), test_labels,
                             chunk_preds, chunk_probs, class_names)
            logger.info("分段预测写入 %s", cp)
        for name, (frame, e_preds, e_probs) in extra_payloads.items():
            ep = preds_path(args.out_dir, f"{run_name}__{name}")
            save_predictions(
                ep, np.arange(len(frame)), frame["label"].to_numpy(), e_preds, e_probs,
                class_names,
                pair_ids=frame["pair_id"].to_numpy() if "pair_id" in frame else None,
            )
            logger.info("额外评测预测写入 %s", ep)
    if args.save_val_preds:
        vp = val_preds_path(args.out_dir, run_name)
        save_predictions(vp, np.arange(len(val_labels)), val_labels,
                         val_preds, val_probs, class_names)
        logger.info("验证集逐条预测写入 %s", vp)
    logger.info("最佳轮次=%d 验证准确率=%.4f 测试准确率=%.4f 宏F1=%.4f",
                best_epoch, best_val, metrics["accuracy"], metrics["macro_f1"])
    logger.info("结果写入 %s", out)


if __name__ == "__main__":
    main()


