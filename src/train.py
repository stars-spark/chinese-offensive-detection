"""神经网络训练入口（流程第 5 步）。

命令行的默认值：
- 截断长度 IMDB 256 / AG News 64；轮数 6 / 4；
- 批大小 64，AdamW + 权重衰减 0.01，前 5% 步数预热后线性衰减，梯度裁剪 1.0；
- 学习率 TextCNN/BiLSTM 1e-3，Transformer 5e-4；dropout 0.5 / 0.1。

每轮在验证集上评估，取验证准确率最高的那一轮参数去测试集评估一次。
测试集不参与选轮或选超参数。结果写入
``results/<dataset>_<model>_<tag>_s<seed>.json``，日志写到 ``logs/``。

--device 默认 cuda；没有显卡时直接报错退出，不静默换 CPU，避免有人
以为在跑 GPU 其实在跑 CPU。冒烟时显式传 ``--device cpu``。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data import (
    DATASETS,
    decontaminate as decontaminate_frame,
    deduplicate_cold_splits,
    eval_set_available,
    load_eval_set,
    load_splits,
    parse_extra_eval,
    stratified_sample_n,
    subsample,
    tokenize_text,
)
from engine import (
    classification_metrics,
    count_parameters,
    evaluate,
    fit,
    make_loader,
    seed_everything,
)
from models import BiLSTM, TextCNN, TransformerClassifier
from result_io import (
    checkpoint_path,
    now_iso,
    preds_path,
    result_json_path,
    save_json,
    save_predictions,
    setup_logger,
    val_preds_path,
)
from vocab import build_vocab, encode_dataset

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("textcnn", "bilstm", "transformer")


class _StoreTrainFrac(argparse.Action):
    """记录 --train_frac 是否被显式给出，用来和 --train_n 查冲突。

    默认值仍是 1.0，所以不传这个参数时 parse_args 的结果和阶段 1 一样；
    只有显式给出时才置 ``_train_frac_given``。
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, "_train_frac_given", True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="训练文本分类模型")
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--tokenizer", choices=("char", "word"), default=None,
                   help="char 按字，word 用 jieba（英文仍用阶段 1 正则）；COLD 默认 char")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_len", type=int, default=None, help="默认按数据集取 256/64")
    p.add_argument("--epochs", type=int, default=None, help="默认按数据集取 6/4")
    p.add_argument("--lr", type=float, default=None, help="默认 CNN/LSTM 1e-3，Transformer 5e-4")
    p.add_argument("--batch_size", type=int, default=64)
    # 阶段 2 的两种抽样方式二选一；默认都不给就是完整训练集（阶段 1 行为）
    p.add_argument("--train_frac", type=float, default=1.0, action=_StoreTrainFrac,
                   help="训练集使用比例，默认 1.0；与 --train_n 不能同时给出")
    p.add_argument("--train_n", type=int, default=None,
                   help="按标签分层抽取的绝对条数，抽样种子等于 --seed")
    p.add_argument("--patience", type=int, default=None,
                   help="早停耐心值；默认 None 表示不早停（阶段 1 行为）")
    p.add_argument("--save_preds", action="store_true",
                   help="把测试集逐条预测写到 results/preds/ 下与 JSON 同名的 parquet")
    p.add_argument("--save_val_preds", action="store_true",
                   help="用最佳轮次模型预测验证集，写到 preds/<run_name>__val.parquet")
    p.add_argument("--save_model", action="store_true",
                   help="把最佳轮次 state_dict 与词表写到 checkpoints/ 下")
    p.add_argument("--decontaminate", dest="decontaminate", action="store_true",
                   default=None, help="剔除污染/重复；COLD 默认开启，英文默认关闭")
    p.add_argument("--no_decontaminate", dest="decontaminate", action="store_false",
                   help="显式关闭去污染（COLD 也关）")
    p.add_argument("--test_limit", type=int, default=None,
                   help="只评测测试集前 N 条；默认不限，冒烟用")
    p.add_argument("--extra_eval", default="",
                   help="逗号分隔的评测集名，训练后逐个评测并写入 extra_eval")
    p.add_argument("--out_dir", default=None,
                   help="结果目录，默认 results/；冒烟用 results/smoke3")
    p.add_argument("--run_suffix", default="",
                   help="文件名后缀，同 tag 下区分 n/max_len/layers 等设置")
    p.add_argument("--pos_enc", choices=("sin", "learned", "none"), default="sin")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=None,
                   help="默认 Transformer 0.1，CNN/LSTM 0.5")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", default="cuda", help="cuda 或 cpu；cuda 不可用会报错退出")
    p.add_argument("--tag", default="main", help="main=正式结果，smoke=冒烟")
    p.set_defaults(_train_frac_given=False)
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """把 None 的默认值按数据集/模型补齐，之后写进 JSON 的 config。

    ``train_frac`` 的默认值放在这里补成 1.0，而不是 argparse 的 default，
    这样才能区分“显式传了 --train_frac”和“没传”，好检查它和 --train_n
    是否冲突。
    """
    ds = DATASETS[args.dataset]
    args.max_len = ds["max_len"] if args.max_len is None else args.max_len
    args.epochs = ds["epochs"] if args.epochs is None else args.epochs
    # COLD 默认按字、默认开启去重；英文默认仍为阶段 1 的正则分词与不去污染
    if args.tokenizer is None:
        args.tokenizer = "char" if args.dataset == "cold" else "word"
    if args.decontaminate is None:
        args.decontaminate = args.dataset == "cold"
    if args.lr is None:
        args.lr = 5e-4 if args.model == "transformer" else 1e-3
    if args.dropout is None:
        args.dropout = 0.1 if args.model == "transformer" else 0.5
    given = getattr(args, "_train_frac_given", args.train_frac != 1.0)
    if args.train_n is not None and given:
        raise SystemExit("--train_n 与 --train_frac 不能同时给出")
    args.train_frac = 1.0 if args.train_frac is None else args.train_frac
    if args.train_frac <= 0 or args.train_frac > 1:
        raise SystemExit("--train_frac 必须在 (0, 1] 内")
    if args.train_n is not None and args.train_n <= 0:
        raise SystemExit("--train_n 必须为正整数")
    if args.patience is not None and args.patience < 1:
        raise SystemExit("--patience 必须 >= 1 或省略")
    if args.test_limit is not None and args.test_limit <= 0:
        raise SystemExit("--test_limit 必须为正整数或省略")
    # 这个内部标记只在解析阶段用，不写进结果 JSON
    if hasattr(args, "_train_frac_given"):
        delattr(args, "_train_frac_given")
    return args


def evaluate_extra_set(model, name: str, vocab, max_len: int,
                      device, batch_size: int, class_names: list[str],
                      tokenizer: str, dataset: str):
    """在额外评测集上跑一次，返回 (DataFrame, 预测, 概率, 指标)。

    与测试集用同一套分词、词表和截断，保证 ``extra_eval`` 的数字和
    ``test_accuracy`` 是同一口径。
    """
    frame = load_eval_set(name)
    token_lists = [tokenize_text(t, dataset, tokenizer) for t in frame["text"]]
    ids, mask, _ = encode_dataset(token_lists, vocab, max_len)
    loader = make_loader(ids, mask, frame["label"].to_numpy(), batch_size, shuffle=False)
    _, _, preds, labels, probs = evaluate(model, loader, device, return_probs=True)
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


def main() -> None:
    args = resolve_defaults(parse_args())
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "要求使用 CUDA，但当前环境找不到可用显卡。"
            "正式运行请在 GPU 机器上跑；冒烟请显式传 --device cpu。"
        )
    device = torch.device(args.device)

    suffix = f"_{args.run_suffix}" if args.run_suffix else ""
    run_name = f"{args.dataset}_{args.model}_{args.tag}{suffix}_s{args.seed}"
    logger = setup_logger(ROOT / "logs" / f"{run_name}.log")
    seed_everything(args.seed)  # 放在建模型和 DataLoader 之前
    logger.info("参数: %s", vars(args))

    splits = load_splits(args.dataset)
    removed = {"train": 0, "val": 0}
    if args.decontaminate and args.dataset == "cold":
        # COLD 的跨划分去重在抽样之前做，剔除条数按官方划分统计
        splits, removed = deduplicate_cold_splits(splits)
    train_df = splits["train"]
    if args.train_n is not None:
        # 学习曲线：按标签分层抽绝对条数，词表随后只用这批样本建
        train_df = stratified_sample_n(train_df, args.train_n, args.seed)
    elif args.train_frac < 1.0:
        # 只对训练部分做分层子采样，验证/测试保持完整
        train_df = subsample(train_df, args.train_frac, args.seed)
    val_df = splits["val"]
    if args.decontaminate and args.dataset != "cold":
        # 英文阶段 3 的去污染放在抽样/划分之后、建词表之前
        train_df, removed["train"] = decontaminate_frame(train_df)
        val_df, removed["val"] = decontaminate_frame(val_df)
    test_df = splits["test"]
    if args.test_limit is not None:
        test_df = test_df.iloc[:args.test_limit].reset_index(drop=True)
    frames = {"train": train_df, "val": val_df, "test": test_df}
    logger.info("分词器=%s 样本数: train=%d val=%d test=%d", args.tokenizer,
                *(len(frames[k]) for k in ("train", "val", "test")))
    if args.decontaminate:
        logger.info("去污染剔除: train=%d val=%d", removed["train"], removed["val"])

    try:
        extra_names = parse_extra_eval(args.extra_eval)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # 每个 split 只分词一次，建词表和编码都复用这批 token
    token_lists = {
        k: [tokenize_text(t, args.dataset, args.tokenizer) for t in frames[k]["text"]]
        for k in frames
    }
    vocab = build_vocab(token_lists["train"], max_size=30000, min_freq=2)
    arrays, trunc_stats = {}, {}
    for split in ("train", "val", "test"):
        ids, mask, stats = encode_dataset(token_lists[split], vocab, args.max_len)
        arrays[split] = {
            "ids": ids,
            "mask": mask,
            "labels": frames[split]["label"].to_numpy(),
        }
        trunc_stats[split] = stats
    logger.info("词表大小=%d 截断比例: train=%.4f val=%.4f test=%.4f",
                len(vocab), *(trunc_stats[k]["truncation_rate"] for k in ("train", "val", "test")))

    loaders = {
        "train": make_loader(arrays["train"]["ids"], arrays["train"]["mask"],
                             arrays["train"]["labels"], args.batch_size, shuffle=True),
        "val": make_loader(arrays["val"]["ids"], arrays["val"]["mask"],
                           arrays["val"]["labels"], args.batch_size, shuffle=False),
        "test": make_loader(arrays["test"]["ids"], arrays["test"]["mask"],
                            arrays["test"]["labels"], args.batch_size, shuffle=False),
    }

    num_classes = DATASETS[args.dataset]["num_classes"]
    if args.model == "textcnn":
        model = TextCNN(len(vocab), num_classes, embed_dim=128, dropout=args.dropout)
    elif args.model == "bilstm":
        model = BiLSTM(len(vocab), num_classes, embed_dim=128, hidden_dim=128, dropout=args.dropout)
    else:
        model = TransformerClassifier(
            len(vocab), num_classes, d_model=args.d_model, layers=args.layers,
            heads=args.heads, d_ff=args.d_ff, dropout=args.dropout,
            max_len=args.max_len, pos_enc=args.pos_enc,
        )
    model.to(device)
    num_params = count_parameters(model)
    logger.info("模型参数量=%d", num_params)

    history, best_epoch, best_val = fit(
        model, loaders["train"], loaders["val"], epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, warmup_frac=args.warmup_frac,
        grad_clip=args.grad_clip, device=device, patience=args.patience,
        log=logger.info,
    )
    # 这里模型已经装回验证集最好的那一轮参数；测试集只评估这一次
    need_probs = args.save_preds or bool(extra_names)
    if need_probs:
        _, _, test_preds, test_labels, test_probs = evaluate(
            model, loaders["test"], device, return_probs=True
        )
    else:
        _, _, test_preds, test_labels = evaluate(model, loaders["test"], device)
        test_probs = None
    class_names = DATASETS[args.dataset]["class_names"]
    metrics = classification_metrics(test_labels, test_preds, class_names)
    fine_grained = fine_grained_stats(frames["test"], test_preds)
    val_preds = val_probs = val_labels = None
    if args.save_val_preds:
        # 验证集预测用同一个最佳轮次模型，不做任何重新选轮
        _, _, val_preds, val_labels, val_probs = evaluate(
            model, loaders["val"], device, return_probs=True
        )

    # 额外评测集用同一个最佳模型、同一套词表和截断逐个跑
    extra_eval, extra_payloads = {}, {}
    for name in extra_names:
        if not eval_set_available(name):
            logger.info("额外评测 %s: 缺文件，跳过", name)
            continue
        try:
            frame, preds, probs, extra_metrics = evaluate_extra_set(
                model, name, vocab, args.max_len, device, args.batch_size,
                class_names, args.tokenizer, args.dataset,
            )
        except FileNotFoundError as exc:
            logger.info("额外评测 %s: 缺文件（%s），跳过", name, exc)
            continue
        extra_eval[name] = {
            "accuracy": extra_metrics["accuracy"],
            "macro_f1": extra_metrics["macro_f1"],
            "n": len(frame),
        }
        if args.dataset == "cold" and "pos_precision" in extra_metrics:
            extra_eval[name]["pos_precision"] = extra_metrics["pos_precision"]
            extra_eval[name]["pos_recall"] = extra_metrics["pos_recall"]
        extra_payloads[name] = (frame, preds, probs)
        logger.info("额外评测 %s: n=%d acc=%.4f macro_f1=%.4f",
                    name, len(frame), extra_metrics["accuracy"], extra_metrics["macro_f1"])

    epoch_times = [h["train_time_sec"] for h in history]
    peaks = [h["gpu_peak_memory_mb"] for h in history if h["gpu_peak_memory_mb"] is not None]
    stopped_epoch = history[-1]["epoch"] if history else 0
    # 保存模型时把“怎么重建模型”也写进去，attention_viz.py 只靠它加载
    model_config = {
        "dataset": args.dataset,
        "model": args.model,
        "num_classes": num_classes,
        "vocab_size": len(vocab),
        "embed_dim": 128 if args.model in ("textcnn", "bilstm") else None,
        "hidden_dim": 128 if args.model == "bilstm" else None,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "max_len": args.max_len,
        "pos_enc": args.pos_enc,
        "class_names": DATASETS[args.dataset]["class_names"],
    }
    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "tag": args.tag,
        "seed": args.seed,
        "run_name": run_name,
        "run_at": now_iso(),
        "config": {
            **vars(args),
            "num_classes": num_classes,
            "vocab_size": len(vocab),
            "embed_dim": model_config["embed_dim"],
            "hidden_dim": model_config["hidden_dim"],
            "d_ff": args.d_ff if args.model == "transformer" else None,
            "optimizer": "AdamW",
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
        },
        "vocab_size": len(vocab),
        "num_params": num_params,
        "truncation_rate": trunc_stats["train"]["truncation_rate"],
        "truncation_rate_by_split": {k: v["truncation_rate"] for k, v in trunc_stats.items()},
        "epochs": args.epochs,
        "stopped_epoch": stopped_epoch,
        "early_stopped": stopped_epoch < args.epochs,
        "n_train_used": len(train_df),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "history": history,
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_per_class_f1": metrics["per_class_f1"],
        "decontaminated": bool(args.decontaminate),
        "decontaminate_removed": removed,
        "extra_eval": extra_eval,
        "train_time_per_epoch_sec": float(np.mean(epoch_times)) if epoch_times else 0.0,
        "total_train_time_sec": float(np.sum(epoch_times)),
        "gpu_peak_memory_mb": float(max(peaks)) if peaks else None,
    }
    if args.dataset != "cold":
        # 英文 JSON 的 config 保持阶段 1 字段，不因新增参数而变
        for key in ("tokenizer", "test_limit", "save_val_preds"):
            payload["config"].pop(key, None)
    if args.dataset == "cold":
        # 中文实验要求：攻击性类精确率/召回率、细粒度每类判为攻击性的比例
        payload["test_pos_precision"] = metrics["pos_precision"]
        payload["test_pos_recall"] = metrics["pos_recall"]
        payload["test_per_class_precision"] = metrics["per_class_precision"]
        payload["test_per_class_recall"] = metrics["per_class_recall"]
        payload["test_fine_grained"] = fine_grained
    out = result_json_path(args.out_dir, run_name)
    save_json(out, payload)
    if args.save_preds:
        pp = preds_path(args.out_dir, run_name)
        save_predictions(
            pp, np.arange(len(test_labels)), test_labels, test_preds, test_probs,
            DATASETS[args.dataset]["class_names"],
        )
        logger.info("逐条预测写入 %s", pp)
        # 额外评测集另存：文件名在原名后加 __<集名>，带 pair_id（若有）
        for name, (frame, preds, probs) in extra_payloads.items():
            ep = preds_path(args.out_dir, f"{run_name}__{name}")
            save_predictions(
                ep, np.arange(len(frame)), frame["label"].to_numpy(), preds, probs,
                DATASETS[args.dataset]["class_names"],
                pair_ids=frame["pair_id"].to_numpy() if "pair_id" in frame else None,
            )
            logger.info("额外评测预测写入 %s", ep)
    if args.save_val_preds:
        vp = val_preds_path(args.out_dir, run_name)
        save_predictions(
            vp, np.arange(len(val_labels)), val_labels, val_preds, val_probs,
            DATASETS[args.dataset]["class_names"],
        )
        logger.info("验证集逐条预测写入 %s", vp)
    if args.save_model:
        cp = checkpoint_path(run_name)
        cp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "vocab": vocab.itos,
            "config": model_config,
            "run_name": run_name,
        }, cp)
        logger.info("最佳轮次模型写入 %s", cp)
    logger.info("最佳轮次=%d 停止轮次=%d 验证准确率=%.4f 测试准确率=%.4f 宏F1=%.4f",
                best_epoch, stopped_epoch, best_val, metrics["accuracy"], metrics["macro_f1"])
    logger.info("结果写入 %s", out)


if __name__ == "__main__":
    main()
