"""TF-IDF + 逻辑回归基线（流程第 4 步）。

1~2 元词组 TF-IDF（sublinear_tf，min_df=2），逻辑回归的 C 在验证集上
从 {0.25, 1, 4} 里选，选好后在官方测试集上只评估一次。

分词直接复用 data.tokenize，和神经网络看到的是同一套词，
这样两种方法的差别主要来自表示和模型，而不是分词。
结果 JSON 的字段与神经网络一致（见 train.py），方便 summarize.py 汇总。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

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
    tokenize_text,
    truncate_text,
)
from engine import classification_metrics
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
C_GRID = (0.25, 1.0, 4.0)


def count_linear_params(model) -> int:
    """非零系数 + 截距。多分类是 OneVsRestClassifier，需要逐个二分类器数。"""
    estimators = getattr(model, "estimators_", [model])
    total = 0
    for est in estimators:
        total += int(np.count_nonzero(est.coef_)) + int(est.intercept_.size)
    return total


def build_vectorizer(tokenizer: str = "word", dataset: str = "imdb") -> TfidfVectorizer:
    """建 TF-IDF 特征提取器。

    sublinear_tf 用 1+log(tf) 代替原始词频，避免一个词重复很多次
    就把权重拉得过大；min_df=2 丢掉只出现一次的词，降维也降噪。
    COLD：字用 analyzer="char"、1~3 gram；词用 jieba、1~2 gram。
    默认参数保持阶段 1 的英文正则分词与 1~2 gram。
    """
    if tokenizer == "char":
        return TfidfVectorizer(
            analyzer="char",
            lowercase=False,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=2,
            dtype=np.float32,
        )
    if tokenizer != "word":
        raise ValueError(f"未知分词器 {tokenizer!r}，可选 char / word")
    return TfidfVectorizer(
        tokenizer=lambda text: tokenize_text(text, dataset, "word"),
        token_pattern=None,  # 已经有 tokenizer，关掉 sklearn 默认的正则
        lowercase=False,     # tokenize 里已经小写化
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        dtype=np.float32,
    )


def fit_and_select(train_texts, train_labels, val_texts, val_labels, seed: int, log,
                   tokenizer: str = "word", dataset: str = "imdb"):
    """拟合向量化器，在验证集上选 C，返回 (vectorizer, 最佳模型, 记录字典)。"""
    vectorizer = build_vectorizer(tokenizer, dataset)
    t0 = time.perf_counter()
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    vectorize_sec = time.perf_counter() - t0
    log(f"TF-IDF 特征数={x_train.shape[1]}，向量化耗时={vectorize_sec:.1f}s")

    best = {"C": None, "val_acc": -1.0, "model": None, "fit_sec": 0.0, "by_C": {}}
    n_classes = len(np.unique(train_labels))
    for c in C_GRID:
        base = LogisticRegression(C=c, max_iter=2000, solver="liblinear", random_state=seed)
        # sklearn 1.9 的 liblinear 不再直接支持多分类，显式包一层一对多
        model = OneVsRestClassifier(base) if n_classes > 2 else base
        t = time.perf_counter()
        model.fit(x_train, train_labels)
        fit_sec = time.perf_counter() - t
        val_acc = float(model.score(x_val, val_labels))
        best["by_C"][str(c)] = {"val_accuracy": val_acc, "fit_sec": fit_sec}
        log(f"C={c:<5} val_acc={val_acc:.4f} fit={fit_sec:.1f}s")
        # 验证准确率相同时取更小的 C（正则更强，模型更简单）
        if val_acc > best["val_acc"]:
            best.update({"C": c, "val_acc": val_acc, "model": model, "fit_sec": fit_sec})
    return vectorizer, best, {"vectorize_sec": vectorize_sec, "n_features": x_train.shape[1]}


def evaluate_extra_set(name: str, vectorizer, model, max_len, class_names,
                       tokenizer: str = "word", dataset: str = "imdb"):
    """在额外评测集上跑一次，返回 (DataFrame, 预测, 概率, 指标)。

    TF-IDF 的截断规则和训练时一致：先按同一分词器截前 max_len 个
    token，再拼回字符串。max_len=None 时不截断。
    """
    frame = load_eval_set(name)
    texts = [truncate_text(t, max_len, tokenizer, dataset) for t in frame["text"]]
    x = vectorizer.transform(texts)
    preds = model.predict(x)
    probs = model.predict_proba(x)
    metrics = classification_metrics(frame["label"].to_numpy(), preds, class_names)
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
    parser = argparse.ArgumentParser(description="TF-IDF + 逻辑回归基线")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--tokenizer", choices=("char", "word"), default=None,
                        help="char 按字 1~3 gram，word 用 jieba 1~2 gram；COLD 默认 char")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="main", help="main=正式结果，smoke=冒烟")
    parser.add_argument("--train_frac", type=float, default=None,
                        help="训练集使用比例，默认 1.0；与 --train_n 不能同时给出")
    parser.add_argument("--train_n", type=int, default=None,
                        help="按标签分层抽取的绝对条数，抽样种子等于 --seed")
    parser.add_argument("--max_len", type=int, default=None,
                        help="只保留前 max_len 个 token（与神经网络同一分词），默认不截断")
    parser.add_argument("--save_preds", action="store_true",
                        help="把测试集逐条预测写到结果目录的 preds/ 下")
    parser.add_argument("--save_val_preds", action="store_true",
                        help="用选定的 C 与最终模型预测验证集，写到 preds/<run_name>__val.parquet")
    parser.add_argument("--test_limit", type=int, default=None,
                        help="只评测测试集前 N 条；默认不限，冒烟用")
    parser.add_argument("--decontaminate", dest="decontaminate", action="store_true",
                        default=None, help="剔除污染/重复；COLD 默认开启，英文默认关闭")
    parser.add_argument("--no_decontaminate", dest="decontaminate", action="store_false",
                        help="显式关闭去污染（COLD 也关）")
    parser.add_argument("--extra_eval", default="",
                        help="逗号分隔的评测集名，训练后逐个评测并写入 extra_eval")
    parser.add_argument("--out_dir", default=None,
                        help="结果目录，默认 results/；冒烟用 results/smoke3")
    parser.add_argument("--run_suffix", default="",
                        help="文件名后缀，同 tag 下区分 n/max_len 等设置")
    parser.add_argument("--aug_csv", default="",
                        help="D1 增强 CSV；默认空，不追加。给了只在去重后加训练集")
    # 下面两个参数只为和神经网络冒烟命令保持同一套写法；TF-IDF 本身在 CPU 上训练，也没有轮次
    parser.add_argument("--device", default="cpu", help="仅为命令统一而接受；sklearn 始终用 CPU")
    parser.add_argument("--epochs", type=int, default=1, help="逻辑回归没有轮次概念，接受但忽略")
    args = parser.parse_args()

    if args.train_n is not None and args.train_frac is not None:
        raise SystemExit("--train_n 与 --train_frac 不能同时给出")
    if args.tokenizer is None:
        args.tokenizer = "char" if args.dataset == "cold" else "word"
    if args.decontaminate is None:
        args.decontaminate = args.dataset == "cold"
    args.train_frac = 1.0 if args.train_frac is None else args.train_frac
    if not 0.0 < args.train_frac <= 1.0:
        raise SystemExit("--train_frac 必须在 (0, 1] 内")
    if args.train_n is not None and args.train_n <= 0:
        raise SystemExit("--train_n 必须为正整数")
    if args.max_len is not None and args.max_len <= 0:
        raise SystemExit("--max_len 必须为正整数或省略")
    if args.test_limit is not None and args.test_limit <= 0:
        raise SystemExit("--test_limit 必须为正整数或省略")

    tag = args.tag
    suffix = f"_{args.run_suffix}" if args.run_suffix else ""
    run_name = f"{args.dataset}_tfidf_{tag}{suffix}_s{args.seed}"
    logger = setup_logger(ROOT / "logs" / f"{run_name}.log")
    logger.info("参数: %s", vars(args))

    try:
        extra_names = parse_extra_eval(args.extra_eval)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    splits = load_splits(args.dataset)
    removed = {"train": 0, "val": 0}
    if args.decontaminate and args.dataset == "cold":
        # COLD 跨划分去重在抽样前做，剔除条数按官方划分统计
        splits, removed = deduplicate_cold_splits(splits)
    train = splits["train"]
    if args.train_n is not None:
        train = stratified_sample_n(train, args.train_n, args.seed)
    elif args.train_frac < 1.0:
        # 只抽训练部分；验证/测试保持完整，评估口径不变
        train = subsample(train, args.train_frac, args.seed)
    # D1：去重/抽样之后追加增强行，只动训练集；默认为空时逐条不变
    train, n_aug = append_aug(train, args.aug_csv)
    if n_aug:
        logger.info("追加增强 %d 条：%s", n_aug, args.aug_csv)
    val = splits["val"]
    test = splits["test"]
    if args.test_limit is not None:
        # 冒烟专用：只取官方测试集前 N 条，正式运行不传
        test = test.iloc[:args.test_limit].reset_index(drop=True)
    if args.decontaminate and args.dataset != "cold":
        # 词表（TF-IDF 的 vocabulary_）在剔除之后拟合
        train, removed["train"] = decontaminate_frame(train)
        val, removed["val"] = decontaminate_frame(val)
    logger.info("分词器=%s 样本数: train=%d val=%d test=%d",
                args.tokenizer, len(train), len(val), len(test))
    if args.decontaminate:
        logger.info("去污染剔除: train=%d val=%d", removed["train"], removed["val"])

    # 截断在分词之后做，保证 TF-IDF 与神经网络看到完全相同的词序列
    split_texts = {}
    trunc_stats = {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        split_texts[name] = [
            truncate_text(t, args.max_len, args.tokenizer, args.dataset)
            for t in frame["text"]
        ]
        if args.max_len is None:
            trunc_stats[name] = None
        else:
            n_trunc = sum(
                len(tokenize_text(t, args.dataset, args.tokenizer)) > args.max_len
                for t in frame["text"]
            )
            trunc_stats[name] = n_trunc / len(frame) if len(frame) else 0.0
    logger.info("截断比例: train=%s val=%s test=%s", *(
        "—" if trunc_stats[k] is None else f"{trunc_stats[k]:.4f}"
        for k in ("train", "val", "test")
    ))

    vectorizer, best, feat_info = fit_and_select(
        split_texts["train"], train["label"].to_numpy(),
        split_texts["val"], val["label"].to_numpy(),
        args.seed, logger.info, args.tokenizer, args.dataset,
    )
    if best["model"] is None:
        raise RuntimeError("没有选出任何 C，检查验证集是否为空")

    x_test = vectorizer.transform(split_texts["test"])
    preds = best["model"].predict(x_test)
    # 二分类用 prob_pos，多分类保留各类概率；与神经网络的 softmax 口径一致
    probs = best["model"].predict_proba(x_test)
    class_names = DATASETS[args.dataset]["class_names"]
    metrics = classification_metrics(
        test["label"].to_numpy(), preds, class_names
    )
    fine_grained = fine_grained_stats(test, preds)
    val_preds = val_probs = None
    if args.save_val_preds:
        # 用最终选定的 C 与同一个向量化器，验证集只预测一次
        x_val = vectorizer.transform(split_texts["val"])
        val_preds = best["model"].predict(x_val)
        val_probs = best["model"].predict_proba(x_val)
    extra_eval, extra_payloads = {}, {}
    for name in extra_names:
        if not eval_set_available(name):
            logger.info("额外评测 %s: 缺文件，跳过", name)
            continue
        try:
            frame, e_preds, e_probs, e_metrics = evaluate_extra_set(
                name, vectorizer, best["model"], args.max_len, class_names,
                args.tokenizer, args.dataset,
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
    # 线性模型的“参数量”按非零系数 + 截距算（稀疏模型不把 0 当参数）
    num_params = count_linear_params(best["model"])
    # 这一个是“训练一个最终模型”的耗时，不含为选 C 额外拟合的另外两个
    fit_sec = feat_info["vectorize_sec"] + best["fit_sec"]

    history = [{
        "epoch": 1,
        "train_loss": None,  # 逻辑回归不记录每步损失，字段保留以统一格式
        "val_loss": None,
        "val_accuracy": best["val_acc"],
        "train_time_sec": fit_sec,
        "gpu_peak_memory_mb": None,
    }]
    payload = {
        "dataset": args.dataset,
        "model": "tfidf",
        "tag": tag,
        "seed": args.seed,
        "run_name": run_name,
        "run_at": now_iso(),
        "config": {
            "dataset": args.dataset,
            "model": "tfidf",
            "seed": args.seed,
            "tag": tag,
            "train_frac": args.train_frac,
            "train_n": args.train_n,
            "max_len": args.max_len,
            "save_preds": args.save_preds,
            "save_val_preds": args.save_val_preds,
            "test_limit": args.test_limit,
            "out_dir": args.out_dir,
            "run_suffix": args.run_suffix,
            "device": "cpu (sklearn)",
            "epochs_requested": args.epochs,
            "tokenizer": args.tokenizer,
            "analyzer": "char" if args.tokenizer == "char" else "tokenizer",
            "ngram_range": [1, 3] if args.tokenizer == "char" else [1, 2],
            "sublinear_tf": True,
            "min_df": 2,
            "C_grid": list(C_GRID),
            "C_selected": best["C"],
            "solver": "liblinear",
            "multiclass": "ovr" if len(np.unique(train["label"])) > 2 else "binary",
            "max_iter": 2000,
            "n_features": feat_info["n_features"],
            "num_classes": len(DATASETS[args.dataset]["class_names"]),
            "n_train": len(train),
            "aug_csv": args.aug_csv or None,
            "aug_n_appended": n_aug,
            "n_val": len(val),
            "n_test": len(test),
        },
        "vocab_size": None,
        "num_params": num_params,
        "truncation_rate": trunc_stats["train"],
        "truncation_rate_by_split": trunc_stats,
        "epochs": 1,
        "stopped_epoch": 1,
        "early_stopped": False,
        "n_train_used": len(train),
        "best_epoch": 1,
        "best_val_accuracy": best["val_acc"],
        "val_accuracy_by_C": best["by_C"],
        "history": history,
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_per_class_f1": metrics["per_class_f1"],
        "decontaminated": bool(args.decontaminate),
        "decontaminate_removed": removed,
        "aug_csv": args.aug_csv or None,
        "aug_n_appended": n_aug,
        "extra_eval": extra_eval,
        "train_time_per_epoch_sec": fit_sec,
        "total_train_time_sec": fit_sec,
        "gpu_peak_memory_mb": None,
    }
    if args.dataset != "cold":
        # 英文 JSON 的 config 保持阶段 1 字段
        for key in ("tokenizer", "analyzer"):
            payload["config"].pop(key, None)
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
        save_predictions(
            pp, np.arange(len(preds)), test["label"].to_numpy(), preds, probs,
            DATASETS[args.dataset]["class_names"],
        )
        logger.info("逐条预测写入 %s", pp)
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
        save_predictions(
            vp, np.arange(len(val_preds)), val["label"].to_numpy(),
            val_preds, val_probs, class_names,
        )
        logger.info("验证集逐条预测写入 %s", vp)
    logger.info("测试准确率=%.4f 宏F1=%.4f", metrics["accuracy"], metrics["macro_f1"])
    logger.info("结果写入 %s", out)


if __name__ == "__main__":
    main()
