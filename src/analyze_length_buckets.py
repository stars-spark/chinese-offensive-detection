"""实验 D：按 IMDB 测试集原始长度分组的准确率。

读取实验 B（max_len=256）和实验 F（DistilBERT）保存的逐条预测 parquet，
按每条评论在**截断之前**的 token 数分四组：≤128、129～256、257～512、
>512。每组内先算每个种子的准确率，再跨种子取均值 ± 样本标准差。

输出：
- ``results/长度分组.md``：样本数和各方法准确率表；
- ``figures/长度分组_准确率.png``：分组柱状图；
- ``results/长度分组.json``：同样的数字，供 plot_stage2.py 写汇总。
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_splits, tokenize
from plot_style import save_figure, setup_chinese_font
from result_io import now_iso

ROOT = Path(__file__).resolve().parents[1]
BUCKETS = ((0, 128, "≤128"), (129, 256, "129～256"),
           (257, 512, "257～512"), (513, None, ">512"))
BUCKET_LABELS = tuple(b[2] for b in BUCKETS)
MODEL_ORDER = {"tfidf": 0, "textcnn": 1, "bilstm": 2, "transformer": 3, "distilbert": 4}
MODEL_TITLE = {
    "tfidf": "TF-IDF + LR", "textcnn": "TextCNN", "bilstm": "BiLSTM",
    "transformer": "Transformer", "distilbert": "DistilBERT",
}


def length_bucket(n_tokens: int) -> str:
    """把原始 token 数映射到四个分组之一，边界归入较低的一组。"""
    if n_tokens <= 128:
        return "≤128"
    if n_tokens <= 256:
        return "129～256"
    if n_tokens <= 512:
        return "257～512"
    return ">512"


def bucket_accuracy(labels, preds, bucket_ids) -> dict[str, float | None]:
    """对每个分组算准确率；该组没有样本时返回 None。"""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    bucket_ids = np.asarray(bucket_ids)
    out = {}
    for i, label in enumerate(BUCKET_LABELS):
        sel = bucket_ids == i
        out[label] = float((labels[sel] == preds[sel]).mean()) if sel.any() else None
    return out


def parse_pred_name(path: Path) -> dict | None:
    """从 ``imdb_transformer_len_ml256_s0.parquet`` 解析方法、seed、后缀。"""
    stem = path.stem
    m = re.match(r"^(?P<prefix>.+)_s(?P<seed>\d+)$", stem)
    if not m:
        return None
    prefix, seed = m.group("prefix"), int(m.group("seed"))
    if prefix.startswith("ag_news_"):
        dataset, rest = "ag_news", prefix[len("ag_news_"):]
    elif prefix.startswith("imdb_"):
        dataset, rest = "imdb", prefix[len("imdb_"):]
    else:
        return None
    parts = rest.split("_")
    model = parts[0]
    if model not in MODEL_ORDER or len(parts) < 2:
        return None
    tag = parts[1]
    suffix = "_".join(parts[2:])
    return {"dataset": dataset, "model": model, "tag": tag,
            "suffix": suffix, "seed": seed}


def discover_preds(tag_len: str, tag_bert: str) -> list[Path]:
    """默认找实验 B 的 max_len=256 和实验 F 的完整 IMDB 预测。"""
    pred_dir = ROOT / "results" / "preds"
    full_n = len(load_splits("imdb")["train"])
    found = []
    for path in sorted(pred_dir.glob("*.parquet")):
        info = parse_pred_name(path)
        if info is None or info["dataset"] != "imdb":
            continue
        if info["tag"] == tag_len and info["model"] in ("tfidf", "textcnn", "bilstm", "transformer"):
            if info["suffix"] in ("", "ml256"):
                found.append(path)
        elif info["tag"] == tag_bert and info["model"] == "distilbert":
            # 只取完整训练集（n=22500）的预测，学习曲线的其它 n 不混进来
            if info["suffix"] in ("", f"n{full_n}"):
                found.append(path)
    return found


def load_imdb_test() -> tuple[np.ndarray, np.ndarray]:
    """返回官方测试集标签和截断前的 token 数，顺序与 parquet 行号一致。"""
    test = load_splits("imdb")["test"]
    labels = test["label"].to_numpy()
    lengths = np.array([len(tokenize(t)) for t in test["text"]], dtype=np.int64)
    return labels, lengths


def aggregate(paths: list[Path], labels: np.ndarray, lengths: np.ndarray) -> dict:
    """把每个预测文件折算成每组的逐种子准确率，再跨种子汇总。"""
    bucket_ids = np.array([BUCKET_LABELS.index(length_bucket(n)) for n in lengths])
    per_method: dict[str, dict[int, dict]] = {}
    used = []
    for path in paths:
        info = parse_pred_name(path)
        if info is None:
            print(f"跳过（文件名无法解析）: {path.name}")
            continue
        df = pd.read_parquet(path)
        missing = {"index", "label", "pred"} - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} 缺少列 {sorted(missing)}")
        idx = df["index"].to_numpy(dtype=np.int64)
        if idx.min() < 0 or idx.max() >= len(labels):
            raise ValueError(f"{path.name} 的 index 超出测试集范围")
        if not np.array_equal(df["label"].to_numpy(), labels[idx]):
            raise ValueError(f"{path.name} 的 label 与官方测试集不一致")
        acc = bucket_accuracy(labels[idx], df["pred"].to_numpy(), bucket_ids[idx])
        method = info["model"]
        per_method.setdefault(method, {})[info["seed"]] = acc
        used.append(f"{path.name}（{MODEL_TITLE[method]}，seed={info['seed']}）")

    stats = {}
    for method in sorted(per_method, key=lambda m: MODEL_ORDER.get(m, 99)):
        by_seed = per_method[method]
        seeds = sorted(by_seed)
        stats[method] = {}
        for label in BUCKET_LABELS:
            vals = [by_seed[s][label] for s in seeds if by_seed[s][label] is not None]
            stats[method][label] = {
                "mean": float(statistics.mean(vals)) if vals else None,
                "std": float(statistics.stdev(vals)) if len(vals) > 1 else None,
                "n_seeds": len(vals),
            }
    counts = {label: int((bucket_ids == i).sum()) for i, label in enumerate(BUCKET_LABELS)}
    return {"stats": stats, "counts": counts, "used_files": used}


def fmt_acc(cell: dict) -> str:
    if cell["mean"] is None:
        return "—"
    if cell["std"] is None:
        return f"{cell['mean'] * 100:.2f}"
    return f"{cell['mean'] * 100:.2f} ± {cell['std'] * 100:.2f}"


def render_md(result: dict, pred_paths: list[Path]) -> str:
    lines = [
        "# IMDB 测试集按原始长度分组的准确率",
        "",
        f"生成时间：{now_iso()}",
        "",
        "分组依据是评论在截断之前的正则分词 token 数；"
        "预测来自实验 B（max_len=256）和实验 F（DistilBERT）。"
        "每组内先算每个种子的准确率，再跨种子取均值 ± 样本标准差（百分数，两位小数）。",
        "",
        "## 分组样本数",
        "",
        "| 分组 | 样本数 |",
        "|---|---:|",
    ]
    for label in BUCKET_LABELS:
        lines.append(f"| {label} | {result['counts'][label]} |")
    lines += [
        "",
        "## 各方法准确率（%）",
        "",
        "| 方法 | " + " | ".join(f"{b}（n={result['counts'][b]}）" for b in BUCKET_LABELS) + " |",
        "|---|" + "---:|" * len(BUCKET_LABELS),
    ]
    for method, cells in result["stats"].items():
        lines.append(
            f"| {MODEL_TITLE.get(method, method)} | "
            + " | ".join(fmt_acc(cells[b]) for b in BUCKET_LABELS) + " |"
        )
    lines += ["", "## 使用的预测文件", ""]
    lines += [f"- `{p.name}`" for p in pred_paths]
    lines += ["", "说明：只有一个种子的方法不显示标准差；"
              "种子数见上方文件名列表。", ""]
    return "\n".join(lines)


def make_figure(result: dict, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    setup_chinese_font()
    methods = list(result["stats"])
    x = np.arange(len(BUCKET_LABELS), dtype=float)
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for i, method in enumerate(methods):
        cells = result["stats"][method]
        means = [0.0 if cells[b]["mean"] is None else cells[b]["mean"] * 100 for b in BUCKET_LABELS]
        errs = [0.0 if cells[b]["std"] is None else cells[b]["std"] * 100 for b in BUCKET_LABELS]
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, means, width,
               yerr=errs, capsize=2.5, label=MODEL_TITLE.get(method, method))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={result['counts'][b]})" for b in BUCKET_LABELS])
    ax.set_xlabel("IMDB 测试集原始 token 数分组")
    ax.set_ylabel("测试准确率（%）")
    ax.set_title("按评论长度分组的准确率（误差棒为种子间标准差）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    save_figure(fig, out_png)


def main() -> None:
    import glob
    import json

    p = argparse.ArgumentParser(description="按 IMDB 测试集原始长度分组统计准确率")
    p.add_argument("--preds", nargs="+", default=None,
                   help="预测 parquet 路径或通配符；不传则自动找实验 B/F 的结果")
    p.add_argument("--tag-len", default="len", help="实验 B 的 tag")
    p.add_argument("--tag-bert", default="bert", help="实验 F 的 tag")
    p.add_argument("--out-md", default="results/长度分组.md")
    p.add_argument("--out-json", default="results/长度分组.json")
    p.add_argument("--out-fig", default="figures/长度分组_准确率.png")
    args = p.parse_args()

    if args.preds:
        paths = []
        for pattern in args.preds:
            q = Path(pattern)
            if not q.is_absolute():
                q = ROOT / q
            paths.extend(Path(x) for x in sorted(glob.glob(str(q))))
    else:
        paths = discover_preds(args.tag_len, args.tag_bert)
    if not paths:
        raise SystemExit("没有找到预测文件；请用 --preds 指定，或先跑实验 B/F 并加 --save_preds")

    labels, lengths = load_imdb_test()
    result = aggregate(paths, labels, lengths)
    if not result["stats"]:
        raise SystemExit("预测文件里没有可识别的方法")

    out_md = Path(args.out_md)
    if not out_md.is_absolute():
        out_md = ROOT / out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(result, paths), encoding="utf-8")

    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = ROOT / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": now_iso(),
            "dataset": "imdb",
            "buckets": [
                {"label": label, "min": lo, "max": hi, "n": result["counts"][label]}
                for lo, hi, label in BUCKETS
            ],
            "methods": result["stats"],
            "files": [x.name for x in paths],
        }, f, ensure_ascii=False, indent=2)

    make_figure(result, Path(args.out_fig) if Path(args.out_fig).is_absolute()
                else ROOT / args.out_fig)
    print(f"写入 {out_md}")
    print(f"写入 {out_json}")
    print(f"写入 {args.out_fig}")


if __name__ == "__main__":
    main()


