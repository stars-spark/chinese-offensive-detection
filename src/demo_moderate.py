"""三级审核演示：TF-IDF（字）→ rbt3 → 人工复核。

第一级在 COLD 训练集上现拟合字级 TF-IDF 与逻辑回归，设置与
baseline_tfidf.py 的 cold/char 一致，C 取正式结果里验证集选出的值。
第二级载入 checkpoints/cold_bert_demo_rbt3_s0，最大长度 192、头部截断。
两个阈值都在验证集上确定：τ1 复用 cascade.select_tau，τ2 取 rbt3
验证集置信度的 10% 分位数。程序只做演示，不写任何结果文件。
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from baseline_tfidf import build_vectorizer
from cascade import confidence_from_prob_pos, select_tau
from data import clean_text, deduplicate_cold_splits, load_splits

ROOT = Path(__file__).resolve().parents[1]
TFIDF_JSON = ROOT / "results" / "cold_tfidf_main_char_s0.json"
TFIDF_VAL = ROOT / "results" / "preds" / "cold_tfidf_main_char_s0__val.parquet"
BERT_VAL = ROOT / "results" / "preds" / "cold_bert_demo_rbt3_s0__val.parquet"
BERT_CKPT = ROOT / "checkpoints" / "cold_bert_demo_rbt3_s0"
BERT_MAX_LEN = 192
TAU2_QUANTILE = 0.10

LEVEL_NAMES = {1: "第一级 TF-IDF", 2: "第二级 rbt3", 3: "人工复核"}
JUDGE_NAMES = {0: "安全", 1: "攻击性"}
LEVEL_STYLES = {1: "green", 2: "yellow", 3: "red"}


def assign_level(tfidf_conf: float, bert_conf: float | None,
                 tau1: float, tau2: float) -> int:
    """按两级阈值分配处理层级：1=TF-IDF，2=rbt3，3=人工。

    第一级置信度不低于 τ1 就直接采信；否则看第二级，第二级置信度
    不低于 τ2 才采信，仍低于 τ2 的交给人工。bert_conf 为 None 表示
    还没跑第二级（第一级已经直接采信）。
    """
    if tfidf_conf >= tau1:
        return 1
    if bert_conf is not None and bert_conf >= tau2:
        return 2
    return 3


def truncate_display(text: str, limit: int = 30) -> str:
    """表格里的单行摘要：压掉换行与连续空白，超过 limit 字加省略号。"""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def format_confidence(conf: float | None) -> str:
    """置信度显示成百分比；人工复核行没有最终判断，显示第二级置信度。"""
    return "—" if conf is None else f"{conf * 100:.1f}%"


def load_tfidf():
    """按 cold/char 设置重新拟合第一级，返回 (向量化器, 分类器, C)。"""
    with open(TFIDF_JSON, encoding="utf-8") as handle:
        c_value = float(json.load(handle)["config"]["C_selected"])
    splits = load_splits("cold")
    splits, _ = deduplicate_cold_splits(splits)
    train = splits["train"]
    vectorizer = build_vectorizer("char", "cold")
    x_train = vectorizer.fit_transform(train["text"].tolist())
    model = LogisticRegression(
        C=c_value, max_iter=2000, solver="liblinear", random_state=0,
    )
    model.fit(x_train, train["label"].to_numpy())
    return vectorizer, model, c_value


def load_thresholds() -> dict:
    """在验证集上定 τ1 与 τ2；两条预测必须逐条对齐。"""
    tfidf = pd.read_parquet(TFIDF_VAL)
    bert = pd.read_parquet(BERT_VAL)
    if len(tfidf) != len(bert) or not np.array_equal(
        tfidf["label"].to_numpy(), bert["label"].to_numpy()
    ):
        raise ValueError("验证集预测没有对齐，无法选阈值")
    conf_tfidf = confidence_from_prob_pos(tfidf["prob_pos"].to_numpy())
    conf_bert = confidence_from_prob_pos(bert["prob_pos"].to_numpy())
    chosen = select_tau(
        tfidf["pred"].to_numpy(), conf_tfidf,
        bert["pred"].to_numpy(), tfidf["label"].to_numpy(),
    )
    tau1 = float(chosen["tau"])
    tau2 = float(np.quantile(conf_bert, TAU2_QUANTILE))
    return {
        "tau1": tau1,
        "tau2": tau2,
        "val_handover": chosen["val_handover"],
        "val_accuracy": chosen["val_accuracy"],
        "n_val": len(tfidf),
    }


def resolve_device(name: str) -> str:
    """auto 有 GPU 就用 GPU，否则退回 CPU；显式 cuda 不可用时报错。"""
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 --device cuda，但这台机器没有可用的 CUDA")
    return name


def load_bert(device: str):
    """载入 rbt3 检查点与分词器，返回 (分词器, 模型, 设备)。"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    tokenizer = AutoTokenizer.from_pretrained(
        BERT_CKPT, local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_CKPT, local_files_only=True,
    )
    model.to(device).eval()
    return tokenizer, model, torch.device(device)


def encode_batch(tokenizer, texts: list[str], max_len: int = BERT_MAX_LEN):
    """按头部截断编码成定长 id/mask，规则与 finetune_bert.encode_split 相同。"""
    import torch

    raw = tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ids = np.full((len(texts), max_len), pad_id, dtype=np.int64)
    mask = np.zeros((len(texts), max_len), dtype=np.int64)
    budget = max_len - 2
    for i, seq in enumerate(raw):
        row = [tokenizer.cls_token_id] + list(seq[:budget]) + [tokenizer.sep_token_id]
        ids[i, :len(row)] = row
        mask[i, :len(row)] = 1
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
    )


def bert_predict(texts: list[str], tokenizer, model, device) -> list[tuple[int, float]]:
    """第二级逐条预测，返回 [(标签, 置信度)]；置信度是 softmax 最大值。"""
    import torch

    ids, mask = encode_batch(tokenizer, texts)
    with torch.no_grad():
        logits = model(
            input_ids=ids.to(device), attention_mask=mask.to(device)
        ).logits.float()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return [
        (int(row.argmax()), float(row.max())) for row in probs
    ]


def tfidf_predict(text: str, vectorizer, model) -> tuple[int, float]:
    """第一级单条预测；空字符串会得到全零特征，仍能正常走完。"""
    probs = model.predict_proba(vectorizer.transform([text]))[0]
    return int(model.classes_[int(probs.argmax())]), float(probs.max())


def clean_for_prediction(text: str) -> str:
    """与训练数据相同的清洗；显示仍用用户原文。"""
    return clean_text(text, "cold")


def classify_one(text: str, tfidf_fn, bert_fn,
                 tau1: float, tau2: float) -> dict:
    """对一条文本跑三级流程，记录最终层级、判断、置信度和耗时。

    tfidf_fn / bert_fn 都接收一条文本并返回 (标签, 置信度)。第二级只在
    第一级交出时才调用；走人工复核时没有最终判断，标签记为 None。
    """
    start = time.perf_counter()
    pred1, conf1 = tfidf_fn(text)
    if conf1 >= tau1:
        level, pred, conf = 1, pred1, conf1
    else:
        pred2, conf2 = bert_fn(text)
        if conf2 >= tau2:
            level, pred, conf = 2, pred2, conf2
        else:
            level, pred, conf = 3, None, conf2
    return {
        "text": text,
        "level": level,
        "pred": pred,
        "conf": conf,
        "tfidf_conf": conf1,
        "elapsed_ms": (time.perf_counter() - start) * 1000.0,
    }


def summarize_results(results: list[dict]) -> dict:
    """统计各层级条数、比例、总耗时与每千条耗时（秒）。"""
    n = len(results)
    counts = {level: 0 for level in (1, 2, 3)}
    for row in results:
        counts[row["level"]] += 1
    total_ms = float(sum(row["elapsed_ms"] for row in results))
    return {
        "n": n,
        "counts": counts,
        "ratios": {k: (v / n if n else 0.0) for k, v in counts.items()},
        "total_ms": total_ms,
        # 1000 条的总毫秒数换算成秒，恰好等于“总毫秒 / 条数”
        "per_1000_sec": (total_ms / n) if n else 0.0,
    }


def _display_width(text: str) -> int:
    """纯文本对齐用：全角/宽字符按 2 列算。"""
    return sum(
        2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        for ch in text
    )


def _pad(text: str, width: int, right: bool = False) -> str:
    gap = max(0, width - _display_width(text))
    return (" " * gap + text) if right else (text + " " * gap)


def _plain_table(results: list[dict]) -> None:
    """rich 不可用时的等宽表格，按显示宽度补空格。"""
    headers = ("序号", "文本", "判断", "置信度", "处理层级", "耗时(ms)")
    widths = (4, 62, 6, 8, 14, 10)
    print(" | ".join(_pad(h, w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for index, row in enumerate(results, 1):
        judge = "待人工" if row["level"] == 3 else JUDGE_NAMES[row["pred"]]
        cells = (
            str(index),
            truncate_display(row["text"]),
            judge,
            format_confidence(row["conf"]),
            LEVEL_NAMES[row["level"]],
            f"{row['elapsed_ms']:.1f}",
        )
        print(" | ".join(
            _pad(cell, width, right=(i in (0, 5)))
            for i, (cell, width) in enumerate(zip(cells, widths))
        ))


def render(results: list[dict], tau1: float, tau2: float,
           load_sec: float, device: str) -> None:
    """打印明细表与汇总；有 rich 用 rich，没有就退回纯文本对齐。"""
    stats = summarize_results(results)
    if stats["n"] == 0:
        print("没有可审核的文本。")
        return
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        _plain_table(results)
        print()
        print(f"总耗时 {stats['total_ms'] / 1000:.3f} s，"
              f"每千条 {stats['per_1000_sec']:.3f} s")
        return

    console = Console(width=122, highlight=False)
    console.print(
        "[bold]三级审核演示[/bold]  第一级 TF-IDF(char) → 第二级 rbt3 → 人工复核"
    )
    console.print(
        f"阈值：τ1={tau1:.3f}（验证集选出）  τ2={tau2:.3f}（rbt3 置信度 10% 分位）"
    )
    console.print(f"设备：{device}    模型与阈值载入：{load_sec:.1f} s")
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold", title_style="bold")
    table.add_column("序号", justify="right", no_wrap=True)
    table.add_column("文本", no_wrap=True)
    table.add_column("判断", no_wrap=True)
    table.add_column("置信度", justify="right", no_wrap=True)
    table.add_column("处理层级", no_wrap=True)
    table.add_column("耗时(ms)", justify="right", no_wrap=True)
    for index, row in enumerate(results, 1):
        judge = "待人工" if row["level"] == 3 else JUDGE_NAMES[row["pred"]]
        style = LEVEL_STYLES[row["level"]]
        table.add_row(
            str(index),
            Text(truncate_display(row["text"])),
            Text(judge, style=style),
            format_confidence(row["conf"]),
            Text(LEVEL_NAMES[row["level"]], style=style),
            f"{row['elapsed_ms']:.1f}",
        )
    console.print(table)
    console.print("[bold]汇总[/bold]")
    for level in (1, 2, 3):
        count = stats["counts"][level]
        console.print(
            f"  {LEVEL_NAMES[level]:<12} {count:>4} 条  "
            f"{stats['ratios'][level] * 100:>6.1f}%"
        )
    console.print(
        f"  总耗时 {stats['total_ms'] / 1000:.3f} s；"
        f"每千条 {stats['per_1000_sec']:.3f} s；"
        f"平均单条 {stats['total_ms'] / stats['n']:.1f} ms"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="三级审核演示：TF-IDF（字）→ rbt3 → 人工复核",
    )
    parser.add_argument(
        "--text", nargs="+", action="append", default=None, metavar="TEXT",
        help="待审核文本，可给多条；该参数也可重复出现",
    )
    parser.add_argument("--csv", type=Path, default=None, help="从 CSV 读文本")
    parser.add_argument("--col", default="text", help="CSV 的文本列名，默认 text")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="默认 auto：有 GPU 用 GPU，否则 CPU",
    )
    return parser


def read_inputs(groups: list[list[str]] | None, csv_path: Path | None,
                column: str) -> list[str]:
    """合并 --text 与 --csv 的输入；列名不对时给出可选的列。"""
    texts: list[str] = []
    for group in groups or []:
        texts.extend(group)
    if csv_path is not None:
        if not csv_path.exists():
            raise SystemExit(f"CSV 不存在：{csv_path}")
        frame = pd.read_csv(csv_path, keep_default_na=False, dtype=str)
        if column not in frame.columns:
            raise SystemExit(
                f"CSV 没有列 {column!r}，可选列：{', '.join(frame.columns)}"
            )
        texts.extend(frame[column].tolist())
    return [str(text) for text in texts]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.text and args.csv is None:
        parser.error("至少给一条 --text，或用 --csv/--col 指定输入")
    raw_texts = read_inputs(args.text, args.csv, args.col)
    if not raw_texts:
        print("输入为空，没有可审核的文本。")
        return

    start = time.perf_counter()
    vectorizer, tfidf_model, c_value = load_tfidf()
    thresholds = load_thresholds()
    device = resolve_device(args.device)
    tokenizer, bert_model, torch_device = load_bert(device)
    # 预热一次，免得第一条走第二级时把首次 kernel 启动算进单条耗时
    bert_predict([""], tokenizer, bert_model, torch_device)
    load_sec = time.perf_counter() - start

    def tfidf_fn(text: str) -> tuple[int, float]:
        return tfidf_predict(text, vectorizer, tfidf_model)

    def bert_fn(text: str) -> tuple[int, float]:
        return bert_predict([text], tokenizer, bert_model, torch_device)[0]

    results = []
    for raw in raw_texts:
        row = classify_one(
            clean_for_prediction(raw), tfidf_fn, bert_fn,
            thresholds["tau1"], thresholds["tau2"],
        )
        row["text"] = raw
        results.append(row)

    render(
        results, thresholds["tau1"], thresholds["tau2"], load_sec,
        f"{device}, TF-IDF C={c_value:g}",
    )


if __name__ == "__main__":
    main()
