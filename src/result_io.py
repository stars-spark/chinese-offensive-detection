"""结果 JSON 与日志的小工具（train.py 和 baseline_tfidf.py 共用）。

两种方法写出的 JSON 字段必须一致，summarize.py 才能统一读。
这里只负责时间戳、日志和 JSON 落盘，不掺训练逻辑。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def now_iso() -> str:
    """带时区的本地时间，方便知道结果是什么时候跑的。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def setup_logger(log_path: Path) -> logging.Logger:
    """同时写文件和终端；重复调用不会叠加 handler。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"exp.{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def save_json(path: Path, payload: dict) -> None:
    """utf-8 + 中文不转义，方便直接打开检查。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_results_dir(out_dir: str | Path | None) -> Path:
    """把 ``--out_dir`` 解析成绝对路径；None 表示原来的 ``results/``。

    冒烟用 ``results/smoke2``，正式运行用默认值，两者互不覆盖。
    """
    path = Path(out_dir) if out_dir else ROOT / "results"
    if not path.is_absolute():
        path = ROOT / path
    return path


def result_json_path(out_dir: str | Path | None, run_name: str) -> Path:
    return resolve_results_dir(out_dir) / f"{run_name}.json"


def preds_path(out_dir: str | Path | None, run_name: str) -> Path:
    """逐条预测与结果 JSON 同名，放在结果目录的 preds/ 下。"""
    return resolve_results_dir(out_dir) / "preds" / f"{run_name}.parquet"


def val_preds_path(out_dir: str | Path | None, run_name: str) -> Path:
    """验证集逐条预测，文件名加 ``__val``，与测试集预测分开。"""
    return preds_path(out_dir, f"{run_name}__val")


def checkpoint_path(run_name: str) -> Path:
    """最佳轮次参数固定放项目根的 checkpoints/，不随 out_dir 改变。"""
    return ROOT / "checkpoints" / f"{run_name}.pt"


def save_predictions(
    path: Path,
    index: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray | None,
    class_names: list[str],
    pair_ids: np.ndarray | None = None,
) -> None:
    """写测试集逐条预测。

    ``index`` 是在官方测试集里的行号（0 起），不是 pandas 索引，便于
    和原始 parquet 对齐。``pair_ids`` 只给阶段 3 的配对评测集用，写在
    label 前面，方便按 pair_id 把原始与改写两份预测对齐。
    二分类写 ``prob_pos``（正类概率），多分类写 ``prob_0``...``prob_k``；
    概率全部是 softmax / predict_proba 的原值。
    """
    data = {"index": np.asarray(index, dtype=np.int64)}
    if pair_ids is not None:
        data["pair_id"] = np.asarray(pair_ids, dtype=np.int64)
    data.update({
        "label": np.asarray(labels, dtype=np.int64),
        "pred": np.asarray(preds, dtype=np.int64),
    })
    if probs is not None:
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[0] != len(data["index"]):
            raise ValueError(f"概率形状 {probs.shape} 与预测条数不符")
        if len(class_names) == 2:
            data["prob_pos"] = probs[:, 1]
        else:
            for i in range(probs.shape[1]):
                data[f"prob_{i}"] = probs[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(path, index=False)
