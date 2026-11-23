"""实验 L：验证集定阈值、测试集评估的级联分析。

读 ``results/preds`` 下 TF-IDF 与 DistilBERT 的验证/测试逐条预测，按
按预定规则定 τ、组级联、算长度分组、成本、McNemar 与稳健性。
成本对照还读实验 J 的 head512、tail512、headtail、chunk 结果。
输出 ``results/级联.md``、``figures/级联.png`` 与
``figures/级联_成本.png``。缺文件时写“缺文件”并继续，不中断。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_length_buckets import BUCKET_LABELS, bucket_accuracy, length_bucket, load_imdb_test
from cascade import (
    cascade_accuracy,
    cascade_predict,
    confidence_from_prob_pos,
    handover_fraction,
    select_tau,
    test_curve,
    visible_pair_ids,
)
from plot_style import save_figure, setup_chinese_font
from result_io import now_iso
from significance import holm_adjust, mcnemar_exact, paired_metrics

ROOT = Path(__file__).resolve().parents[1]
TFIDF_NAME = "imdb_tfidf_cascade_s0"
SEEDS = (0, 1, 2)
MAX_LENS = (256, 512)
FAMILIES = (("contrast", "对比集"), ("cad", "CAD"))
LONG_TAG = "long"
J_READS = ("head512", "tail512", "headtail")
READ_TITLES = {
    "tfidf": "TF-IDF",
    "distilbert_256": "DistilBERT 256",
    "head512": "head512",
    "tail512": "tail512",
    "headtail": "headtail",
    "chunk": "chunk",
}
READ_ORDER = ("tfidf", "distilbert_256", "head512", "tail512", "headtail", "chunk")


def bert_run_name(max_len: int, seed: int) -> str:
    return f"imdb_distilbert_cascade_ml{max_len}_s{seed}"


def _resolve(path: str | Path | None, default: Path) -> Path:
    p = Path(path) if path else default
    return p if p.is_absolute() else ROOT / p


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_preds(preds_dir: Path, name: str) -> pd.DataFrame | None:
    path = preds_dir / f"{name}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    missing = {"index", "label", "pred", "prob_pos"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} 缺少列 {sorted(missing)}")
    if not frame["index"].is_unique:
        raise ValueError(f"{path.name} 的 index 有重复")
    return frame


def _align_on_index(tfidf: pd.DataFrame | None, bert: pd.DataFrame | None) -> dict | None:
    """按官方测试集/验证集的行号对齐两份预测。"""
    if tfidf is None or bert is None:
        return None
    merged = tfidf.merge(bert, on="index", suffixes=("_tfidf", "_bert"))
    if merged.empty:
        return None
    if not np.array_equal(merged["label_tfidf"], merged["label_bert"]):
        raise ValueError("TF-IDF 与 DistilBERT 的 label 在同一 index 上不一致")
    return {
        "index": merged["index"].to_numpy(dtype=np.int64),
        "label": merged["label_tfidf"].to_numpy(dtype=np.int64),
        "tfidf_pred": merged["pred_tfidf"].to_numpy(dtype=np.int64),
        "bert_pred": merged["pred_bert"].to_numpy(dtype=np.int64),
        "conf": confidence_from_prob_pos(merged["prob_pos_tfidf"].to_numpy()),
    }


def _fmt_pct(value: float | None, std: float | None = None, signed: bool = False) -> str:
    if value is None:
        return "—"
    text = f"{value * 100:+.2f}" if signed else f"{value * 100:.2f}"
    if std is not None:
        text += f" ± {std * 100:.2f}"
    return text


def _fmt_pp(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:+.2f}"


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    mean = float(statistics.mean(values))
    std = float(statistics.stdev(values)) if len(values) > 1 else None
    return mean, std


def load_groups(preds_dir: Path) -> tuple[dict | None, dict | None, dict, list[str]]:
    """读全部测试/验证预测，逐种子定 τ；缺文件的种子只记 missing。"""
    tfidf_test = _read_preds(preds_dir, TFIDF_NAME)
    tfidf_val = _read_preds(preds_dir, f"{TFIDF_NAME}__val")
    groups, missing = {}, []
    for max_len in MAX_LENS:
        records = []
        for seed in SEEDS:
            name = bert_run_name(max_len, seed)
            bert_test = _read_preds(preds_dir, name)
            bert_val = _read_preds(preds_dir, f"{name}__val")
            wanted = [f"{TFIDF_NAME}.parquet", f"{TFIDF_NAME}__val.parquet",
                      f"{name}.parquet", f"{name}__val.parquet"]
            absent = [x for x in wanted if not (preds_dir / x).exists()]
            if absent:
                records.append({"max_len": max_len, "seed": seed,
                                "run_name": name, "missing": absent})
                missing.extend(absent)
                continue
            test = _align_on_index(tfidf_test, bert_test)
            val = _align_on_index(tfidf_val, bert_val)
            if test is None or val is None:
                records.append({"max_len": max_len, "seed": seed,
                                "run_name": name, "missing": ["对齐后为空"]})
                continue
            chosen = select_tau(val["tfidf_pred"], val["conf"],
                                val["bert_pred"], val["label"])
            pred = cascade_predict(test["tfidf_pred"], test["conf"],
                                   test["bert_pred"], chosen["tau"])
            records.append({
                "max_len": max_len, "seed": seed, "run_name": name,
                "test": test, "val": val,
                "tau": chosen["tau"], "nominal": chosen["nominal"],
                "val_acc": chosen["val_accuracy"],
                "val_handover": chosen["val_handover"],
                "test_handover": handover_fraction(test["conf"], chosen["tau"]),
                "cascade_pred": pred,
                "cascade_acc": float((pred == test["label"]).mean()),
                "bert_acc": float((test["bert_pred"] == test["label"]).mean()),
                "tfidf_acc": float((test["tfidf_pred"] == test["label"]).mean()),
            })
        groups[max_len] = records
    return tfidf_test, tfidf_val, groups, sorted(set(missing))


def valid_records(records: list[dict]) -> list[dict]:
    return [r for r in records if "test" in r]


def reproduction_section(tfidf_test: pd.DataFrame | None, groups: dict) -> list[str]:
    """新跑结果与实验 F/J、阶段 1 的已有结果逐位对照。"""
    lines = ["## 复现检查", ""]
    old_tfidf = _read_json(ROOT / "results" / "imdb_tfidf_main_s0.json")
    old_acc = None if old_tfidf is None else old_tfidf.get("test_accuracy")
    if tfidf_test is None or old_acc is None:
        lines += ["TF-IDF：缺文件。", ""]
    else:
        new_acc = float((tfidf_test["pred"] == tfidf_test["label"]).mean())
        diff = new_acc - old_acc
        lines += [
            f"TF-IDF（n={len(tfidf_test)}）：新跑 {new_acc * 100:.4f}%，"
            f"阶段 1 的 `imdb_tfidf_main_s0.json` 为 {old_acc * 100:.4f}%，"
            f"差值 {_fmt_pp(diff)} pp，逐位相同："
            f"{'是' if new_acc == old_acc else '否'}。",
            "",
        ]
    lines += [
        "| 组 | 种子 | 新跑测试准确率(%) | 已有结果(%) | 差值(pp) | 逐位相同 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for max_len in MAX_LENS:
        for record in groups.get(max_len, []):
            if "test" not in record:
                lines.append(f"| ml{max_len} | {record['seed']} | 缺文件 | 缺文件 | — | — |")
                continue
            old_path = (ROOT / "results" / f"imdb_distilbert_bert_n22500_s{record['seed']}.json"
                        if max_len == 256 else
                        ROOT / "results" / f"imdb_distilbert_long_head512_s{record['seed']}.json")
            old = _read_json(old_path)
            if old is None or old.get("test_accuracy") is None:
                lines.append(f"| ml{max_len} | {record['seed']} | "
                             f"{record['bert_acc'] * 100:.4f} | 缺文件 | — | — |")
                continue
            old_acc = float(old["test_accuracy"])
            same = record["bert_acc"] == old_acc
            lines.append(
                f"| ml{max_len} | {record['seed']} | {record['bert_acc'] * 100:.4f} | "
                f"{old_acc * 100:.4f} | {_fmt_pp(record['bert_acc'] - old_acc)} | "
                f"{'是' if same else '否'} |"
            )
    checked = [r for ml in MAX_LENS for r in valid_records(groups.get(ml, []))]
    zero_ok = all(np.array_equal(
        cascade_predict(r["test"]["tfidf_pred"], r["test"]["conf"],
                        r["test"]["bert_pred"], 0.0), r["test"]["tfidf_pred"])
        for r in checked)
    full_ok = all(np.array_equal(
        cascade_predict(r["test"]["tfidf_pred"], r["test"]["conf"],
                        r["test"]["bert_pred"], float("inf")), r["test"]["bert_pred"])
        for r in checked)
    lines += [
        "",
        f"端点检查（{len(checked)} 个可用种子）：交出 0% 时逐条等于 TF-IDF："
        f"{'是' if checked and zero_ok else '否或无法检查'}；"
        f"交出 100% 时逐条等于 DistilBERT："
        f"{'是' if checked and full_ok else '否或无法检查'}。",
        "",
    ]
    return lines


def length_section(groups: dict) -> tuple[list[str], dict, dict]:
    """四个长度组；返回 Markdown、逐组统计和分组样本数。"""
    labels, lengths = load_imdb_test()
    bucket_ids = np.array([BUCKET_LABELS.index(length_bucket(int(n))) for n in lengths])
    checked = [r for ml in MAX_LENS for r in valid_records(groups.get(ml, []))]
    if checked:
        used_idx = checked[0]["test"]["index"]
        counts = {b: int((bucket_ids[used_idx] == i).sum()) for i, b in enumerate(BUCKET_LABELS)}
    else:
        counts = {b: int((bucket_ids == i).sum()) for i, b in enumerate(BUCKET_LABELS)}
    stats: dict[tuple[int, str], dict[str, dict]] = {}
    for max_len in MAX_LENS:
        for method in ("cascade", "bert"):
            stats[(max_len, method)] = {}
            for bucket in BUCKET_LABELS:
                values = []
                for record in valid_records(groups.get(max_len, [])):
                    idx = record["test"]["index"]
                    pred = (record["cascade_pred"] if method == "cascade"
                            else record["test"]["bert_pred"])
                    cell = bucket_accuracy(
                        record["test"]["label"], pred, bucket_ids[idx]
                    )
                    if cell[bucket] is not None:
                        values.append(cell[bucket])
                mean, std = _mean_std(values)
                stats[(max_len, method)][bucket] = {
                    "mean": mean, "std": std, "n_seeds": len(values),
                }
    lines = [
        "## 四个长度组",
        "",
        "分组复用 `analyze_length_buckets.length_bucket`：按截断前的正则分词 "
        "token 数分为 ≤128、129～256、257～512、>512。每组内先算每个种子的"
        "准确率，再跨种子取均值 ± 样本标准差。",
        "",
        "| 分组 | " + " | ".join(f"{b}（n={counts[b]}）" for b in BUCKET_LABELS) + " |",
        "|---|" + "---:|" * len(BUCKET_LABELS),
    ]
    for max_len in MAX_LENS:
        for method, title in (("cascade", "级联"), ("bert", "DistilBERT")):
            cells = stats[(max_len, method)]
            lines.append(
                f"| ml{max_len} {title} | " + " | ".join(
                    _fmt_pct(cells[b]["mean"], cells[b]["std"]) for b in BUCKET_LABELS
                ) + " |"
            )
    lines += ["", "说明：只有一个可用种子时不显示标准差。", ""]
    return lines, stats, counts


def validation_section(groups: dict) -> list[str]:
    """验证集选出的 τ、交出比例，以及对应的测试集结果。"""
    lines = [
        "## 验证集选定的 τ、交出比例与测试准确率",
        "",
        "τ 是验证集置信度 c=max(p,1−p) 的分位数；测试集沿用同一个 τ，"
        "实际交出比例按 c<τ 重新计算，不直接搬验证集的比例。",
        "",
        "| 组 | 种子 | τ | 名义档位(%) | 验证交出比例(%) | 验证准确率(%) | "
        "测试交出比例(%) | 测试准确率(%) | DistilBERT 测试准确率(%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for max_len in MAX_LENS:
        for record in groups.get(max_len, []):
            if "test" not in record:
                lines.append(f"| ml{max_len} | {record['seed']} | " + " | ".join(
                    ["缺文件"] * 8) + " |")
                continue
            lines.append(
                f"| ml{max_len} | {record['seed']} | {record['tau']:.6f} | "
                f"{record['nominal'] * 100:.2f} | {record['val_handover'] * 100:.2f} | "
                f"{record['val_acc'] * 100:.2f} | {record['test_handover'] * 100:.2f} | "
                f"{record['cascade_acc'] * 100:.2f} | {record['bert_acc'] * 100:.2f} |"
            )
    lines += [""]
    return lines


def load_long_readings(out_dir: Path) -> tuple[dict, list[str]]:
    """读实验 J 的准确率；chunk 用 head512 JSON 里的分段评测字段。"""
    readings, missing = {}, []
    for key in J_READS:
        values = []
        for seed in SEEDS:
            path = out_dir / f"imdb_distilbert_{LONG_TAG}_{key}_s{seed}.json"
            payload = _read_json(path)
            value = None if payload is None else payload.get("test_accuracy")
            if value is None:
                missing.append(path.name)
            else:
                values.append(float(value))
        mean, std = _mean_std(values)
        readings[key] = {"acc": mean, "std": std, "n": len(values)}
    values = []
    for seed in SEEDS:
        path = out_dir / f"imdb_distilbert_{LONG_TAG}_head512_s{seed}.json"
        payload = _read_json(path)
        value = None if payload is None else payload.get("chunk_test_accuracy")
        if value is None:
            missing.append(path.name)
        else:
            values.append(float(value))
    mean, std = _mean_std(values)
    readings["chunk"] = {"acc": mean, "std": std, "n": len(values)}
    return readings, sorted(set(missing))


def _bench_ms(bench: dict | None, key: str) -> float | None:
    if not bench:
        return None
    info = bench.get(key, {})
    if isinstance(info, dict) and info.get("ms_per_1k") is not None:
        return float(info["ms_per_1k"])
    top = bench.get("ms_per_1k", {})
    return None if top.get(key) is None else float(top[key])


def cost_section(bench: dict | None, groups: dict,
                 out_dir: Path) -> tuple[list[str], dict]:
    """按预定公式算级联耗时，并加入实验 J 的四种读法。"""
    lines = ["## 成本", ""]
    if bench is None:
        lines += ["缺文件：`results/bench_cascade.json`。", ""]
        return lines, {}
    lines += [
        f"测速机器：GPU {bench.get('gpu')}；CPU {bench.get('cpu')}；"
        f"torch 线程数 {bench.get('torch_num_threads')}（cpu_count="
        f"{bench.get('cpu_count')}）；torch {bench.get('torch_version')}；"
        f"sklearn {bench.get('sklearn_version')}。",
        "DistilBERT：批大小 " + str(bench.get("batch_size")) +
        "，FP32，含分词，预热 " + str(bench.get("warmup_batches")) +
        " 批，测 " + str(bench.get("reps")) + " 次取中位数。"
        "TF-IDF：向量化加 `predict_proba`，单进程，测 " +
        str(bench.get("reps")) + " 次取中位数。"
        "chunk：510 子词一段、相邻段重叠 128，计时含分词与切段，测 " +
        str(bench.get("reps")) + " 次取中位数。",
        "",
    ]
    tfidf_ms = _bench_ms(bench, "tfidf")
    bert_ms = {ml: _bench_ms(bench, f"distilbert_{ml}") for ml in MAX_LENS}
    chunk_ms = _bench_ms(bench, "chunk")
    chunk_info = bench.get("chunk") or {}
    if (chunk_info.get("n_segments") is not None
            and chunk_info.get("segments_per_review") is not None):
        lines += [
            f"chunk 段数：{chunk_info.get('n_segments')} 段，平均 "
            f"{chunk_info.get('segments_per_review'):.4f} 段/条，最多 "
            f"{chunk_info.get('max_segments')} 段。",
            "",
        ]

    readings, long_missing = load_long_readings(out_dir)
    all_records = [r for ml in MAX_LENS for r in valid_records(groups.get(ml, []))]
    tf_values = sorted({r["tfidf_acc"] for r in all_records})
    tf_mean, tf_std = _mean_std(tf_values)
    d256_mean, d256_std = _mean_std(
        [r["bert_acc"] for r in valid_records(groups.get(256, []))]
    )
    readings["tfidf"] = {"acc": tf_mean, "std": tf_std,
                         "n": len(tf_values), "ms": tfidf_ms}
    readings["distilbert_256"] = {"acc": d256_mean, "std": d256_std,
                                  "n": len(valid_records(groups.get(256, []))),
                                  "ms": bert_ms[256]}
    for key in J_READS:
        readings[key]["ms"] = bert_ms[512]
    readings["chunk"]["ms"] = chunk_ms

    stats = {"tfidf_ms": tfidf_ms, "bert_ms": bert_ms, "chunk_ms": chunk_ms,
             "readings": readings, "long_missing": long_missing, "selected": {}}
    lines += [
        "| 方法/读法 | 测试准确率(%) | 每千条 ms |",
        "|---|---:|---:|",
    ]
    for key in READ_ORDER:
        cell = readings.get(key, {})
        lines.append(f"| {READ_TITLES[key]} | {_fmt_pct(cell.get('acc'), cell.get('std'))} | "
                     f"{_fmt_num(cell.get('ms'))} |")

    lines += [
        "",
        "级联每千条耗时 = TF-IDF 每千条 + 交出比例 × DistilBERT 每千条。",
        "",
        "| 组 | 平均交出比例(%) | 级联准确率(%) | TF-IDF ms/千条 | "
        "DistilBERT ms/千条 | 级联 ms/千条 | 级联 / DistilBERT |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ml in MAX_LENS:
        records = valid_records(groups.get(ml, []))
        handovers = [r["test_handover"] for r in records]
        accs = [r["cascade_acc"] for r in records]
        mean_handover, hand_std = _mean_std(handovers)
        mean_acc, acc_std = _mean_std(accs)
        base = bert_ms[ml]
        cascade_ms = (None if tfidf_ms is None or base is None or mean_handover is None
                      else tfidf_ms + mean_handover * base)
        ratio = None if cascade_ms is None or not base else cascade_ms / base
        cost_std = None if hand_std is None or base is None else hand_std * base
        stats[ml] = {"handover": mean_handover, "handover_std": hand_std,
                     "acc": mean_acc, "acc_std": acc_std,
                     "tfidf_ms": tfidf_ms, "bert_ms": base,
                     "cascade_ms": cascade_ms, "cascade_ms_std": cost_std,
                     "ratio": ratio}
        stats["selected"][ml] = stats[ml]
        lines.append(
            f"| ml{ml} | {'—' if mean_handover is None else f'{mean_handover * 100:.2f}'} | "
            f"{_fmt_pct(mean_acc, acc_std)} | "
            f"{_fmt_num(tfidf_ms)} | {_fmt_num(base)} | {_fmt_num(cascade_ms)} | "
            f"{_fmt_num(ratio, 3)} |"
        )
    lines += [""]
    return lines, stats


def mcnemar_section(groups: dict) -> tuple[list[str], dict]:
    """级联对 DistilBERT 单独，逐种子精确 McNemar；全部 p 值做 Holm。"""
    runs = []
    for max_len in MAX_LENS:
        for record in valid_records(groups.get(max_len, [])):
            labels = record["test"]["label"]
            right_c = record["cascade_pred"] == labels
            right_b = record["test"]["bert_pred"] == labels
            only_c = int((right_c & ~right_b).sum())
            only_b = int((~right_c & right_b).sum())
            runs.append({
                "max_len": max_len, "seed": record["seed"],
                "cascade_acc": record["cascade_acc"],
                "bert_acc": record["bert_acc"],
                "only_cascade": only_c, "only_bert": only_b,
                "p": mcnemar_exact(only_c, only_b),
            })
    adjusted = holm_adjust([run["p"] for run in runs]) if runs else []
    for run, p_adj in zip(runs, adjusted):
        run["p_adj"] = p_adj
    lines = [
        "## McNemar 与 Holm 校正",
        "",
        f"级联对 DistilBERT 单独，逐种子做精确 McNemar（复用 "
        f"`significance.mcnemar_exact`）。{len(runs)} 个 p 值一起做 Holm 校正"
        "（`significance.holm_adjust`）；校正后 p<0.05 的种子数与方向如下。",
        "",
        "| 组 | 种子 | 级联准确率(%) | DistilBERT(%) | 差值(pp) | "
        "仅级联对 | 仅 DistilBERT 对 | 原始 p | 校正 p | 方向 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in runs:
        direction = "级联高" if run["cascade_acc"] > run["bert_acc"] else (
            "级联低" if run["cascade_acc"] < run["bert_acc"] else "持平")
        lines.append(
            f"| ml{run['max_len']} | {run['seed']} | "
            f"{run['cascade_acc'] * 100:.2f} | {run['bert_acc'] * 100:.2f} | "
            f"{_fmt_pp(run['cascade_acc'] - run['bert_acc'])} | "
            f"{run['only_cascade']} | {run['only_bert']} | "
            f"{run['p']:.4g} | {run['p_adj']:.4g} | {direction} |"
        )
    lines += ["", "| 组 | 可用种子数 | 校正后 p<0.05 的种子数 | 方向 |",
              "|---|---:|---:|---|"]
    summary = {}
    for max_len in MAX_LENS:
        subset = [r for r in runs if r["max_len"] == max_len]
        n_sig = sum(1 for r in subset if r["p_adj"] < 0.05)
        signs = {r["cascade_acc"] > r["bert_acc"] for r in subset}
        direction = ("全部级联高" if signs == {True} else
                     "全部级联低" if signs == {False} else "不一致/不完整")
        summary[max_len] = {"runs": subset, "n_seeds": len(subset),
                            "n_sig": n_sig, "direction": direction}
        lines.append(f"| ml{max_len} | {len(subset)} | {n_sig} | {direction} |")
    lines += ["", f"Holm 校正的检验数：{len(runs)}。", ""]
    return lines, summary


def _compose_cascade_frame(tfidf: pd.DataFrame | None, bert: pd.DataFrame | None,
                           tau: float) -> tuple[pd.DataFrame | None, float | None]:
    """把 TF-IDF 与某个种子的 DistilBERT 预测按 pair_id 合成级联。"""
    if tfidf is None or bert is None:
        return None, None
    merged = tfidf.merge(bert[["pair_id", "pred"]], on="pair_id", suffixes=("", "_bert"))
    conf = confidence_from_prob_pos(merged["prob_pos"].to_numpy())
    hand = conf < tau
    frame = pd.DataFrame({
        "pair_id": merged["pair_id"].to_numpy(dtype=np.int64),
        "label": merged["label"].to_numpy(dtype=np.int64),
        "pred": np.where(hand, merged["pred_bert"], merged["pred"]).astype(np.int64),
    })
    return frame, float(hand.mean())


def _robust_metrics(orig: pd.DataFrame | None, edit: pd.DataFrame | None,
                    ids: list[int], handover: float | None = None) -> dict | None:
    if orig is None or edit is None:
        return None
    base = paired_metrics(orig, edit, ids)
    if base["n"] == 0:
        return None
    keep = set(int(x) for x in ids)
    left = orig[orig["pair_id"].isin(keep)][["pair_id", "pred"]]
    right = edit[edit["pair_id"].isin(keep)][["pair_id", "pred"]]
    merged = left.merge(right, on="pair_id", suffixes=("_orig", "_edit"))
    return {
        "n": base["n"], "orig": base["orig_acc"], "edit": base["edit_acc"],
        "drop": base["drop"], "consistency": base["consistency"],
        "changed": float((merged["pred_orig"] != merged["pred_edit"]).mean()),
        "handover": handover,
    }


def _agg_robust(runs: list[dict], key: str) -> tuple[float | None, float | None]:
    return _mean_std([r[key] for r in runs if r.get(key) is not None])


def robustness_section(preds_dir: Path, pairs_path: Path, robust_tag: str,
                       tau_mean: float | None) -> tuple[list[str], dict]:
    """阶段 3 主口径下的 TF-IDF / DistilBERT / 级联对比。"""
    lines = ["## 稳健性", ""]
    pairs = _read_json(pairs_path)
    if pairs is None:
        lines += [f"缺文件：`{pairs_path}`。", ""]
        return lines, {}
    if tau_mean is None:
        lines += ["缺 ml256 的验证集预测，无法取三个种子的 τ 均值。", ""]
        return lines, {}
    tfidf_name = f"imdb_tfidf_{robust_tag}_s0"
    bert_names = [f"imdb_distilbert_{robust_tag}_s{s}" for s in SEEDS]
    lines += [
        f"主口径读 `{pairs_path}` 的 visible 子集，剔除 excluded_no_diff；"
        f"τ 取 ml256 三个种子验证集阈值的均值 {tau_mean:.6f}。"
        "阶段 3 的 TF-IDF 与阶段 4 的 TF-IDF 训练集只差 648 条，这是近似。",
        "",
    ]
    stats: dict[str, dict] = {}
    for family, family_title in FAMILIES:
        ids = visible_pair_ids(pairs, family)
        tfidf_orig = _read_preds(preds_dir, f"{tfidf_name}__{family}_orig")
        tfidf_edit = _read_preds(preds_dir, f"{tfidf_name}__{family}_edit")
        tfidf_run = _robust_metrics(tfidf_orig, tfidf_edit, ids)
        bert_runs, cascade_runs = [], []
        for name in bert_names:
            bert_orig = _read_preds(preds_dir, f"{name}__{family}_orig")
            bert_edit = _read_preds(preds_dir, f"{name}__{family}_edit")
            bert_run = _robust_metrics(bert_orig, bert_edit, ids)
            if bert_run is not None:
                bert_runs.append(bert_run)
            c_orig, _ = _compose_cascade_frame(tfidf_orig, bert_orig, tau_mean)
            c_edit, handover = _compose_cascade_frame(tfidf_edit, bert_edit, tau_mean)
            cascade_run = _robust_metrics(c_orig, c_edit, ids, handover)
            if cascade_run is not None:
                cascade_runs.append(cascade_run)

        # TF-IDF “有把握地判错”：c≥τ 且判错的改写评论占比
        confident_wrong = None
        if tfidf_edit is not None:
            keep = tfidf_edit[tfidf_edit["pair_id"].isin(set(ids))]
            conf = confidence_from_prob_pos(keep["prob_pos"].to_numpy())
            wrong = keep["pred"].to_numpy() != keep["label"].to_numpy()
            confident_wrong = float(((conf >= tau_mean) & wrong).mean())

        family_stats = {
            "n": len(ids),
            "tfidf": tfidf_run,
            "bert_runs": bert_runs,
            "cascade_runs": cascade_runs,
            "confident_wrong": confident_wrong,
        }
        for method, runs in (("tfidf", [tfidf_run] if tfidf_run else []),
                             ("bert", bert_runs), ("cascade", cascade_runs)):
            family_stats[method] = {
                key: _agg_robust(runs, key) for key in
                ("orig", "edit", "drop", "changed", "handover")
            } if runs else None
        stats[family] = family_stats

    lines += [
        "| 家族 | 方法 | n | 原始准确率(%) | 改写准确率(%) | 改写降幅(pp) | "
        "改后预测变了(%) | 改写集交出比例(%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, family_title in FAMILIES:
        family_stats = stats[family]
        for method, title in (("tfidf", "TF-IDF"), ("bert", "DistilBERT"),
                              ("cascade", "级联")):
            cells = family_stats.get(method)
            if not cells:
                lines.append(f"| {family_title} | {title} | 缺文件 | 缺文件 | 缺文件 | "
                             f"缺文件 | 缺文件 | 缺文件 |")
                continue
            handover = cells["handover"][0]
            lines.append(
                f"| {family_title} | {title} | {family_stats['n']} | "
                f"{_fmt_pct(cells['orig'][0], cells['orig'][1])} | "
                f"{_fmt_pct(cells['edit'][0], cells['edit'][1])} | "
                f"{_fmt_pp(cells['drop'][0])}"
                + (f" ± {cells['drop'][1] * 100:.2f}" if cells['drop'][1] is not None else "")
                + f" | {_fmt_pct(cells['changed'][0], cells['changed'][1])} | "
                f"{'—' if handover is None else f'{handover * 100:.2f}'} |"
            )
    lines += ["", "| 家族 | n | τ 均值 | TF-IDF 有把握判错（改写集，%） |",
              "|---|---:|---:|---:|"]
    for family, family_title in FAMILIES:
        value = stats[family].get("confident_wrong")
        lines.append(f"| {family_title} | {stats[family]['n']} | {tau_mean:.6f} | "
                     f"{'—' if value is None else f'{value * 100:.2f}'} |")
    lines += ["", "说明：DistilBERT 与 TF-IDF 单列的交出比例为“—”；"
              "改后预测变了 = 同一 pair_id 的原始与改写预测不相等。", ""]
    return lines, stats


def _accuracy_cells(groups: dict, max_len: int, key: str) -> tuple:
    values = [r[key] for r in valid_records(groups.get(max_len, []))]
    return _mean_std(values)


def expectations_section(groups: dict, length_stats: dict, cost_stats: dict,
                         mcnemar_summary: dict, robust_stats: dict) -> list[str]:
    """C1～C6 所需的全部数字；只列数字，不写判语。"""
    lines = ["## 预期 C1～C6 所需的数字", "",
             "差异判法：差值大于两组标准差中较大者的 2 倍，且大于 0.5 个百分点。", ""]
    for max_len, label in ((256, "C1"), (512, "C2")):
        c_mean, c_std = _accuracy_cells(groups, max_len, "cascade_acc")
        b_mean, b_std = _accuracy_cells(groups, max_len, "bert_acc")
        diff = None if c_mean is None or b_mean is None else c_mean - b_mean
        larger = max(c_std or 0.0, b_std or 0.0)
        summary = mcnemar_summary.get(max_len, {})
        p_list = ", ".join(f"s{r['seed']}={r['p_adj']:.4g}"
                           for r in summary.get("runs", []))
        lines += [
            f"**{label}（ml{max_len}）**",
            f"- 级联测试准确率 {_fmt_pct(c_mean, c_std)}%；DistilBERT 单独 "
            f"{_fmt_pct(b_mean, b_std)}%；差值 {_fmt_pp(diff)} pp。",
            f"- 2×较大标准差 = {larger * 200:.2f} pp；0.5 pp 门槛；"
            f"判法要求差值同时大于两者。",
            f"- McNemar 校正后 p<0.05 的种子数 {summary.get('n_sig', 0)}/"
            f"{summary.get('n_seeds', 0)}；方向：{summary.get('direction', '缺文件')}；"
            f"校正 p：{p_list or '缺文件'}。",
            "",
        ]

    lines += ["**C3（ml256 四个长度组的提升）**", "",
              "| 分组 | 级联准确率(%) | DistilBERT 准确率(%) | 差值(pp) |",
              "|---|---:|---:|---:|"]
    diffs = {}
    for bucket in BUCKET_LABELS:
        c_cell = length_stats.get((256, "cascade"), {}).get(bucket, {})
        b_cell = length_stats.get((256, "bert"), {}).get(bucket, {})
        c_mean, b_mean = c_cell.get("mean"), b_cell.get("mean")
        diff = None if c_mean is None or b_mean is None else c_mean - b_mean
        diffs[bucket] = diff
        lines.append(
            f"| {bucket} | {_fmt_pct(c_mean, c_cell.get('std'))} | "
            f"{_fmt_pct(b_mean, b_cell.get('std'))} | {_fmt_pp(diff)} |"
        )
    lines += ["", "短组差值：≤128 " + _fmt_pp(diffs.get("≤128")) +
              " pp，129～256 " + _fmt_pp(diffs.get("129～256")) +
              " pp；长组差值：257～512 " + _fmt_pp(diffs.get("257～512")) +
              " pp，>512 " + _fmt_pp(diffs.get(">512")) + " pp。", ""]

    records = valid_records(groups.get(256, []))
    val_handovers = [r["val_handover"] for r in records]
    test_handovers = [r["test_handover"] for r in records]
    val_mean, val_std = _mean_std(val_handovers)
    test_mean, test_std = _mean_std(test_handovers)
    cost = cost_stats.get(256, {})
    lines += [
        "**C4（ml256 交出比例与成本）**",
        f"- 验证集交出比例 {_fmt_pct(val_mean, val_std)}%（最大 "
        f"{'—' if not val_handovers else f'{max(val_handovers) * 100:.2f}'}%）；"
        f"测试集实际交出比例 {_fmt_pct(test_mean, test_std)}%（最大 "
        f"{'—' if not test_handovers else f'{max(test_handovers) * 100:.2f}'}%）。",
        f"- 成本：TF-IDF {_fmt_num(cost.get('tfidf_ms'))} ms/千条，"
        f"DistilBERT {_fmt_num(cost.get('bert_ms'))} ms/千条，"
        f"级联 {_fmt_num(cost.get('cascade_ms'))} ms/千条，"
        f"级联/DistilBERT = {_fmt_num(cost.get('ratio'), 3)}。",
        "- 判法门槛：交出比例 40%；级联每千条耗时 / DistilBERT 每千条 = 0.5。",
        "",
    ]

    lines += ["**C5（稳健性，主口径）**", "",
              "| 家族 | 级联改写降幅(pp) | DistilBERT 改写降幅(pp) | 差值(pp) | "
              "2×较大标准差(pp) | 0.5pp 门槛 |",
              "|---|---:|---:|---:|---:|---:|"]
    for family, family_title in FAMILIES:
        family_stats = robust_stats.get(family, {})
        cascade = family_stats.get("cascade")
        bert = family_stats.get("bert")
        if not cascade or not bert:
            lines.append(f"| {family_title} | 缺文件 | 缺文件 | 缺文件 | 缺文件 | 0.50 |")
            continue
        c_mean, c_std = cascade["drop"]
        b_mean, b_std = bert["drop"]
        diff = c_mean - b_mean
        larger = max(c_std or 0.0, b_std or 0.0)
        lines.append(
            f"| {family_title} | {_fmt_pct(c_mean, c_std, signed=True)} | "
            f"{_fmt_pct(b_mean, b_std, signed=True)} | {_fmt_pp(diff)} | "
            f"{larger * 200:.2f} | 0.50 |"
        )

    lines += ["", "**C6（ml256 级联与 chunk）**"]
    selected = cost_stats.get("selected", {}).get(256)
    chunk = cost_stats.get("readings", {}).get("chunk")
    if (not selected or not chunk or selected.get("acc") is None
            or chunk.get("acc") is None):
        lines += ["- 缺文件：需要 ml256 级联准确率、chunk 准确率与 chunk 测速。", ""]
    else:
        diff = selected["acc"] - chunk["acc"]
        larger = max(selected.get("acc_std") or 0.0, chunk.get("std") or 0.0)
        cost_ratio = (None if not chunk.get("ms")
                      else selected["cascade_ms"] / chunk["ms"])
        lines += [
            f"- 测试准确率：级联 {_fmt_pct(selected['acc'], selected.get('acc_std'))}%，"
            f"chunk {_fmt_pct(chunk['acc'], chunk.get('std'))}%，"
            f"差值（级联 − chunk）{_fmt_pp(diff)} pp；"
            f"2×较大标准差 {larger * 200:.2f} pp；1 pp 门槛。",
            f"- 每千条耗时：级联 {_fmt_num(selected.get('cascade_ms'))} ms，"
            f"chunk {_fmt_num(chunk.get('ms'))} ms，"
            f"耗时比 {_fmt_num(cost_ratio, 3)}；0.25 门槛。",
            "",
        ]
    return lines


def _fmt_num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def make_figure(groups: dict, out_fig: Path) -> None:
    import matplotlib.pyplot as plt

    setup_chinese_font()
    colors = {256: "#1f77b4", 512: "#d62728"}
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    any_data = False
    for max_len in MAX_LENS:
        records = valid_records(groups.get(max_len, []))
        if not records:
            continue
        any_data = True
        color = colors[max_len]
        curves = [test_curve(r["test"]["tfidf_pred"], r["test"]["conf"],
                             r["test"]["bert_pred"], r["test"]["label"])
                  for r in records]
        x = np.mean([c["handover"] for c in curves], axis=0) * 100
        y = np.mean([c["accuracy"] for c in curves], axis=0) * 100
        ax.plot(x, y, color=color, lw=1.8, label=f"级联曲线（截断 {max_len}）")
        bert_mean, _ = _mean_std([r["bert_acc"] for r in records])
        if bert_mean is not None:
            ax.axhline(bert_mean * 100, color=color, ls="--", lw=1.1,
                       label=f"DistilBERT {max_len} 单独")
        xs = [r["test_handover"] * 100 for r in records]
        ys = [r["cascade_acc"] * 100 for r in records]
        ax.scatter(xs, ys, s=28, facecolors="none", edgecolors=color,
                   linewidths=1.2, label=f"验证集选定点（截断 {max_len}）", zorder=4)
        ax.scatter([np.mean(xs)], [np.mean(ys)], marker="*", s=150,
                   color=color, edgecolors="black", linewidths=0.5, zorder=5)
    if not any_data:
        ax.text(0.5, 0.5, "缺文件", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
    ax.set_xlim(0, 100)
    ax.set_xlabel("测试集交出比例（%）")
    ax.set_ylabel("测试准确率（%）")
    ax.set_title("级联：交出比例对测试准确率")
    ax.grid(alpha=0.3)
    if any_data:
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, out_fig)


def make_cost_figure(groups: dict, cost_stats: dict, out_fig: Path) -> None:
    """准确率对每千条耗时；横轴对数，含实验 J 的四种读法。"""
    import matplotlib.pyplot as plt

    setup_chinese_font()
    readings = cost_stats.get("readings", {})
    style = {
        "tfidf": ("#7f7f7f", "s"),
        "distilbert_256": ("#1f77b4", "o"),
        "head512": ("#d62728", "^"),
        "tail512": ("#ff7f0e", "v"),
        "headtail": ("#9467bd", "D"),
        "chunk": ("#2ca02c", "P"),
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    any_data = False
    for key in READ_ORDER:
        cell = readings.get(key) or {}
        if cell.get("acc") is None or cell.get("ms") is None:
            continue
        color, marker = style[key]
        ax.errorbar([cell["ms"]], [cell["acc"] * 100],
                    yerr=(cell.get("std") or 0.0) * 100,
                    fmt=marker, color=color, ms=7, capsize=2.5,
                    linestyle="none", label=READ_TITLES[key])
        any_data = True

    colors = {256: "#1f77b4", 512: "#d62728"}
    tfidf_ms = cost_stats.get("tfidf_ms")
    for max_len in MAX_LENS:
        records = valid_records(groups.get(max_len, []))
        base = (cost_stats.get("bert_ms") or {}).get(max_len)
        if not records or base is None or tfidf_ms is None:
            continue
        curves = [test_curve(r["test"]["tfidf_pred"], r["test"]["conf"],
                             r["test"]["bert_pred"], r["test"]["label"])
                  for r in records]
        hand = np.mean([c["handover"] for c in curves], axis=0)
        acc = np.mean([c["accuracy"] for c in curves], axis=0) * 100
        if len(curves) > 1:
            std = np.std([c["accuracy"] for c in curves], axis=0, ddof=1) * 100
        else:
            std = np.zeros_like(acc)
        x = tfidf_ms + hand * base
        ax.plot(x, acc, color=colors[max_len], lw=1.7,
                label=f"级联 {max_len} 曲线")
        ax.fill_between(x, acc - std, acc + std, color=colors[max_len],
                        alpha=0.12, linewidth=0)
        selected = (cost_stats.get("selected") or {}).get(max_len)
        if selected and selected.get("cascade_ms") is not None:
            ax.errorbar([selected["cascade_ms"]], [selected["acc"] * 100],
                        xerr=selected.get("cascade_ms_std") or 0.0,
                        yerr=(selected.get("acc_std") or 0.0) * 100,
                        fmt="*", ms=16, color=colors[max_len],
                        markeredgecolor="black", capsize=2.5,
                        label=f"验证集选定点 {max_len}")
        any_data = True

    if not any_data:
        ax.text(0.5, 0.5, "缺文件", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
    elif not any((readings.get(k) or {}).get("acc") is not None
                 for k in J_READS + ("chunk",)):
        ax.text(0.02, 0.98, "实验 J 结果缺文件", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="0.45")
    ax.set_xscale("log")
    ax.set_xlabel("每千条耗时（毫秒，对数刻度）")
    ax.set_ylabel("测试准确率（%）")
    ax.set_title("级联与实验 J 读法：准确率对每千条耗时")
    ax.grid(alpha=0.3, which="both")
    if any_data:
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    save_figure(fig, out_fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="实验 L 级联分析")
    parser.add_argument("--out-dir", default="results", help="结果目录，默认 results")
    parser.add_argument("--preds-dir", default=None,
                        help="预测目录，默认 <out-dir>/preds")
    parser.add_argument("--out-md", default=None, help="默认 <out-dir>/级联.md")
    parser.add_argument("--out-fig", default=None,
                        help="默认正式跑为 figures/级联.png；换 out-dir 时放 figures/<目录名>/")
    parser.add_argument("--bench", default="results/bench_cascade.json",
                        help="测速 JSON，默认 results/bench_cascade.json")
    parser.add_argument("--visible-pairs", default="results/robust_visible_pairs.json",
                        help="稳健性可见配对清单")
    parser.add_argument("--robust-tag", default="robust",
                        help="阶段 3 去污染模型的 tag，默认 robust")
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir, ROOT / "results")
    preds_dir = _resolve(args.preds_dir, out_dir / "preds")
    out_md = (_resolve(args.out_md, out_dir / "级联.md")
              if args.out_md else out_dir / "级联.md")
    if args.out_fig:
        out_fig = _resolve(args.out_fig, ROOT / "figures" / "级联.png")
    elif out_dir == ROOT / "results":
        out_fig = ROOT / "figures" / "级联.png"
    else:
        out_fig = ROOT / "figures" / out_dir.name / "级联.png"
    bench_path = _resolve(args.bench, ROOT / "results" / "bench_cascade.json")
    pairs_path = _resolve(args.visible_pairs, ROOT / "results" / "robust_visible_pairs.json")

    tfidf_test, tfidf_val, groups, missing = load_groups(preds_dir)
    bench = _read_json(bench_path)
    tau_values = [r["tau"] for r in valid_records(groups.get(256, []))
                  if np.isfinite(r["tau"])]
    tau_mean = float(np.mean(tau_values)) if tau_values else None

    lines = [
        "# 实验 L：TF-IDF 先判、拿不准交给 DistilBERT 的级联",
        "",
        f"生成时间：{now_iso()}",
        "",
        f"结果目录 `{out_dir}`，预测目录 `{preds_dir}`。"
        "阈值只在验证集上选，测试集沿用同一个 τ；测试曲线只作描述。",
        "",
    ]
    lines += reproduction_section(tfidf_test, groups)
    lines += validation_section(groups)
    length_lines, length_stats, _ = length_section(groups)
    lines += length_lines
    cost_lines, cost_stats = cost_section(bench, groups, out_dir)
    lines += cost_lines
    mcnemar_lines, mcnemar_summary = mcnemar_section(groups)
    lines += mcnemar_lines
    robust_lines, robust_stats = robustness_section(
        preds_dir, pairs_path, args.robust_tag, tau_mean
    )
    lines += robust_lines
    lines += expectations_section(groups, length_stats, cost_stats,
                                  mcnemar_summary, robust_stats)
    missing = sorted(set(missing) | set(cost_stats.get("long_missing", [])))
    if missing:
        lines += ["## 缺文件", ""] + [f"- `{name}`" for name in missing] + [""]
    else:
        lines += ["## 缺文件", "", "无。", ""]

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    make_figure(groups, out_fig)
    out_cost_fig = out_fig.parent / "级联_成本.png"
    make_cost_figure(groups, cost_stats, out_cost_fig)
    print(f"写入 {out_md}")
    print(f"写入 {out_fig}")
    print(f"写入 {out_cost_fig}")
    if missing:
        print(f"缺文件 {len(missing)} 个")


if __name__ == "__main__":
    main()





