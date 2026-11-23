"""C 组结果分析：读 results/ 下的 JSON 与逐条预测，写 C_结果.md 和 C_*.png。

只做描述统计，不写“符合/不符合”。缺文件的地方写“缺文件”并继续；
默认读正式结果，冒烟时用 ``--results_dir results/smoke_c --figures_dir
figures/smoke_c``。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from cascade import (
    cascade_predict,
    confidence_from_prob_pos,
    select_tau,
    tau_for_fraction,
)
from data import deduplicate_cold_splits, load_eval_set, load_splits
from plot_style import save_figure, setup_chinese_font
import matplotlib.pyplot as plt
from result_io import now_iso
from significance import holm_adjust, mcnemar_exact

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
CONFLICT_FUNCS = ("profanity_nh", "target_indiv_nh", "target_obj_nh")
EXPRESSION_NAMES = {0: "非仇恨", 1: "显式仇恨", 2: "隐式仇恨", 3: "转述"}
CLOAK_KINDS = ("pinyin", "homophone", "chaizi", "symbol", "delete")
CLOAK_NAMES = {"pinyin": "拼音", "homophone": "同音字",
               "chaizi": "拆字", "symbol": "插符号", "delete": "删词（对照）"}

# (内部名, 报告名, 文件名前缀, 种子)
METHODS = [
    ("tfidf_char", "TF-IDF（字）", "cold_tfidf_main_char", (0,)),
    ("tfidf_word", "TF-IDF（词）", "cold_tfidf_main_word", (0,)),
    ("textcnn_char", "TextCNN（字）", "cold_textcnn_main_char", SEEDS),
    ("textcnn_word", "TextCNN（词）", "cold_textcnn_main_word", SEEDS),
    ("bilstm_char", "BiLSTM（字）", "cold_bilstm_main_char", SEEDS),
    ("bilstm_word", "BiLSTM（词）", "cold_bilstm_main_word", SEEDS),
    ("transformer_char", "Transformer（字）", "cold_transformer_main_char", SEEDS),
    ("rbt3", "rbt3", "cold_bert_rbt3", SEEDS),
    ("roberta", "RoBERTa-wwm-ext", "cold_bert_roberta", SEEDS),
]
METHOD_LABEL = {key: label for key, label, _, _ in METHODS}
METHOD_PREFIX = {key: prefix for key, _, prefix, _ in METHODS}


def _read_json(results_dir: Path, name: str) -> dict | None:
    path = results_dir / f"{name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_pred(results_dir: Path, name: str) -> pd.DataFrame | None:
    path = results_dir / "preds" / f"{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    values = [float(v) for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], None
    return float(statistics.mean(values)), float(statistics.stdev(values))


def _fmt(mean: float | None, std: float | None = None, digits: int = 2) -> str:
    if mean is None:
        return "—"
    text = f"{mean * 100:.{digits}f}"
    if std is not None:
        text += f" ± {std * 100:.{digits}f}"
    return text


def _fmt_signed(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value * 100:+.{digits}f}"


def _macro_f1(labels, preds) -> float:
    return float(f1_score(labels, preds, average="macro", zero_division=0))


def _pos_pr(labels, preds) -> tuple[float | None, float | None]:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    tp = int(((preds == 1) & (labels == 1)).sum())
    pp = int((preds == 1).sum())
    ap = int((labels == 1).sum())
    return (tp / pp if pp else None), (tp / ap if ap else None)


def _run_seed_preds(results_dir: Path, name: str) -> list[dict]:
    """返回某方法每个可用种子的 JSON 与测试预测。"""
    out = []
    for seed in SEEDS:
        run = f"{name}_s{seed}"
        js = _read_json(results_dir, run)
        pred = _read_pred(results_dir, run)
        if js is not None or pred is not None:
            out.append({"seed": seed, "run": run, "json": js, "pred": pred})
    return out


def _method_runs(results_dir: Path, key: str) -> list[dict]:
    prefix = METHOD_PREFIX[key]
    seeds = dict((k, s) for k, _, _, s in METHODS)[key]
    out = []
    for seed in seeds:
        run = f"{prefix}_s{seed}"
        js = _read_json(results_dir, run)
        pred = _read_pred(results_dir, run)
        if js is not None or pred is not None:
            out.append({"seed": seed, "run": run, "json": js, "pred": pred})
    return out


def _test_labels() -> pd.DataFrame:
    return load_splits("cold")["test"]


def _align(frame: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """按 index 把预测并到数据表上；缺 index 时按行号。"""
    if pred is None or frame is None:
        return pd.DataFrame()
    left = frame.reset_index(drop=True)
    if "index" in pred.columns:
        left = left.copy()
        left["index"] = np.arange(len(left), dtype=np.int64)
        merged = left.merge(pred, on="index", how="inner", suffixes=("", "_pred"))
    else:
        merged = left.copy()
        merged["pred"] = pred["pred"].to_numpy()[:len(left)]
    return merged


def main_section(results_dir: Path):
    """主测试表与细粒度误判率。"""
    test = _test_labels()
    rows, fine_rows = [], []
    for key, label, _, _ in METHODS:
        runs = _method_runs(results_dir, key)
        acc, f1, prec, rec = [], [], [], []
        anti, other = [], []
        for run in runs:
            js, pred = run["json"], run["pred"]
            if js is not None:
                acc.append(js.get("test_accuracy"))
                f1.append(js.get("test_macro_f1"))
                prec.append(js.get("test_pos_precision"))
                rec.append(js.get("test_pos_recall"))
            merged = _align(test, pred)
            if not merged.empty:
                y = merged["label"].to_numpy()
                p = merged["pred"].to_numpy()
                if js is None:
                    acc.append(float((y == p).mean()))
                    f1.append(_macro_f1(y, p))
                    pr, rc = _pos_pr(y, p)
                    prec.append(pr)
                    rec.append(rc)
                fg = merged["fine-grained-label"].to_numpy()
                anti.append(float((p[fg == 3] == 1).mean()) if (fg == 3).any() else None)
                other.append(float((p[fg == 0] == 1).mean()) if (fg == 0).any() else None)
        if runs:
            rows.append({
                "key": key, "label": label, "n": len(runs),
                "acc": _mean_std(acc), "f1": _mean_std(f1),
                "prec": _mean_std(prec), "rec": _mean_std(rec),
            })
            fine_rows.append({"label": label, "n": len(runs),
                              "anti": _mean_std(anti), "other": _mean_std(other)})
    lines = ["## 1. COLD 主测试", "",
             "均值 ± 样本标准差（多个种子）；TF-IDF 只有种子 0。“攻击性”为正类。", "",
             "| 方法 | 种子数 | 准确率 | 宏 F1 | 攻击性精确率 | 攻击性召回率 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['n']} | {_fmt(*r['acc'])} | {_fmt(*r['f1'])} | "
            f"{_fmt(*r['prec'])} | {_fmt(*r['rec'])} |"
        )
    if not rows:
        lines.append("| 缺文件 | — | — | — | — | — |")
    # 冒烟时测试集被截断，表头 n 按实际参与评测的 index 统计
    counts = {}
    for key, _, _, _ in METHODS:
        for run in _method_runs(results_dir, key):
            if run["pred"] is not None and "index" in run["pred"]:
                idx = run["pred"]["index"].to_numpy()
                fg = test.iloc[idx]["fine-grained-label"].to_numpy()
                counts = {c: int((fg == c).sum()) for c in (0, 1, 2, 3)}
                break
        if counts:
            break
    lines += ["", "### 细粒度误判率", "",
              "每类中被判为攻击性的比例；n 为实际参与评测的条数。", "",
              f"| 方法 | 其他安全 0（n={counts.get(0, 0)}） | 攻击个人 1（n={counts.get(1, 0)}） | "
              f"攻击群体 2（n={counts.get(2, 0)}） | 反歧视 3（n={counts.get(3, 0)}） |",
              "|---|---:|---:|---:|---:|"]
    # 细粒度每类的判为攻击性比例，按方法×类别重算一遍
    for key, label, _, _ in METHODS:
        runs = _method_runs(results_dir, key)
        cells = {c: [] for c in (0, 1, 2, 3)}
        for run in runs:
            merged = _align(test, run["pred"])
            if merged.empty:
                continue
            fg = merged["fine-grained-label"].to_numpy()
            p = merged["pred"].to_numpy()
            for c in cells:
                if (fg == c).any():
                    cells[c].append(float((p[fg == c] == 1).mean()))
        vals = [_fmt(*_mean_std(cells[c])) for c in (0, 1, 2, 3)]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    if not any(_method_runs(results_dir, k) for k, _, _, _ in METHODS):
        lines.append("| 缺文件 | — | — | — | — |")
    return lines, rows, fine_rows


def hatecheck_section(results_dir: Path):
    """HateCheck 分功能表；三个口径冲突类单独成表。"""
    try:
        frame = load_eval_set("mhc_zh")
    except Exception as exc:  # noqa: BLE001 - 缺文件时继续
        return [f"## 2. HateCheck", "", f"缺文件：{exc}", ""], None
    funcs = sorted(frame["functionality"].unique())
    normal = [f for f in funcs if f not in CONFLICT_FUNCS]
    conflict = [f for f in funcs if f in CONFLICT_FUNCS]
    table = {}
    for key, label, _, _ in METHODS:
        per_func, overall = {}, []
        for run in _method_runs(results_dir, key):
            merged = _align(frame, _read_pred(results_dir, f"{run['run']}__mhc_zh"))
            if merged.empty:
                continue
            for f in funcs:
                sel = merged["functionality"] == f
                if sel.any():
                    per_func.setdefault(f, []).append(
                        float((merged.loc[sel, "pred"] == merged.loc[sel, "label"]).mean())
                    )
            sel = merged["functionality"].isin(normal)
            if sel.any():
                overall.append(
                    float((merged.loc[sel, "pred"] == merged.loc[sel, "label"]).mean())
                )
        table[key] = {"per": per_func, "overall": _mean_std(overall)}
    if not any(v["per"] for v in table.values()):
        return ["## 2. HateCheck", "", "缺文件：HateCheck 逐条预测", ""], None

    def render(rows, title):
        lines = [f"### {title}", "",
                 "| 功能 | n | " + " | ".join(METHOD_LABEL[k] for k, _, _, _ in METHODS) + " |",
                 "|---|---:|" + "---:|" * len(METHODS)]
        for f in rows:
            n = int((frame["functionality"] == f).sum())
            vals = []
            for key, _, _, _ in METHODS:
                vals.append(_fmt(*_mean_std(table[key]["per"].get(f, []))))
            lines.append(f"| {f} | {n} | " + " | ".join(vals) + " |")
        return lines

    lines = ["## 2. HateCheck 分功能准确率", "",
             "口径冲突类不计入总体；总体只算其余功能。", ""]
    lines += render(normal, "非冲突功能")
    lines += ["", f"**非冲突总体**：" + "；".join(
        f"{METHOD_LABEL[k]} {_fmt(*table[k]['overall'])}"
        for k, _, _, _ in METHODS), ""]
    lines += render(conflict, "口径冲突类（单独，不计入总体）")
    lines.append("")
    return lines, table


def toxicn_section(results_dir: Path):
    """ToxiCN 总体、按平台、按表达方式。"""
    try:
        frame = load_eval_set("toxicn_test")
    except Exception as exc:  # noqa: BLE001
        return [f"## 3. ToxiCN", "", f"缺文件：{exc}", ""], None
    groups = {"总体": [None], "平台": sorted(frame["platform"].unique()),
              "表达方式": sorted(frame["expression"].unique())}
    stats = {}
    for key, label, _, _ in METHODS:
        per = {}
        for run in _method_runs(results_dir, key):
            merged = _align(frame, _read_pred(results_dir, f"{run['run']}__toxicn_test"))
            if merged.empty:
                continue
            for level, values in groups.items():
                for value in values:
                    name = level if value is None else str(value)
                    sel = np.ones(len(merged), dtype=bool) if value is None else (
                        merged["platform"].to_numpy() == value
                        if level == "平台" else merged["expression"].to_numpy() == value
                    )
                    if sel.any():
                        y = merged.loc[sel, "label"].to_numpy()
                        p = merged.loc[sel, "pred"].to_numpy()
                        per.setdefault(name, {"acc": [], "f1": []})
                        per[name]["acc"].append(float((y == p).mean()))
                        per[name]["f1"].append(_macro_f1(y, p))
        stats[key] = per
    if not any(stats.values()):
        return ["## 3. ToxiCN", "", "缺文件：ToxiCN 逐条预测", ""], None
    lines = ["## 3. ToxiCN 跨平台测试", "",
             "单元格为“准确率 / 宏 F1”，多种子给均值。", ""]
    for level, values in groups.items():
        lines += [f"### {level}", "",
                  "| 分组 | n | " + " | ".join(METHOD_LABEL[k] for k, _, _, _ in METHODS) + " |",
                  "|---|---:|" + "---:|" * len(METHODS)]
        for value in values:
            name = level if value is None else str(value)
            if value is None:
                n = len(frame)
            elif level == "平台":
                n = int((frame["platform"] == value).sum())
            else:
                n = int((frame["expression"] == value).sum())
            disp = "总体" if value is None else (
                name if level == "平台" else EXPRESSION_NAMES.get(int(value), name))
            cells = []
            for key, _, _, _ in METHODS:
                item = stats[key].get(name)
                if not item:
                    cells.append("—")
                else:
                    acc = _mean_std(item["acc"])[0]
                    f1 = _mean_std(item["f1"])[0]
                    cells.append("—" if acc is None else f"{acc * 100:.2f} / {f1 * 100:.2f}")
            lines.append(f"| {disp} | {n} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines, stats


def cloak_section(results_dir: Path):
    """四种规避变换的配对召回下降。"""
    records = {}
    for kind in CLOAK_KINDS:
        try:
            orig_frame = load_eval_set(f"cloak_{kind}_orig")
            edit_frame = load_eval_set(f"cloak_{kind}_edit")
        except FileNotFoundError:
            continue
        for key, label, _, _ in METHODS:
            drops, origs, edits = [], [], []
            for run in _method_runs(results_dir, key):
                po = _read_pred(results_dir, f"{run['run']}__cloak_{kind}_orig")
                pe = _read_pred(results_dir, f"{run['run']}__cloak_{kind}_edit")
                if po is None or pe is None:
                    continue
                mo = _align(orig_frame, po)
                me = _align(edit_frame, pe)
                if mo.empty or me.empty:
                    continue
                merged = mo.merge(me, on="pair_id", suffixes=("_o", "_e"))
                merged = merged[merged["label_o"].to_numpy() == 1]
                if merged.empty:
                    continue
                ro = float((merged["pred_o"].to_numpy() == 1).mean())
                re = float((merged["pred_e"].to_numpy() == 1).mean())
                origs.append(ro)
                edits.append(re)
                drops.append(ro - re)
            if drops:
                records[(kind, key)] = {
                    "drop": _mean_std(drops), "orig": _mean_std(origs),
                    "edit": _mean_std(edits), "n": len(drops),
                }
    lines = ["## 4. 规避变换的召回下降", "",
             "召回下降 = 配对原评论召回率 − 变体召回率；只算标签为攻击性的配对。", ""]
    if not records:
        lines += ["缺文件：自制规避集或其逐条预测。", ""]
        return lines, records
    lines += ["| 变换 | " + " | ".join(METHOD_LABEL[k] for k, _, _, _ in METHODS) + " |",
              "|---|" + "---:|" * len(METHODS)]
    for kind in CLOAK_KINDS:
        cells = []
        for key, _, _, _ in METHODS:
            item = records.get((kind, key))
            cells.append("—" if item is None else _fmt_signed(item["drop"][0]))
        lines.append(f"| {CLOAK_NAMES[kind]} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines, records


def swap_section(results_dir: Path):
    """群体词互换：判断一致率与概率变化绝对值。"""
    try:
        orig_frame = load_eval_set("swap_orig")
        edit_frame = load_eval_set("swap_edit")
    except FileNotFoundError:
        return ["## 5. 群体词互换", "", "缺文件：自制互换集。", ""], {}
    out = {}
    for key, label, _, _ in METHODS:
        cons, delta = [], []
        for run in _method_runs(results_dir, key):
            po = _read_pred(results_dir, f"{run['run']}__swap_orig")
            pe = _read_pred(results_dir, f"{run['run']}__swap_edit")
            if po is None or pe is None:
                continue
            mo = _align(orig_frame, po)
            me = _align(edit_frame, pe)
            if mo.empty or me.empty:
                continue
            merged = mo.merge(me, on="pair_id", suffixes=("_o", "_e"))
            if merged.empty:
                continue
            cons.append(float((merged["pred_o"] == merged["pred_e"]).mean()))
            if "prob_pos_o" in merged and "prob_pos_e" in merged:
                delta.append(float(
                    np.abs(merged["prob_pos_e"].to_numpy()
                           - merged["prob_pos_o"].to_numpy()).mean()
                ))
        if cons:
            out[key] = {"cons": _mean_std(cons), "delta": _mean_std(delta),
                        "n": len(cons)}
    lines = ["## 5. 群体词互换一致率", "",
             "一致率 = 判断不变的配对比例；概率变化 = |prob_pos(变体) − prob_pos(原句)| 的均值。",
             "",
             "| 方法 | 种子数 | 一致率 | 概率变化 |",
             "|---|---:|---:|---:|"]
    if not out:
        lines.append("| 缺文件 | — | — | — |")
    for key, _, _, _ in METHODS:
        item = out.get(key)
        if item is None:
            lines.append(f"| {METHOD_LABEL[key]} | — | 缺文件 | — |")
        else:
            lines.append(f"| {METHOD_LABEL[key]} | {item['n']} | "
                         f"{_fmt(*item['cons'])} | "
                         f"{'—' if item['delta'][0] is None else _fmt(*item['delta'])} |")
    lines.append("")
    return lines, out


def _learning_runs(results_dir: Path, key: str, n: int | None) -> list[dict]:
    if n is None:
        prefix = METHOD_PREFIX[key]
    elif key == "rbt3":
        prefix = f"cold_bert_n{n}_rbt3"
    else:
        # run_C.sh 的 q2 后缀是 n<n>_<char|word>，文件名里 n 在分词器前面
        model = {"tfidf_char": "tfidf", "textcnn_char": "textcnn",
                 "transformer_char": "transformer"}[key]
        prefix = f"cold_{model}_main_n{n}_char"
    seeds = dict((k, s) for k, _, _, s in METHODS)[key]
    out = []
    for seed in seeds:
        name = f"{prefix}_s{seed}"
        js = _read_json(results_dir, name)
        if js is not None:
            out.append({"seed": seed, "run": name, "json": js})
    return out


def learning_section(results_dir: Path):
    """C2 学习曲线：n=1000/3000/10000 与全量。"""
    full_n = len(deduplicate_cold_splits(load_splits("cold"))[0]["train"])
    grid = [(1000, "1000"), (3000, "3000"), (10000, "10000"), (None, "全量")]
    methods = [("tfidf_char", "TF-IDF（字）"), ("textcnn_char", "TextCNN（字）"),
               ("transformer_char", "Transformer（字）"), ("rbt3", "rbt3")]
    data = {}
    for key, label in methods:
        for n, nlabel in grid:
            vals = [r["json"].get("test_macro_f1") for r in _learning_runs(results_dir, key, n)]
            vals = [v for v in vals if v is not None]
            if vals:
                data[(key, nlabel)] = (_mean_std(vals), len(vals))
    lines = ["## 6. 学习曲线", "",
             f"全量训练集 {full_n} 条（COLD 去重后）。单元格为宏 F1 均值 ± 标准差。", "",
             "| n | " + " | ".join(label for _, label in methods) + " |",
             "|---:|" + "---:|" * len(methods)]
    for _, nlabel in grid:
        cells = []
        for key, _ in methods:
            item = data.get((key, nlabel))
            cells.append("缺文件" if item is None else _fmt(*item[0]))
        lines.append(f"| {nlabel} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines, data


def _auto_accuracy(pred, conf, labels, fraction: float) -> float | None:
    """把置信度最低的 fraction 交给人工（假设判对），返回自动部分准确率。"""
    if fraction >= 1.0:
        return None
    tau = tau_for_fraction(conf, fraction)
    keep = np.asarray(conf) >= tau
    if not keep.any():
        return None
    return float((np.asarray(pred)[keep] == np.asarray(labels)[keep]).mean())


def _anti_bias_rate(test: pd.DataFrame, pred: pd.DataFrame) -> float | None:
    merged = _align(test, pred)
    if merged.empty:
        return None
    fg = merged["fine-grained-label"].to_numpy()
    sel = fg == 3
    return float((merged.loc[sel, "pred"] == 1).mean()) if sel.any() else None


def cascade_section(results_dir: Path):
    """三级审核：TF-IDF（字）→ rbt3/RoBERTa → 人工。"""
    tfidf_name = "cold_tfidf_main_char_s0"
    tfidf_test = _read_pred(results_dir, tfidf_name)
    tfidf_val = _read_pred(results_dir, f"{tfidf_name}__val")
    if tfidf_test is None or tfidf_val is None:
        return ["## 7. 三级审核级联", "", "缺文件：TF-IDF（字）的测试/验证预测。", ""], {}
    test = _test_labels()
    val = load_splits("cold")["val"]
    fracs = np.linspace(0.0, 1.0, 21)
    curves, table = {}, []
    tfidf_anti = _anti_bias_rate(test, tfidf_test)
    for bert_key, bert_label in (("rbt3", "rbt3"), ("roberta", "RoBERTa-wwm-ext")):
        seed_rows, seed_curves = [], []
        for seed in SEEDS:
            name = f"cold_bert_{bert_key}_s{seed}"
            btest = _read_pred(results_dir, name)
            bval = _read_pred(results_dir, f"{name}__val")
            if btest is None or bval is None:
                continue
            val_conf = confidence_from_prob_pos(tfidf_val["prob_pos"].to_numpy())
            chosen = select_tau(
                tfidf_val["pred"].to_numpy(), val_conf,
                bval["pred"].to_numpy(), bval["label"].to_numpy(),
            )
            tau = chosen["tau"]
            mask = confidence_from_prob_pos(tfidf_test["prob_pos"].to_numpy()) < tau
            bert_conf = confidence_from_prob_pos(btest["prob_pos"].to_numpy())
            tfidf_conf = confidence_from_prob_pos(tfidf_test["prob_pos"].to_numpy())
            mixed_conf = np.where(mask, bert_conf, tfidf_conf)
            pred = cascade_predict(
                tfidf_test["pred"].to_numpy(), tfidf_conf,
                btest["pred"].to_numpy(), tau,
            )
            merged = _align(test, pd.DataFrame({
                "index": btest["index"].to_numpy(), "pred": pred,
                "label": btest["label"].to_numpy(),
            }))
            y = merged["label"].to_numpy()
            p = merged["pred"].to_numpy()
            fg = merged["fine-grained-label"].to_numpy()
            bmerged = _align(test, btest)
            seed_rows.append({
                "seed": seed,
                "tfidf_acc": float((merged["label"].to_numpy()
                                   == tfidf_test.set_index("index").loc[
                                       merged["index"], "pred"].to_numpy()).mean()),
                "bert_acc": float((bmerged["label"] == bmerged["pred"]).mean()),
                "cascade_acc": float((y == p).mean()),
                "cascade_f1": _macro_f1(y, p),
                "bert_f1": _macro_f1(bmerged["label"], bmerged["pred"]),
                "anti": float((p[fg == 3] == 1).mean()) if (fg == 3).any() else None,
                "bert_anti": (float((bmerged.loc[bmerged["fine-grained-label"] == 3,
                                                 "pred"] == 1).mean())
                              if (bmerged["fine-grained-label"] == 3).any() else None),
                "val_frac": chosen["fraction"],
                "test_frac": float(mask.mean()),
            })
            seed_curves.append([
                _auto_accuracy(pred, mixed_conf, y, f) for f in fracs
            ])
        if not seed_rows:
            continue
        table.append({"label": bert_label, "rows": seed_rows})
        if seed_curves:
            arr = np.array([[np.nan if v is None else v for v in row]
                            for row in seed_curves], dtype=float)
            mean_curve = np.array([
                np.nan if np.all(np.isnan(arr[:, j])) else np.nanmean(arr[:, j])
                for j in range(arr.shape[1])
            ])
            curves[bert_label] = (fracs, mean_curve)
    lines = ["## 7. 三级审核级联", "",
             "第一级 TF-IDF（字），第二级 BERT；τ 在验证集上按 1% 档选。"
             "“人工”把第二级置信度最低的 k% 交给人工并假设判对。", ""]
    if not table:
        lines += ["缺文件：TF-IDF 与 BERT 的验证/测试预测。", ""]
        return lines, curves
    lines += ["| 第二级 | 单独 TF-IDF 准确率 | 单独 BERT 准确率 | 级联准确率 | 级联宏 F1 | "
              "验证交出比例 | 测试交出比例 | 反歧视误判（TF-IDF/ BERT/ 级联） |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in table:
        rows = item["rows"]

        def avg(k):
            vals = [r[k] for r in rows if r[k] is not None]
            return float(np.mean(vals)) if vals else None
        anti = (f"{_fmt(tfidf_anti)} / {_fmt(avg('bert_anti'))} / {_fmt(avg('anti'))}")
        lines.append(
            f"| {item['label']} | {_fmt(avg('tfidf_acc'))} | {_fmt(avg('bert_acc'))} | "
            f"{_fmt(avg('cascade_acc'))} | {_fmt(avg('cascade_f1'))} | "
            f"{_fmt(avg('val_frac'))} | {_fmt(avg('test_frac'))} | {anti} |"
        )
    lines += ["", "人工比例与自动部分准确率：", "",
              "| 人工比例 | " + " | ".join(curves) + " |",
              "|---:|" + "---:|" * len(curves)]
    for f in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
        cells = []
        for label, (fracs_arr, acc) in curves.items():
            idx = int(np.argmin(np.abs(fracs_arr - f)))
            cells.append(_fmt(None if np.isnan(acc[idx]) else acc[idx]))
        lines.append(f"| {f * 100:.0f}% | " + " | ".join(cells) + " |")
    lines.append("")
    return lines, curves


def _paired_from_index(pa: pd.DataFrame, pb: pd.DataFrame):
    merged = pa.merge(pb, on="index", suffixes=("_a", "_b"))
    if merged.empty:
        return []
    if not np.array_equal(merged["label_a"], merged["label_b"]):
        raise ValueError("两模型在同一 index 上的标签不一致")
    return [(merged["pred_a"].to_numpy(), merged["label_a"].to_numpy(),
             merged["pred_b"].to_numpy())]


def _paired_from_pair_id(pa: pd.DataFrame, pb: pd.DataFrame):
    merged = pa.merge(pb, on="pair_id", suffixes=("_a", "_b"))
    if merged.empty:
        return []
    if not np.array_equal(merged["label_a"], merged["label_b"]):
        raise ValueError("配对集两边的标签不一致")
    return [(merged["pred_a"].to_numpy(), merged["label_a"].to_numpy(),
             merged["pred_b"].to_numpy())]


def _counts(a_pred, labels, b_pred) -> tuple[int, int]:
    a_ok = np.asarray(a_pred) == np.asarray(labels)
    b_ok = np.asarray(b_pred) == np.asarray(labels)
    return int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())


def _main_pairs(results_dir: Path, a_name: str, b_name: str, b_seeds=SEEDS):
    pa = _read_pred(results_dir, a_name)
    if pa is None:
        return []
    out = []
    for seed in b_seeds:
        pb = _read_pred(results_dir, f"{b_name}_s{seed}")
        if pb is not None:
            out.extend(_paired_from_index(pa, pb))
    return out


def significance_section(results_dir: Path):
    """精确 McNemar + Holm；三个种子都显著且方向一致才算显著。"""
    test = _test_labels()
    tfidf_char = _read_pred(results_dir, "cold_tfidf_main_char_s0")
    tfidf_word = _read_pred(results_dir, "cold_tfidf_main_word_s0")
    comparisons = []

    def add(name, pairs):
        if pairs:
            comparisons.append((name, pairs))

    if tfidf_char is not None:
        pairs = []
        for seed in SEEDS:
            r = _read_pred(results_dir, f"cold_bert_roberta_s{seed}")
            if r is not None:
                pairs.extend(_paired_from_index(r, tfidf_char))
        add("RoBERTa 对 TF-IDF（字）", pairs)
        add("TF-IDF（字）对 TF-IDF（词）",
            _paired_from_index(tfidf_char, tfidf_word) if tfidf_word is not None else [])
    pairs = []
    for seed in SEEDS:
        a = _read_pred(results_dir, f"cold_bert_roberta_s{seed}")
        b = _read_pred(results_dir, f"cold_bert_rbt3_s{seed}")
        if a is not None and b is not None:
            pairs.extend(_paired_from_index(a, b))
    add("RoBERTa 对 rbt3", pairs)
    pairs = []
    for seed in SEEDS:
        a = _read_pred(results_dir, f"cold_textcnn_main_char_s{seed}")
        b = _read_pred(results_dir, f"cold_textcnn_main_word_s{seed}")
        if a is not None and b is not None:
            pairs.extend(_paired_from_index(a, b))
    add("TextCNN（字）对 TextCNN（词）", pairs)
    # 每种方法在每种规避变换上，原评论对变体
    for kind in CLOAK_KINDS:
        for key, label, _, _ in METHODS:
            pairs = []
            for run in _method_runs(results_dir, key):
                po = _read_pred(results_dir, f"{run['run']}__cloak_{kind}_orig")
                pe = _read_pred(results_dir, f"{run['run']}__cloak_{kind}_edit")
                if po is not None and pe is not None:
                    pairs.extend(_paired_from_pair_id(po, pe))
            add(f"{label}：{CLOAK_NAMES[kind]} 原句对变体", pairs)

    flat = []
    records = []
    for cname, pairs in comparisons:
        rec = {"name": cname, "pairs": []}
        for a_pred, labels, b_pred in pairs:
            b, c = _counts(a_pred, labels, b_pred)
            rec["pairs"].append({"b": b, "c": c, "p": mcnemar_exact(b, c)})
            flat.append(rec["pairs"][-1]["p"])
        if rec["pairs"]:
            records.append(rec)
    adjusted = holm_adjust(flat) if flat else []
    cursor = 0
    for rec in records:
        for item in rec["pairs"]:
            item["adj"] = adjusted[cursor]
            cursor += 1
        rec["b"] = sum(x["b"] for x in rec["pairs"])
        rec["c"] = sum(x["c"] for x in rec["pairs"])
        rec["max_p"] = max(x["p"] for x in rec["pairs"])
        rec["max_adj"] = max(x["adj"] for x in rec["pairs"])
        need = 3 if len(rec["pairs"]) >= 3 else 1
        same = (all(x["b"] > x["c"] for x in rec["pairs"])
                or all(x["b"] < x["c"] for x in rec["pairs"]))
        rec["sig"] = (len(rec["pairs"]) >= need and same
                      and all(x["adj"] < 0.05 for x in rec["pairs"]))
        rec["direction"] = ("前者优" if rec["b"] > rec["c"] else
                            "后者优" if rec["c"] > rec["b"] else "持平")
        rec["need"] = need
    lines = ["## 8. 显著性（精确 McNemar + Holm）", "",
             "b = 前者对、后者错；c = 前者错、后者对。Holm 对全部比较的全部种子一起校正。"
             "三个种子都校正后 p<0.05 且方向一致才算显著；只有单种子的比较按单种子判。", "",
             "| 比较 | 配对数 | b | c | 最大原始 p | Holm p | 方向 | 显著 |",
             "|---|---:|---:|---:|---:|---:|---|---|"]
    real = records
    if not real:
        lines.append("| 缺文件 | — | — | — | — | — | — | — |")
    for rec in real:
        lines.append(
            f"| {rec['name']} | {len(rec['pairs'])} | {rec['b']} | {rec['c']} | "
            f"{rec['max_p']:.4g} | {rec['max_adj']:.4g} | {rec['direction']} | "
            f"{'是' if rec['sig'] else ('单种子不显著' if rec['need'] == 1 else '否')} |"
        )
    lines.append("")
    return lines, real


def sanity_section(main_rows, cloak_data, results_dir: Path):
    """合理性检查。"""
    lines = ["## 9. 合理性检查", ""]
    by_key = {r["key"]: r for r in main_rows}
    flags = []
    low = [(r["label"], r["acc"][0]) for r in main_rows
           if r["acc"][0] is not None and r["acc"][0] < 0.70]
    if low:
        detail = "；".join(f"{a} {b * 100:.2f}%" for a, b in low)
        lines.append(f"- COLD 准确率低于 70%：{detail}。")
        flags.append("准确率低于 70%：" + detail)
    else:
        lines.append("- COLD 准确率低于 70%：无。")
    rbt3 = by_key.get("rbt3", {}).get("acc", (None, None))[0]
    roberta = by_key.get("roberta", {}).get("acc", (None, None))[0]
    if rbt3 is not None and roberta is not None:
        diff = roberta - rbt3
        lines.append(f"- RoBERTa 减 rbt3 的准确率差：{diff * 100:+.2f} 个百分点"
                     + ("（低于 -2，需检查）" if diff < -0.02 else "（未低于 -2）"))
        if diff < -0.02:
            flags.append("RoBERTa 低于 rbt3 超过 2 个百分点")
    else:
        lines.append("- RoBERTa 与 rbt3：缺文件，无法比较。")
    # 规避集原评论召回 vs COLD 攻击性类召回
    worst = None
    for (kind, key), item in cloak_data.items():
        main_rec = by_key.get(key, {}).get("rec", (None, None))[0]
        if main_rec is None or item["orig"][0] is None:
            continue
        diff = abs(item["orig"][0] - main_rec)
        if worst is None or diff > worst[0]:
            worst = (diff, kind, key, item["orig"][0], main_rec)
    if worst is None:
        lines.append("- 规避集原评论召回与 COLD 攻击性类召回：缺文件，无法比较。")
    else:
        lines.append(f"- 规避集原评论召回与 COLD 攻击性类召回的最大差：{worst[0] * 100:.2f} "
                     f"个百分点（{CLOAK_NAMES[worst[1]]}/{METHOD_LABEL[worst[2]]}）"
                     + ("，超过 5，需检查" if worst[0] > 0.05 else "，未超过 5"))
        if worst[0] > 0.05:
            flags.append("规避集原评论召回与 COLD 攻击性类召回相差超过 5 个百分点")
    # 去重条数与登记口径
    _, actual = deduplicate_cold_splits(load_splits("cold"))
    reported = set()
    for key, _, _, _ in METHODS:
        for run in _method_runs(results_dir, key):
            js = run["json"] or {}
            rem = js.get("decontaminate_removed")
            if rem:
                reported.add((int(rem.get("train", -1)), int(rem.get("val", -1))))
    lines.append(f"- COLD 去重实测：train 剔除 {actual['train']} 条，dev 剔除 {actual['val']} 条；"
                 f"结果 JSON 记录：{sorted(reported) if reported else '缺文件'}。")
    if reported and reported != {(actual["train"], actual["val"])}:
        flags.append("去重条数与实测不一致")
    lines += ["", "**需检查项**：" + ("；".join(flags) if flags else "无。"), ""]
    return lines


def fig_main(rows, figures_dir: Path) -> Path | None:
    avail = [r for r in rows if r["f1"][0] is not None]
    if not avail:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(avail))
    vals = [r["f1"][0] * 100 for r in avail]
    errs = [0 if r["f1"][1] is None else r["f1"][1] * 100 for r in avail]
    ax.bar(x, vals, yerr=errs, capsize=3, color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in avail], rotation=25, ha="right")
    ax.set_ylabel("宏 F1（%）")
    ax.set_ylim(max(0, min(vals) - 10), 100)
    ax.set_title("COLD 测试集宏 F1")
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, figures_dir / "C_主测试.png")


def fig_fine(fine_rows, figures_dir: Path) -> Path | None:
    avail = [r for r in fine_rows if r["anti"][0] is not None]
    if not avail:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(avail))
    w = 0.38
    anti = [r["anti"][0] * 100 for r in avail]
    other = [0 if r["other"][0] is None else r["other"][0] * 100 for r in avail]
    ax.bar(x - w / 2, anti, w, label="反歧视言论（3）", color="#08589E")
    ax.bar(x + w / 2, other, w, label="其他安全类（0）", color="#7BCCC4")
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in avail], rotation=25, ha="right")
    ax.set_ylabel("被判为攻击性言论的比例（%）")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, figures_dir / "C_细粒度误判率.png")


def fig_toxicn(stats, figures_dir: Path) -> Path | None:
    if not stats:
        return None
    platforms = ["zhihu", "tieba"]
    methods = [k for k, _, _, _ in METHODS if k in stats and stats[k]]
    if not methods:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(platforms))
    w = 0.8 / len(methods)
    for i, key in enumerate(methods):
        vals = []
        for p in platforms:
            item = stats[key].get(p)
            vals.append(0 if not item else float(np.mean(item["acc"])) * 100)
        ax.bar(x + (i - len(methods) / 2 + 0.5) * w, vals, w, label=METHOD_LABEL[key])
    ax.set_xticks(x)
    ax.set_xticklabels(["知乎", "贴吧"])
    ax.set_ylabel("准确率（%）")
    ax.set_title("ToxiCN 按平台")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, figures_dir / "C_ToxiCN.png")


def fig_cloak(records, figures_dir: Path) -> Path | None:
    methods = [k for k, _, _, _ in METHODS if any((kind, k) in records for kind in CLOAK_KINDS)]
    if not methods:
        return None
    matrix = np.full((len(methods), len(CLOAK_KINDS)), np.nan)
    for i, key in enumerate(methods):
        for j, kind in enumerate(CLOAK_KINDS):
            item = records.get((kind, key))
            if item and item["drop"][0] is not None:
                matrix[i, j] = item["drop"][0] * 100
    fig, ax = plt.subplots(figsize=(7.5, 6))
    data = np.nan_to_num(matrix, nan=0.0)
    im = ax.imshow(data, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(CLOAK_KINDS)))
    ax.set_xticklabels([CLOAK_NAMES[k] for k in CLOAK_KINDS])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABEL[k] for k in methods])
    for i in range(len(methods)):
        for j in range(len(CLOAK_KINDS)):
            val = "—" if np.isnan(matrix[i, j]) else f"{matrix[i, j]:+.1f}"
            ax.text(j, i, val, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="召回下降（百分点）")
    ax.set_title("四种规避变换的召回下降")
    return save_figure(fig, figures_dir / "C_规避召回下降.png")


def fig_learning(data, figures_dir: Path) -> Path | None:
    if not data:
        return None
    methods = [("tfidf_char", "TF-IDF（字）"), ("textcnn_char", "TextCNN（字）"),
               ("transformer_char", "Transformer（字）"), ("rbt3", "rbt3")]
    ns = ["1000", "3000", "10000", "全量"]
    xs = [1000, 3000, 10000, 25653]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for key, label in methods:
        ys, es = [], []
        for n in ns:
            item = data.get((key, n))
            if item is None:
                ys.append(np.nan)
                es.append(0)
            else:
                ys.append(item[0][0] * 100)
                es.append(0 if item[0][1] is None else item[0][1] * 100)
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(ns)
    ax.set_xlabel("训练集条数")
    ax.set_ylabel("宏 F1（%）")
    ax.set_title("学习曲线")
    ax.legend()
    ax.grid(alpha=0.25)
    return save_figure(fig, figures_dir / "C_学习曲线.png")


def fig_cascade(curves, figures_dir: Path) -> Path | None:
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for label, (fracs, acc) in curves.items():
        ax.plot(fracs * 100, acc * 100, marker="o", markersize=3, label=label)
    ax.set_xlabel("交给人工的比例（%）")
    ax.set_ylabel("自动部分准确率（%）")
    ax.legend()
    ax.grid(alpha=0.25)
    return save_figure(fig, figures_dir / "C_级联.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="C 组中文实验结果分析")
    parser.add_argument("--results_dir", default=None, help="默认 results/")
    parser.add_argument("--figures_dir", default=None, help="默认 figures/")
    parser.add_argument("--out", default=None, help="默认 <results_dir>/C_结果.md")
    args = parser.parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else ROOT / "results"
    figures_dir = Path(args.figures_dir) if args.figures_dir else ROOT / "figures"
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    if not figures_dir.is_absolute():
        figures_dir = ROOT / figures_dir
    out = Path(args.out) if args.out else results_dir / "C_结果.md"
    if not out.is_absolute():
        out = ROOT / out
    setup_chinese_font()

    main_lines, main_rows, fine_rows = main_section(results_dir)
    hc_lines, hc_stats = hatecheck_section(results_dir)
    tox_lines, tox_stats = toxicn_section(results_dir)
    cloak_lines, cloak_data = cloak_section(results_dir)
    swap_lines, swap_data = swap_section(results_dir)
    learn_lines, learn_data = learning_section(results_dir)
    casc_lines, casc_curves = cascade_section(results_dir)
    sig_lines, sig_recs = significance_section(results_dir)
    san_lines = sanity_section(main_rows, cloak_data, results_dir)

    missing = []
    if not main_rows:
        missing.append("COLD 主测试结果")
    if hc_stats is None:
        missing.append("HateCheck 逐条预测")
    if tox_stats is None:
        missing.append("ToxiCN 逐条预测")
    if not cloak_data:
        missing.append("自制规避集或逐条预测")
    if not swap_data:
        missing.append("自制互换集或逐条预测")
    if not learn_data:
        missing.append("学习曲线结果")
    if not casc_curves:
        missing.append("级联所需的验证/测试预测")

    figures = [
        fig_main(main_rows, figures_dir),
        fig_fine(fine_rows, figures_dir),
        fig_toxicn(tox_stats, figures_dir),
        fig_cloak(cloak_data, figures_dir),
        fig_learning(learn_data, figures_dir),
        fig_cascade(casc_curves, figures_dir),
    ]
    lines = ["# C 组：中文攻击性言论识别结果", "",
             f"生成时间：{now_iso()}；结果目录：`{results_dir}`。", "",
             "HateCheck 的总体只算非冲突功能。", ""]
    lines += main_lines + [""] + hc_lines + [""] + tox_lines + [""] + cloak_lines
    lines += [""] + swap_lines + [""] + learn_lines + [""] + casc_lines
    lines += [""] + sig_lines + [""] + san_lines
    lines += ["## 10. 缺文件清单", ""]
    lines += [f"- {x}" for x in missing] if missing else ["- 无。"]
    lines += ["", "图：" + "；".join(f"`{p}`" for p in figures if p) if any(figures) else "图：缺文件。", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"写入 {out}")
    for p in figures:
        if p is not None:
            print(f"写入 {p}")


if __name__ == "__main__":
    main()
