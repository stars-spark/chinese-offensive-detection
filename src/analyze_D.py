"""D 组判定用数字：只算数，不写“符合 / 不符合”。

读 D1 的 rbt3/TF-IDF 结果、D2 的归因 JSON、D3 的大模型 JSON 与 C 组已有
结果，按预先定好的判法给出差值和“有差异”标记。输出
``results/D_判定数字.md``。

用法：python src/analyze_D.py [--results results] [--out results/D_判定数字.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEL = ROOT / "data_selfmade"
KINDS = ("pinyin", "homophone", "chaizi", "symbol")
BASE, AUG_ALL, AUG_HELD = "cold_bert_rbt3", "cold_bert_aug_all", "cold_bert_aug_held"
TFIDF = {
    "base": ("cold_tfidf_main_char", (0,)),
    "augall": ("cold_tfidf_main_aug_all", (0,)),
    "augheld": ("cold_tfidf_main_aug_held", (0,)),
}
CONDITIONS = [("base", BASE, (0, 1, 2)),
              ("augall", AUG_ALL, (0, 1, 2)),
              ("augheld", AUG_HELD, (0, 1, 2))]


class Results:
    def __init__(self, root: Path):
        self.root = root

    def js(self, name: str) -> dict | None:
        path = self.root / f"{name}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def pr(self, name: str) -> pd.DataFrame | None:
        path = self.root / "preds" / f"{name}.parquet"
        return pd.read_parquet(path) if path.exists() else None


def run_name(prefix: str, seed) -> str:
    return prefix if seed is None else f"{prefix}_s{seed}"


def ms(values) -> tuple[float | None, float | None, int]:
    """均值、样本标准差（百分点）、有效个数。"""
    arr = np.asarray([float(v) for v in values
                      if v is not None and np.isfinite(float(v))], dtype=np.float64)
    if len(arr) == 0:
        return None, None, 0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std, int(len(arr))


def decision(a, b) -> tuple[float | None, bool | None]:
    """判法：|差| > 2×较大标准差，且 > 0.5 个百分点。"""
    if a[2] == 0 or b[2] == 0:
        return None, None
    diff = a[0] - b[0]
    threshold = max(2 * max(a[1] or 0.0, b[1] or 0.0), 0.5)
    return float(diff), bool(abs(diff) > threshold)


def fmt(x) -> str:
    if x[2] == 0:
        return "缺"
    if x[2] == 1:
        return f"{x[0]:.2f}"
    return f"{x[0]:.2f} ± {x[1]:.2f}"


def _num(value) -> float:
    return float("nan") if value is None else float(value)


def fmt_diff(d, sig) -> str:
    if d is None:
        return "缺"
    return f"{d:+.2f}（{'有差异' if sig else '无法区分'}）"


def field_names(names, key: str) -> tuple:
    vals = []
    for name in names:
        js = R.js(name)
        if js is not None and js.get(key) is not None:
            vals.append(float(js[key]) * 100)
    return ms(vals)


def field(prefix: str, seeds, key: str) -> tuple:
    return field_names([run_name(prefix, s) for s in seeds], key)


def anti_bias_names(names, key: str = "test_fine_grained") -> tuple:
    vals = []
    for name in names:
        js = R.js(name)
        if js is None:
            continue
        item = (js.get(key) or {}).get("3")
        if item is not None:
            vals.append(float(item["rate_pred_offensive"]) * 100)
    return ms(vals)


def anti_bias(prefix: str, seeds, key: str = "test_fine_grained") -> tuple:
    return anti_bias_names([run_name(prefix, s) for s in seeds], key)


def _cloak_keep(kind: str, subset: str, unseen_vocab: set[str] | None = None) -> set[int]:
    meta = pd.read_csv(SEL / f"cloak_{kind}.csv")
    only = meta.matched_words.astype(str).str.split("|").apply(
        lambda x: set(x) == {"恶心"})
    keep = set(meta.pair_id[~only])
    if subset == "unseen":
        vocab = unseen_vocab or set()
        hit = meta.matched_words.astype(str).str.split("|").apply(
            lambda x: bool(set(x) & vocab))
        keep &= set(meta.pair_id[~hit])
    return keep


def recall_drop_names(names, kind: str, subset: str = "main",
                      unseen_vocab: set[str] | None = None) -> tuple:
    """配对原评论召回 − 变体召回（百分点）；只算全部匹配词为攻击性的配对。"""
    try:
        keep = _cloak_keep(kind, subset, unseen_vocab)
    except FileNotFoundError:
        return ms([])
    drops = []
    for name in names:
        po = R.pr(f"{name}__cloak_{kind}_orig")
        pe = R.pr(f"{name}__cloak_{kind}_edit")
        if po is None or pe is None:
            continue
        po, pe = po[po.pair_id.isin(keep)], pe[pe.pair_id.isin(keep)]
        if len(po) == 0 or len(pe) == 0:
            continue
        drops.append(float((po.pred == 1).mean() - (pe.pred == 1).mean()) * 100)
    return ms(drops)


def recall_drop(prefix: str, seeds, kind: str, subset: str = "main",
                unseen_vocab: set[str] | None = None) -> tuple:
    return recall_drop_names([run_name(prefix, s) for s in seeds], kind,
                             subset, unseen_vocab)


def aug_vocab(condition: str) -> set[str]:
    """增强数据里出现过的词表词；用于划“未见词”子集。"""
    vocab = set()
    for path in sorted(SEL.glob(f"aug_{condition}_s*.csv")):
        frame = pd.read_csv(path, keep_default_na=False)
        for words in frame["matched_words"].astype(str):
            vocab.update(w for w in words.split("|") if w)
    return vocab


def swap_consistency_names(names) -> tuple:
    vals = []
    for name in names:
        po = R.pr(f"{name}__swap_orig")
        pe = R.pr(f"{name}__swap_edit")
        if po is None or pe is None:
            continue
        merged = po.merge(pe, on="pair_id", suffixes=("_o", "_e"))
        if len(merged):
            vals.append(float((merged.pred_o == merged.pred_e).mean()) * 100)
    return ms(vals)


def swap_consistency(prefix: str, seeds) -> tuple:
    return swap_consistency_names([run_name(prefix, s) for s in seeds])


def extra_metric_names(names, name: str, key: str = "macro_f1") -> tuple:
    vals = []
    for run in names:
        js = R.js(run)
        item = (js or {}).get("extra_eval", {}).get(name)
        if item is not None:
            vals.append(float(item[key]) * 100)
    return ms(vals)


def extra_metric(prefix: str, seeds, name: str, key: str = "macro_f1") -> tuple:
    return extra_metric_names([run_name(prefix, s) for s in seeds], name, key)


R: Results  # 在 main 里赋值，供上面的小函数共用


def section_d1() -> list[str]:
    lines = ["## H10～H12 D1 数据增强", "",
             "召回下降 = 配对原评论召回 − 变体召回，单位百分点；"
             "主口径排除唯一改动词为“恶心”的配对。", ""]
    drops = {}
    for cond, prefix, seeds in CONDITIONS:
        for kind in KINDS:
            drops[(cond, kind)] = recall_drop(prefix, seeds, kind)
    lines += ["### 四种变体集主口径的召回下降", "",
              "| 条件 | " + " | ".join(KINDS) + " |",
              "|---|" + "---:|" * len(KINDS)]
    for cond, _, _ in CONDITIONS:
        cells = [fmt(drops[(cond, k)]) for k in KINDS]
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines += ["", "### 相对 base 的差值", "",
              "| 条件 | " + " | ".join(KINDS) + " |",
              "|---|" + "---|" * len(KINDS)]
    for cond in ("augall", "augheld"):
        cells = [fmt_diff(*decision(drops[(cond, k)], drops[("base", k)]))
                 for k in KINDS]
        lines.append(f"| {cond} − base | " + " | ".join(cells) + " |")

    lines += ["", "### H11 held 改善量 / all 改善量", "",
              "| 留出类型 | base−all | base−held | 比值 |", "|---|---:|---:|---:|"]
    for kind in ("homophone", "chaizi"):
        b, a, h = (drops[("base", kind)][0], drops[("augall", kind)][0],
                   drops[("augheld", kind)][0])
        if None in (b, a, h):
            lines.append(f"| {kind} | 缺 | 缺 | 缺 |")
            continue
        denom = b - a
        ratio = (b - h) / denom if abs(denom) > 1e-9 else None
        lines.append(f"| {kind} | {b - a:+.2f} | {b - h:+.2f} | "
                     f"{'—' if ratio is None else f'{ratio:.3f}'} |")

    lines += ["", "### COLD 测试宏 F1 与反歧视误判率", "",
              "| 条件 | 宏 F1 | 准确率 | 反歧视误判率 |",
              "|---|---:|---:|---:|"]
    f1, acc, anti = {}, {}, {}
    for cond, prefix, seeds in CONDITIONS:
        f1[cond] = field(prefix, seeds, "test_macro_f1")
        acc[cond] = field(prefix, seeds, "test_accuracy")
        anti[cond] = anti_bias(prefix, seeds)
        lines.append(f"| {cond} | {fmt(f1[cond])} | {fmt(acc[cond])} | "
                     f"{fmt(anti[cond])} |")
    lines.append("")
    for cond in ("augall", "augheld"):
        d1, s1 = decision(f1[cond], f1["base"])
        d2, s2 = decision(anti[cond], anti["base"])
        lines.append(f"- {cond} − base 宏 F1：{fmt_diff(d1, s1)}")
        lines.append(f"- {cond} − base 反歧视误判率：{fmt_diff(d2, s2)}")

    lines += ["", "### TF-IDF（字）三个条件（只照报，不参与判定）", "",
              "| 条件 | 宏 F1 | 反歧视误判率 | " + " | ".join(KINDS) + " |",
              "|---|---:|---:|" + "---:|" * len(KINDS)]
    for cond, (prefix, seeds) in TFIDF.items():
        cells = [fmt(recall_drop(prefix, seeds, k)) for k in KINDS]
        lines.append(f"| {cond} | {fmt(field(prefix, seeds, 'test_macro_f1'))} | "
                     f"{fmt(anti_bias(prefix, seeds))} | " + " | ".join(cells) + " |")

    lines += ["", "### 照报：互换一致率、HateCheck、ToxiCN", "",
              "| 条件 | 互换一致率 | HateCheck 宏 F1 | ToxiCN 宏 F1 |",
              "|---|---:|---:|---:|"]
    named = [(cond, prefix, seeds) for cond, prefix, seeds in CONDITIONS]
    named += [(f"tfidf_{k}", p, s) for k, (p, s) in TFIDF.items()]
    for cond, prefix, seeds in named:
        lines.append(
            f"| {cond} | {fmt(swap_consistency(prefix, seeds))} | "
            f"{fmt(extra_metric(prefix, seeds, 'mhc_zh'))} | "
            f"{fmt(extra_metric(prefix, seeds, 'toxicn_test'))} |")

    vocab = aug_vocab("all")
    lines += ["", "### 未见词子集（改动词在 aug_all 增强数据里从未出现）", "",
              "| 条件 | " + " | ".join(KINDS) + " |",
              "|---|" + "---:|" * len(KINDS)]
    for cond, prefix, seeds in CONDITIONS:
        cells = [fmt(recall_drop(prefix, seeds, k, "unseen", vocab))
                 for k in KINDS]
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    counts = {k: len(_cloak_keep(k, "unseen", vocab)) for k in KINDS}
    lines.append("")
    lines.append("- 未见词配对数：" + "，".join(
        f"{k} {counts[k]}" for k in KINDS) +
        f"；aug_all 增强数据词表 {len(vocab)} 个词。")
    lines.append("")
    return lines


def section_d2() -> list[str]:
    lines = ["## H13～H14 D2 归因", "",
             "富集比 = 群体词字正归因份额 ÷ 群体词字占正文份额；"
             "Mann-Whitney 为单侧 A > B。", "",
             "| 种子 | A n | A 中位数 (Q1–Q3) | B 中位数 | C 中位数 | "
             "A vs B p |", "|---:|---:|---:|---:|---:|---:|"]
    for seed in (0, 1, 2):
        path = R.root / f"D2_归因_s{seed}.json"
        if not path.exists():
            lines.append(f"| {seed} | 缺 | 缺 | 缺 | 缺 | 缺 |")
            continue
        with open(path, encoding="utf-8") as f:
            js = json.load(f)
        groups = js.get("groups", {})
        cells = []
        for group in ("A", "B", "C"):
            item = (groups.get(group) or {}).get("enrichment") or {}
            if item.get("median") is None:
                cells.append("缺")
            elif group == "A":
                cells.append(f"{item['median']:.2f} "
                             f"({item['q1']:.2f}–{item['q3']:.2f})")
            else:
                cells.append(f"{item['median']:.2f}")
        p = (js.get("mann_whitney", {}).get("A_vs_B") or {}).get("p")
        n_a = (groups.get("A") or {}).get("n", 0)
        lines.append(f"| {seed} | {n_a} | {cells[0]} | {cells[1]} | {cells[2]} | "
                     f"{'缺' if p is None else f'{p:.4g}'} |")

    lines += ["", "| 种子 | 50 步完整性中位数 | 实际步数 | 是否升 200 步 | "
              "实际完整性中位数 | 删最高 3 字下降 | 随机删 3 字下降 |",
              "|---:|---:|---:|---|---:|---:|---:|"]
    for seed in (0, 1, 2):
        path = R.root / f"D2_归因_s{seed}.json"
        if not path.exists():
            lines.append(f"| {seed} | 缺 | 缺 | 缺 | 缺 | 缺 | 缺 |")
            continue
        with open(path, encoding="utf-8") as f:
            js = json.load(f)
        comp = js.get("completeness", {})
        faith = js.get("faithfulness", {})
        lines.append(
            f"| {seed} | {_num(comp.get('median_at_initial_steps')):.4f} | "
            f"{comp.get('steps_used', 0)} | "
            f"{'是' if comp.get('escalated') else '否'} | "
            f"{_num(comp.get('median_used')):.4f} | "
            f"{_num(faith.get('top3_median_drop')):.4f} | "
            f"{_num(faith.get('random3_median_drop')):.4f} |")
    lines.append("")
    return lines


def _bench_rbt3_ms() -> float | None:
    path = R.root / "bench_cascade_cold_rbt3.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        js = json.load(f)
    return (js.get("ms_per_1k") or {}).get("bert_192")


def section_d3() -> list[str]:
    lines = ["## H15～H18 D3 大模型零样本", "",
             "大模型只有一次运行，标准差按 0 计；rbt3 用三个种子。", "",
             "| 模型 | 宏 F1 | 反歧视误判率 | " + " | ".join(KINDS) +
             " | ms/千条 | 相对 rbt3 倍数 |",
             "|---|---:|---:|" + "---:|" * len(KINDS) + "---:|---:|"]
    rbt_f1 = field(BASE, (0, 1, 2), "test_macro_f1")
    rbt_anti = anti_bias(BASE, (0, 1, 2))
    rbt_ms = _bench_rbt3_ms()
    lines.append(
        f"| rbt3 | {fmt(rbt_f1)} | {fmt(rbt_anti)} | " +
        " | ".join(fmt(recall_drop(BASE, (0, 1, 2), k)) for k in KINDS) +
        f" | {'缺' if rbt_ms is None else f'{rbt_ms:.2f}'} | 1.00 |")

    qwen_info = {}
    for name in ("qwen15b", "qwen3b", "qwen7b", "qwen14b"):
        prefix = f"cold_llm_{name}"
        js = R.js(prefix)
        if js is None:
            lines.append(f"| {name} | 缺 | 缺 | " + " | ".join(["缺"] * len(KINDS)) +
                         " | 缺 | 缺 |")
            continue
        f1 = ms([float(js["test_macro_f1"]) * 100])
        anti = anti_bias(prefix, (None,))
        drops = {k: recall_drop(prefix, (None,), k) for k in KINDS}
        # 一律从原始计时重算：早期版本把“秒/千条”误记为 ms_per_1k
        bench = js.get("bench") or {}
        ms_per_1k = (bench["median_sec"] / bench["n"] * 1_000_000.0
                     if bench.get("median_sec") and bench.get("n") else None)
        ratio = (ms_per_1k / rbt_ms
                 if ms_per_1k is not None and rbt_ms else None)
        qwen_info[name] = {"f1": f1, "anti": anti, "drops": drops,
                           "ms": ms_per_1k, "ratio": ratio, "js": js}
        lines.append(
            f"| {name} | {fmt(f1)} | {fmt(anti)} | " +
            " | ".join(fmt(drops[k]) for k in KINDS) +
            f" | {'缺' if ms_per_1k is None else f'{ms_per_1k:.2f}'} | "
            f"{'缺' if ratio is None else f'{ratio:.2f}'} |")
    lines.append("")
    for name, info in qwen_info.items():
        d1, s1 = decision(info["f1"], rbt_f1)
        d2, s2 = decision(info["anti"], rbt_anti)
        lines.append(f"- {name} − rbt3 宏 F1：{fmt_diff(d1, s1)}")
        lines.append(f"- {name} − rbt3 反歧视误判率：{fmt_diff(d2, s2)}")
        for kind in KINDS:
            d, sig = decision(info["drops"][kind], recall_drop(BASE, (0, 1, 2), kind))
            lines.append(f"- {name} − rbt3 {kind} 召回下降：{fmt_diff(d, sig)}")
    lines.append("")

    lines += ["### 阈值 0.5 照报", "",
              "| 模型 | 宏 F1@0.5 | 准确率@0.5 | 反歧视误判@0.5 |",
              "|---|---:|---:|---:|"]
    lines.append(f"| rbt3 | {fmt(rbt_f1)} | "
                 f"{fmt(field(BASE, (0, 1, 2), 'test_accuracy'))} | "
                 f"{fmt(rbt_anti)} |")
    for name, info in qwen_info.items():
        js = info["js"]
        f1 = ms([float(js["test_macro_f1_at_0.5"]) * 100])
        acc = ms([float(js["test_accuracy_at_0.5"]) * 100])
        anti = anti_bias(f"cold_llm_{name}", (None,), "test_fine_grained_at_0.5")
        lines.append(f"| {name} | {fmt(f1)} | {fmt(acc)} | {fmt(anti)} |")
    lines.append("")
    return lines


def _bench_roberta_ms() -> float | None:
    path = R.root / "bench_cascade_cold_roberta.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        js = json.load(f)
    return (js.get("ms_per_1k") or {}).get("bert_192")


def _bench_from_json(js: dict | None) -> float | None:
    """一律从 median_sec / n 重算，兼容早期把秒/千条写成 ms_per_1k 的 JSON。"""
    if not js:
        return None
    bench = js.get("bench") or {}
    if bench.get("median_sec") and bench.get("n"):
        return float(bench["median_sec"]) / float(bench["n"]) * 1_000_000.0
    return None


def bench_ms_names(names) -> tuple:
    vals = []
    for name in names:
        value = _bench_from_json(R.js(name))
        if value is not None:
            vals.append(value)
    return ms(vals)


def d4_ms(prefix: str, seeds) -> tuple:
    return bench_ms_names([run_name(prefix, s) for s in seeds])


def row_names(names, ms_override=None) -> dict:
    return {
        "f1": field_names(names, "test_macro_f1"),
        "acc": field_names(names, "test_accuracy"),
        "anti": anti_bias_names(names),
        "drops": {k: recall_drop_names(names, k) for k in KINDS},
        "mhc": extra_metric_names(names, "mhc_zh"),
        "toxicn": extra_metric_names(names, "toxicn_test"),
        "swap": swap_consistency_names(names),
        "ms": (ms_override if ms_override is not None
               else bench_ms_names(names)),
    }


def d4_row(prefix: str, seeds, ms_override=None) -> dict:
    return row_names([run_name(prefix, s) for s in seeds], ms_override)


def _d4_table_row(label: str, row: dict) -> str:
    cells = [fmt(row["f1"]), fmt(row["acc"]), fmt(row["anti"])]
    cells += [fmt(row["drops"][k]) for k in KINDS]
    cells.append(fmt(row["ms"]))
    return f"| {label} | " + " | ".join(cells) + " |"


def _d4_diff_cell(a: tuple, b: tuple) -> str:
    return fmt_diff(*decision(a, b))


def section_d4() -> list[str]:
    lines = ["## H21～H23 D4 LoRA 微调", "",
             "LoRA 三个种子，均值 ± 样本标准差；零样本只有一次运行，标准差按 0 计。"
             "召回下降 = 配对原评论召回 − 变体召回（百分点），主口径排除唯一"
             "改动词为“恶心”的配对。", ""]
    seven = "cold_llmft_qwen7b"
    small = "cold_llmft_qwen15b"
    zero = "cold_llm_qwen7b"
    rbt_ms = _bench_rbt3_ms()
    roberta_ms = _bench_roberta_ms()
    rows = {
        "rbt3": d4_row(BASE, (0, 1, 2),
                       ms([rbt_ms]) if rbt_ms is not None else ms([])),
        "RoBERTa": d4_row("cold_bert_roberta", (0, 1, 2),
                          ms([roberta_ms]) if roberta_ms is not None else ms([])),
        "Qwen-7B 零样本": d4_row(zero, (None,)),
        "Qwen-7B LoRA": d4_row(seven, (0, 1, 2)),
        "Qwen-1.5B LoRA": d4_row(small, (0, 1, 2)),
    }

    lines += ["### 主口径", "",
              "| 模型 | 宏 F1 | 准确率 | 反歧视误判率 | " +
              " | ".join(KINDS) + " | ms/千条 |",
              "|---|---:|---:|---:|" + "---:|" * len(KINDS) + "---:|"]
    for label, row in rows.items():
        lines.append(_d4_table_row(label, row))

    lines += ["", "### 照报项", "",
              "| 模型 | HateCheck 宏 F1 | ToxiCN 宏 F1 | 互换一致率 |",
              "|---|---:|---:|---:|"]
    for label, row in rows.items():
        lines.append(f"| {label} | {fmt(row['mhc'])} | {fmt(row['toxicn'])} | "
                     f"{fmt(row['swap'])} |")

    lines += ["", "### 与 rbt3、RoBERTa、Qwen-7B 零样本的差值", "",
              "单元格为“差值（有差异 / 无法区分）”；“有差异”按上述判法。", "",
              "| 比较 | Δ宏 F1 | Δ准确率 | Δ反歧视误判率 | " +
              " | ".join(f"Δ{k}" for k in KINDS) + " |",
              "|---|---:|---:|---:|" + "---:|" * len(KINDS)]
    comparisons = [
        ("7B LoRA − rbt3", rows["Qwen-7B LoRA"], rows["rbt3"]),
        ("7B LoRA − RoBERTa", rows["Qwen-7B LoRA"], rows["RoBERTa"]),
        ("7B LoRA − Qwen-7B 零样本", rows["Qwen-7B LoRA"],
         rows["Qwen-7B 零样本"]),
        ("1.5B LoRA − rbt3", rows["Qwen-1.5B LoRA"], rows["rbt3"]),
        ("1.5B LoRA − RoBERTa", rows["Qwen-1.5B LoRA"], rows["RoBERTa"]),
    ]
    for label, a, b in comparisons:
        cells = [_d4_diff_cell(a["f1"], b["f1"]),
                 _d4_diff_cell(a["acc"], b["acc"]),
                 _d4_diff_cell(a["anti"], b["anti"])]
        cells += [_d4_diff_cell(a["drops"][k], b["drops"][k]) for k in KINDS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", "### H21～H23 判定数字", ""]
    lines.append("- H21 7B LoRA − rbt3 宏 F1：" +
                 _d4_diff_cell(rows["Qwen-7B LoRA"]["f1"], rows["rbt3"]["f1"]))
    lines.append("- H22 7B LoRA − Qwen-7B 零样本 反歧视误判率：" +
                 _d4_diff_cell(rows["Qwen-7B LoRA"]["anti"],
                               rows["Qwen-7B 零样本"]["anti"]))
    for kind in ("pinyin", "homophone"):
        lines.append(f"- H23 {kind} 召回下降 7B LoRA − rbt3：" +
                     _d4_diff_cell(rows["Qwen-7B LoRA"]["drops"][kind],
                                   rows["rbt3"]["drops"][kind]))
    lines.append("")
    return lines


def section_d57() -> list[str]:
    """H24～H28：D5 带提示词微调、D6 增强微调、D7 少样本。"""
    lines = ["## H24～H28 D5～D7", "",
             "D5/D6 为种子 0～2，D7 为三套少样本例子（shot_seed 0～2）；"
             "均值 ± 样本标准差。零样本只有一次运行，标准差按 0 计。"
             "召回下降 = 配对原评论召回 − 变体召回（百分点），主口径排除"
             "唯一改动词为“恶心”的配对。", ""]
    names = {
        "D3 零样本": ["cold_llm_qwen7b"],
        "D4 分类头 LoRA": [f"cold_llmft_qwen7b_s{s}" for s in (0, 1, 2)],
        "D5 带提示词 LoRA": [f"cold_llmftp_qwen7b_s{s}" for s in (0, 1, 2)],
        "D6 增强 LoRA": [f"cold_llmft_qwen7b_aug_s{s}" for s in (0, 1, 2)],
        "D7 少样本": [f"cold_llm_qwen7b_fs{s}" for s in (0, 1, 2)],
    }
    rows = {label: row_names(run_names) for label, run_names in names.items()}

    lines += ["### 主口径", "",
              "| 条件 | 宏 F1 | 准确率 | 反歧视误判率 | " +
              " | ".join(KINDS) + " | HateCheck | ToxiCN | ms/千条 |",
              "|---|---:|---:|---:|" + "---:|" * len(KINDS) + "---:|---:|---:|"]
    for label, row in rows.items():
        cells = [fmt(row["f1"]), fmt(row["acc"]), fmt(row["anti"])]
        cells += [fmt(row["drops"][k]) for k in KINDS]
        cells += [fmt(row["mhc"]), fmt(row["toxicn"]), fmt(row["ms"])]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", "### 差值（有差异 / 无法区分）", "",
              "| 比较 | Δ宏 F1 | Δ准确率 | Δ反歧视误判率 | " +
              " | ".join(f"Δ{k}" for k in KINDS) + " |",
              "|---|---:|---:|---:|" + "---:|" * len(KINDS)]
    comparisons = [
        ("D5 − D4", rows["D5 带提示词 LoRA"], rows["D4 分类头 LoRA"]),
        ("D5 − D3 零样本", rows["D5 带提示词 LoRA"], rows["D3 零样本"]),
        ("D6 − D4", rows["D6 增强 LoRA"], rows["D4 分类头 LoRA"]),
        ("D7 − D3 零样本", rows["D7 少样本"], rows["D3 零样本"]),
    ]
    for label, a, b in comparisons:
        cells = [_d4_diff_cell(a["f1"], b["f1"]),
                 _d4_diff_cell(a["acc"], b["acc"]),
                 _d4_diff_cell(a["anti"], b["anti"])]
        cells += [_d4_diff_cell(a["drops"][k], b["drops"][k]) for k in KINDS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    d5, d4 = rows["D5 带提示词 LoRA"], rows["D4 分类头 LoRA"]
    d6, zero = rows["D6 增强 LoRA"], rows["D3 零样本"]
    d7 = rows["D7 少样本"]
    lines += ["", "### H24～H28 判定数字", ""]
    lines.append("- H24 D5 − D4 反歧视误判率：" +
                 _d4_diff_cell(d5["anti"], d4["anti"]))
    lines.append("- H25 D5 − 零样本 反歧视误判率：" +
                 _d4_diff_cell(d5["anti"], zero["anti"]))
    for kind in ("pinyin", "homophone", "chaizi"):
        lines.append(f"- H26 {kind} 召回下降 D6 − D4：" +
                     _d4_diff_cell(d6["drops"][kind], d4["drops"][kind]))
    lines.append("- H27 宏 F1 D6 − D4：" + _d4_diff_cell(d6["f1"], d4["f1"]))
    lines.append("- H28 宏 F1 D7 − 零样本：" + _d4_diff_cell(d7["f1"], zero["f1"]))
    lines.append("")
    return lines


def main() -> None:
    global R
    parser = argparse.ArgumentParser(description="D 组判定用数字")
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    R = Results(Path(args.results))
    out = Path(args.out) if args.out else Path(args.results) / "D_判定数字.md"
    lines = ["# D 组判定用数字（analyze_D.py 自动生成）", "",
             "只列数字与“有差异”标记。", ""]
    lines += section_d1()
    lines += section_d2()
    lines += section_d3()
    lines += section_d4()
    lines += section_d57()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"写入 {out}")


if __name__ == "__main__":
    main()

