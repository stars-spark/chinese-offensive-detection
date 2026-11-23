"""实验 K：配对显著性检验（精确 McNemar + 配对自助法 + Holm 校正）。

只读逐条预测，不训练模型。每个比较按种子 i 对种子 i 配对；只有一份
预测的 TF-IDF 固定用种子 0，去配对方每个种子。第 1～3 组与第 4 组的
主口径（“改动全部可见”子集）放在一起做一次 Holm 校正，校正后
p < 0.05 才算显著；第 4 组的补充口径（“全部配对”）单列，不参与校正。
两种口径都剔除分词后无差异的配对。

精确 McNemar 只用两边判断不同的样本：b = A 对 B 错，c = A 错 B 对，
在 b+c 次独立试验里做双侧二项检验（p=0.5）。配对自助法对每个样本的
“A 对减 B 对”（-1/0/1）重采样 10000 次，取 2.5%/97.5% 分位数为 95%
置信区间。重采样用多项分布等价实现，避免 25000×10000 的大矩阵。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from result_io import now_iso

ROOT = Path(__file__).resolve().parents[1]
BOOT_N = 10000
BOOT_SEED = 0
SCOPES = ("visible", "all")


def mcnemar_exact(b: int, c: int) -> float:
    """精确 McNemar 的双侧 p 值；b、c 是两种“一边对一边错”的条数。"""
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    return float(binomtest(int(b), n, 0.5, alternative="two-sided").pvalue)


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm 逐步校正，返回与输入同顺序的校正后 p 值。

    先按原始 p 升序排，第 rank 个乘 (m-rank)，再取累计最大值保证
    单调，最后截到 1。累计最大值不能省，否则后面的 p 可能反而更小。
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p, kind="stable")):
        value = min(1.0, float(p[idx]) * (m - rank))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def bootstrap_ci(diff: np.ndarray, n_boot: int = BOOT_N,
                 seed: int = BOOT_SEED) -> tuple[float, float]:
    """配对自助法的 95% 置信区间，输入是逐样本的 -1/0/1 差值。

    对经验分布做 n 次有放回抽样，等价于每轮抽一个三项多项分布
    （-1、0、+1 的条数）。这样只生成 n_boot 组计数，不用保存
    n_boot×n 的索引矩阵，结果与逐样本重采样相同。
    """
    diff = np.asarray(diff, dtype=np.float64)
    if len(diff) == 0:
        return (float("nan"), float("nan"))
    values, counts = np.unique(diff, return_counts=True)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(diff), counts / len(diff), size=n_boot)
    boot = draws @ values / len(diff)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def paired_consistency(orig: pd.DataFrame, edit: pd.DataFrame) -> tuple[float, int]:
    """按 pair_id 对齐原始/改写预测，算“两边都判对”的配对比例。"""
    if "pair_id" not in orig.columns or "pair_id" not in edit.columns:
        raise ValueError("配对一致率需要 pair_id 列")
    merged = orig.merge(edit, on="pair_id", suffixes=("_orig", "_edit"))
    both = ((merged["pred_orig"] == merged["label_orig"])
            & (merged["pred_edit"] == merged["label_edit"]))
    return float(both.mean()), len(merged)


def load_visible_pairs(path: Path) -> dict:
    """读 ``robust_visible_pairs.py`` 生成的可见配对清单。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for family in ("contrast", "cad"):
        if family not in data:
            raise ValueError(f"{path} 缺少 {family} 的可见配对清单")
        data[family].setdefault("visible", [])
        data[family].setdefault("excluded_no_diff", [])
    return data


def scope_pair_ids(pairs: dict, family: str, scope: str) -> list[int]:
    """返回某口径下参与统计的 pair_id；两种口径都剔除无差异配对。

    ``visible`` 是主口径“改动全部可见”子集；``all`` 是补充口径“全部配对”，
    按清单里的 ``n_total`` 取全部 pair_id。``excluded_no_diff`` 一律剔除。
    """
    if scope not in SCOPES:
        raise ValueError(f"未知口径 {scope!r}，应为 {SCOPES}")
    if family not in pairs:
        raise ValueError(f"可见配对清单缺少 {family}")
    meta = pairs[family]
    excluded = {int(x) for x in meta.get("excluded_no_diff", [])}
    if scope == "visible":
        ids = [int(x) for x in meta.get("visible", [])]
    else:
        ids = list(range(int(meta.get("n_total", 0))))
    return [pid for pid in ids if pid not in excluded]


def paired_metrics(orig: pd.DataFrame | None, edit: pd.DataFrame | None,
                   pair_ids: list[int]) -> dict:
    """在指定配对子集上算原始/改写准确率、降幅和配对一致率。

    原始与改写按 ``pair_id`` 内连接，再只保留 ``pair_ids`` 里的对。这样
    四个指标用的是同一批配对，样本数也一致。预测缺失或子集为空时返回
    ``n=0`` 与四个 ``None``，由调用方显示为“—”。
    """
    empty = {"n": 0, "orig_acc": None, "edit_acc": None,
             "drop": None, "consistency": None}
    if orig is None or edit is None:
        return empty
    if not orig["pair_id"].is_unique or not edit["pair_id"].is_unique:
        raise ValueError("原始或改写预测的 pair_id 有重复，无法一一配对")
    merged = orig.merge(edit, on="pair_id", suffixes=("_orig", "_edit"))
    keep = {int(x) for x in pair_ids}
    merged = merged[merged["pair_id"].isin(keep)]
    if merged.empty:
        return empty
    correct_orig = (merged["pred_orig"] == merged["label_orig"]).to_numpy()
    correct_edit = (merged["pred_edit"] == merged["label_edit"]).to_numpy()
    return {
        "n": int(len(merged)),
        "orig_acc": float(correct_orig.mean()),
        "edit_acc": float(correct_edit.mean()),
        "drop": float(correct_orig.mean() - correct_edit.mean()),
        "consistency": float((correct_orig & correct_edit).mean()),
    }


def _group4_specs(tag: str) -> list[dict]:
    """实验 I 第 4 组：每个比较拆成主口径与补充口径两份。

    两种口径都用 ``pair_id`` 对齐并剔除无差异配对；只有主口径的 p 值
    进入 Holm 校正。TF-IDF 与 TF-IDF(256) 只有种子 0。
    """
    specs = []
    methods = [("tfidf", "TF-IDF"), ("tfidf_ml256", "TF-IDF(256)"),
               ("textcnn", "TextCNN"), ("bilstm", "BiLSTM"),
               ("transformer", "Transformer"), ("distilbert", "DistilBERT")]
    for method, label in methods:
        seeds = [0] if method in ("tfidf", "tfidf_ml256") else [0, 1, 2]
        if method == "tfidf_ml256":
            base = f"imdb_tfidf_{tag}_ml256_s{{seed}}"
        else:
            base = f"imdb_{method}_{tag}_s{{seed}}"
        for family, fam_label in (("contrast", "对比集"), ("cad", "CAD")):
            for scope, scope_label, holm in (
                    ("visible", "主口径：全部可见", True),
                    ("all", "补充口径：全部配对", False)):
                specs.append({
                    "group": "4 实验 I 改写" if holm else "4 实验 I 改写（补充）",
                    "name": f"{label} {fam_label} 原始 vs 改写（{scope_label}）",
                    "a": f"{base}__{family}_orig",
                    "b": f"{base}__{family}_edit",
                    "seeds": seeds, "key": "pair_id", "family": family,
                    "scope": scope, "holm": holm,
                })
    return specs


def build_comparisons(tag: str = "robust") -> list[dict]:
    """预先列出的比较；第 1～3 组加第 4 组的主/补充两种口径。

    ``a``/``b`` 是预测文件名（不含 .parquet），``{seed}`` 在运行时替换；
    ``key`` 指明用哪一列对齐：普通测试集用 index，配对评测集用 pair_id。
    """
    specs = [
        # 1. IMDB 完整训练集，截断 256
        {"group": "1 IMDB 完整/256", "name": "TF-IDF vs Transformer",
         "a": "imdb_tfidf_len_ml256_s0", "b": "imdb_transformer_len_ml256_s{seed}",
         "seeds": [0, 1, 2], "key": "index"},
        {"group": "1 IMDB 完整/256", "name": "TF-IDF vs DistilBERT",
         "a": "imdb_tfidf_len_ml256_s0", "b": "imdb_distilbert_bert_n22500_s{seed}",
         "seeds": [0, 1, 2], "key": "index"},
        {"group": "1 IMDB 完整/256", "name": "Transformer vs TextCNN",
         "a": "imdb_transformer_len_ml256_s{seed}", "b": "imdb_textcnn_len_ml256_s{seed}",
         "seeds": [0, 1, 2], "key": "index"},
        {"group": "1 IMDB 完整/256", "name": "Transformer vs BiLSTM",
         "a": "imdb_transformer_len_ml256_s{seed}", "b": "imdb_bilstm_len_ml256_s{seed}",
         "seeds": [0, 1, 2], "key": "index"},
        # 2. AG News 完整训练集（DistilBERT 复用实验 F，清单里没有它的比较）
        {"group": "2 AG News 完整", "name": "TF-IDF vs Transformer",
         "a": "ag_news_tfidf_agsig_s0", "b": "ag_news_transformer_agsig_s{seed}",
         "seeds": [0, 1, 2], "key": "index"},
        {"group": "2 AG News 完整", "name": "Transformer vs TextCNN",
         "a": "ag_news_transformer_agsig_s{seed}", "b": "ag_news_textcnn_agsig_s{seed}",
         "seeds": [0, 1, 2], "key": "index"},
        # 3. 实验 J
        {"group": "3 长文本", "name": "headtail vs head512",
         "a": "imdb_distilbert_long_headtail_s{seed}",
         "b": "imdb_distilbert_long_head512_s{seed}",
         "seeds": [0, 1, 2], "key": "index"},
        {"group": "3 长文本", "name": "head512 vs head256",
         "a": "imdb_distilbert_long_head512_s{seed}",
         "b": "imdb_distilbert_bert_n22500_s{seed}",
         "seeds": [0, 1, 2], "key": "index"},
    ]
    return specs + _group4_specs(tag)


COMPARISONS = build_comparisons()


def load_pred(path: Path, key: str) -> pd.DataFrame:
    """读一份预测，只保留对齐键、标签、预测三列。"""
    frame = pd.read_parquet(path)
    missing = {"label", "pred", key} - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} 缺少列 {sorted(missing)}")
    out = frame[[key, "label", "pred"]].copy()
    if not out[key].is_unique:
        raise ValueError(f"{path.name} 的 {key} 有重复，无法一一配对")
    return out


def analyze_comparison(spec: dict, preds_dir: Path, pairs: dict | None = None) -> dict:
    """对一个比较逐种子跑 McNemar，返回记录（缺文件时记 missing）。

    带 ``family``/``scope`` 的第 4 组比较先按口径筛配对；``pairs`` 是
    可见配对清单。筛选后为空时记 ``empty``，不产生 p 值。
    """
    record = {"spec": spec, "runs": []}
    for seed in spec["seeds"]:
        names = [spec["a"].format(seed=seed), spec["b"].format(seed=seed)]
        paths = [preds_dir / f"{name}.parquet" for name in names]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            record["runs"].append({"seed": seed, "missing": missing})
            continue
        left = load_pred(paths[0], spec["key"])
        right = load_pred(paths[1], spec["key"])
        merged = left.merge(right, on=spec["key"], suffixes=("_a", "_b"))
        if len(merged) != len(left) or len(merged) != len(right):
            raise ValueError(f"{spec['name']} 种子 {seed} 的配对键没有一一对应")
        if spec.get("family") and spec.get("scope"):
            if pairs is None:
                raise ValueError(f"{spec['name']} 需要可见配对清单")
            ids = scope_pair_ids(pairs, spec["family"], spec["scope"])
            merged = merged[merged[spec["key"]].isin(set(ids))].reset_index(drop=True)
            if merged.empty:
                record["runs"].append({"seed": seed, "empty": True})
                continue
        correct_a = (merged["pred_a"] == merged["label_a"]).to_numpy()
        correct_b = (merged["pred_b"] == merged["label_b"]).to_numpy()
        diff = correct_a.astype(np.int64) - correct_b.astype(np.int64)
        b = int((correct_a & ~correct_b).sum())
        c = int((~correct_a & correct_b).sum())
        record["runs"].append({
            "seed": seed,
            "n": len(merged),
            "acc_a": float(correct_a.mean()),
            "acc_b": float(correct_b.mean()),
            "diff": diff,
            "b": b, "c": c,
            "p": mcnemar_exact(b, c),
            "files": [p.name for p in paths],
        })
    return record


def apply_holm(records: list[dict]) -> int:
    """只把主口径与第 1～3 组的 p 值一起做 Holm 校正。

    补充口径（``spec['holm'] == False``）即使有 p 值也不进入校正，
    也不写 ``p_adj``。返回参与校正的检验数。
    """
    refs = [run for rec in records if rec["spec"].get("holm", True)
            for run in rec["runs"] if "p" in run]
    adjusted = holm_adjust([run["p"] for run in refs])
    for run, padj in zip(refs, adjusted):
        run["p_adj"] = padj
    return len(refs)


def _fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}"


def _fmt_pp(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:+.2f}"


def _run_by_seed(record: dict, seed: int) -> dict | None:
    for run in record["runs"]:
        if run["seed"] == seed and "p" in run:
            return run
    return None


def _seed_sig_text(record: dict, adjusted: bool = True) -> str:
    tested = [run for run in record["runs"] if "p" in run]
    if not tested:
        return "缺文件"
    if len(tested) != len(record["spec"]["seeds"]):
        return "不完整"
    if len(tested) == 1:
        return "仅1种子"
    key = "p_adj" if adjusted else "p"
    if not all(run[key] < 0.05 for run in tested):
        return "否"
    signs = {run["acc_a"] > run["acc_b"] for run in tested}
    return "是" if len(signs) == 1 else "是，但方向不一致"


def _p_cell(run: dict | None, adjusted: bool) -> str:
    if run is None:
        return "—"
    if adjusted:
        return f"{run['p']:.4g} / {run['p_adj']:.4g}"
    return f"{run['p']:.4g}"


def _render_rows(records: list[dict], adjusted: bool) -> list[str]:
    rows = []
    for rec in records:
        spec = rec["spec"]
        runs = [run for run in rec["runs"] if "p" in run]
        if not runs:
            rows.append(
                f"| {spec['group']} | {spec['name']} | 缺文件 | 缺文件 | 缺文件 | "
                f"缺文件 | 缺文件 | 缺文件 | 缺文件 | 缺文件 |"
            )
            continue
        acc_a = float(np.mean([run["acc_a"] for run in runs]))
        acc_b = float(np.mean([run["acc_b"] for run in runs]))
        delta = float(np.mean([run["acc_a"] - run["acc_b"] for run in runs]))
        first = runs[0]
        lo, hi = bootstrap_ci(first["diff"])
        p_cells = [_p_cell(_run_by_seed(rec, seed), adjusted) for seed in (0, 1, 2)]
        rows.append(
            f"| {spec['group']} | {spec['name']} | {_fmt_pct(acc_a)} | {_fmt_pct(acc_b)} | "
            f"{_fmt_pp(delta)} | [{lo * 100:+.2f}, {hi * 100:+.2f}] | "
            + " | ".join(p_cells) + f" | {_seed_sig_text(rec, adjusted)} |"
        )
    return rows


def render_md(records: list[dict], n_tests: int, preds_dir: Path) -> str:
    main_records = [rec for rec in records if rec["spec"].get("holm", True)]
    supp_records = [rec for rec in records if not rec["spec"].get("holm", True)]
    lines = [
        "# 显著性检验",
        "",
        f"生成时间：{now_iso()}",
        "",
        "精确 McNemar 用双侧二项检验（p=0.5），只用两边判断不同的样本；"
        "配对自助法 10000 次、种子 0，给出准确率之差的 95% 置信区间。",
        f"第 1～3 组与第 4 组主口径共 {n_tests} 个可用 p 值一起做 Holm 校正，"
        "校正后 p < 0.05 才算显著。第 4 组补充口径单列，不参与校正。",
        "主表的准确率、差值是各可用种子的均值；置信区间只取第一个可用种子（通常是 s0），"
        "与均值差值不对应。逐种子的准确率与区间见文末明细。"
        "“是，但方向不一致”表示三个种子都显著、但 A、B 谁高在种子间不一致，不能说哪个方法更好。",
        "",
        "## 主口径与第 1～3 组（参与 Holm 校正）",
        "",
        "| 组 | 比较 | A 准确率(%) | B 准确率(%) | 差值(pp) | s0 的 95% CI(pp) | "
        "s0 原始/校正 p | s1 原始/校正 p | s2 原始/校正 p | 三个种子都显著 |",
        "|---|---|---:|---:|---:|---|---|---|---|---|",
    ]
    lines += _render_rows(main_records, adjusted=True)
    lines += [
        "",
        "## 补充口径（不参与 Holm 校正）",
        "",
        "补充口径，不参与校正；下表 p 值为未校正 p 值。",
        "",
        "| 组 | 比较 | A 准确率(%) | B 准确率(%) | 差值(pp) | s0 的 95% CI(pp) | "
        "s0 未校正 p | s1 未校正 p | s2 未校正 p | 三个种子原 p<0.05 |",
        "|---|---|---:|---:|---:|---|---|---|---|---|",
    ]
    lines += _render_rows(supp_records, adjusted=False)

    missing_all = sorted({
        name for rec in records for run in rec["runs"] for name in run.get("missing", [])
    })
    lines += ["", "## 缺文件", ""]
    lines += [f"- `{name}`" for name in missing_all] if missing_all else ["无。"]
    lines += ["", "## 逐种子明细（参与校正）", "",
              "| 比较 | 种子 | A 准确率(%) | B 准确率(%) | 差值(pp) | 95% CI(pp) | 原始 p | 校正 p |",
              "|---|---:|---:|---:|---:|---|---:|---:|"]
    for rec in main_records:
        for run in rec["runs"]:
            if "p" not in run:
                continue
            lo, hi = bootstrap_ci(run["diff"])
            lines.append(
                f"| {rec['spec']['name']} | {run['seed']} | {_fmt_pct(run['acc_a'])} | "
                f"{_fmt_pct(run['acc_b'])} | {_fmt_pp(run['acc_a'] - run['acc_b'])} | "
                f"[{lo * 100:+.2f}, {hi * 100:+.2f}] | {run['p']:.4g} | {run['p_adj']:.4g} |"
            )
    lines += ["", "## 逐种子明细（补充口径，未校正）", "",
              "| 比较 | 种子 | A 准确率(%) | B 准确率(%) | 差值(pp) | 95% CI(pp) | 未校正 p |",
              "|---|---:|---:|---:|---:|---|---:|"]
    for rec in supp_records:
        for run in rec["runs"]:
            if "p" not in run:
                continue
            lo, hi = bootstrap_ci(run["diff"])
            lines.append(
                f"| {rec['spec']['name']} | {run['seed']} | {_fmt_pct(run['acc_a'])} | "
                f"{_fmt_pct(run['acc_b'])} | {_fmt_pp(run['acc_a'] - run['acc_b'])} | "
                f"[{lo * 100:+.2f}, {hi * 100:+.2f}] | {run['p']:.4g} |"
            )
    lines += ["", f"预测目录：`{preds_dir}`", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="实验 K 配对显著性检验")
    parser.add_argument("--out-dir", default="results",
                        help="结果目录，默认 results；冒烟用 results/smoke3")
    parser.add_argument("--preds-dir", default=None,
                        help="预测目录，默认 <out-dir>/preds")
    parser.add_argument("--out-md", default=None,
                        help="Markdown 输出，默认 <out-dir>/显著性.md")
    parser.add_argument("--tag", default="robust",
                        help="实验 I 的 tag，默认 robust；冒烟用 smoke3")
    parser.add_argument("--visible-pairs", default="results/robust_visible_pairs.json",
                        help="可见配对清单，默认 results/robust_visible_pairs.json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    preds_dir = Path(args.preds_dir) if args.preds_dir else out_dir / "preds"
    if not preds_dir.is_absolute():
        preds_dir = ROOT / preds_dir
    out_md = Path(args.out_md) if args.out_md else out_dir / "显著性.md"
    if not out_md.is_absolute():
        out_md = ROOT / out_md
    pairs_path = Path(args.visible_pairs)
    if not pairs_path.is_absolute():
        pairs_path = ROOT / pairs_path
    pairs = load_visible_pairs(pairs_path)

    records = [analyze_comparison(spec, preds_dir, pairs)
               for spec in build_comparisons(args.tag)]
    n_tests = apply_holm(records)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(records, n_tests, preds_dir), encoding="utf-8")

    n_missing = sum(1 for rec in records if not any("p" in run for run in rec["runs"]))
    print(f"写入 {out_md}")
    print(f"比较 {len(records)} 个，其中 {n_missing} 个缺文件；参与 Holm 校正的检验 {n_tests} 个")


if __name__ == "__main__":
    main()
