"""D1 变体词训练增强：每条含词表词的训练评论生成恰好一个变体。

匹配规则完全沿用 ``make_cloak.py``：先用 jieba 词边界与最长的词表词定位，
再用它的 ``apply_transform`` 做四种变换之一。与 ``make_cloak`` 的区别只有
原料和抽样：这里读 COLD **训练集**，先按 C 组口径跨划分去重，再逐条抽一个
变换类型；安全评论也增强，标签保持原评论的标签。

条件 ``all`` 允许 pinyin / homophone / chaizi / symbol，条件 ``held`` 只允许
pinyin / symbol。抽到的类型若对该评论做不了（例如拆字表里没有），按随机
顺序改试其余允许类型；全都不行才跳过并计数。
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from data import deduplicate_cold_splits, load_splits
from make_cloak import (
    KINDS,
    apply_transform,
    build_homophone_map,
    build_word_index,
    find_matches,
    load_chaizi,
    load_lexicon,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data_selfmade"
# 条件 -> 允许的变换类型；held 留出 homophone 与 chaizi 两种训练时没见过的写法
CONDITIONS = {"all": tuple(KINDS), "held": ("pinyin", "symbol")}
COLUMNS = ["text", "label", "topic", "kind", "matched_words", "src_text"]


def allowed_kinds(condition: str) -> tuple[str, ...]:
    """返回条件允许的变换类型，未知条件直接报错。"""
    if condition not in CONDITIONS:
        raise ValueError(f"未知条件 {condition!r}，可选 {sorted(CONDITIONS)}")
    return CONDITIONS[condition]


def load_resources() -> dict:
    """一次性加载词表、同音字表与拆字表，六个文件共用。"""
    words = load_lexicon()
    by_len, lengths = build_word_index(words)
    return {
        "by_len": by_len,
        "lengths": lengths,
        "homophones": build_homophone_map(),
        "chaizi": load_chaizi(),
    }


def make_variant(text: str, matches, condition: str, rng: random.Random,
                 resources: dict):
    """给一条评论抽一个能成功的变换。

    先按均匀分布抽一个允许类型，做不了就把其余类型随机排序后依次试。
    返回 ``(变体, 实际类型)``；全部失败返回 ``None``。
    """
    kinds = list(allowed_kinds(condition))
    first = rng.choice(kinds)
    rest = [k for k in kinds if k != first]
    rng.shuffle(rest)
    for kind in [first] + rest:
        edit, changed, _ = apply_transform(
            text, matches, kind,
            homophones=resources["homophones"],
            chaizi=resources["chaizi"],
            rng=rng,
        )
        if changed:
            return edit, kind
    return None


def build_augment(splits: dict[str, pd.DataFrame], condition: str, seed: int,
                  resources: dict | None = None):
    """去重后增强训练集，返回 ``(行列表, 统计)``。

    去重调用 ``deduplicate_cold_splits``，和 C 组训练时完全同一口径；
    验证集和测试集不参与增强。每条命中的源评论最多生成一行。
    去重只剔除跨划分重复，划分内部的重复行保留，因此这些行各生成一个变体。
    """
    allowed_kinds(condition)
    if resources is None:
        resources = load_resources()
    dedup, removed = deduplicate_cold_splits(splits)
    train = dedup["train"]
    rng = random.Random(seed)
    rows: list[dict] = []
    skipped = 0
    n_matched = 0
    by_kind = Counter()
    by_label = Counter()
    for _, row in train.iterrows():
        text = str(row["text"])
        matches = find_matches(text, resources["by_len"], resources["lengths"])
        if not matches:
            continue
        n_matched += 1
        picked = make_variant(text, matches, condition, rng, resources)
        if picked is None:
            skipped += 1
            continue
        edit, kind = picked
        label = int(row["label"])
        rows.append({
            "text": edit,
            "label": label,
            "topic": str(row["topic"]) if "topic" in row else "",
            "kind": kind,
            "matched_words": "|".join(word for _, _, word in matches),
            "src_text": text,
        })
        by_kind[kind] += 1
        by_label[label] += 1
    stats = {
        "condition": condition,
        "seed": seed,
        "n_matched": n_matched,
        "n_generated": len(rows),
        "n_skipped": skipped,
        "by_kind": dict(sorted(by_kind.items())),
        "by_label": {str(k): int(by_label.get(k, 0)) for k in (0, 1)},
        "dedup_removed": removed,
    }
    return rows, stats


def write_outputs(conditions, seeds, out_dir: Path = OUT_DIR) -> list[dict]:
    """生成所有条件/种子文件，返回每个文件的统计。"""
    splits = load_splits("cold")
    resources = load_resources()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for condition in conditions:
        for seed in seeds:
            rows, stats = build_augment(splits, condition, seed, resources)
            path = out_dir / f"aug_{condition}_s{seed}.csv"
            pd.DataFrame(rows, columns=COLUMNS).to_csv(
                path, index=False, encoding="utf-8")
            stats["file"] = path.name
            all_stats.append(stats)
            print(f"{path.name}: {stats['n_generated']} 条，"
                  f"跳过 {stats['n_skipped']} 条")
    return all_stats


def render_stats(all_stats: list[dict]) -> str:
    """把统计写成 markdown；数字全部来自 build_augment 的返回值。"""
    lines = ["# D1 增强数据统计（make_aug.py 自动生成）", "",
             "先按 C 组口径去重，再对含词表词的训练评论各生成一个变体；"
             "标签保留原评论标签，安全评论也增强。", "",
             "| 文件 | 条件 | 种子 | 命中源评论 | 生成 | 跳过 | 攻击性 | 安全 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in all_stats:
        lines.append(
            f"| {s['file']} | {s['condition']} | {s['seed']} | "
            f"{s['n_matched']} | {s['n_generated']} | {s['n_skipped']} | "
            f"{s['by_label']['1']} | {s['by_label']['0']} |"
        )
    lines += ["", "## 按变换类型", "",
              "| 文件 | " + " | ".join(KINDS) + " |",
              "|---|" + "---:|" * len(KINDS)]
    for s in all_stats:
        cells = [str(s["by_kind"].get(k, 0)) for k in KINDS]
        lines.append(f"| {s['file']} | " + " | ".join(cells) + " |")
    if all_stats:
        removed = all_stats[0]["dedup_removed"]
        lines += ["", f"去重剔除：train={removed['train']}，"
                      f"val={removed['val']}，test={removed['test']}。"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="D1 变体词训练增强")
    parser.add_argument("--conditions", default="all,held",
                        help="逗号分隔，可选 all / held，默认两个都生成")
    parser.add_argument("--seeds", default="0,1,2",
                        help="逗号分隔的种子，默认 0,1,2")
    parser.add_argument("--out_dir", default=str(OUT_DIR),
                        help="输出目录，默认 data_selfmade/；冒烟可改")
    args = parser.parse_args()
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    all_stats = write_outputs(conditions, seeds, out_dir)
    stats_path = out_dir / "aug_统计.md"
    stats_path.write_text(render_stats(all_stats), encoding="utf-8")
    print(f"统计写入 {stats_path}")


if __name__ == "__main__":
    main()
