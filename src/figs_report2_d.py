"""报告 2 的 D 组图。数字全部经 analyze_D 的函数从结果 JSON 与逐条预测重算。

输出：
  figures/R2_D1_增强.png    三个训练条件下四种变体词的召回下降（主口径）
  figures/R2_D_大模型.png   零样本规模曲线、LoRA、带提示词 LoRA、少样本：宏 F1、立场区分比、耗时
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analyze_D as A  # noqa: E402
from plot_style import PALETTE, save_figure, setup_chinese_font  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

FIG = ROOT / "figures"
KIND_NAMES = {"pinyin": "拼音", "homophone": "同音字", "chaizi": "拆字", "symbol": "插符号"}
C_BASE, C_ALL, C_HELD = PALETTE["gray"], PALETTE["teal"], PALETTE["cyan"]


def fig_d1():
    conds = [("不增强", "cold_bert_rbt3", C_BASE), ("四种都增强", "cold_bert_aug_all", C_ALL),
             ("只增强拼音、插符号（同音字、拆字训练时未见）", "cold_bert_aug_held", C_HELD)]
    x = np.arange(len(A.KINDS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, (label, prefix, color) in enumerate(conds):
        stats = [A.recall_drop(prefix, (0, 1, 2), k) for k in A.KINDS]
        mu = [s[0] for s in stats]
        sd = [s[1] for s in stats]
        ax.bar(x + (i - 1) * w, mu, w, yerr=sd, capsize=3, color=color, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [KIND_NAMES[k] for k in A.KINDS])
    ax.set_ylabel("攻击性召回下降（百分点）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    save_figure(fig, FIG / "R2_D1_增强.png")
    plt.close(fig)


def fine_rate(names, cls: str):
    vals = []
    for n in names:
        js = A.R.js(n)
        item = ((js or {}).get("test_fine_grained") or {}).get(cls)
        if item is not None:
            vals.append(float(item["rate_pred_offensive"]) * 100)
    return A.ms(vals)[0]


def point(names):
    f1 = A.field_names(names, "test_macro_f1")[0]
    f1 = None if f1 is None else (f1 * 100 if f1 <= 1 else f1)
    anti, other = fine_rate(names, "3"), fine_rate(names, "0")
    ratio = anti / other if anti is not None and other else None
    cost = A.bench_ms_names(names)[0]
    return f1, ratio, cost


def fig_llm():
    sizes = {"qwen15b": 1.5, "qwen3b": 3, "qwen7b": 7, "qwen14b": 14}
    zs = {k: point([f"cold_llm_{k}"]) for k in sizes}
    groups = [
        ("零样本", [(sizes[k], zs[k]) for k in sizes], PALETTE["navy"], "o-"),
        ("LoRA 微调", [(1.5, point([f"cold_llmft_qwen15b_s{s}" for s in (0, 1, 2)])),
                     (7, point([f"cold_llmft_qwen7b_s{s}" for s in (0, 1, 2)]))], PALETTE["teal"], "s--"),
        ("带提示词 LoRA", [(7, point([f"cold_llmftp_qwen7b_s{s}" for s in (0, 1, 2)]))], PALETTE["green"], "D"),
        ("少样本（8 例）", [(7, point([f"cold_llm_qwen7b_fs{s}" for s in (0, 1, 2)]))], PALETTE["blue"], "^"),
    ]
    refs = [("rbt3", point([f"cold_bert_rbt3_s{s}" for s in (0, 1, 2)])),
            ("RoBERTa", point([f"cold_bert_roberta_s{s}" for s in (0, 1, 2)]))]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    panels = [(0, "COLD 测试宏 F1（%）", "越高越好"), (1, "反歧视误判率 ÷ 普通安全误判率", "越低越能分清立场"),
              (2, "每千条推理耗时（ms，对数刻度）", "越低越省")]
    for ax, (idx, ylabel, note) in zip(axes, panels):
        for label, pts, color, style in groups:
            pts = [(s, p[idx]) for s, p in pts if p[idx] is not None]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, style, color=color, ms=7, label=label)
        for name, p in refs:
            if p[idx] is not None:
                ax.axhline(p[idx], color="gray", ls=":", lw=1.2)
                ax.text(15.5, p[idx], name, fontsize=8, color="gray", va="center")
        ax.set_xscale("log")
        ax.set_xticks([1.5, 3, 7, 14], ["1.5B", "3B", "7B", "14B"])
        ax.minorticks_off()
        ax.set_xlim(1.2, 24)
        ax.set_xlabel("Qwen2.5 参数量")
        ax.set_ylabel(ylabel)
        ax.set_title(note, fontsize=9, color="dimgray")
        ax.grid(alpha=0.3)
        if idx == 2:
            ax.set_yscale("log")
            ax.set_ylim(400, 4e5)  # 把 rbt3、RoBERTa 的参照线也画进来
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save_figure(fig, FIG / "R2_D_大模型.png")
    plt.close(fig)


if __name__ == "__main__":
    A.R = A.Results(ROOT / "results")  # analyze_D 的函数读模块级的 R
    setup_chinese_font()
    fig_d1()
    fig_llm()
    print("完成")
