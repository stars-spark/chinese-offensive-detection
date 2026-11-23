"""报告 2 的 8 张数据插图，统一样式。

每张图按进版宽度作图（与 main.tex 里 \\includegraphics 的 width 一致），进版不缩放，
图内刻度 8 磅、轴标签 9 磅；保存前查文字与线条、柱子、图例的重叠，有重叠就不写文件。
数字全部从结果 JSON 与逐条预测重算，取数函数沿用 figs_report2、figs_report2_d、
analyze_cold 与 fig_d2。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from plot_style import CMAP_ATTR, CMAP_SEQ, PALETTE, REF_COLOR, paper_style, save_checked  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import analyze_cold as AC  # noqa: E402
import figs_report2 as F2  # noqa: E402
import figs_report2_d as FD  # noqa: E402
from cascade import cascade_predict, confidence_from_prob_pos, tau_for_fraction  # noqa: E402

FIG = ROOT / "figures"
NAVY, TEAL, BLUE, GREEN, CYAN = (PALETTE[k] for k in ("navy", "teal", "blue", "green", "cyan"))


def two_line(name: str) -> str:
    """“TextCNN（字）”拆成两行，横轴标签不用斜排。"""
    return name.replace("（", "\n（").replace("RoBERTa-wwm-ext", "RoBERTa\n-wwm-ext")


def luminance(rgba) -> float:
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3]]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def text_color(rgba) -> str:
    """按 WCAG 对比度在黑字、白字里取对比度高的那个。"""
    y = luminance(rgba)
    return "white" if 1.05 / (y + 0.05) > (y + 0.05) / 0.05 else "black"


def lightness(rgba) -> float:
    """CIE L*，0 为黑、100 为白。"""
    y = luminance(rgba)
    return 116 * y ** (1 / 3) - 16 if y > 0.008856 else 903.3 * y


def even_cmap(name: str = "GnBu", l_hi: float = 93.0):
    """把色带按明度重新采样成等步长：数值每差一点，深浅变化看起来一样；最浅端截到 L*=l_hi。"""
    from matplotlib.colors import LinearSegmentedColormap
    base = plt.get_cmap(name)
    t = np.linspace(0, 1, 1001)
    L = np.array([lightness(base(x)) for x in t])          # 单调递减
    targets = np.linspace(min(l_hi, L[0]), L[-1], 256)
    ts = np.interp(-targets, -L, t)
    return LinearSegmentedColormap.from_list(f"{name}_even", [base(x) for x in ts])


def ylab(ax, text: str):
    """纵轴标签竖排时整串逆时针转 90°，所以“越高越好”写成 →、“越低越好”写成 ←，转过来正好朝上、朝下。"""
    ax.set_ylabel(text.replace("↑", "→").replace("↓", "←"))


def plain_log_ticks(ax, ticks):
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()


def top_legend(fig, ax, ncol: int, **kw):
    h, l = ax.get_legend_handles_labels()
    rows = -(-len(h) // ncol)        # 按行排：matplotlib 按列填，先把次序转置
    idx = [r + c * rows for r in range(rows) for c in range(ncol) if r + c * rows < len(h)]
    idx = [i for c in range(ncol) for i in idx[c::ncol]]
    h, l = [h[i] for i in idx], [l[i] for i in idx]
    fig.legend(h, l, loc="outside upper center", ncol=ncol, frameon=False,
               columnspacing=1.4, handletextpad=0.5, **kw)


# ---------------- 图 2 学习曲线 ----------------
def fig_lc():
    ns = [1000, 3000, 10000, 25653]
    series = [("TF-IDF（字）", "tfidf", NAVY, "o"), ("TextCNN（字）", "textcnn", TEAL, "s"),
              ("Transformer（字）", "transformer", BLUE, "^"), ("rbt3，训练 3 轮", "rbt3", GREEN, "D")]

    def name(kind, n, s):
        if kind == "rbt3":
            return f"cold_bert_n{n}_rbt3_s{s}" if n < 25653 else f"cold_bert_rbt3_s{s}"
        if kind == "tfidf":
            return f"cold_tfidf_main_n{n}_char_s{s}" if n < 25653 else "cold_tfidf_main_char_s0"
        return f"cold_{kind}_main_n{n}_char_s{s}" if n < 25653 else f"cold_{kind}_main_char_s{s}"

    fig, ax = plt.subplots(figsize=(0.75 * 6.30, 3.0))
    for label, kind, color, marker in series:
        mu, sd = [], []
        for n in ns:
            seeds = (0,) if kind == "tfidf" and n == 25653 else (0, 1, 2)
            m, s = F2.ms([F2.js(name(kind, n, s))["test_macro_f1"] * 100 for s in seeds])
            mu.append(m); sd.append(s)
        ax.errorbar(ns, mu, yerr=sd, marker=marker, color=color, label=label, ms=4.5, lw=1.4)
    m10, s10 = F2.ms([F2.js(f"cold_bert_n1000ep10_rbt3_s{s}")["test_macro_f1"] * 100 for s in (0, 1, 2)])
    ax.errorbar([1000], [m10], yerr=[s10], marker="*", ms=9, color=GREEN, mfc="white", mew=1.2,
                linestyle="none", label="rbt3，训练 10 轮")
    ax.set_xscale("log")
    ax.set_xticks(ns, ["1000", "3000", "10000", "25653"])
    ax.minorticks_off()
    ax.set_xlim(800, 32000)
    ax.set_xlabel("训练条数（对数刻度）")
    ylab(ax, "测试宏 F1（%）↑")
    ax.grid(axis="y")
    top_legend(fig, ax, ncol=3)
    save_checked(fig, FIG / "R2_学习曲线.png", 0.75)


# ---------------- 图 3 两类安全言论的误判率 ----------------
def fig_fine():
    _, _, rows = AC.main_section(ROOT / "results")
    rows = [r for r in rows if r["anti"][0] is not None]
    x = np.arange(len(rows))
    w = 0.38
    anti = [r["anti"][0] * 100 for r in rows]
    other = [0 if r["other"][0] is None else r["other"][0] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(6.30, 2.7))
    ax.bar(x - w / 2, anti, w, color=NAVY, label="反歧视言论")
    ax.bar(x + w / 2, other, w, color=CYAN, label="其他安全言论")
    for xs, vals in ((x - w / 2, anti), (x + w / 2, other)):
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [two_line(r["label"]) for r in rows])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 80)
    ylab(ax, "判为攻击性的比例（%）↓")
    ax.grid(axis="y")
    top_legend(fig, ax, ncol=2)
    save_checked(fig, FIG / "C_细粒度误判率.png", 1.0)


# ---------------- 图 4 变体词召回下降热图 ----------------
def fig_cloak():
    mat = np.zeros((len(F2.M), len(F2.KINDS)))
    for j, (kd, _) in enumerate(F2.KINDS):
        meta = pd.read_csv(ROOT / f"data_selfmade/cloak_{kd}.csv")
        only = meta.matched_words.astype(str).str.split("|").apply(lambda x: set(x) == {"恶心"})
        keep = set(meta.pair_id[~only])
        for i, (_, p, seeds) in enumerate(F2.M):
            d = []
            for s in seeds:
                o, e = F2.pr(f"{p}_s{s}__cloak_{kd}_orig"), F2.pr(f"{p}_s{s}__cloak_{kd}_edit")
                o, e = o[o.pair_id.isin(keep)], e[e.pair_id.isin(keep)]
                d.append(((o.pred == 1).mean() - (e.pred == 1).mean()) * 100)
            mat[i, j] = np.mean(d)
    cmap = even_cmap("GnBu")
    vmax = 30.0
    fig, ax = plt.subplots(figsize=(0.78 * 6.30, 3.4))
    mesh = ax.pcolormesh(mat, cmap=cmap, vmin=0, vmax=vmax, edgecolors="white", linewidth=1.2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            rgba = cmap(min(max(mat[i, j], 0), vmax) / vmax)
            t = ax.text(j + 0.5, i + 0.5, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
                        color=text_color(rgba))
            t._in_bar = True
    ax.set_xticks(np.arange(len(F2.KINDS)) + 0.5, [k[1] for k in F2.KINDS])
    ax.set_yticks(np.arange(len(F2.M)) + 0.5, [m[0] for m in F2.M])
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.02, aspect=25)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2)
    cb.set_label("召回下降（百分点）")
    save_checked(fig, FIG / "R2_规避主口径.png", 0.78)


# ---------------- 图 5 级联的准确率与耗时 ----------------
def fig_cascade():
    bench = {k: F2.js(f"bench_cascade_cold_{k}")["ms_per_1k"] for k in ("rbt3", "roberta")}
    t = F2.pr("cold_tfidf_main_char_s0")
    c = confidence_from_prob_pos(t.prob_pos.values)
    fig, ax = plt.subplots(figsize=(0.72 * 6.30, 2.9))
    marks = []
    for key, label, color, marker in (("rbt3", "TF-IDF → rbt3", TEAL, "s"), ("roberta", "TF-IDF → RoBERTa", NAVY, "o")):
        tf_ms, b_ms = bench[key]["tfidf"], bench[key]["bert_192"]
        bert = [F2.pr(f"cold_bert_{key}_s{s}").pred.values for s in (0, 1, 2)]
        fr = np.linspace(0, 1, 21)
        xs = tf_ms + fr * b_ms
        ys = [np.mean([(cascade_predict(t.pred.values, c, b, tau_for_fraction(c, f)) == t.label.values).mean() * 100
                       for b in bert]) for f in fr]
        ax.plot(xs, ys, "-", marker=marker, ms=3, color=color, label=label)
        for f in (0.2, 0.4):
            k = int(round(f * 20))
            ax.plot(xs[k], ys[k], marker=marker, ms=7, mfc="white", mec=color, mew=1.3, linestyle="none")
            marks.append((key, f, xs[k], ys[k], color))
    # 交出比例的标注放在曲线下方的空白处，用短引线指向空心点
    offs = {("roberta", 0.2): (14, -16), ("roberta", 0.4): (6, -20),
            ("rbt3", 0.2): (-8, -24), ("rbt3", 0.4): (8, -22)}
    for key, f, x, y, color in marks:
        ax.annotate(f"交出 {int(f * 100)}%", (x, y), xytext=offs[(key, f)], textcoords="offset points",
                    fontsize=7.5, color=color, ha="center", va="top",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.6, shrinkA=0, shrinkB=4))
    ax.set_xscale("log")
    plain_log_ticks(ax, [100, 300, 1000, 3000])
    ax.set_xlabel("每千条评论推理耗时（ms，对数刻度）↓")
    ylab(ax, "测试准确率（%）↑")
    ax.grid(axis="y")
    top_legend(fig, ax, ncol=2)
    save_checked(fig, FIG / "R2_级联取舍.png", 0.72)


# ---------------- 图 6 交给人工的比例 ----------------
def fig_human():
    _, curves = AC.cascade_section(ROOT / "results")
    fig, ax = plt.subplots(figsize=(0.66 * 6.30, 2.7))
    style = {"rbt3": (TEAL, "s"), "RoBERTa-wwm-ext": (NAVY, "o")}
    for label, (fracs, acc) in curves.items():
        color, marker = style.get(label, (BLUE, "^"))
        ax.plot(np.asarray(fracs) * 100, np.asarray(acc) * 100, marker=marker, ms=3, color=color, label=label)
    ax.set_xlabel("交给人工的比例（%）")
    ylab(ax, "其余评论的准确率（%）↑")
    ax.set_xlim(-2, 100)
    ax.grid(axis="y")
    top_legend(fig, ax, ncol=2)
    save_checked(fig, FIG / "C_级联.png", 0.66)


# ---------------- 图 7 变体词增强 ----------------
def fig_aug():
    A = FD.A
    conds = [("不增强", "cold_bert_rbt3", NAVY), ("四种都增强", "cold_bert_aug_all", TEAL),
             ("只增强拼音、插符号", "cold_bert_aug_held", CYAN)]
    x = np.arange(len(A.KINDS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(0.75 * 6.30, 2.8))
    for i, (label, prefix, color) in enumerate(conds):
        stats = [A.recall_drop(prefix, (0, 1, 2), k) for k in A.KINDS]
        ax.bar(x + (i - 1) * w, [s[0] for s in stats], w, yerr=[s[1] for s in stats], color=color, label=label,
               error_kw=dict(lw=0.8, capsize=2, capthick=0.8, ecolor="#333333"))
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, [FD.KIND_NAMES[k] for k in A.KINDS])
    ax.tick_params(axis="x", length=0)
    ylab(ax, "攻击性召回下降（百分点）↓")
    ax.grid(axis="y")
    top_legend(fig, ax, ncol=3)
    save_checked(fig, FIG / "R2_D1_增强.png", 0.75)


# ---------------- 图 8 反歧视言论的字级归因 ----------------
def fig_attr():
    import fig_d2
    frame = pd.read_parquet(fig_d2.PRED_PATH)
    picked, _, _ = fig_d2.pick_rows(frame, fig_d2.load_veto())   # 与原图同一选择规则，选出同样 6 条
    rows = list(picked.iterrows())
    max_len = max(len(str(r["chars"])) for _, r in rows)
    width, row_h, right, x0 = 6.30, 0.30, 0.55, 0.05     # 单位英寸；右侧一栏放富集比
    cell = min(0.135, (width - right - x0) / max_len)     # 按最长一条定每字宽
    fsize = cell * 72 * 0.88
    height = row_h * len(rows) + 0.32
    fig = plt.figure(figsize=(width, height), layout="none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    cmap = plt.get_cmap(CMAP_ATTR)
    assert x0 + max_len * cell <= width - right, "评论太长，放不下"
    hx = width - right / 2
    ax.text(hx, 0.16, "富集比", ha="center", va="center", fontsize=8)
    for k, (_, r) in enumerate(rows):
        text, mask = str(r["chars"]), str(r["group_mask"])
        attrs = np.maximum(np.asarray(json.loads(r["char_attrs"]), float), 0.0)
        peak = attrs.max() if len(attrs) and attrs.max() > 0 else 1.0
        y = 0.32 + k * row_h + row_h / 2
        for i, ch in enumerate(text):
            frac = attrs[i] / peak
            ax.add_patch(Rectangle((x0 + i * cell, y - 0.11), cell, 0.22, facecolor=cmap(0.08 + 0.92 * frac),
                                   edgecolor="none"))
            t = ax.text(x0 + (i + 0.5) * cell, y, ch, ha="center", va="center", fontsize=fsize,
                        color=text_color(cmap(0.08 + 0.92 * frac)))
            t._in_bar = True
        # 群体词：连续的字合成一个黑框
        i = 0
        while i < len(mask):
            if mask[i] == "1":
                j = i
                while j < len(mask) and mask[j] == "1":
                    j += 1
                ax.add_patch(Rectangle((x0 + i * cell, y - 0.11), (j - i) * cell, 0.22, fill=False,
                                       edgecolor="black", lw=0.9))
                i = j
            else:
                i += 1
        ax.text(hx, y, f"{r['enrichment']:.2f}" if pd.notna(r["enrichment"]) else "—",
                ha="center", va="center", fontsize=8)
    save_checked(fig, FIG / "R2_D2_归因.png", 1.0)


# ---------------- 图 9 Qwen2.5 各规模 ----------------
def fig_llm():
    P = FD.point
    sizes = {"qwen15b": 1.5, "qwen3b": 3, "qwen7b": 7, "qwen14b": 14}
    groups = [
        ("零样本", [(sizes[k], P([f"cold_llm_{k}"])) for k in sizes], NAVY, dict(marker="o", ls="-")),
        ("LoRA 微调", [(1.5, P([f"cold_llmft_qwen15b_s{s}" for s in (0, 1, 2)])),
                     (7, P([f"cold_llmft_qwen7b_s{s}" for s in (0, 1, 2)]))], TEAL, dict(marker="s", ls="--")),
        ("带提示词 LoRA", [(7, P([f"cold_llmftp_qwen7b_s{s}" for s in (0, 1, 2)]))], GREEN, dict(marker="D", ls="none")),
        ("少样本（8 例）", [(7, P([f"cold_llm_qwen7b_fs{s}" for s in (0, 1, 2)]))], BLUE, dict(marker="^", ls="none")),
    ]
    refs = [("rbt3", P([f"cold_bert_rbt3_s{s}" for s in (0, 1, 2)]), ":"),
            ("RoBERTa", P([f"cold_bert_roberta_s{s}" for s in (0, 1, 2)]), "-.")]
    fig, axes = plt.subplots(1, 3, figsize=(6.30, 2.45))
    ylabels = ["COLD 测试宏 F1（%）↑", "误判率之比 ↓", "每千条耗时（ms）↓"]
    for idx, (ax, yl) in enumerate(zip(axes, ylabels)):
        for name, p, ls in refs:
            if p[idx] is not None:
                ax.axhline(p[idx], color=REF_COLOR, ls=ls, lw=1.0, label=f"{name}（参照）")
        for label, pts, color, st in groups:
            pts = [(s, p[idx]) for s, p in pts if p[idx] is not None]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=color, ms=5, label=label, **st)
        ax.set_xscale("log")
        ax.set_xticks([1.5, 3, 7, 14], ["1.5B", "3B", "7B", "14B"])
        ax.minorticks_off()
        ax.set_xlim(1.2, 18)
        ax.set_xlabel("Qwen2.5 参数量")
        ylab(ax, yl)
        ax.grid(axis="y")
        if idx == 2:
            ax.set_yscale("log")
            ax.set_ylim(400, 4e5)
    h, l = axes[0].get_legend_handles_labels()
    order = [2, 3, 4, 5, 0, 1]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="outside upper center", ncol=6,
               frameon=False, columnspacing=1.0, handletextpad=0.4, handlelength=1.8)
    save_checked(fig, FIG / "R2_D_大模型.png", 1.0)


ALL = {"lc": fig_lc, "fine": fig_fine, "cloak": fig_cloak, "cascade": fig_cascade, "human": fig_human,
       "aug": fig_aug, "attr": fig_attr, "llm": fig_llm}

if __name__ == "__main__":
    paper_style()
    FD.A.R = FD.A.Results(ROOT / "results")
    names = sys.argv[1:] or list(ALL)
    failed = []
    for n in names:
        try:
            ALL[n]()
            print("完成", n)
        except AssertionError as exc:
            failed.append(n)
            print("未通过", n, exc)
    sys.exit(1 if failed else 0)
