"""报告 2 专用图表。数字全部从结果 JSON 与逐条预测重算。

输出：
  figures/R2_规避主口径.png   非仅恶心子集的召回下降热图（主口径）
  figures/R2_级联取舍.png     准确率对每千条耗时（对数横轴）
  figures/R2_学习曲线.png     宏 F1 随训练量变化，含 rbt3 训练 10 轮的补充点
  figures/R2_演示截图.png     demo/示例输出.txt 渲染成终端风格图片
  results/R2_主表.tex         主对比表（LaTeX 片段）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from plot_style import CMAP_SEQ, PALETTE, save_figure, setup_chinese_font  # noqa: E402
from cascade import cascade_predict, confidence_from_prob_pos, tau_for_fraction  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RES, PRED, FIG = ROOT / "results", ROOT / "results/preds", ROOT / "figures"
M = [("TF-IDF（字）", "cold_tfidf_main_char", (0,)), ("TF-IDF（词）", "cold_tfidf_main_word", (0,)),
     ("TextCNN（字）", "cold_textcnn_main_char", (0, 1, 2)), ("TextCNN（词）", "cold_textcnn_main_word", (0, 1, 2)),
     ("BiLSTM（字）", "cold_bilstm_main_char", (0, 1, 2)), ("BiLSTM（词）", "cold_bilstm_main_word", (0, 1, 2)),
     ("Transformer（字）", "cold_transformer_main_char", (0, 1, 2)),
     ("rbt3", "cold_bert_rbt3", (0, 1, 2)), ("RoBERTa-wwm-ext", "cold_bert_roberta", (0, 1, 2))]
KINDS = [("pinyin", "拼音"), ("homophone", "同音字"), ("chaizi", "拆字"), ("symbol", "插符号"), ("delete", "删词（对照）")]


def js(n):
    return json.load(open(RES / f"{n}.json", encoding="utf-8"))


def pr(n):
    return pd.read_parquet(PRED / f"{n}.parquet")


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def fig_cloak():
    mat = np.zeros((len(M), len(KINDS)))
    for j, (kd, _) in enumerate(KINDS):
        meta = pd.read_csv(ROOT / f"data_selfmade/cloak_{kd}.csv")
        only = meta.matched_words.astype(str).str.split("|").apply(lambda x: set(x) == {"恶心"})
        keep = set(meta.pair_id[~only])
        for i, (_, p, seeds) in enumerate(M):
            d = []
            for s in seeds:
                o, e = pr(f"{p}_s{s}__cloak_{kd}_orig"), pr(f"{p}_s{s}__cloak_{kd}_edit")
                o, e = o[o.pair_id.isin(keep)], e[e.pair_id.isin(keep)]
                d.append(((o.pred == 1).mean() - (e.pred == 1).mean()) * 100)
            mat[i, j] = np.mean(d)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(mat, cmap=CMAP_SEQ, aspect="auto", vmin=0)
    ax.set_xticks(range(len(KINDS)), [k[1] for k in KINDS])
    ax.set_yticks(range(len(M)), [m[0] for m in M])
    for i in range(len(M)):
        for j in range(len(KINDS)):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] > 20 else "black")
    fig.colorbar(im, ax=ax, label="召回下降（百分点）")
    save_figure(fig, FIG / "R2_规避主口径.png")
    plt.close(fig)


def fig_cascade():
    bench = {"rbt3": js("bench_cascade_cold_rbt3")["ms_per_1k"], "roberta": js("bench_cascade_cold_roberta")["ms_per_1k"]}
    t = pr("cold_tfidf_main_char_s0")
    c = confidence_from_prob_pos(t.prob_pos.values)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, name, color in (("rbt3", "TF-IDF → rbt3", PALETTE["teal"]), ("roberta", "TF-IDF → RoBERTa", PALETTE["navy"])):
        tf_ms, b_ms = bench[key]["tfidf"], bench[key]["bert_192"]
        xs, ys = [], []
        for f in np.linspace(0, 1, 21):
            acc = [(cascade_predict(t.pred.values, c, pr(f"cold_bert_{key}_s{s}").pred.values,
                                    tau_for_fraction(c, f)) == t.label.values).mean() * 100 for s in (0, 1, 2)]
            xs.append(tf_ms + f * b_ms); ys.append(np.mean(acc))
        ax.plot(xs, ys, "-o", ms=3, color=color, label=name)
        for f in (0.2, 0.4):
            k = int(round(f * 20))
            ax.annotate(f"交出 {int(f*100)}%", (xs[k], ys[k]), textcoords="offset points", xytext=(4, -12), fontsize=8, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("每千条评论推理耗时（ms，对数刻度）")
    ax.set_ylabel("测试准确率（%）")
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, FIG / "R2_级联取舍.png")
    plt.close(fig)


def fig_lc():
    ns = [1000, 3000, 10000, 25653]
    series = [("TF-IDF（字）", lambda n, s: f"cold_tfidf_main_n{n}_char_s{s}" if n < 25653 else "cold_tfidf_main_char_s0"),
              ("TextCNN（字）", lambda n, s: f"cold_textcnn_main_n{n}_char_s{s}" if n < 25653 else f"cold_textcnn_main_char_s{s}"),
              ("Transformer（字）", lambda n, s: f"cold_transformer_main_n{n}_char_s{s}" if n < 25653 else f"cold_transformer_main_char_s{s}"),
              ("rbt3（训练 3 轮）", lambda n, s: f"cold_bert_n{n}_rbt3_s{s}" if n < 25653 else f"cold_bert_rbt3_s{s}")]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, fn in series:
        mu, sd = zip(*[ms([js(fn(n, s))["test_macro_f1"] * 100 for s in ((0,) if (name.startswith("TF") and n == 25653) else (0, 1, 2))]) for n in ns])
        ax.errorbar(ns, mu, yerr=sd, marker="o", ms=4, capsize=3, label=name)
    m10, s10 = ms([js(f"cold_bert_n1000ep10_rbt3_s{s}")["test_macro_f1"] * 100 for s in (0, 1, 2)])
    ax.errorbar([1000], [m10], yerr=[s10], marker="*", ms=11, color=PALETTE["navy"], capsize=3, linestyle="none", label="rbt3（训练 10 轮）")
    ax.set_xscale("log")
    ax.set_xlabel("训练样本数（对数刻度）")
    ax.set_ylabel("测试宏 F1（%）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save_figure(fig, FIG / "R2_学习曲线.png")
    plt.close(fig)


def fig_demo():
    from PIL import Image, ImageDraw, ImageFont
    lines = (ROOT / "demo/示例输出.txt").read_text(encoding="utf-8").rstrip().splitlines()
    from rich.cells import cell_len, get_character_cell_size
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20, index=7)  # Mono CJK SC
    # 按终端的格子逐字摆放：rich 排表时汉字占 2 格、其余多占 1 格，
    # 整行一次画出时 ⊙、… 等字形比格子宽，会把后面的列往右推，所以逐字定位
    cw = font.getlength("a")
    narrow = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 20)
    narrow = ImageFont.truetype(narrow.path, int(20 * cw / narrow.getlength("a")))  # 字宽对齐到一格
    lines = [l.rstrip() for l in lines]
    w = int(max(cell_len(l) for l in lines) * cw) + 40
    img = Image.new("RGB", (w, 30 * len(lines) + 40), (30, 30, 30))
    d = ImageDraw.Draw(img)
    for i, l in enumerate(lines):
        col = 0
        for ch in l:
            n = get_character_cell_size(ch)
            base = 20 + 30 * i + 21  # 各字共用基线
            if ch == "━":  # 表头下的粗横线画成连续的线
                d.rectangle([20 + col * cw, base - 8, 20 + (col + n) * cw, base - 6], fill=(220, 220, 220))
            elif n and not ch.isspace():
                f, gw = font, font.getlength(ch)
                if gw > n * cw + 1:  # 比格子宽的字形（⊙、→、… 等）改用等宽西文字体，正好占一格，不盖住邻字
                    f = narrow
                    gw = f.getlength(ch)
                gx = 20 + col * cw + max(0.0, (n * cw - gw) / 2)  # 窄于格子的字形居中
                d.text((gx, base), ch, font=f, fill=(220, 220, 220), anchor="ls")
            col += n
    img.save(FIG / "R2_演示截图.png")


def table_main():
    rows = []
    for name, p, seeds in M:
        g = lambda k: ms([js(f"{p}_s{s}")[k] * 100 for s in seeds])
        f1, acc, rec, pre = g("test_macro_f1"), g("test_accuracy"), g("test_pos_recall"), g("test_pos_precision")
        cell = lambda x: f"{x[0]:.2f}" + (f" $\\pm$ {x[1]:.2f}" if len(seeds) > 1 else "")
        rows.append(f"{name} & {cell(acc)} & {cell(f1)} & {cell(pre)} & {cell(rec)} \\\\")
    (RES / "R2_主表.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    setup_chinese_font()
    fig_cloak(); fig_cascade(); fig_lc(); fig_demo(); table_main()
    print("完成")
