"""C 组预期 H1～H9 的判定用数字。

只算各预期需要的数字，并按差异判法给出"有差异 / 无法区分"，
最终"符合 / 不符合"另行判定。
差异判法：|均值差| > 2 × 两组标准差中较大者，且 > 0.5 个百分点。单种子的标准差记 0。
用法：python src/judge_C.py [--results results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEL = ROOT / "data_selfmade"
M = [  # 键, 名称, 结果名前缀（不含 _s{seed}）, 种子
    ("tfidf_char", "TF-IDF（字）", "cold_tfidf_main_char", (0,)),
    ("tfidf_word", "TF-IDF（词）", "cold_tfidf_main_word", (0,)),
    ("textcnn_char", "TextCNN（字）", "cold_textcnn_main_char", (0, 1, 2)),
    ("textcnn_word", "TextCNN（词）", "cold_textcnn_main_word", (0, 1, 2)),
    ("bilstm_char", "BiLSTM（字）", "cold_bilstm_main_char", (0, 1, 2)),
    ("bilstm_word", "BiLSTM（词）", "cold_bilstm_main_word", (0, 1, 2)),
    ("transformer_char", "Transformer（字）", "cold_transformer_main_char", (0, 1, 2)),
    ("rbt3", "rbt3", "cold_bert_rbt3", (0, 1, 2)),
    ("roberta", "RoBERTa-wwm-ext", "cold_bert_roberta", (0, 1, 2)),
]
NAME = {k: n for k, n, _, _ in M}
KINDS = ("pinyin", "homophone", "chaizi", "symbol", "delete")


def ms(vals):
    v = np.asarray([x for x in vals if x is not None], float) * 100
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)) if len(v) else (np.nan, 0.0, 0)


def diff(a, b):
    """a、b 为 (均值, 标准差, n)；返回 (差, 是否有差异)。"""
    d = a[0] - b[0]
    return d, bool(abs(d) > max(2 * max(a[1], b[1]), 0.5))


def fmt(x):
    return f"{x[0]:.2f} ± {x[1]:.2f}" if x[2] > 1 else f"{x[0]:.2f}"


class R:
    def __init__(self, d: Path):
        self.d = d

    def js(self, name):
        p = self.d / f"{name}.json"
        return json.load(open(p, encoding="utf-8")) if p.exists() else None

    def pr(self, name):
        p = self.d / "preds" / f"{name}.parquet"
        return pd.read_parquet(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args()
    r = R(Path(a.results))
    out = ["# C 组判定用数字（judge_C.py 自动生成）", ""]

    # H1 主测试宏 F1
    f1 = {k: ms([(r.js(f"{p}_s{s}") or {}).get("test_macro_f1") for s in seeds]) for k, _, p, seeds in M}
    out += ["## H1 主测试宏 F1（%）", "", "| 方法 | 宏 F1 | 准确率 | 攻击性召回 |", "|---|---:|---:|---:|"]
    for k, n, p, seeds in M:
        acc = ms([(r.js(f"{p}_s{s}") or {}).get("test_accuracy") for s in seeds])
        rec = ms([(r.js(f"{p}_s{s}") or {}).get("test_pos_recall") for s in seeds])
        out.append(f"| {n} | {fmt(f1[k])} | {fmt(acc)} | {fmt(rec)} |")
    d, s = diff(f1["roberta"], f1["rbt3"])
    out += ["", f"- RoBERTa − rbt3 = {d:+.2f}（{'有差异' if s else '无法区分'}）"]
    for k in [x for x in NAME if x not in ("rbt3", "roberta")]:
        for b in ("rbt3", "roberta"):
            d, s = diff(f1[b], f1[k])
            out.append(f"- {NAME[b]} − {NAME[k]} = {d:+.2f}（{'有差异' if s else '无法区分'}）")

    # H2 细粒度误判率
    out += ["", "## H2 被判为攻击性的比例（%）：反歧视类（3）对其他安全类（0）", "",
            "| 方法 | 反歧视 | 其他安全 | 差 | 判 |", "|---|---:|---:|---:|---|"]
    fg = {}
    for k, n, p, seeds in M:
        g = lambda c: ms([((r.js(f"{p}_s{s}") or {}).get("test_fine_grained") or {}).get(c, {}).get("rate_pred_offensive") for s in seeds])
        fg[k] = (g("3"), g("0"))
        d, s = diff(*fg[k])
        out.append(f"| {n} | {fmt(fg[k][0])} | {fmt(fg[k][1])} | {d:+.2f} | {'有差异' if s else '无法区分'} |")
    d, s = diff(fg["tfidf_char"][0], fg["roberta"][0])
    out.append(f"\n- 反歧视误判率 TF-IDF（字）− RoBERTa = {d:+.2f}（{'有差异' if s else '无法区分'}）")

    # H3～H5 规避：召回下降（原 − 改），按子集
    sets = {kd: pd.read_csv(SEL / f"cloak_{kd}.csv") for kd in KINDS}
    drops = {}
    out += ["", "## H3～H5 规避集召回下降（原评论召回 − 变体召回，百分点）", ""]
    for subset in ("非仅恶心", "仅恶心", "全部"):
        out += [f"### 子集：{subset}", "", "| 方法 | " + " | ".join(KINDS) + " |", "|---|" + "---:|" * len(KINDS)]
        for k, n, p, seeds in M:
            cells = []
            for kd in KINDS:
                meta = sets[kd]
                only = meta.matched_words.astype(str).str.split("|").apply(lambda x: set(x) == {"恶心"})
                keep = set(meta.pair_id[~only] if subset == "非仅恶心" else meta.pair_id[only] if subset == "仅恶心" else meta.pair_id)
                o_r, e_r = [], []
                for s in seeds:
                    po, pe = r.pr(f"{p}_s{s}__cloak_{kd}_orig"), r.pr(f"{p}_s{s}__cloak_{kd}_edit")
                    if po is None or pe is None:
                        continue
                    po, pe = po[po.pair_id.isin(keep)], pe[pe.pair_id.isin(keep)]
                    o_r.append((po.pred == 1).mean()); e_r.append((pe.pred == 1).mean())
                o, e = ms(o_r), ms(e_r)
                dd, sig = diff(o, e)
                drops[(subset, k, kd)] = (o, e, dd, sig)
                cells.append(f"{dd:+.2f}{'*' if sig else ''}" if o[2] else "缺")
            out.append(f"| {n} | " + " | ".join(cells) + " |")
        out += ["", "（* 表示原评论召回与变体召回有差异）", ""]
    for kd in ("pinyin",):
        rb, tf = drops[("非仅恶心", "roberta", kd)], drops[("非仅恶心", "tfidf_char", kd)]
        out.append(f"- H4 拼音：RoBERTa 下降 {rb[2]:+.2f}，TF-IDF（字）下降 {tf[2]:+.2f}")
    hc, hw = drops[("非仅恶心", "tfidf_char", "homophone")], drops[("非仅恶心", "tfidf_word", "homophone")]
    out.append(f"- H5 同音字：TF-IDF（字）下降 {hc[2]:+.2f}，TF-IDF（词）下降 {hw[2]:+.2f}，差 {hw[2] - hc[2]:+.2f}")

    # H6 互换一致率
    out += ["", "## H6 群体词互换一致率（%）与平均概率变化", "", "| 方法 | 一致率 | 平均 |Δp| |", "|---|---:|---:|"]
    for k, n, p, seeds in M:
        c, dp = [], []
        for s in seeds:
            po, pe = r.pr(f"{p}_s{s}__swap_orig"), r.pr(f"{p}_s{s}__swap_edit")
            if po is None or pe is None:
                continue
            m = po.merge(pe, on="pair_id", suffixes=("_o", "_e"))
            c.append((m.pred_o == m.pred_e).mean()); dp.append((m.prob_pos_o - m.prob_pos_e).abs().mean())
        out.append(f"| {n} | {fmt(ms(c))} | {np.mean(dp) if dp else float('nan'):.4f} |")

    # H7 ToxiCN
    tox = pd.DataFrame(json.load(open(ROOT / "data/external/toxicn/raw/ToxiCN_ex/ToxiCN/data/test.json", encoding="utf-8")))
    out += ["", "## H7 跨平台：COLD 宏 F1 对 ToxiCN 宏 F1；ToxiCN 显式（1）对隐式（2）的召回", "",
            "| 方法 | COLD | ToxiCN | 判 | |", "|---|---:|---:|---|---|"]
    from sklearn.metrics import f1_score
    for k, n, p, seeds in M:
        tf1 = []
        for s in seeds:
            pt = r.pr(f"{p}_s{s}__toxicn_test")
            if pt is not None:
                tf1.append(f1_score(pt.label, pt.pred, average="macro"))
        d, sg = diff(f1[k], ms(tf1))
        out.append(f"| {n} | {fmt(f1[k])} | {fmt(ms(tf1))} | 下降 {d:+.2f}（{'有差异' if sg else '无法区分'}） | |")
    out += ["", "（显式、隐式的分组召回取 `C_结果.md` 中 analyze_cold 的 ToxiCN 表，它保留了 expression 字段与行号的对应）"]

    # H8 学习曲线差距
    out += ["", "## H8 rbt3 − TF-IDF（字）宏 F1 差距", ""]
    for nn in ("1000", "3000", "10000", "全量"):
        if nn == "全量":
            g = [((r.js(f"cold_bert_rbt3_s{s}") or {}).get("test_macro_f1"), (r.js("cold_tfidf_main_char_s0") or {}).get("test_macro_f1")) for s in (0, 1, 2)]
        else:
            g = [((r.js(f"cold_bert_n{nn}_rbt3_s{s}") or {}).get("test_macro_f1"), (r.js(f"cold_tfidf_main_n{nn}_char_s{s}") or {}).get("test_macro_f1")) for s in (0, 1, 2)]
        gaps = [x - y for x, y in g if x is not None and y is not None]
        out.append(f"- n={nn}：差距 {fmt(ms(gaps))}（{len(gaps)} 个种子）")

    # H9 级联：TF-IDF（字）→ RoBERTa，阈值在验证集上选
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from cascade import confidence_from_prob_pos, select_tau, cascade_predict
    fine = pd.read_csv(ROOT / "data/external/coldataset/raw/COLDataset/test.csv")["fine-grained-label"].to_numpy()
    tv, tt = r.pr("cold_tfidf_main_char_s0__val"), r.pr("cold_tfidf_main_char_s0")
    out += ["", "## H9 级联 TF-IDF（字）→ RoBERTa（验证集定阈值）", "",
            "| 种子 | 交出比例(%) | 级联宏 F1 | RoBERTa 宏 F1 | 级联反歧视误判 | RoBERTa 反歧视误判 |", "|---:|---:|---:|---:|---:|---:|"]
    cf, rf, ca, ra = [], [], [], []
    for s in (0, 1, 2):
        bv, bt = r.pr(f"cold_bert_roberta_s{s}__val"), r.pr(f"cold_bert_roberta_s{s}")
        if any(x is None for x in (tv, tt, bv, bt)):
            continue
        sel = select_tau(tv.pred.values, confidence_from_prob_pos(tv.prob_pos.values), bv.pred.values, tv.label.values)
        ct = confidence_from_prob_pos(tt.prob_pos.values)
        pred = cascade_predict(tt.pred.values, ct, bt.pred.values, sel["tau"])
        lab = tt.label.values; fg3 = fine[tt["index"].values] == 3
        cf.append(f1_score(lab, pred, average="macro")); rf.append(f1_score(lab, bt.pred.values, average="macro"))
        ca.append(pred[fg3].mean()); ra.append(bt.pred.values[fg3].mean())
        out.append(f"| {s} | {(ct < sel['tau']).mean()*100:.1f} | {cf[-1]*100:.2f} | {rf[-1]*100:.2f} | {ca[-1]*100:.2f} | {ra[-1]*100:.2f} |")
    if cf:
        d1, s1 = diff(ms(cf), ms(rf)); d2, s2 = diff(ms(ca), ms(ra))
        out += ["", f"- 级联 − RoBERTa 宏 F1 = {d1:+.2f}（{'有差异' if s1 else '无法区分'}）",
                f"- 级联 − RoBERTa 反歧视误判率 = {d2:+.2f}（{'有差异' if s2 else '无法区分'}）"]

    Path(a.results, "C_判定数字.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
