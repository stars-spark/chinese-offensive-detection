"""D2 示例图：种子 0 的 A 组中取 6 条画字级归因热力图。

选择规则：只取长度不超过 50 字的 A 组评论，用 ``random.Random(0)`` 打乱后
依次取 6 条，跳过 ``data_selfmade/D2_否决.txt`` 中列出的测试集 index。
每个字的底色深浅对应它朝攻击性方向的正归因，群体词字加黑色边框。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from plot_style import save_figure, setup_chinese_font
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "results" / "preds" / "D2_归因_s0.parquet"
VETO_PATH = ROOT / "data_selfmade" / "D2_否决.txt"
OUT_PATH = ROOT / "figures" / "D2_归因示例.png"
N_PICK = 6
MAX_CHARS = 50


def load_veto(path: Path = VETO_PATH) -> set[int]:
    """每行一个测试集 index；没有文件就不跳过。"""
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.add(int(line))
        except ValueError:
            continue
    return out


def pick_rows(frame: pd.DataFrame, veto: set[int], n: int = N_PICK):
    """按 random.Random(0) 打乱后顺延取 n 条，跳过被否决的 index。

    返回 (选中的行, 选中 index 列表, 被否决 index 列表)，供记录用。
    """
    pool = frame[(frame["group"] == "A")
                 & (frame["chars"].astype(str).str.len() <= MAX_CHARS)]
    order = list(range(len(pool)))
    random.Random(0).shuffle(order)
    picked, picked_idx, vetoed = [], [], []
    for pos in order:
        row = pool.iloc[pos]
        index = int(row["index"])
        if index in veto:
            vetoed.append(index)
            continue
        picked.append(row)
        picked_idx.append(index)
        if len(picked) >= n:
            break
    return pd.DataFrame(picked), picked_idx, vetoed


def draw_row(ax, text: str, attrs: np.ndarray, group_mask: str,
             y: float, cmap, title: str) -> None:
    """一条评论一行：字底色按正归因，群体词字加边框。"""
    positive = np.maximum(attrs, 0.0)
    peak = float(positive.max()) if len(positive) else 0.0
    for i, ch in enumerate(text):
        frac = positive[i] / peak if peak > 0 else 0.0
        is_group = i < len(group_mask) and group_mask[i] == "1"
        ax.text(
            i + 0.5, y, ch, ha="center", va="center", fontsize=11,
            bbox={
                "boxstyle": "square,pad=0.08",
                "facecolor": cmap(frac),
                "edgecolor": "black" if is_group else "none",
                "linewidth": 1.2 if is_group else 0.0,
            },
        )
    ax.text(len(text) + 0.8, y, title, ha="left", va="center", fontsize=8)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="D2 归因示例图")
    parser.add_argument("--pred", default=str(PRED_PATH))
    parser.add_argument("--veto", default=str(VETO_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--record", default=str(ROOT / "results" / "D2_归因示例_选择.json"))
    args = parser.parse_args()
    setup_chinese_font()
    frame = pd.read_parquet(args.pred)
    picked, picked_idx, vetoed = pick_rows(frame, load_veto(Path(args.veto)))
    if picked.empty:
        raise SystemExit("没有可画的 A 组样本，检查 parquet 或否决文件")
    width = int(picked["chars"].astype(str).str.len().max()) + 18
    fig, ax = plt.subplots(figsize=(min(16.0, 0.24 * width + 4),
                                   1.05 * len(picked) + 0.8))
    cmap = plt.get_cmap("GnBu")
    for y, (_, row) in enumerate(picked.iterrows()):
        attrs = np.asarray(json.loads(row["char_attrs"]), dtype=np.float64)
        title = (f"index={int(row['index'])}  "
                 f"富集比={row['enrichment']:.2f}"
                 if pd.notna(row["enrichment"]) else f"index={int(row['index'])}")
        draw_row(ax, str(row["chars"]), attrs, str(row["group_mask"]),
                 float(len(picked) - 1 - y), cmap, title)
    ax.set_xlim(0, width)
    ax.set_ylim(-0.6, len(picked) - 0.4)
    ax.axis("off")
    # 报告用图不画图内标题，说明放在 LaTeX 图题里
    save_figure(fig, args.out)
    record = {
        "pred": str(args.pred),
        "veto_file": str(args.veto),
        "picked_indices": picked_idx,
        "vetoed_indices": vetoed,
        "n_picked": len(picked_idx),
    }
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    Path(args.record).write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {args.out}，{len(picked)} 条")
    print("index: " + ", ".join(str(int(x)) for x in picked["index"]))


if __name__ == "__main__":
    main()
