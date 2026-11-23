"""绘图公共设置（阶段 2 的图共用）。

图里的文字用中文，字体按任务要求指定 ``Noto Sans CJK TC``。matplotlib
默认字体不含中文，不设置会变成一排方块。所有图都用无界面的 Agg 后端，
服务器上没有显示器也能保存 PNG。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT = "Noto Sans CJK TC"

# 报告统一配色：蓝绿色系，取自 ColorBrewer 的 GnBu / YlGnBu 色带（github.com/axismaps/colorbrewer），
# 由深到浅排列，类别色按顺序取用；参照线、基线用中性灰。
PALETTE = {
    "navy": "#08589E",   # 深蓝
    "blue": "#2B8CBE",   # 蓝
    "teal": "#1A9E8F",   # 青绿
    "green": "#41AB5D",  # 绿
    "cyan": "#7BCCC4",   # 浅青
    "sky": "#A8DDB5",    # 浅绿
    "gray": "#9AA5B1",   # 中性灰，基线与参照
}
CYCLE = [PALETTE[k] for k in ("navy", "teal", "blue", "green", "cyan", "sky")]
CMAP_SEQ = "YlGnBu"     # 热图等连续量
CMAP_ATTR = "GnBu"      # 归因深浅


def setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [FONT, "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=CYCLE)


def save_figure(fig, path: str | Path) -> Path:
    """保存并关闭，自动建父目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
