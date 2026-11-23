"""绘图公共设置。

图里的文字用中文，字体指定 ``Noto Sans CJK TC``。matplotlib
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


# ---------------- 报告插图：按进版尺寸作图，保存前查重叠 ----------------
TEXT_WIDTH_IN = 6.30          # A4、页边距 2.5 cm 的正文宽度
REF_COLOR = PALETTE["blue"]   # 参照线用蓝色，不用灰色


def paper_style() -> None:
    """按进版尺寸作图：刻度 8 磅、轴标签 9 磅，去上右边框，网格压在数据下。"""
    setup_chinese_font()
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "savefig.bbox": None,
        "figure.constrained_layout.use": True,
        "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.axisbelow": True, "axes.grid": False,
        "grid.color": "#E3E8EE", "grid.linewidth": 0.6,
        "lines.linewidth": 1.4, "lines.markersize": 4.5,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "errorbar.capsize": 2.5,
    })


def _bbox(artist, renderer):
    try:
        return artist.get_window_extent(renderer)
    except Exception:
        return None


def _line_points(line):
    """线段按 1 像素间距加密取样，免得直线穿过扁宽文字框时漏检。"""
    import numpy as np
    xy = line.get_transform().transform(np.column_stack(line.get_data()).astype(float))
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 2:
        return xy
    out = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(2, int(np.hypot(*(b - a))) + 1)
        out.append(np.linspace(a, b, n)[1:])
    return np.vstack(out)


def check_overlap(fig, pad: float = 1.0, gap: float = 3.0) -> list[str]:
    """文字与文字、文字与数据线、文字与柱子、图例与数据之间的重叠，空表示干净。

    不查：带不透明底的文字（有意挖空线条）、坐标轴之外的刻度标签与网格线、热图格内的数字。
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    legends = [lg for ax in fig.axes for lg in ([ax.get_legend()] if ax.get_legend() else [])] + list(fig.legends)
    legend_kids = {id(a) for lg in legends for a in lg.findobj()}
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
             and id(t) not in legend_kids and not getattr(t, "_skip_overlap", False)]
    boxes = [(t, Text.get_window_extent(t, r)) for t in texts]   # 注释的引线不算进文字框
    boxes = [(t, b.expanded(1, 1).padded(-pad)) for t, b in boxes if b is not None and b.width > 0]
    hits = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            both_cells = getattr(boxes[i][0], "_in_bar", False) and getattr(boxes[j][0], "_in_bar", False)
            g = 0.0 if both_cells else gap      # 格内的字（热图数字、逐字归因）只要求不相交
            if boxes[i][1].padded(g).overlaps(boxes[j][1]):   # 其余文字之间至少留 gap 像素
                hits.append(f"文字相叠：{boxes[i][0].get_text()!r} / {boxes[j][0].get_text()!r}")
    import numpy as np
    for ax in fig.axes:
        pts = [_line_points(ln) for ln in ax.get_lines()
               if ln.get_visible() and id(ln) not in legend_kids]
        for c in ax.collections:          # 误差线等 LineCollection
            if c.get_visible() and hasattr(c, "get_segments") and id(c) not in legend_kids:
                for seg in c.get_segments():
                    if len(seg) >= 2:
                        pts.append(_line_points(Line2D(*np.asarray(seg).T, transform=c.get_transform())))
        bar_boxes = [_bbox(p, r) for p in ax.patches
                     if isinstance(p, Rectangle) and p.get_visible() and p.get_height() != 0]
        own = set(ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label, ax.title])
        for t, b in boxes:
            if t.axes is not ax or t in own:
                continue
            bp = t.get_bbox_patch()
            if bp is not None and bp.get_facecolor()[3] > 0.9:   # 有意挖空
                continue
            if any(len(p) and b.count_contains(p) for p in pts):
                hits.append(f"文字压线：{t.get_text()!r}")
            if not getattr(t, "_in_bar", False) and any(bb is not None and b.overlaps(bb) for bb in bar_boxes):
                hits.append(f"文字压柱：{t.get_text()!r}")
    for lg in legends:
        lb = _bbox(lg.get_frame(), r)
        for ax in fig.axes:
            if lb is None or not lb.overlaps(ax.bbox):
                continue
            for ln in ax.get_lines():
                if id(ln) in legend_kids or not ln.get_visible():
                    continue
                p = _line_points(ln)
                if len(p) and lb.count_contains(p):
                    hits.append(f"图例压线：{ln.get_label()!r}")
            for p in ax.patches:
                pb = _bbox(p, r)
                if pb is not None and p.get_visible() and lb.overlaps(pb) and id(p) not in legend_kids:
                    hits.append("图例压柱")
                    break
    return sorted(set(hits))


def save_checked(fig, path: str | Path, width_frac: float) -> Path:
    """检查宽度与重叠，都过了才写文件；宽度按正文宽的比例给定，进版不缩放。"""
    want = width_frac * TEXT_WIDTH_IN
    got = fig.get_size_inches()[0]
    if abs(got - want) > 0.02:
        raise AssertionError(f"{path}: 画布宽 {got:.2f} in，进版宽 {want:.2f} in")
    hits = check_overlap(fig)
    if hits:
        plt.close(fig)
        raise AssertionError(f"{path} 有重叠，未保存：\n  " + "\n  ".join(hits))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
