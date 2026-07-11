"""Аннотации для matplotlib."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


def annotate_bars(
    ax: plt.Axes,
    fmt: str = '{:.0f}',
    threshold: float | None = None,
    fontsize: float = 8,
    padding: float = 3,
    color: str = '#333333',
) -> None:
    """Подписать значения на всех барах axes.

    Args:
        ax:        Axes с барами.
        fmt:       строка форматирования ('{:.1%}' для процентов и т.п.).
        threshold: не подписывать бары с abs(height) < threshold.
        fontsize:  размер шрифта подписи.
        padding:   отступ от вершины бара в points.
        color:     цвет текста подписи.

    """
    for patch in ax.patches:
        h = patch.get_height()
        if h == 0:
            continue
        if threshold is not None and abs(h) < threshold:
            continue
        x = patch.get_x() + patch.get_width() / 2
        va = 'bottom' if h >= 0 else 'top'
        sign = 1 if h >= 0 else -1
        offset = mtransforms.ScaledTranslation(0, sign * padding / 72, ax.figure.dpi_scale_trans)
        ax.text(
            x, h, fmt.format(h),
            ha='center', va=va, fontsize=fontsize, color=color,
            transform=ax.transData + offset,
        )
