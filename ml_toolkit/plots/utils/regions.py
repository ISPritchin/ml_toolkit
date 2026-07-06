"""Заливки вертикальных областей и управление spines."""
from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


def hide_spines(
    ax: plt.Axes,
    which: Sequence[str] = ('top', 'right'),
) -> None:
    """Скрывает указанные рамки (spines)."""
    for spine in which:
        ax.spines[spine].set_visible(False)


_REGION_LOC: dict[str, tuple[float, float, str, str]] = {
    'top-left':      (0.04, 0.96, 'left',   'top'),
    'top-center':    (0.50, 0.96, 'center', 'top'),
    'top-right':     (0.96, 0.96, 'right',  'top'),
    'center-left':   (0.04, 0.50, 'left',   'center'),
    'center':        (0.50, 0.50, 'center', 'center'),
    'center-right':  (0.96, 0.50, 'right',  'center'),
    'bottom-left':   (0.04, 0.04, 'left',   'bottom'),
    'bottom-center': (0.50, 0.04, 'center', 'bottom'),
    'bottom-right':  (0.96, 0.04, 'right',  'bottom'),
}


def fill_region(
    ax: plt.Axes,
    x1: float,
    x2: float,
    label: str = '',
    color: str = 'steelblue',
    alpha: float = 0.15,
    label_loc: str = 'top-center',
    fontsize: float = 9,
    **span_kwargs,
) -> None:
    """Заливка вертикальной области с необязательной подписью.

    Args:
        label_loc: положение подписи внутри области —
            'top-left' | 'top-center' | 'top-right' |
            'center-left' | 'center' | 'center-right' |
            'bottom-left' | 'bottom-center' | 'bottom-right'
    """
    ax.axvspan(x1, x2, color=color, alpha=alpha, **span_kwargs)
    if not label:
        return
    xf, yf, ha, va = _REGION_LOC.get(label_loc, _REGION_LOC['top-center'])
    x_data = x1 + xf * (x2 - x1)
    trans  = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(x_data, yf, label, transform=trans, color=color,
            fontsize=fontsize, ha=ha, va=va)
