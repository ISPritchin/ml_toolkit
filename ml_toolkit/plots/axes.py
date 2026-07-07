"""Утилиты для осей: symmetrize_ylim, log_axis."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def symmetrize_ylim(
    ax: plt.Axes,
    center: float = 0.0,
    margin: float = 0.05,
) -> None:
    """Сделать ось Y симметричной вокруг center (полезно для residuals/returns).

    Args:
        margin: относительный отступ сверху и снизу (доля от диапазона).

    """
    yl = ax.get_ylim()
    half = max(abs(yl[0] - center), abs(yl[1] - center))
    pad = half * margin
    ax.set_ylim(center - half - pad, center + half + pad)


def log_axis(
    ax: plt.Axes,
    axis: str = 'y',
    base: int = 10,
    subs: tuple[float, ...] | None = None,
    fmt: str = 'auto',
) -> None:
    """Log-ось с правильными minor-тиками и форматом без научной нотации.

    Args:
        axis:  'x' | 'y' | 'both'.
        subs:  позиции minor-тиков; авто если None.
        fmt:   'auto' — без научной нотации | 'sci' — оставить как есть.

    """
    _subs = subs or (2, 3, 4, 5, 6, 7, 8, 9)

    def _setup(target_axis):
        if base == 10:
            target_axis.set_major_locator(mticker.LogLocator(base=10, numticks=10))
            target_axis.set_minor_locator(mticker.LogLocator(base=10, subs=_subs, numticks=50))
            target_axis.set_minor_formatter(mticker.NullFormatter())
        if fmt == 'auto':
            target_axis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f'{x:g}')
            )

    if axis in ('y', 'both'):
        ax.set_yscale('log', base=base)
        _setup(ax.yaxis)
    if axis in ('x', 'both'):
        ax.set_xscale('log', base=base)
        _setup(ax.xaxis)
