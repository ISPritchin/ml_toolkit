"""Опорные линии с подписями и биссектриса для matplotlib."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


def add_bisector(
    ax: plt.Axes,
    color: str = '#888888',
    lw: float = 1.0,
    linestyle: str = '--',
    label: str | None = 'y = x',
    **kwargs,
) -> None:
    """Рисует биссектрису y=x в текущих пределах осей."""
    xl, yl = ax.get_xlim(), ax.get_ylim()
    lo, hi = min(xl[0], yl[0]), max(xl[1], yl[1])
    ax.plot([lo, hi], [lo, hi], color=color, lw=lw, linestyle=linestyle, label=label, **kwargs)
    ax.set_xlim(xl)
    ax.set_ylim(yl)


def add_vline(
    ax: plt.Axes,
    x: float,
    label: str = '',
    loc: str = 'top',
    color: str = '#888888',
    lw: float = 1.0,
    linestyle: str = '--',
    fontsize: float = 9,
    rotation: int = 90,
    **line_kwargs,
) -> None:
    """Вертикальная линия с подписью.

    Args:
        ax:           Axes для отрисовки.
        x:            координата X линии.
        label:        текст подписи; '' — без подписи.
        loc:          положение подписи вдоль линии — 'top' | 'bottom' | 'center'.
        color:        цвет линии и подписи.
        lw:           толщина линии.
        linestyle:    стиль линии matplotlib.
        fontsize:     размер шрифта подписи.
        rotation:     угол поворота текста в градусах.
                      90  — вдоль линии (по умолчанию).
                      0   — перпендикулярно линии (горизонтально).
        **line_kwargs: дополнительные kwargs для ax.axvline.

    """
    ax.axvline(x, color=color, lw=lw, linestyle=linestyle, **line_kwargs)
    if not label:
        return
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    y_frac = {'top': 0.97, 'bottom': 0.03, 'center': 0.50}.get(loc, 0.97)
    va     = {'top': 'top', 'bottom': 'bottom', 'center': 'center'}.get(loc, 'top')
    ax.text(x, y_frac, f' {label}', transform=trans, color=color,
            fontsize=fontsize, va=va, ha='left', rotation=rotation)


def add_hline(
    ax: plt.Axes,
    y: float,
    label: str = '',
    loc: str = 'right',
    color: str = '#888888',
    lw: float = 1.0,
    linestyle: str = '--',
    fontsize: float = 9,
    rotation: int = 0,
    **line_kwargs,
) -> None:
    """Горизонтальная линия с подписью.

    Args:
        ax:           Axes для отрисовки.
        y:            координата Y линии.
        label:        текст подписи; '' — без подписи.
        loc:          положение подписи вдоль линии — 'right' | 'left' | 'center'.
        color:        цвет линии и подписи.
        lw:           толщина линии.
        linestyle:    стиль линии matplotlib.
        fontsize:     размер шрифта подписи.
        rotation:     угол поворота текста в градусах.
                      0   — вдоль линии (по умолчанию).
                      90  — перпендикулярно линии (вертикально).
        **line_kwargs: дополнительные kwargs для ax.axhline.

    """
    ax.axhline(y, color=color, lw=lw, linestyle=linestyle, **line_kwargs)
    if not label:
        return
    trans  = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    x_frac = {'right': 0.97, 'left': 0.03, 'center': 0.50}.get(loc, 0.97)
    if rotation == 90:
        ha = 'center'
        va = 'bottom'
    else:
        ha = {'right': 'right', 'left': 'left', 'center': 'center'}.get(loc, 'right')
        va = 'bottom'
    ax.text(x_frac, y, f' {label}', transform=trans, color=color,
            fontsize=fontsize, va=va, ha=ha, rotation=rotation)
