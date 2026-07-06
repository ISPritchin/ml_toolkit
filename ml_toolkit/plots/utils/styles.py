"""Цветовые палитры и предустановленные стили осей matplotlib."""
from __future__ import annotations

import matplotlib.path as mpath
import matplotlib.pyplot as plt

from .regions import hide_spines


# ── Закруглённые стрелки для стиля 'presentation' (стиль черчения) ────────────
# Квадратичные кривые Безье: кончик в (0,0), крылья уходят влево и раскрываются.
_AH_VERTS: list[tuple[float, float]] = [
    ( 0.0,  0.00),
    (-0.8,  0.06),
    (-1.0,  0.38),
    ( 0.0,  0.00),
    (-0.8, -0.06),
    (-1.0, -0.38),
]
_AH_CODES = [
    mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3,
    mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3,
]
_ARROWHEAD_RIGHT = mpath.Path(_AH_VERTS, _AH_CODES)
_ARROWHEAD_UP    = mpath.Path([(-y, x) for x, y in _AH_VERTS], _AH_CODES)


PALETTES: dict[str, list[str]] = {
    'corporate': ['#0057B8', '#FF6B00', '#00A651', '#D62728', '#9467BD', '#8C564B'],
    'muted':     ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948'],
    'pastel':    ['#AEC6CF', '#FFD1DC', '#B5EAD7', '#FFDAC1', '#E2F0CB', '#C7CEEA'],
    'dark':      ['#1B2A4A', '#C0392B', '#1A7F54', '#E67E22', '#6C3483', '#2E86C1'],
    'diverging': ['#D73027', '#FC8D59', '#FEE090', '#91BFDB', '#4575B4'],
}


def apply_style(
    ax: plt.Axes,
    style: str = 'clean',
    palette: str | list[str] | None = None,
    grid: bool = True,
) -> None:
    """Применяет предустановленный стиль к осям.

    Args:
        style:   'clean'        — без верхней/правой рамки, светлый фон.
                 'minimal'      — только левая ось, без сетки.
                 'dark'         — тёмный фон.
                 'presentation' — крупные шрифты, жирные линии.
        palette: ключ из PALETTES или список hex-цветов.
        grid:    показать сетку.
    """
    from cycler import cycler  # noqa: PLC0415

    if style == 'clean':
        hide_spines(ax, ('top', 'right'))
        ax.set_facecolor('#FAFAFA')
        if grid:
            ax.grid(True, alpha=0.25, linewidth=0.6)

    elif style == 'minimal':
        hide_spines(ax, ('top', 'right', 'bottom'))
        ax.tick_params(bottom=False)
        ax.grid(False)

    elif style == 'dark':
        ax.set_facecolor('#1C1C1C')
        if ax.get_figure() is not None:
            ax.get_figure().set_facecolor('#1C1C1C')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
        ax.tick_params(colors='#AAAAAA')
        ax.xaxis.label.set_color('#CCCCCC')
        ax.yaxis.label.set_color('#CCCCCC')
        ax.title.set_color('#FFFFFF')
        if grid:
            ax.grid(True, color='#333333', linewidth=0.6)

    elif style == 'presentation':
        hide_spines(ax, ('top', 'right'))
        for lbl in (ax.xaxis.label, ax.yaxis.label):
            lbl.set_fontsize(14)
        ax.title.set_fontsize(15)
        ax.tick_params(labelsize=13)
        for name in ('left', 'bottom'):
            ax.spines[name].set_linewidth(1.5)
        if grid:
            ax.grid(True, alpha=0.2, linewidth=0.8)
        _ah_kw = dict(ls='', transform=ax.transAxes, clip_on=False,
                      markersize=13, markeredgecolor='black',
                      markerfacecolor='none', markeredgewidth=1.5)
        ax.plot(1, 0, marker=_ARROWHEAD_RIGHT, **_ah_kw)
        ax.plot(0, 1, marker=_ARROWHEAD_UP,    **_ah_kw)

    if palette is not None:
        colors = PALETTES[palette] if isinstance(palette, str) else palette
        ax.set_prop_cycle(cycler(color=colors))
