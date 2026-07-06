"""Оверлеи для временных рядов."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np


def add_period_bands(
    ax: plt.Axes,
    periods: list[tuple],
    labels: list[str] | None = None,
    colors: list[str] | str | None = None,
    alpha: float = 0.15,
    fontsize: float = 8,
    label_loc: str = 'top',
    rotation: int = 0,
) -> None:
    """Набор именованных периодов как вертикальные заливки.

    Args:
        periods:   список пар (x_start, x_end).
        labels:    подписи периодов; None — без подписей.
        colors:    один цвет для всех или список цветов.
        label_loc: 'top' | 'center' | 'bottom'.
    """
    _default_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']
    n = len(periods)
    if colors is None:
        clrs = [_default_colors[i % len(_default_colors)] for i in range(n)]
    elif isinstance(colors, str):
        clrs = [colors] * n
    else:
        clrs = list(colors)

    lbls = labels or [''] * n
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    for (x1, x2), label, color in zip(periods, lbls, clrs):
        ax.axvspan(x1, x2, color=color, alpha=alpha, linewidth=0)
        if label:
            xm = x1 + (x2 - x1) / 2
            y_frac = {'top': 0.97, 'center': 0.50, 'bottom': 0.03}.get(label_loc, 0.97)
            va = {'top': 'top', 'center': 'center', 'bottom': 'bottom'}.get(label_loc, 'top')
            ax.text(xm, y_frac, label, transform=trans,
                    ha='center', va=va, fontsize=fontsize, color=color, rotation=rotation)


def add_forecast_region(
    ax: plt.Axes,
    x_split,
    label_actual: str = 'Факт',
    label_forecast: str = 'Прогноз',
    color: str = '#4E79A7',
    alpha: float = 0.08,
    fontsize: float = 9,
) -> None:
    """Вертикальная черта + заливка правой части как «прогноз».

    Args:
        x_split: значение x, начиная с которого идёт прогноз.
    """
    ax.axvline(x_split, color=color, lw=1.2, linestyle='--', alpha=0.8)
    xl = ax.get_xlim()
    ax.axvspan(x_split, xl[1], color=color, alpha=alpha, linewidth=0)

    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    if label_actual:
        ax.text(x_split, 0.97, f'  {label_actual}', transform=trans,
                ha='right', va='top', fontsize=fontsize, color=color, alpha=0.7)
    if label_forecast:
        ax.text(x_split, 0.97, f'  {label_forecast}', transform=trans,
                ha='left', va='top', fontsize=fontsize, color=color)


def add_event_markers(
    ax: plt.Axes,
    events: dict,
    y_frac: float = 1.02,
    marker: str = 'v',
    markersize: float = 7,
    color: str = '#E15759',
    fontsize: float = 7,
    rotation: int = 45,
) -> None:
    """Маркеры именованных событий над осью X.

    Args:
        events: {x_value: label_str} — значение оси X → название события.
        y_frac: положение маркера в долях axes (>1 → над графиком).
    """
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for x_val, label in events.items():
        ax.plot(x_val, y_frac, marker=marker, markersize=markersize,
                color=color, transform=trans, clip_on=False)
        ax.text(x_val, y_frac + 0.02, label, transform=trans,
                ha='left', va='bottom', fontsize=fontsize, color=color,
                rotation=rotation, clip_on=False)
