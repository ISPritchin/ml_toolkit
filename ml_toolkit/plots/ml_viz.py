"""ML-специфичные визуализации: пороги классификации, confusion quadrants."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


def add_threshold_band(
    ax: plt.Axes,
    threshold: float,
    precision: float | None = None,
    recall: float | None = None,
    color: str = '#E67E22',
    lw: float = 1.4,
    alpha: float = 0.12,
    band_width: float | None = None,
    fontsize: float = 8,
    axis: str = 'x',
) -> None:
    """Полоса вокруг классификационного порога с precision/recall.

    Args:
        ax:         Axes для отрисовки.
        threshold:  значение порога.
        precision:  precision при этом пороге (для подписи).
        recall:     recall при этом пороге (для подписи).
        color:      цвет линии/полосы/подписи.
        lw:         толщина линии порога.
        alpha:      прозрачность закрашенной полосы.
        band_width: ширина закрашенной полосы ±; None — только линия.
        fontsize:   размер шрифта подписи.
        axis:       'x' (threshold по оси X) | 'y'.

    """
    _vline = ax.axvline if axis == 'x' else ax.axhline
    _vspan = ax.axvspan if axis == 'x' else ax.axhspan

    _vline(threshold, color=color, lw=lw, linestyle='--', zorder=5)

    if band_width is not None:
        _vspan(threshold - band_width, threshold + band_width,
               color=color, alpha=alpha, linewidth=0)

    parts = [f'thr={threshold:.3g}']
    if precision is not None:
        parts.append(f'P={precision:.2%}')
    if recall is not None:
        parts.append(f'R={recall:.2%}')
    label = '  '.join(parts)

    if axis == 'x':
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(threshold, 0.97, f'  {label}', transform=trans,
                ha='left', va='top', fontsize=fontsize, color=color)
    else:
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(0.97, threshold, f'  {label}', transform=trans,
                ha='right', va='bottom', fontsize=fontsize, color=color)


def add_confusion_quadrant_labels(
    ax: plt.Axes,
    threshold_x: float | None = None,
    threshold_y: float | None = None,
    labels: dict[str, str] | None = None,
    fontsize: float = 9,
    color: str = '#AAAAAA',
    alpha: float = 0.7,
) -> None:
    """Текстовые метки TP/FP/TN/FN в четырёх квадрантах scatter.

    Args:
        ax:          Axes со scatter-графиком.
        threshold_x: граница по X (авто — середина xlim).
        threshold_y: граница по Y (авто — середина ylim).
        labels:      {quadrant: text}; quadrant ∈ {'TL','TR','BL','BR'}.
        fontsize:    размер шрифта меток.
        color:       цвет разделительных линий и меток.
        alpha:       прозрачность меток.

    """
    xl, yl = ax.get_xlim(), ax.get_ylim()
    tx = threshold_x if threshold_x is not None else (xl[0] + xl[1]) / 2
    ty = threshold_y if threshold_y is not None else (yl[0] + yl[1]) / 2

    ax.axvline(tx, color=color, lw=0.8, linestyle=':', alpha=0.5)
    ax.axhline(ty, color=color, lw=0.8, linestyle=':', alpha=0.5)

    _default_labels = {'TR': 'TP', 'TL': 'FN', 'BR': 'FP', 'BL': 'TN'}
    lbls = {**_default_labels, **(labels or {})}

    _positions = {
        'TR': (0.97, 0.97, 'right', 'top'),
        'TL': (0.03, 0.97, 'left',  'top'),
        'BR': (0.97, 0.03, 'right', 'bottom'),
        'BL': (0.03, 0.03, 'left',  'bottom'),
    }
    for key, text in lbls.items():
        x_a, y_a, ha, va = _positions[key]
        ax.text(x_a, y_a, text, transform=ax.transAxes,
                ha=ha, va=va, fontsize=fontsize, color=color, alpha=alpha,
                fontweight='bold')
