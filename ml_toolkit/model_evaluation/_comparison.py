"""Multi-model comparison utilities (classification and regression)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ._base import BaseEvaluator

logger = logging.getLogger(__name__)


def compare_models(
    evaluators: dict[str, BaseEvaluator],
    split: str,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Compute metrics for multiple models on a single split.

    Args:
        evaluators: {model_name: evaluator} — split must be registered in each.
        split:      Split name.
        metrics:    Metric names (default: registered in the first evaluator).

    Returns:
        DataFrame with index=metrics, columns=model_names.

    """
    if not evaluators:
        raise ValueError('evaluators must not be empty')

    first_ev = next(iter(evaluators.values()))
    metric_names = first_ev._resolve_metric_names(metrics)
    metric_fns = {m: first_ev._metrics[m] for m in metric_names}

    data: dict[str, list[float]] = {}
    for model_name, ev in evaluators.items():
        if split not in ev._splits:
            raise ValueError(
                f'Split {split!r} not registered in evaluator {model_name!r}. '
                f'Available: {list(ev._splits)}'
            )
        y_true, y_second = ev._splits[split]
        vals = []
        for m, fn in metric_fns.items():
            try:
                vals.append(float(fn(y_true, y_second)))
            except Exception as exc:
                logger.debug('Metric %r failed for %r: %s', m, model_name, exc)
                vals.append(float('nan'))
        data[model_name] = vals

    return pd.DataFrame(data, index=metric_names)


# ── Internal panel helpers ─────────────────────────────────────────────────────

def _annotate_bars(ax_: Any, bars: Any, vals: Any, fmt: str = '{:.4g}') -> None:
    """Value annotations above/below each bar, scaled to current y-range."""
    ylims = ax_.get_ylim()
    # Используем фактический y_range (без clamp к 1e-9) для offset.
    # Если offset > y_range, текст улетел бы за пределы axes → raster overflow.
    y_range = abs(ylims[1] - ylims[0])
    offset = y_range * 0.03
    for bar, val in zip(bars, vals):
        if np.isnan(val):
            continue
        ax_.text(
            bar.get_x() + bar.get_width() / 2,
            val + offset if val >= 0 else val - offset,
            fmt.format(val),
            ha='center', va='bottom' if val >= 0 else 'top', fontsize=7.5,
        )


def _draw_cmp_panel(
    ax_: Any,
    row: pd.Series,
    model_names: list[str],
    palette: list,
) -> None:
    """Bars for one metric, one bar per model."""
    x = np.arange(len(model_names))
    bars = ax_.bar(x, row.values.astype(float), color=palette, alpha=0.85, width=0.65)
    ax_.axhline(0, color='black', lw=0.6, alpha=0.4)
    _annotate_bars(ax_, bars, row.values.astype(float))
    ax_.set_xticks(x)
    ax_.set_xticklabels(model_names, rotation=20, ha='right', fontsize=8)
    ax_.set_ylabel('Value', fontsize=8)


def _draw_delta_panel(
    ax_: Any,
    row: pd.Series,
    model_names: list[str],
    palette: list,
) -> None:
    """Delta bars for one metric vs reference, one bar per model."""
    x = np.arange(len(model_names))
    vals = row.values.astype(float)
    bar_colors = [c if v >= 0 else '#ef9a9a' for c, v in zip(palette, vals)]
    bars = ax_.bar(x, vals, color=bar_colors, alpha=0.85, width=0.65)
    ax_.axhline(0, color='black', lw=1.0, linestyle='--', alpha=0.5)
    _annotate_bars(ax_, bars, vals, fmt='{:+.4g}')
    ax_.set_xticks(x)
    ax_.set_xticklabels(model_names, rotation=20, ha='right', fontsize=8)
    ax_.set_ylabel('Δ value', fontsize=8)


def _facet_figure(n_metrics: int, ncols: int, cell_w: float = 4.5, cell_h: float = 4.0):
    """Create a grid figure; return (fig, flat_axes_list)."""
    import matplotlib.pyplot as plt
    n_cols = min(ncols, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, ax_arr = plt.subplots(n_rows, n_cols,
                                figsize=(cell_w * n_cols, cell_h * n_rows),
                                squeeze=False)
    flat = [ax_arr[r][c] for r in range(n_rows) for c in range(n_cols)]
    for i in range(n_metrics, len(flat)):
        flat[i].set_visible(False)
    return fig, flat[:n_metrics]


def _save_facet(fig: Any, path: Any, n_legend_rows: int = 1) -> None:
    import matplotlib.pyplot as plt
    # suptitle(y>1) + tight_layout вызывают бесконечное расширение фигуры.
    # Используем subplots_adjust с фиксированными долями, без tight_layout.
    bottom_frac = max(0.06, 0.06 * n_legend_rows)
    fig.subplots_adjust(top=0.93, bottom=bottom_frac, hspace=0.45, wspace=0.35)
    if path is not None:
        fig.savefig(path, dpi=100)
    else:
        plt.show()
    plt.close(fig)


# ── Public API ────────────────────────────────────────────────────────────────

def plot_model_comparison(
    evaluators: dict[str, BaseEvaluator],
    split: str,
    metrics: list[str] | None = None,
    sort_by: str | None = None,
    ncols: int = 3,
    ax: Any = None,
    path: Any = None,
) -> None:
    """Faceted bar chart: one panel per metric, models as bars.

    Each panel uses its own y-scale, so metrics on different scales (e.g. MAE=50
    alongside R²=0.9) are all equally readable.

    When exactly one metric is requested, the chart fits on a single Axes and the
    ax= parameter works for composition. For multiple metrics a faceted grid is
    always created and ax= is ignored.

    Args:
        evaluators: {model_name: evaluator}.
        split:      Split name.
        metrics:    Metric names (default: registered in first evaluator).
        sort_by:    Sort metrics by this model's values (descending).
        ncols:      Columns in the facet grid (default 3).
        ax:         Existing Axes — used only when len(metrics)==1.
        path:       Output path (optional).

    """
    from matplotlib.patches import Patch

    df = compare_models(evaluators, split, metrics)
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    n_metrics, n_models = len(df), len(df.columns)
    ev0 = next(iter(evaluators.values()))
    palette = ev0._palette(n_models)
    model_names = list(df.columns)

    if n_metrics == 1 and ax is not None:
        fig, ax_, created = ev0._prepare_ax(ax, figsize=(5, 4))
        _draw_cmp_panel(ax_, df.iloc[0], model_names, palette)
        ax_.set_title(f'{df.index[0]} — {split}')
        ev0._finalize(fig, path, created)
        return

    fig, axes = _facet_figure(n_metrics, ncols)
    for ax_, metric_name in zip(axes, df.index):
        _draw_cmp_panel(ax_, df.loc[metric_name], model_names, palette)
        ax_.set_title(metric_name, fontsize=9, fontweight='bold')

    legend_handles = [Patch(color=c, label=n) for c, n in zip(palette, model_names)]
    ncol_legend = min(n_models, 5)
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=ncol_legend, fontsize=9,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f'Model Comparison — {split}', fontsize=11, y=0.99)
    _save_facet(fig, path, n_legend_rows=max(1, (n_models + ncol_legend - 1) // ncol_legend))


def plot_model_heatmap(
    evaluators: dict[str, BaseEvaluator],
    split: str,
    metrics: list[str] | None = None,
    normalize_rows: bool = True,
    ax: Any = None,
    path: Any = None,
) -> None:
    """Heatmap of metrics × models with per-row normalisation.

    Raw values are shown as text; colour encodes relative rank within each metric
    row (green = highest, red = lowest). Normalisation does NOT account for metric
    direction (higher vs lower is better) — use the bar chart or delta plot for
    directional interpretation.

    Args:
        evaluators:     {model_name: evaluator}.
        split:          Split name.
        metrics:        Metrics to show.
        normalize_rows: Normalise each metric row to [0, 1] for colour encoding.
        ax:             Existing Axes (optional).
        path:           Output path (optional).

    """
    df = compare_models(evaluators, split, metrics)

    ev0 = next(iter(evaluators.values()))
    fig, ax_, created = ev0._prepare_ax(
        ax, figsize=(max(5, len(df.columns) * 1.5), max(3, len(df) * 0.65))
    )

    if normalize_rows:
        row_min = df.min(axis=1)
        row_max = df.max(axis=1)
        denom = (row_max - row_min).replace(0.0, np.nan)
        display = (df.T - row_min).T.div(denom, axis=0).fillna(0.5)
    else:
        display = (df - df.min().min()) / (df.max().max() - df.min().min() + 1e-12)

    vals = display.values.astype(float)
    ax_.imshow(vals, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

    ax_.set_xticks(range(len(df.columns)))
    ax_.set_xticklabels(df.columns, rotation=30, ha='right', fontsize=9)
    ax_.set_yticks(range(len(df.index)))
    ax_.set_yticklabels(df.index, fontsize=9)

    for r in range(len(df.index)):
        for c in range(len(df.columns)):
            raw = df.iloc[r, c]
            brightness = vals[r, c]
            text_color = 'white' if brightness < 0.25 or brightness > 0.82 else 'black'
            ax_.text(c, r, f'{raw:.4g}' if not np.isnan(raw) else 'NaN',
                     ha='center', va='center', fontsize=8, color=text_color)

    norm_note = ' (row-normalised colour)' if normalize_rows else ''
    ax_.set_title(f'Model Heatmap{norm_note} — {split}')
    ev0._finalize(fig, path, created, tight=True)


def plot_model_delta(
    evaluators: dict[str, BaseEvaluator],
    ref: str,
    split: str,
    metrics: list[str] | None = None,
    ncols: int = 3,
    ax: Any = None,
    path: Any = None,
) -> None:
    """Faceted delta bar chart: one panel per metric, showing Δ vs a reference model.

    Each panel has its own y-scale so small differences (e.g. ΔR²=0.02) and large
    differences (e.g. ΔMAE=8) are equally visible. Model-coloured bars = above
    reference; red bars = below reference.

    When exactly one metric is requested the chart fits on a single Axes and ax=
    works for composition. For multiple metrics a faceted grid is always created.

    Args:
        evaluators: {model_name: evaluator} — must include ref.
        ref:        Name of the baseline model.
        split:      Split name.
        metrics:    Metrics to show.
        ncols:      Columns in the facet grid (default 3).
        ax:         Existing Axes — used only when len(metrics)==1.
        path:       Output path (optional).

    """
    from matplotlib.patches import Patch

    if ref not in evaluators:
        raise ValueError(f'ref={ref!r} not in evaluators. Keys: {list(evaluators)}')

    df = compare_models(evaluators, split, metrics)
    ref_vals = df[ref]
    delta = df.drop(columns=[ref]).subtract(ref_vals, axis=0)

    n_metrics, n_others = len(delta.index), len(delta.columns)
    ev0 = next(iter(evaluators.values()))

    all_names = list(evaluators.keys())
    all_colors = ev0._palette(len(all_names))
    palette = [c for name, c in zip(all_names, all_colors) if name != ref]
    other_names = list(delta.columns)

    if n_metrics == 1 and ax is not None:
        fig, ax_, created = ev0._prepare_ax(ax, figsize=(5, 4))
        _draw_delta_panel(ax_, delta.iloc[0], other_names, palette)
        ax_.set_title(f'Δ {delta.index[0]} vs "{ref}" — {split}')
        ev0._finalize(fig, path, created)
        return

    fig, axes = _facet_figure(n_metrics, ncols)
    for ax_, metric_name in zip(axes, delta.index):
        _draw_delta_panel(ax_, delta.loc[metric_name], other_names, palette)
        ax_.set_title(metric_name, fontsize=9, fontweight='bold')

    legend_handles = [Patch(color=c, label=n) for c, n in zip(palette, other_names)]
    ncol_legend = min(n_others, 5)
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=ncol_legend, fontsize=9,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f'Model Delta vs "{ref}" — {split}', fontsize=11, y=0.99)
    _save_facet(fig, path, n_legend_rows=max(1, (n_others + ncol_legend - 1) // ncol_legend))
