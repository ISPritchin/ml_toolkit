"""Regression evaluator — metrics and visualisations."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._base import BaseEvaluator, SavePath, logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# ── Preset metric functions ────────────────────────────────────────────────────

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(y_true == 0, 1.0, np.abs(y_true))
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')


def _medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(y_true - y_pred)))


def _max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).max())


REGRESSION_PRESETS: dict[str, Callable] = {
    'mae':       _mae,
    'mse':       _mse,
    'rmse':      _rmse,
    'mape':      _mape,
    'smape':     _smape,
    'r2':        _r2,
    'medae':     _medae,
    'max_error': _max_error,
}

_REG_DEFAULT_METRICS = ['mae', 'rmse', 'r2', 'mape', 'medae']

# ── RegressionEvaluator ────────────────────────────────────────────────────────

class RegressionEvaluator(BaseEvaluator):
    """Metrics and visualisations for regression models.

    Usage::

        ev = RegressionEvaluator()
        ev.add('train', y_true_train, y_pred_train)
        ev.add('valid', y_true_valid, y_pred_valid)
        ev.add('test',  y_true_test,  y_pred_test)

        ev.add_default_metrics()
        ev.add_metric('max_error')
        ev.add_metric(lambda yt, yp: float(np.percentile(np.abs(yt - yp), 90)),
                      name='p90_abs_error')

        print(ev.metrics())
        ev.plot_actual_vs_predicted()
        ev.report('report.html')
    """

    _AVAILABLE_PRESETS = REGRESSION_PRESETS
    _DEFAULT_METRIC_NAMES = _REG_DEFAULT_METRICS

    # ── Plots ─────────────────────────────────────────────────────────────────

    def plot_actual_vs_predicted(
        self,
        splits: list[str] | None = None,
        axes: list[Axes] | None = None,
        path: SavePath | None = None,
    ) -> None:
        """Scatter of actual vs predicted with identity line — one panel per split.

        Pass axes=[ax1, ax2, ...] to draw into existing Axes.
        """
        split_names = self._resolve_splits(splits)
        fig, ax_list, created = self._prepare_axes_grid(
            axes, len(split_names), figsize=(5 * len(split_names), 5)
        )
        for ax_, sname, color in zip(ax_list, split_names, self._palette(len(split_names)), strict=False):
            y, p = self._splits[sname]
            ax_.scatter(y, p, alpha=0.3, s=12, color=color)
            lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
            ax_.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
            mae = float(np.abs(y - p).mean())
            ax_.set_title(f'{sname}  (MAE={mae:.4g})')
            ax_.set_xlabel('Actual')
            ax_.set_ylabel('Predicted')
        if created:
            fig.suptitle('Actual vs Predicted')
        self._finalize(fig, path, created, tight=True)

    def plot_residuals_distribution(
        self,
        splits: list[str] | None = None,
        axes: list[Axes] | None = None,
        path: SavePath | None = None,
    ) -> None:
        """Histogram of residuals (actual − predicted) — one panel per split.

        Pass axes=[ax1, ax2, ...] to draw into existing Axes.
        """
        split_names = self._resolve_splits(splits)
        fig, ax_list, created = self._prepare_axes_grid(
            axes, len(split_names), figsize=(5 * len(split_names), 4)
        )
        for ax_, sname, color in zip(ax_list, split_names, self._palette(len(split_names)), strict=False):
            y, p = self._splits[sname]
            residuals = y - p
            ax_.hist(residuals, bins=40, color=color, alpha=0.7, density=True)
            ax_.axvline(0, color='k', linestyle='--', lw=0.8)
            ax_.set_title(f'{sname}  (bias={residuals.mean():.4g})')
            ax_.set_xlabel('Residual (actual − predicted)')
        if created:
            fig.suptitle('Residuals Distribution')
        self._finalize(fig, path, created, tight=True)

    def plot_residuals_vs_predicted(
        self,
        splits: list[str] | None = None,
        axes: list[Axes] | None = None,
        path: SavePath | None = None,
    ) -> None:
        """Scatter of residuals vs predicted — reveals heteroscedasticity.

        Pass axes=[ax1, ax2, ...] to draw into existing Axes.
        """
        split_names = self._resolve_splits(splits)
        fig, ax_list, created = self._prepare_axes_grid(
            axes, len(split_names), figsize=(5 * len(split_names), 4)
        )
        for ax_, sname, color in zip(ax_list, split_names, self._palette(len(split_names)), strict=False):
            y, p = self._splits[sname]
            residuals = y - p
            ax_.scatter(p, residuals, alpha=0.3, s=12, color=color)
            ax_.axhline(0, color='k', linestyle='--', lw=0.8)
            ax_.set_title(sname)
            ax_.set_xlabel('Predicted')
            ax_.set_ylabel('Residual')
        if created:
            fig.suptitle('Residuals vs Predicted')
        self._finalize(fig, path, created, tight=True)

    def plot_error_percentile(
        self, splits: list[str] | None = None, ax: Axes | None = None, path: SavePath | None = None
    ) -> None:
        """Sorted absolute errors — shows where the model is worst.

        Pass ax= to draw on an existing Axes.
        """
        split_names = self._resolve_splits(splits)
        fig, ax_, created = self._prepare_ax(ax, figsize=(8, 5))
        for sname, color in zip(split_names, self._palette(len(split_names)), strict=False):
            y, p = self._splits[sname]
            abs_err = np.sort(np.abs(y - p))
            pcts = np.arange(1, len(abs_err) + 1) / len(abs_err)
            ax_.plot(pcts, abs_err, color=color, label=sname)
        ax_.set_xlabel('Percentile of samples')
        ax_.set_ylabel('Absolute error')
        ax_.set_title('Absolute Error by Percentile')
        ax_.legend()
        self._finalize(fig, path, created)

    def plot_prediction_error_bins(
        self,
        split: str,
        n_bins: int = 10,
        ax: Axes | None = None,
        path: SavePath | None = None,
    ) -> None:
        """Mean absolute error per quantile bin of actual values.

        Shows whether the model is systematically worse for high/low actuals.
        Pass ax= to draw on an existing Axes.
        """
        y, p = self._splits[split]
        quantiles = np.quantile(y, np.linspace(0, 1, n_bins + 1))
        mae_per_bin, centers = [], []
        for i in range(n_bins):
            lo, hi = quantiles[i], quantiles[i + 1]
            mask = (y >= lo) & (y <= hi) if i == n_bins - 1 else (y >= lo) & (y < hi)
            if mask.sum() == 0:
                continue
            mae_per_bin.append(float(np.abs(y[mask] - p[mask]).mean()))
            centers.append(float((lo + hi) / 2))

        fig, ax_, created = self._prepare_ax(ax, figsize=(8, 4))
        ax_.bar(range(len(centers)), mae_per_bin, color='steelblue')
        ax_.set_xticks(range(len(centers)))
        ax_.set_xticklabels([f'{c:.3g}' for c in centers], rotation=45, ha='right')
        ax_.set_xlabel('Bin center (actual value)')
        ax_.set_ylabel('MAE')
        ax_.set_title(f'MAE per Actual-Value Bin — {split}')
        self._finalize(fig, path, created)

    # ── HTML report ───────────────────────────────────────────────────────────

    def report(self, path: str) -> None:
        """Save a self-contained HTML report with metrics table and all plots."""
        split_names = list(self._splits)
        sections = []

        if self._metrics and split_names:
            sections.append(self._metrics_html())

        plot_calls: list[tuple[str, dict]] = [
            ('plot_actual_vs_predicted', {}),
            ('plot_residuals_distribution', {}),
            ('plot_residuals_vs_predicted', {}),
            ('plot_error_percentile', {}),
        ]
        for sname in split_names:
            plot_calls.append(('plot_prediction_error_bins', {'split': sname}))

        for method_name, kwargs in plot_calls:
            section = self._plot_to_section(method_name, kwargs)
            if section is not None:
                sections.append(section)

        html = self._render_html(sections, 'Regression Evaluation Report')
        with Path(path).open('w', encoding='utf-8') as f:
            f.write(html)
        logger.info('Report saved to %s', path)
