"""Shared base for evaluation classes."""
from __future__ import annotations

import base64
from collections.abc import Callable
import io
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseEvaluator:
    """Shared split/metric registry and computation logic.

    Subclasses define:
        _AVAILABLE_PRESETS: dict[str, Callable]  — preset name → metric fn
        _DEFAULT_METRIC_NAMES: list[str]         — names added by add_default_metrics()
    """

    _AVAILABLE_PRESETS: dict[str, Callable] = {}
    _DEFAULT_METRIC_NAMES: list[str] = []

    def __init__(self) -> None:
        self._splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._metrics: dict[str, Callable] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def add(self, name: str, y_true: Any, y_second: Any) -> BaseEvaluator:
        """Register a split. y_second is y_proba (cls) or y_pred (reg). Returns self."""
        self._splits[name] = (np.asarray(y_true), np.asarray(y_second))
        return self

    def add_metric(
        self,
        name_or_fn: str | Callable,
        name: str | None = None,
    ) -> BaseEvaluator:
        """Register one metric.

        Args:
            name_or_fn: Preset name (str) or callable (y_true, y_second) → float.
            name: Display name; required when name_or_fn is callable.

        """
        if isinstance(name_or_fn, str):
            if name_or_fn not in self._AVAILABLE_PRESETS:
                raise ValueError(
                    f'Unknown preset {name_or_fn!r}. '
                    f'Available: {sorted(self._AVAILABLE_PRESETS)}'
                )
            self._metrics[name_or_fn] = self._AVAILABLE_PRESETS[name_or_fn]
        elif callable(name_or_fn):
            if name is None:
                raise ValueError('name= is required when passing a callable')
            self._metrics[name] = name_or_fn
        else:
            raise TypeError(f'Expected str or callable, got {type(name_or_fn).__name__!r}')
        return self

    def add_metrics(self, metrics: dict[str, str | Callable]) -> BaseEvaluator:
        """Register multiple metrics from {display_name: preset_str_or_callable}."""
        for k, v in metrics.items():
            if isinstance(v, str):
                if v not in self._AVAILABLE_PRESETS:
                    raise ValueError(f'Unknown preset {v!r} for metric {k!r}')
                self._metrics[k] = self._AVAILABLE_PRESETS[v]
            elif callable(v):
                self._metrics[k] = v
            else:
                raise TypeError(
                    f'Metric {k!r}: expected str or callable, got {type(v).__name__!r}'
                )
        return self

    def add_default_metrics(self) -> BaseEvaluator:
        """Add the class-defined default metric set."""
        for name in self._DEFAULT_METRIC_NAMES:
            self._metrics[name] = self._AVAILABLE_PRESETS[name]
        return self

    # ── Computation ───────────────────────────────────────────────────────────

    def _resolve_splits(self, splits: list[str] | None) -> list[str]:
        if splits is None:
            return list(self._splits)
        missing = [s for s in splits if s not in self._splits]
        if missing:
            raise ValueError(f'Unknown splits: {missing}. Registered: {list(self._splits)}')
        return list(splits)

    def _resolve_metric_names(self, metrics: list[str] | None) -> list[str]:
        if metrics is None:
            return list(self._metrics)
        missing = [m for m in metrics if m not in self._metrics]
        if missing:
            raise ValueError(f'Unknown metrics: {missing}. Registered: {list(self._metrics)}')
        return list(metrics)

    def _compute(self, metric_name: str, split_name: str) -> float:
        fn = self._metrics[metric_name]
        y_true, y_second = self._splits[split_name]
        try:
            return float(fn(y_true, y_second))
        except Exception as exc:
            logger.debug('Metric %r on split %r failed: %s', metric_name, split_name, exc)
            return float('nan')

    def metrics(
        self,
        splits: list[str] | None = None,
        metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute metrics table. Rows = metrics, columns = splits."""
        split_names = self._resolve_splits(splits)
        metric_names = self._resolve_metric_names(metrics)
        data = {
            split: [self._compute(m, split) for m in metric_names]
            for split in split_names
        }
        return pd.DataFrame(data, index=metric_names)

    def compare_splits(self, ref: str, target: str) -> pd.DataFrame:
        """Compare all metrics: metric | ref | target | delta | ratio."""
        metric_names = list(self._metrics)
        ref_vals = [self._compute(m, ref) for m in metric_names]
        tgt_vals = [self._compute(m, target) for m in metric_names]
        df = pd.DataFrame({ref: ref_vals, target: tgt_vals}, index=metric_names)
        df['delta'] = df[target] - df[ref]
        df['ratio'] = df[target] / df[ref].replace(0.0, np.nan)
        return df

    # ── Plot helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_ax(
        ax: Any | None,
        figsize: tuple[float, float],
    ) -> tuple[Any, Any, bool]:
        """Return (fig_or_None, ax, created_by_us).

        If caller passed an ax, return it as-is and mark created=False so
        _finalize() becomes a no-op. Otherwise create a new figure.
        """
        import matplotlib.pyplot as plt
        if ax is not None:
            return None, ax, False
        fig, axis = plt.subplots(figsize=figsize)
        return fig, axis, True

    @staticmethod
    def _prepare_axes_grid(
        axes: list | None,
        n: int,
        figsize: tuple[float, float],
    ) -> tuple[Any, list, bool]:
        """Return (fig_or_None, axes_list, created_by_us) for n-panel layouts."""
        import matplotlib.pyplot as plt
        if axes is not None:
            if len(axes) < n:
                raise ValueError(f'Expected at least {n} axes, got {len(axes)}')
            return None, list(axes[:n]), False
        fig, ax_arr = plt.subplots(1, n, figsize=figsize, squeeze=False)
        return fig, list(ax_arr[0]), True

    @staticmethod
    def _finalize(
        fig: Any,
        path: Any,
        created: bool,
        tight: bool = False,
    ) -> None:
        """Save/show only if we created the figure; no-op when caller owns it."""
        import matplotlib.pyplot as plt
        if not created:
            return
        if tight:
            plt.tight_layout()
        if path is not None:
            fig.savefig(path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close(fig)

    @staticmethod
    def _palette(n: int) -> list:
        import matplotlib.pyplot as plt
        return [plt.get_cmap('tab10')(i % 10) for i in range(n)]

    # ── Bootstrap ─────────────────────────────────────────────────────────────

    def _bootstrap_samples(
        self,
        split: str,
        n_iter: int,
        metric_names: list[str],
        seed: int | None,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        y_true, y_second = self._splits[split]
        n = len(y_true)
        samples: dict[str, np.ndarray] = {
            m: np.full(n_iter, np.nan) for m in metric_names
        }
        for i in range(n_iter):
            idx = rng.integers(0, n, size=n)
            y_b, s_b = y_true[idx], y_second[idx]
            for m in metric_names:
                try:
                    samples[m][i] = float(self._metrics[m](y_b, s_b))
                except Exception:
                    pass
        return samples

    def bootstrap_metrics(
        self,
        split: str,
        n_iter: int = 1000,
        ci: float = 0.95,
        metrics: list[str] | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Bootstrap confidence intervals for all registered metrics on a split.

        Returns DataFrame: index=metric, columns=[mean, std, ci_low, ci_high].
        """
        metric_names = self._resolve_metric_names(metrics)
        samples = self._bootstrap_samples(split, n_iter, metric_names, seed)
        alpha = (1 - ci) / 2
        rows: dict[str, dict] = {}
        for m, vals in samples.items():
            valid = vals[~np.isnan(vals)]
            rows[m] = {
                'mean':    float(np.nanmean(vals)),
                'std':     float(np.nanstd(vals)),
                'ci_low':  float(np.percentile(valid, alpha * 100)) if len(valid) else np.nan,
                'ci_high': float(np.percentile(valid, (1 - alpha) * 100)) if len(valid) else np.nan,
            }
        return pd.DataFrame(rows).T

    def _draw_ci_panel(
        self,
        ax_: Any,
        df: pd.DataFrame,
        split: str,
        show_point_estimate: bool,
    ) -> None:
        """Draw CI bars for a subset of metrics onto an existing Axes."""
        y_pos = np.arange(len(df))
        ax_.barh(y_pos, df['mean'], height=0.55,
                 color='steelblue', alpha=0.70, label='Bootstrap mean')
        ax_.errorbar(
            df['mean'], y_pos,
            xerr=[df['mean'] - df['ci_low'], df['ci_high'] - df['mean']],
            fmt='none', color='#222222', capsize=5, lw=1.8, capthick=1.5,
        )
        if show_point_estimate:
            point_vals = [self._compute(m, split) for m in df.index]
            ax_.scatter(point_vals, y_pos, color='crimson', s=45,
                        zorder=5, marker='D', label='Point estimate')
        ax_.set_yticks(y_pos)
        ax_.set_yticklabels(df.index, fontsize=9)
        ax_.invert_yaxis()
        ax_.set_xlabel('Metric value')
        ax_.legend(fontsize=8)

    def plot_bootstrap_ci(
        self,
        split: str,
        n_iter: int = 1000,
        ci: float = 0.95,
        metrics: list[str] | None = None,
        seed: int | None = None,
        show_point_estimate: bool = True,
        ax: Any = None,
        path: Any = None,
    ) -> None:
        """Horizontal CI bars for each metric (bootstrap). Pass ax= for composition.

        When ax= is not provided and metrics span very different scales (e.g. a lift
        metric ~4–5 mixed with 0–1 metrics), the plot automatically splits into two
        side-by-side panels so neither group is compressed. Metrics with |mean| > 2
        are treated as 'large-scale'.

        Args:
            split:               Split name.
            n_iter:              Bootstrap iterations.
            ci:                  Confidence level (0–1).
            metrics:             Subset of registered metrics (default: all).
            seed:                Random seed for reproducibility.
            show_point_estimate: Overlay a diamond marker at the point estimate.
            ax:                  Existing Axes (optional); disables auto-split.
            path:                Output path (optional).

        """
        import matplotlib.pyplot as plt

        metric_names = self._resolve_metric_names(metrics)
        df = self.bootstrap_metrics(split, n_iter, ci, metric_names, seed)

        # Auto-split by scale only when we own the figure
        if ax is None:
            small = df[df['mean'].abs() <= 2.0]
            large = df[df['mean'].abs() > 2.0]

            if not small.empty and not large.empty:
                total = len(df)
                fig, (ax_s, ax_l) = plt.subplots(
                    1, 2, figsize=(14, max(4, total * 0.55)),
                )
                self._draw_ci_panel(ax_s, small, split, show_point_estimate)
                self._draw_ci_panel(ax_l, large, split, show_point_estimate)
                ax_s.set_title(f'Bootstrap {ci:.0%} CI  (n={n_iter}) — {split}')
                ax_l.set_title(f'Large-scale metrics — {split}')
                self._finalize(fig, path, True, tight=True)
                return

            fig, ax_, created = self._prepare_ax(None, figsize=(9, max(4, len(df) * 0.55)))
        else:
            fig, ax_, created = self._prepare_ax(ax, figsize=(9, max(4, len(df) * 0.55)))

        self._draw_ci_panel(ax_, df, split, show_point_estimate)
        ax_.set_title(f'Bootstrap {ci:.0%} CI  (n={n_iter}) — {split}')
        self._finalize(fig, path, created)

    def plot_bootstrap_distributions(
        self,
        split: str,
        metrics: list[str] | None = None,
        n_iter: int = 1000,
        ci: float = 0.95,
        seed: int | None = None,
        axes: list | None = None,
        path: Any = None,
    ) -> None:
        """Grid of bootstrap histograms — one panel per metric.

        Shows distribution, CI band, bootstrap mean (navy), and point estimate
        on the full sample (crimson dashed).

        Pass axes= (flat list, len ≥ n_metrics) to draw into existing Axes.
        """
        import matplotlib.pyplot as plt

        metric_names = self._resolve_metric_names(metrics)
        n_m = len(metric_names)
        samples = self._bootstrap_samples(split, n_iter, metric_names, seed)
        alpha = (1 - ci) / 2

        if axes is not None:
            if len(axes) < n_m:
                raise ValueError(f'Expected at least {n_m} axes, got {len(axes)}')
            fig, ax_list, created = None, list(axes[:n_m]), False
        else:
            ncols = min(3, n_m)
            nrows = max(1, (n_m + ncols - 1) // ncols)
            fig, ax_arr = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False
            )
            ax_flat = [ax_arr[r][c] for r in range(nrows) for c in range(ncols)]
            for i in range(n_m, len(ax_flat)):
                ax_flat[i].set_visible(False)
            ax_list, created = ax_flat[:n_m], True

        for ax_, m in zip(ax_list, metric_names):
            vals = samples[m]
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                ax_.set_title(f'{m} (all NaN)')
                continue
            ci_lo = float(np.percentile(valid, alpha * 100))
            ci_hi = float(np.percentile(valid, (1 - alpha) * 100))
            mu = float(np.mean(valid))
            point_est = self._compute(m, split)

            ax_.hist(valid, bins=40, color='steelblue', alpha=0.60, density=True)
            ax_.axvspan(ci_lo, ci_hi, alpha=0.12, color='steelblue')
            ax_.axvline(ci_lo, color='steelblue', lw=1.0, linestyle='--')
            ax_.axvline(ci_hi, color='steelblue', lw=1.0, linestyle='--')
            ax_.axvline(mu, color='navy', lw=1.8, label=f'mean={mu:.4f}')
            ax_.axvline(point_est, color='crimson', lw=1.5, linestyle=':',
                        label=f'point={point_est:.4f}')
            ax_.set_title(m, fontsize=9)
            ax_.legend(fontsize=6.5)
            ax_.set_xlabel('Value', fontsize=8)

        if created:
            fig.suptitle(
                f'Bootstrap Distributions  (n={n_iter}, {ci:.0%} CI) — {split}', y=1.01
            )
        self._finalize(fig, path, created, tight=True)

    # ── HTML report helpers ───────────────────────────────────────────────────

    def _metrics_html(self) -> str:
        try:
            return (
                '<h2>Metrics</h2>'
                + self.metrics().to_html(float_format='%.4f', classes='t')
            )
        except Exception as exc:
            return f'<p>Metrics table failed: {exc}</p>'

    def _plot_to_section(self, method_name: str, kwargs: dict) -> str | None:
        try:
            buf = io.BytesIO()
            getattr(self, method_name)(**kwargs, path=buf)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode()
            title = method_name.replace('plot_', '').replace('_', ' ').title()
            extra = ', '.join(f'{k}={v!r}' for k, v in kwargs.items())
            heading = f'{title} — {extra}' if extra else title
            return (
                f'<h2>{heading}</h2>'
                f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%"/>'
            )
        except Exception as exc:
            logger.debug('%s(%s) skipped in report: %s', method_name, kwargs, exc)
            return None

    @staticmethod
    def _render_html(sections: list[str], title: str) -> str:
        return (
            '<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<title>{title}</title>'
            '<style>'
            'body{font-family:sans-serif;margin:2em;max-width:1200px}'
            'h2{margin-top:2em;border-bottom:1px solid #ccc}'
            'table.t{border-collapse:collapse}'
            'table.t td,table.t th{border:1px solid #ccc;padding:4px 10px;text-align:right}'
            'table.t th{background:#f5f5f5;text-align:center}'
            '</style></head><body>'
            + ''.join(f'<section>{s}</section>' for s in sections)
            + '</body></html>'
        )
