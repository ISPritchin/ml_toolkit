"""Classification evaluator — metrics, threshold analysis, visualisations."""
from __future__ import annotations

from collections.abc import Callable
import functools
from typing import Any

import numpy as np
import pandas as pd

from ._base import BaseEvaluator, logger

# ── Preset metric functions ────────────────────────────────────────────────────

def _roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if y_proba.ndim == 1:
        return float(roc_auc_score(y_true, y_proba))
    return float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'))


def _pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if y_proba.ndim == 1:
        return float(average_precision_score(y_true, y_proba))
    return float(average_precision_score(y_true, y_proba, average='macro'))


def _log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from sklearn.metrics import log_loss
    return float(log_loss(y_true, y_proba))


def _brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from sklearn.metrics import brier_score_loss
    if y_proba.ndim == 1:
        return float(brier_score_loss(y_true, y_proba))
    classes = np.unique(y_true)
    return float(np.mean([
        brier_score_loss((y_true == cls).astype(int), y_proba[:, k])
        for k, cls in enumerate(classes)
    ]))


def _ks(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if y_proba.ndim > 1:
        raise ValueError('KS statistic is only defined for binary classification')
    pos = np.sort(y_proba[y_true == 1])
    neg = np.sort(y_proba[y_true == 0])
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    all_t = np.sort(np.unique(y_proba))
    cdf_pos = np.searchsorted(pos, all_t, side='right') / len(pos)
    cdf_neg = np.searchsorted(neg, all_t, side='right') / len(neg)
    return float(np.abs(cdf_pos - cdf_neg).max())


def _gini(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return 2.0 * _roc_auc(y_true, y_proba) - 1.0


def _mcc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from sklearn.metrics import matthews_corrcoef
    pred = y_proba.argmax(axis=1) if y_proba.ndim > 1 else (y_proba >= 0.5).astype(int)
    return float(matthews_corrcoef(y_true, pred))


def _ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    if y_proba.ndim > 1:
        classes = np.unique(y_true)
        return float(np.mean([
            _ece((y_true == cls).astype(int), y_proba[:, k], n_bins)
            for k, cls in enumerate(classes)
        ]))
    p, y = y_proba, y_true
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total, n = 0.0, len(y)
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1])
        if i == n_bins - 1:
            mask |= p == 1.0
        if mask.sum() == 0:
            continue
        total += mask.sum() / n * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def _accuracy(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from sklearn.metrics import accuracy_score
    pred = y_proba.argmax(axis=1) if y_proba.ndim > 1 else (y_proba >= 0.5).astype(int)
    return float(accuracy_score(y_true, pred))


def _balanced_accuracy(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    pred = y_proba.argmax(axis=1) if y_proba.ndim > 1 else (y_proba >= 0.5).astype(int)
    return float(balanced_accuracy_score(y_true, pred))


def _f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    if y_proba.ndim > 1:
        return float(f1_score(y_true, y_proba.argmax(axis=1), average='macro', zero_division=0))
    return float(f1_score(y_true, (y_proba >= 0.5).astype(int), zero_division=0))


def _precision(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from sklearn.metrics import precision_score
    if y_proba.ndim > 1:
        return float(precision_score(y_true, y_proba.argmax(axis=1), average='macro', zero_division=0))
    return float(precision_score(y_true, (y_proba >= 0.5).astype(int), zero_division=0))


def _recall(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from sklearn.metrics import recall_score
    if y_proba.ndim > 1:
        return float(recall_score(y_true, y_proba.argmax(axis=1), average='macro', zero_division=0))
    return float(recall_score(y_true, (y_proba >= 0.5).astype(int), zero_division=0))


def _cohen_kappa(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score
    pred = y_proba.argmax(axis=1) if y_proba.ndim > 1 else (y_proba >= 0.5).astype(int)
    return float(cohen_kappa_score(y_true, pred))


CLASSIFICATION_PRESETS: dict[str, Callable] = {
    'roc_auc':           _roc_auc,
    'pr_auc':            _pr_auc,
    'log_loss':          _log_loss,
    'brier':             _brier,
    'ks':                _ks,
    'gini':              _gini,
    'mcc':               _mcc,
    'ece':               _ece,
    'accuracy':          _accuracy,
    'balanced_accuracy': _balanced_accuracy,
    'f1':                _f1,
    'precision':         _precision,
    'recall':            _recall,
    'cohen_kappa':       _cohen_kappa,
}

_CLS_DEFAULT_METRICS = [
    'roc_auc', 'pr_auc', 'log_loss', 'brier', 'ks', 'gini', 'mcc', 'ece',
]

# ── Factory functions ──────────────────────────────────────────────────────────

def precision_at_k(k: float) -> Callable:
    """Returns (y_true, y_proba) → precision in top-k predictions.

    Args:
        k: Number of objects (int) or fraction of sample (float ∈ (0, 1]).

    """
    from ml_toolkit.models._utils import precision_at_k as _pak
    return functools.partial(_pak, k=k)


def recall_at_k(k: float) -> Callable:
    """Returns (y_true, y_proba) → recall in top-k predictions."""
    from ml_toolkit.models._utils import recall_at_k as _rak
    return functools.partial(_rak, k=k)


def lift_at_k(k: float) -> Callable:
    """Returns (y_true, y_proba) → lift = precision@k / base_rate."""
    from ml_toolkit.models._utils import precision_at_k as _pak

    def _fn(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        base = np.asarray(y_true).mean()
        return _pak(y_true, y_proba, k=k) / base if base > 0 else 0.0

    return _fn


def f1_at_threshold(t: float) -> Callable:
    """Returns (y_true, y_proba) → F1 at fixed decision threshold t."""
    from sklearn.metrics import f1_score as _f1s

    def _fn(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        return float(_f1s(y_true, (np.asarray(y_proba) >= t).astype(int), zero_division=0))

    return _fn


# ── ClassificationEvaluator ────────────────────────────────────────────────────

class ClassificationEvaluator(BaseEvaluator):
    """Metrics and visualisations for classification models.

    Usage::

        ev = ClassificationEvaluator(task='binary')
        ev.add('train', y_true_train, y_proba_train)
        ev.add('valid', y_true_valid, y_proba_valid)
        ev.add('test',  y_true_test,  y_proba_test)

        ev.add_default_metrics()
        ev.add_metric(precision_at_k(0.10), name='precision@10%')
        ev.add_metric(precision_at_k(0.20), name='precision@20%')

        print(ev.metrics(splits=['valid', 'test']))
        ev.plot_roc(splits=['valid', 'test'])
        ev.report('report.html')
    """

    _AVAILABLE_PRESETS = CLASSIFICATION_PRESETS
    _DEFAULT_METRIC_NAMES = _CLS_DEFAULT_METRICS

    def __init__(self, task: str = 'binary') -> None:
        if task not in ('binary', 'multiclass'):
            raise ValueError(f"task must be 'binary' or 'multiclass', got {task!r}")
        super().__init__()
        self._task = task

    # ── Threshold & stability analysis ────────────────────────────────────────

    def psi(
        self, ref: str, target: str, n_bins: int = 10
    ) -> tuple[float, pd.DataFrame]:
        """Population Stability Index between two splits (binary only).

        Returns:
            (total_psi, bin_df) — bin_df columns: bin, {ref}_pct, {target}_pct, psi.

        """
        self._require_binary('psi')
        _, p_ref = self._splits[ref]
        _, p_tgt = self._splits[target]

        edges = np.linspace(0.0, 1.0, n_bins + 1)
        edges[0], edges[-1] = -np.inf, np.inf
        ref_cnt = np.histogram(p_ref, bins=edges)[0]
        tgt_cnt = np.histogram(p_tgt, bins=edges)[0]
        ref_pct = ref_cnt / ref_cnt.sum()
        tgt_pct = tgt_cnt / tgt_cnt.sum()

        eps = 1e-8
        psi_vals = (tgt_pct - ref_pct) * np.log((tgt_pct + eps) / (ref_pct + eps))

        real_edges = np.linspace(0.0, 1.0, n_bins + 1)
        labels = [f'[{real_edges[i]:.2f},{real_edges[i+1]:.2f})' for i in range(n_bins)]
        df = pd.DataFrame({
            'bin': labels,
            f'{ref}_pct': ref_pct,
            f'{target}_pct': tgt_pct,
            'psi': psi_vals,
        })
        return float(psi_vals.sum()), df

    def threshold_scan(self, split: str, n_points: int = 200) -> pd.DataFrame:
        """Scan precision/recall/f1/accuracy/specificity across thresholds (binary only)."""
        self._require_binary('threshold_scan')
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )
        y, p = self._splits[split]
        thresholds = np.linspace(0.01, 0.99, n_points)
        n_neg = (y == 0).sum()
        rows = []
        for t in thresholds:
            pred = (p >= t).astype(int)
            tn = int(((y == 0) & (pred == 0)).sum())
            rows.append({
                'threshold':   float(t),
                'precision':   float(precision_score(y, pred, zero_division=0)),
                'recall':      float(recall_score(y, pred, zero_division=0)),
                'f1':          float(f1_score(y, pred, zero_division=0)),
                'accuracy':    float(accuracy_score(y, pred)),
                'specificity': float(tn / n_neg) if n_neg > 0 else float('nan'),
            })
        return pd.DataFrame(rows)

    def best_threshold(
        self, metric: str = 'f1', split: str = 'valid'
    ) -> dict[str, float]:
        """Threshold that maximises metric on the given split (binary only)."""
        df = self.threshold_scan(split)
        if metric not in df.columns:
            raise ValueError(f'metric must be one of {list(df.columns[1:])}, got {metric!r}')
        row = df.iloc[df[metric].idxmax()]
        return {col: float(row[col]) for col in df.columns}

    # ── Plot helpers ──────────────────────────────────────────────────────────

    def _require_binary(self, method: str) -> None:
        if self._task != 'binary':
            raise ValueError(f'{method}() is only available for task="binary"')

    # ── Plots ─────────────────────────────────────────────────────────────────

    def plot_roc(
        self, splits: list[str] | None = None, ax: Any = None, path: Any = None
    ) -> None:
        """ROC curve for each split. Pass ax= to draw on an existing Axes."""
        from sklearn.metrics import roc_auc_score, roc_curve

        split_names = self._resolve_splits(splits)
        fig, ax_, created = self._prepare_ax(ax, figsize=(7, 6))
        for sname, color in zip(split_names, self._palette(len(split_names))):
            y, p = self._splits[sname]
            p1 = p if p.ndim == 1 else p[:, 1]
            fpr, tpr, _ = roc_curve(y, p1)
            ax_.plot(fpr, tpr, color=color, label=f'{sname} (AUC={roc_auc_score(y, p1):.3f})')
        ax_.plot([0, 1], [0, 1], 'k--', lw=0.8)
        ax_.set_xlabel('False Positive Rate')
        ax_.set_ylabel('True Positive Rate')
        ax_.set_title('ROC Curve')
        ax_.legend()
        self._finalize(fig, path, created)

    def plot_pr(
        self, splits: list[str] | None = None, ax: Any = None, path: Any = None
    ) -> None:
        """Precision-Recall curve for each split. Pass ax= to draw on an existing Axes."""
        from sklearn.metrics import average_precision_score, precision_recall_curve

        split_names = self._resolve_splits(splits)
        fig, ax_, created = self._prepare_ax(ax, figsize=(7, 6))
        for sname, color in zip(split_names, self._palette(len(split_names))):
            y, p = self._splits[sname]
            p1 = p if p.ndim == 1 else p[:, 1]
            prec, rec, _ = precision_recall_curve(y, p1)
            ax_.plot(rec, prec, color=color,
                     label=f'{sname} (AP={average_precision_score(y, p1):.3f})')
        baseline = np.asarray(self._splits[split_names[0]][0]).mean()
        ax_.axhline(baseline, color='k', linestyle='--', lw=0.8,
                    label=f'baseline ({baseline:.3f})')
        ax_.set_xlabel('Recall')
        ax_.set_ylabel('Precision')
        ax_.set_title('Precision-Recall Curve')
        ax_.legend()
        self._finalize(fig, path, created)

    def plot_score_distribution(
        self,
        splits: list[str] | None = None,
        axes: list | None = None,
        path: Any = None,
    ) -> None:
        """Histogram of scores by class — one panel per split.

        Pass axes=[ax1, ax2, ...] to draw into existing Axes (one per split).
        """
        split_names = self._resolve_splits(splits)
        fig, ax_list, created = self._prepare_axes_grid(
            axes, len(split_names), figsize=(5 * len(split_names), 4)
        )
        for ax_, sname in zip(ax_list, split_names):
            y, p = self._splits[sname]
            p1 = p if p.ndim == 1 else p[:, 1]
            for cls, label in [(0, 'Negative'), (1, 'Positive')]:
                mask = y == cls
                if mask.sum() > 0:
                    ax_.hist(p1[mask], bins=30, alpha=0.55, density=True, label=label)
            ax_.set_title(sname)
            ax_.set_xlabel('Score')
            ax_.legend()
        if created:
            fig.suptitle('Score Distribution by Class')
        self._finalize(fig, path, created, tight=True)

    def plot_score_cdf(
        self,
        splits: list[str] | None = None,
        axes: list | None = None,
        path: Any = None,
    ) -> None:
        """CDF of scores by class — one panel per split.

        Maximum vertical gap between curves equals the KS statistic.
        Pass axes=[ax1, ax2, ...] to draw into existing Axes.
        """
        split_names = self._resolve_splits(splits)
        fig, ax_list, created = self._prepare_axes_grid(
            axes, len(split_names), figsize=(5 * len(split_names), 4)
        )
        for ax_, sname in zip(ax_list, split_names):
            y, p = self._splits[sname]
            p1 = p if p.ndim == 1 else p[:, 1]
            for cls, label, ls in [(0, 'Negative', '-'), (1, 'Positive', '--')]:
                mask = y == cls
                if mask.sum() == 0:
                    continue
                sp = np.sort(p1[mask])
                ax_.plot(sp, np.arange(1, len(sp) + 1) / len(sp), linestyle=ls, label=label)
            ax_.set_title(sname)
            ax_.set_xlabel('Score')
            ax_.set_ylabel('CDF')
            ax_.legend()
        if created:
            fig.suptitle('Score CDF by Class')
        self._finalize(fig, path, created, tight=True)

    def plot_calibration(
        self,
        splits: list[str] | None = None,
        n_bins: int = 10,
        ax: Any = None,
        path: Any = None,
    ) -> None:
        """Reliability diagram for each split. Pass ax= to draw on an existing Axes."""
        split_names = self._resolve_splits(splits)
        fig, ax_, created = self._prepare_ax(ax, figsize=(7, 6))
        ax_.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Perfect calibration')
        for sname, color in zip(split_names, self._palette(len(split_names))):
            y, p = self._splits[sname]
            p1 = p if p.ndim == 1 else p[:, 1]
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            means, fracs = [], []
            for i in range(n_bins):
                mask = (p1 >= bins[i]) & (p1 < bins[i + 1])
                if i == n_bins - 1:
                    mask |= p1 == 1.0
                if mask.sum() == 0:
                    continue
                means.append(p1[mask].mean())
                fracs.append(y[mask].mean())
            ax_.plot(means, fracs, 'o-', color=color, label=sname)
        ax_.set_xlabel('Mean predicted probability')
        ax_.set_ylabel('Fraction of positives')
        ax_.set_title('Calibration Curve')
        ax_.legend()
        self._finalize(fig, path, created)

    def plot_confusion_matrix(
        self,
        split: str,
        threshold: float = 0.5,
        normalize: str | None = None,
        ax: Any = None,
        path: Any = None,
    ) -> None:
        """Confusion matrix. normalize: 'true' | 'pred' | None.

        Pass ax= to draw on an existing Axes.
        """
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        y, p = self._splits[split]
        pred = p.argmax(axis=1) if p.ndim > 1 else (p >= threshold).astype(int)
        cm = confusion_matrix(y, pred, normalize=normalize)
        fig, ax_, created = self._prepare_ax(ax, figsize=(6, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax_, colorbar=True)
        norm_label = f' normalize={normalize}' if normalize else ''
        ax_.set_title(f'Confusion Matrix — {split}{norm_label}')
        self._finalize(fig, path, created)

    def plot_lift(
        self, splits: list[str] | None = None, ax: Any = None, path: Any = None
    ) -> None:
        """Lift curve (binary only). Pass ax= to draw on an existing Axes."""
        self._require_binary('plot_lift')
        split_names = self._resolve_splits(splits)
        fig, ax_, created = self._prepare_ax(ax, figsize=(7, 5))
        ax_.axhline(1.0, color='k', linestyle='--', lw=0.8, label='Baseline')
        for sname, color in zip(split_names, self._palette(len(split_names))):
            y, p = self._splits[sname]
            base_rate = y.mean()
            if base_rate == 0:
                continue
            idx = np.argsort(p)[::-1]
            y_s = y[idx]
            n = len(y)
            lift = (np.cumsum(y_s) / np.arange(1, n + 1)) / base_rate
            ax_.plot(np.arange(1, n + 1) / n, lift, color=color, label=sname)
        ax_.set_xlabel('Fraction of sample (descending score)')
        ax_.set_ylabel('Lift')
        ax_.set_title('Lift Curve')
        ax_.legend()
        self._finalize(fig, path, created)

    def plot_gains(
        self, splits: list[str] | None = None, ax: Any = None, path: Any = None
    ) -> None:
        """Cumulative gains curve (binary only). Pass ax= to draw on an existing Axes."""
        self._require_binary('plot_gains')
        split_names = self._resolve_splits(splits)
        fig, ax_, created = self._prepare_ax(ax, figsize=(7, 5))
        ax_.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Baseline')
        for sname, color in zip(split_names, self._palette(len(split_names))):
            y, p = self._splits[sname]
            total_pos = y.sum()
            if total_pos == 0:
                continue
            idx = np.argsort(p)[::-1]
            gains = np.cumsum(y[idx]) / total_pos
            ax_.plot(np.arange(1, len(y) + 1) / len(y), gains, color=color, label=sname)
        ax_.set_xlabel('Fraction of sample (descending score)')
        ax_.set_ylabel('Fraction of positives captured')
        ax_.set_title('Cumulative Gains Curve')
        ax_.legend()
        self._finalize(fig, path, created)

    def plot_precision_recall_at_k(
        self,
        splits: list[str] | None = None,
        k_frac: float | list[float] | None = None,
        k_n: int | list[int] | None = None,
        min_precision: float | None = None,
        show_f1: bool = False,
        show_counts_axis: bool = False,
        ax: Any = None,
        path: Any = None,
    ) -> None:
        """Precision and recall curves as the top-k selection grows (binary only).

        Encoding:
          - **Color** = metric: precision (blue), recall (orange-red), F1 (green).
          - **Line style** = split rank: last split (test) → thick solid; second-to-last
            (valid) → dashed; earlier splits → thin dotted.

        Args:
            splits:            Splits to plot (default: all).
            k_frac:            float or list[float] — draw vertical marker(s) at these
                               fractions; annotates (precision, recall) per split.
            k_n:               int or list[int] — same markers specified as absolute counts.
            min_precision:     Draw a horizontal reference line at this level.
                               For each split, marks the maximum k where precision ≥ value.
            show_f1:           Add F1@k curve (green).
            show_counts_axis:  Add a secondary top X-axis with absolute object counts.
            ax:                Existing Axes to draw on (optional).
            path:              Output path (optional).

        """
        from matplotlib.lines import Line2D

        self._require_binary('plot_precision_recall_at_k')
        split_names = self._resolve_splits(splits)
        fig, ax_, created = self._prepare_ax(ax, figsize=(9, 6))

        # Metric colors (fixed, independent of splits)
        C_PREC = '#1565C0'   # deep blue
        C_REC  = '#C62828'   # deep red
        C_F1   = '#2E7D32'   # deep green

        # Line style + width by split rank (last = test = thick solid)
        def _ls(rank: int, n: int) -> tuple[str, float]:
            if rank == n - 1:   return '-',  2.4   # test
            if rank == n - 2:   return '--', 1.6   # valid
            return ':',  1.2                        # train / earlier

        # Normalise scalar markers → lists
        k_fracs: list[float] = (
            [k_frac] if isinstance(k_frac, (int, float)) else list(k_frac or [])
        )
        k_ns: list[int] = (
            [k_n] if isinstance(k_n, int) else list(k_n or [])
        )

        # Store curves for marker annotation: {sname: (frac_range, prec, rec)}
        curves: dict[str, tuple] = {}

        min_prec_line_drawn = False

        for rank, sname in enumerate(split_names):
            y, p = self._splits[sname]
            n = len(y)
            total_pos = int(y.sum())
            if total_pos == 0:
                continue

            idx = np.argsort(p)[::-1]
            cum_pos = np.cumsum(y[idx])
            k_range = np.arange(1, n + 1)
            frac_range = k_range / n

            prec_curve = cum_pos / k_range
            rec_curve  = cum_pos / total_pos

            ls, lw = _ls(rank, len(split_names))
            curves[sname] = (frac_range, prec_curve, rec_curve, n)

            ax_.plot(frac_range, prec_curve, color=C_PREC, lw=lw, linestyle=ls)
            ax_.plot(frac_range, rec_curve,  color=C_REC,  lw=lw, linestyle=ls)

            if show_f1:
                denom = prec_curve + rec_curve
                f1_curve = np.where(denom > 0,
                                    2 * prec_curve * rec_curve / denom, 0.0)
                ax_.plot(frac_range, f1_curve, color=C_F1, lw=lw, linestyle=ls)

            # ── min_precision crossing ─────────────────────────────────────────
            if min_precision is not None:
                if not min_prec_line_drawn:
                    ax_.axhline(min_precision, color='#B71C1C', linestyle='--',
                                lw=1.0, alpha=0.6, zorder=0)
                    ax_.text(0.01, min_precision + 0.015,
                             f'min precision = {min_precision:.2f}',
                             fontsize=7, color='#B71C1C', va='bottom')
                    min_prec_line_drawn = True
                valid_mask = prec_curve >= min_precision
                if valid_mask.any():
                    cross_frac = frac_range[valid_mask][-1]
                    cross_n = int(round(cross_frac * n))
                    ax_.axvline(cross_frac, color='grey', linestyle=ls,
                                lw=lw * 0.7, alpha=0.7, zorder=0)
                    ax_.scatter([cross_frac], [min_precision],
                                color=C_PREC, s=50, zorder=6)
                    # Offset label vertically per split rank to avoid overlap
                    label_y = min_precision - 0.07 - rank * 0.07
                    ax_.annotate(
                        f'{sname}: {cross_frac:.1%} ({cross_n:,})',
                        xy=(cross_frac, min_precision),
                        xytext=(cross_frac + 0.01, max(0.02, label_y)),
                        fontsize=6.5, color='#555555',
                        arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.6),
                    )

        # ── k_frac markers: vertical line + annotation box ────────────────────
        # Boxes are staggered vertically so adjacent markers don't overlap
        for m_idx, kf in enumerate(k_fracs):
            ax_.axvline(kf, color='grey', linestyle=':', lw=0.9, zorder=0)
            lines = [f'k = {kf:.0%}']
            for sname, (fr, pc, rc, n) in curves.items():
                ki = max(0, min(n - 1, int(round(kf * n)) - 1))
                lines.append(f'{sname}: p={pc[ki]:.2f}  r={rc[ki]:.2f}')
            # Alternate box y-position: odd markers go higher
            box_y = 0.62 if m_idx % 2 == 0 else 0.38
            ax_.text(kf + 0.01, box_y, '\n'.join(lines),
                     fontsize=6.5, color='#333333', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white',
                               ec='grey', alpha=0.88))

        # ── k_n markers: vertical line + annotation box ────────────────────────
        for m_idx, kn in enumerate(k_ns):
            _fr, _pc, _rc, first_n = next(iter(curves.values()))
            kf = min(first_n - 1, kn - 1) / first_n
            ax_.axvline(kf, color='grey', linestyle=':', lw=0.9, zorder=0)
            ax_.text(kf + 0.005, 0.02, f'n={kn:,}',
                     fontsize=6.5, color='grey', rotation=90, va='bottom')
            lines = [f'n = {kn:,}']
            for sname, (fr, pc, rc, n) in curves.items():
                ki = min(n - 1, kn - 1)
                lines.append(f'{sname}: p={pc[ki]:.2f}  r={rc[ki]:.2f}')
            box_y = 0.62 if m_idx % 2 == 0 else 0.38
            ax_.text(kf + 0.01, box_y, '\n'.join(lines),
                     fontsize=6.5, color='#333333', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white',
                               ec='grey', alpha=0.88))

        # Base rate
        y0, _ = self._splits[split_names[0]]
        ax_.axhline(float(y0.mean()), color='grey', linestyle=':', lw=0.8, alpha=0.5)
        ax_.text(0.01, float(y0.mean()) + 0.01, f'base rate ({y0.mean():.3f})',
                 fontsize=6.5, color='grey', va='bottom')

        # ── Custom legend (metrics + splits) ──────────────────────────────────
        metric_handles = [
            Line2D([0], [0], color=C_PREC, lw=2.0, label='Precision'),
            Line2D([0], [0], color=C_REC,  lw=2.0, label='Recall'),
        ]
        if show_f1:
            metric_handles.append(Line2D([0], [0], color=C_F1, lw=2.0, label='F1'))

        split_handles = []
        for rank, sname in enumerate(split_names):
            ls_val, lw_val = _ls(rank, len(split_names))
            split_handles.append(
                Line2D([0], [0], color='grey', lw=lw_val, linestyle=ls_val, label=sname)
            )
        # Two-group legend
        ax_.legend(
            handles=metric_handles + split_handles,
            fontsize=8, loc='upper right',
            title='metric  |  split',
            title_fontsize=7,
        )

        ax_.set_xlabel('Fraction of population selected (descending score)')
        ax_.set_ylabel('Precision / Recall')
        ax_.set_title('Precision & Recall at k')
        ax_.set_xlim(0, 1)
        ax_.set_ylim(0, 1.05)

        # Optional secondary top axis: absolute counts
        if show_counts_axis:
            n_ref = len(self._splits[split_names[0]][0])
            ax2 = ax_.twiny()
            ax2.set_xlim(0, n_ref)
            ax2.set_xlabel('Number of records selected', fontsize=9)
            nice_ticks = np.linspace(0, n_ref, 6, dtype=int)
            ax2.set_xticks(nice_ticks)
            ax2.set_xticklabels([f'{t:,}' for t in nice_ticks])

        self._finalize(fig, path, created)

    def plot_decile_bar(
        self, split: str, ax: Any = None, path: Any = None
    ) -> None:
        """Positive rate per score decile. Decile 1 = highest scores (binary only).

        Pass ax= to draw on an existing Axes.
        """
        self._require_binary('plot_decile_bar')
        y, p = self._splits[split]
        idx = np.argsort(p)[::-1]
        n, ds = len(y), max(1, len(y) // 10)
        rates = [
            y[idx[d * ds: (d + 1) * ds if d < 9 else n]].mean()
            for d in range(10)
        ]
        fig, ax_, created = self._prepare_ax(ax, figsize=(8, 4))
        ax_.bar(range(1, 11), rates, color='steelblue')
        ax_.axhline(y.mean(), color='red', linestyle='--', lw=1.0,
                    label=f'Overall rate ({y.mean():.3f})')
        ax_.set_xlabel('Decile (1 = highest score)')
        ax_.set_ylabel('Positive rate')
        ax_.set_title(f'Decile Analysis — {split}')
        ax_.legend()
        self._finalize(fig, path, created)

    def plot_threshold_scan(
        self,
        split: str,
        metrics: list[str] | None = None,
        ax: Any = None,
        path: Any = None,
    ) -> None:
        """Precision/recall/f1 vs threshold with optimal F1 marker (binary only).

        Pass ax= to draw on an existing Axes.
        """
        df = self.threshold_scan(split)
        show = metrics if metrics is not None else ['precision', 'recall', 'f1']
        fig, ax_, created = self._prepare_ax(ax, figsize=(8, 5))
        for m in show:
            ax_.plot(df['threshold'], df[m], label=m)
        if 'f1' in df.columns:
            best_t = float(df.loc[df['f1'].idxmax(), 'threshold'])
            ax_.axvline(best_t, color='k', linestyle=':', lw=0.8,
                        label=f'Best F1 @ t={best_t:.2f}')
        ax_.set_xlabel('Threshold')
        ax_.set_ylabel('Score')
        ax_.set_title(f'Threshold Scan — {split}')
        ax_.legend()
        self._finalize(fig, path, created)

    def plot_ks(self, split: str, ax: Any = None, path: Any = None) -> None:
        """KS plot: positive vs negative CDF with KS statistic (binary only).

        Pass ax= to draw on an existing Axes.
        """
        self._require_binary('plot_ks')
        y, p = self._splits[split]
        pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
        all_t = np.sort(np.unique(p))
        cdf_pos = np.searchsorted(pos, all_t, side='right') / len(pos)
        cdf_neg = np.searchsorted(neg, all_t, side='right') / len(neg)
        diffs = np.abs(cdf_pos - cdf_neg)
        ks_t, ks_stat = all_t[diffs.argmax()], diffs.max()

        fig, ax_, created = self._prepare_ax(ax, figsize=(7, 5))
        ax_.plot(all_t, cdf_pos, label='Positive CDF')
        ax_.plot(all_t, cdf_neg, label='Negative CDF')
        ax_.axvline(ks_t, color='red', linestyle='--', lw=0.8,
                    label=f'KS={ks_stat:.3f} @ t={ks_t:.3f}')
        ax_.set_xlabel('Score')
        ax_.set_ylabel('CDF')
        ax_.set_title(f'KS Plot — {split}')
        ax_.legend()
        self._finalize(fig, path, created)

    def plot_psi(
        self,
        ref: str,
        target: str,
        n_bins: int = 10,
        axes: list | None = None,
        path: Any = None,
    ) -> None:
        """Score distribution shift and per-bin PSI bar (binary only).

        Pass axes=[ax_left, ax_right] to draw into existing Axes.
        """
        total_psi, df = self.psi(ref, target, n_bins)
        x, w = np.arange(n_bins), 0.38
        fig, ax_list, created = self._prepare_axes_grid(axes, 2, figsize=(13, 4))
        ax0, ax1 = ax_list[0], ax_list[1]

        ax0.bar(x - w / 2, df[f'{ref}_pct'], w, label=ref)
        ax0.bar(x + w / 2, df[f'{target}_pct'], w, label=target)
        ax0.set_xticks(x)
        ax0.set_xticklabels(df['bin'], rotation=45, ha='right', fontsize=7)
        ax0.set_title('Score Distribution by Bin')
        ax0.legend()

        ax1.bar(x, df['psi'], color='coral')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['bin'], rotation=45, ha='right', fontsize=7)
        ax1.set_title(f'PSI per Bin  (total PSI = {total_psi:.4f})')

        self._finalize(fig, path, created, tight=True)

    def plot_roc_ovr(
        self, splits: list[str] | None = None, path: Any = None
    ) -> None:
        """One-vs-Rest ROC curves per class — one figure per split (multiclass only).

        Does not support ax= because the number of subplots equals n_classes.
        """
        from sklearn.metrics import roc_auc_score, roc_curve

        if self._task != 'multiclass':
            raise ValueError('plot_roc_ovr() is only available for task="multiclass"')
        for sname in self._resolve_splits(splits):
            y, p = self._splits[sname]
            classes = np.unique(y)
            fig, ax_, created = self._prepare_ax(None, figsize=(7, 6))
            ax_.plot([0, 1], [0, 1], 'k--', lw=0.8)
            for k, (cls, color) in enumerate(zip(classes, self._palette(len(classes)))):
                y_bin = (y == cls).astype(int)
                fpr, tpr, _ = roc_curve(y_bin, p[:, k])
                ax_.plot(fpr, tpr, color=color,
                         label=f'Class {cls} (AUC={roc_auc_score(y_bin, p[:, k]):.3f})')
            ax_.set_xlabel('FPR')
            ax_.set_ylabel('TPR')
            ax_.set_title(f'OvR ROC Curves — {sname}')
            ax_.legend()
            self._finalize(fig, path, created)

    def plot_metrics_per_class(
        self, split: str, ax: Any = None, path: Any = None
    ) -> None:
        """Precision/recall/F1 bar chart per class (multiclass only).

        Pass ax= to draw on an existing Axes.
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        if self._task != 'multiclass':
            raise ValueError('plot_metrics_per_class() is only available for task="multiclass"')
        y, p = self._splits[split]
        pred = p.argmax(axis=1)
        classes = np.unique(y)
        prec = precision_score(y, pred, average=None, zero_division=0)
        rec  = recall_score(y, pred, average=None, zero_division=0)
        f1   = f1_score(y, pred, average=None, zero_division=0)

        x, w = np.arange(len(classes)), 0.25
        fig, ax_, created = self._prepare_ax(ax, figsize=(max(6, len(classes) * 2), 5))
        ax_.bar(x - w, prec, w, label='Precision')
        ax_.bar(x,     rec,  w, label='Recall')
        ax_.bar(x + w, f1,   w, label='F1')
        ax_.set_xticks(x)
        ax_.set_xticklabels([f'Class {c}' for c in classes])
        ax_.set_ylim(0, 1.05)
        ax_.set_title(f'Per-Class Metrics — {split}')
        ax_.legend()
        self._finalize(fig, path, created)

    # ── HTML report ───────────────────────────────────────────────────────────

    def report(self, path: str) -> None:
        """Save a self-contained HTML report with metrics table and all plots."""
        split_names = list(self._splits)
        sections = []

        if self._metrics and split_names:
            sections.append(self._metrics_html())

        if self._task == 'binary':
            plot_calls = [
                ('plot_roc', {}),
                ('plot_pr', {}),
                ('plot_precision_recall_at_k', {'show_f1': True, 'show_counts_axis': True}),
                ('plot_score_distribution', {}),
                ('plot_score_cdf', {}),
                ('plot_calibration', {}),
                ('plot_lift', {}),
                ('plot_gains', {}),
            ]
            for sname in split_names:
                plot_calls += [
                    ('plot_confusion_matrix', {'split': sname}),
                    ('plot_ks', {'split': sname}),
                    ('plot_decile_bar', {'split': sname}),
                    ('plot_threshold_scan', {'split': sname}),
                ]
            if len(split_names) >= 2:
                plot_calls.append(
                    ('plot_psi', {'ref': split_names[-2], 'target': split_names[-1]})
                )
        else:
            plot_calls = [('plot_roc_ovr', {}), ('plot_calibration', {})]
            for sname in split_names:
                plot_calls += [
                    ('plot_confusion_matrix', {'split': sname}),
                    ('plot_metrics_per_class', {'split': sname}),
                ]

        for method_name, kwargs in plot_calls:
            section = self._plot_to_section(method_name, kwargs)
            if section is not None:
                sections.append(section)

        html = self._render_html(sections, 'Classification Evaluation Report')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info('Report saved to %s', path)


# Backward-compatible alias
ModelEvaluator = ClassificationEvaluator
