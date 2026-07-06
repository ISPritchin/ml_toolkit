"""Error analysis: FN/FP profiling against a feature matrix."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """Profile model errors (FN/FP/TP) against raw feature values.

    Designed for binary classifiers on imbalanced data where understanding
    *which* positives are missed is more informative than aggregate metrics.

    Usage::

        from ml_toolkit.model_evaluation import ErrorAnalyzer

        ea = ErrorAnalyzer(y_valid, proba_valid, X_valid)
        print(ea.summary())                      # TP/FP/FN/TN counts at best-F1 threshold
        ea.profile(top_n=20)                     # feature means by segment, sorted by FN-TP gap
        ea.plot_fn_vs_tp()                       # Cohen's d bar chart
        ea.plot_score_distribution()             # score histogram for positive class only
        ea.worst_fn(n=10)                        # rows with lowest score among false negatives

    Args:
        y_true:    Ground-truth binary labels.
        y_proba:   Predicted probabilities for the positive class.
        X:         Feature DataFrame aligned with y_true (same index length).
        threshold: Decision threshold. If None, automatically chosen as the
                   threshold that maximises F1 on (y_true, y_proba).
    """

    def __init__(
        self,
        y_true: np.ndarray | pd.Series,
        y_proba: np.ndarray | pd.Series,
        X: pd.DataFrame,
        threshold: float | None = None,
    ) -> None:
        self._y = np.asarray(y_true)
        self._p = np.asarray(y_proba)
        self._X = X.reset_index(drop=True)
        self._threshold_override = threshold
        self._threshold_cache: float | None = None

    # ── Threshold ──────────────────────────────────────────────────────────────

    @property
    def threshold(self) -> float:
        """Decision threshold (auto best-F1 or constructor override)."""
        if self._threshold_override is not None:
            return self._threshold_override
        if self._threshold_cache is None:
            self._threshold_cache = self._best_f1_threshold()
        return self._threshold_cache

    def _best_f1_threshold(self) -> float:
        from sklearn.metrics import f1_score

        ts = np.linspace(0.005, 0.995, 500)
        scores = [
            f1_score(self._y, (self._p >= t).astype(int), zero_division=0)
            for t in ts
        ]
        best = float(ts[int(np.argmax(scores))])
        logger.debug("Auto threshold (best F1): %.4f", best)
        return best

    # ── Segment assignment ─────────────────────────────────────────────────────

    def segments(self, threshold: float | None = None) -> pd.Series:
        """Return a Series of 'TP'/'FP'/'FN'/'TN' for each row."""
        t = threshold if threshold is not None else self.threshold
        pred = (self._p >= t).astype(int)
        y = self._y
        seg = np.where(
            (y == 1) & (pred == 1), "TP",
            np.where(
                (y == 1) & (pred == 0), "FN",
                np.where((y == 0) & (pred == 1), "FP", "TN"),
            ),
        )
        return pd.Series(seg, index=self._X.index, name="segment")

    # ── Summary ────────────────────────────────────────────────────────────────

    def summary(self, threshold: float | None = None) -> pd.DataFrame:
        """TP/FP/FN/TN counts + precision, recall, F1 at the given threshold.

        Returns a single-row DataFrame with key error breakdown metrics.
        """
        t = threshold if threshold is not None else self.threshold
        pred = (self._p >= t).astype(int)
        y = self._y

        tp = int(((y == 1) & (pred == 1)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
        fn_rate = fn / (tp + fn) if (tp + fn) > 0 else float("nan")

        return pd.DataFrame(
            {
                "threshold": [round(t, 4)],
                "TP": [tp],
                "FP": [fp],
                "FN": [fn],
                "TN": [tn],
                "precision": [round(prec, 4)],
                "recall": [round(rec, 4)],
                "f1": [round(f1, 4)],
                "fn_rate": [round(fn_rate, 4)],
            }
        )

    # ── Feature profile ────────────────────────────────────────────────────────

    def profile(
        self,
        threshold: float | None = None,
        top_n: int = 25,
        include_fp: bool = True,
    ) -> pd.DataFrame:
        """Feature means by error segment, sorted by FN–TP discriminability.

        Columns: mean per segment (TP, FN, and optionally FP) plus a
        ``|FN-TP|/σ`` column (standardised mean difference).

        Args:
            threshold: Decision threshold (default: auto best-F1).
            top_n:     Number of features to return, ranked by |FN-TP|/σ.
            include_fp: Include FP column in output.

        Returns:
            DataFrame indexed by feature name with one column per segment.
        """
        segs = self.segments(threshold)
        show = ["TP", "FN"] + (["FP"] if include_fp else [])

        parts = {}
        for seg in show:
            mask = segs == seg
            n = int(mask.sum())
            label = f"{seg} (n={n})"
            parts[label] = self._X[mask].mean() if n > 0 else pd.Series(
                np.nan, index=self._X.columns
            )

        df = pd.DataFrame(parts)

        tp_col = next((c for c in df.columns if c.startswith("TP")), None)
        fn_col = next((c for c in df.columns if c.startswith("FN")), None)

        if tp_col and fn_col:
            std = self._X.std().replace(0, np.nan)
            df["|FN-TP|/σ"] = (df[fn_col] - df[tp_col]).abs() / std
            df = df.dropna(subset=["|FN-TP|/σ"]).nlargest(top_n, "|FN-TP|/σ")

        return df

    # ── Worst false negatives ──────────────────────────────────────────────────

    def worst_fn(self, n: int = 10, threshold: float | None = None) -> pd.DataFrame:
        """Return the n false negatives with the lowest model score.

        These are the positives the model is *most confident are negative* —
        the hardest cases to recover by threshold tuning.

        Returns:
            DataFrame with feature columns + '_score' and '_segment'.
        """
        segs = self.segments(threshold)
        fn_mask = segs == "FN"
        if not fn_mask.any():
            return pd.DataFrame(columns=list(self._X.columns) + ["_score", "_segment"])

        fn_df = self._X[fn_mask].copy()
        fn_df["_score"] = self._p[fn_mask.values]
        fn_df["_segment"] = "FN"
        return fn_df.nsmallest(n, "_score")

    # ── Plots ──────────────────────────────────────────────────────────────────

    def plot_fn_vs_tp(
        self,
        threshold: float | None = None,
        top_n: int = 25,
        path: str | None = None,
    ) -> None:
        """Horizontal bar chart of Cohen's d (FN − TP) for top discriminating features.

        Red bars: feature is *higher* in missed positives (FN).
        Blue bars: feature is *lower* in missed positives (FN).

        This directly answers: "what makes a positive client invisible to the model?"
        """
        import matplotlib.pyplot as plt

        segs = self.segments(threshold)
        tp_mask = (segs == "TP").values
        fn_mask = (segs == "FN").values
        n_tp, n_fn = int(tp_mask.sum()), int(fn_mask.sum())
        t = threshold if threshold is not None else self.threshold

        if n_fn == 0:
            logger.warning("No FN at threshold=%.4f — nothing to plot.", t)
            return
        if n_tp == 0:
            logger.warning("No TP at threshold=%.4f — cannot compute FN vs TP.", t)
            return

        X_tp = self._X.values[tp_mask]
        X_fn = self._X.values[fn_mask]

        mean_tp = X_tp.mean(axis=0)
        mean_fn = X_fn.mean(axis=0)

        # pooled std (Cohen's d denominator)
        var_tp = X_tp.var(axis=0) * (n_tp - 1) if n_tp > 1 else np.zeros(X_tp.shape[1])
        var_fn = X_fn.var(axis=0) * (n_fn - 1) if n_fn > 1 else np.zeros(X_fn.shape[1])
        pooled_std = np.sqrt((var_tp + var_fn) / max(n_tp + n_fn - 2, 1)) + 1e-9

        d = (mean_fn - mean_tp) / pooled_std
        feat_names = np.array(self._X.columns)
        top_idx = np.argsort(np.abs(d))[-top_n:]
        d_top = d[top_idx]
        names_top = feat_names[top_idx]
        order = np.argsort(d_top)
        d_plot = d_top[order]
        names_plot = names_top[order]

        fig, ax = plt.subplots(figsize=(9, max(5, len(d_plot) * 0.4)))
        colors = ["#C62828" if v > 0 else "#1565C0" for v in d_plot]
        ax.barh(names_plot, d_plot, color=colors, alpha=0.82)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Cohen's d  (FN − TP)   [красный = у FN выше, синий = у FN ниже]")
        ax.set_title(
            f"Профиль ошибок: FN vs TP   "
            f"(TP={n_tp}, FN={n_fn}, threshold={t:.4f})"
        )
        plt.tight_layout()
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)

    def plot_score_distribution(
        self,
        threshold: float | None = None,
        n_bins: int = 40,
        log_scale: bool = False,
        path: str | None = None,
    ) -> None:
        """Score histogram for the **positive class only**, split by TP (blue) / FN (red).

        With < 1% positive rate this is far more informative than the full
        score distribution: shows whether FNs cluster near the threshold
        (easy to recover) or are deep in the negative zone (structural misses).

        Args:
            log_scale: Use log-scale on the y-axis (useful when n_FN << n_TP or vice versa).
        """
        import matplotlib.pyplot as plt

        t = threshold if threshold is not None else self.threshold
        pos_mask = self._y == 1
        pos_scores = self._p[pos_mask]
        fn_scores = pos_scores[pos_scores < t]
        tp_scores = pos_scores[pos_scores >= t]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(fn_scores, bins=n_bins, color="#C62828", alpha=0.75,
                label=f"FN  n={len(fn_scores)}  ({len(fn_scores)/max(len(pos_scores),1):.0%})")
        ax.hist(tp_scores, bins=n_bins, color="#1565C0", alpha=0.75,
                label=f"TP  n={len(tp_scores)}  ({len(tp_scores)/max(len(pos_scores),1):.0%})")
        ax.axvline(t, color="black", linestyle="--", lw=1.5, label=f"threshold={t:.4f}")
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count (positives only)")
        ax.set_title("Распределение скоров — только позитивный класс")
        ax.legend()
        plt.tight_layout()
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)

    def plot_score_buckets(
        self,
        n_buckets: int = 5,
        top_n: int = 15,
        threshold: float | None = None,
        path: str | None = None,
    ) -> pd.DataFrame:
        """Heatmap of feature means across score buckets (positive class only).

        Splits positive-class predictions into n_buckets equal-count score bands,
        then shows mean feature value per band for the top_n most variable features.
        Bucket 1 = lowest scores (near-FN), Bucket n = highest scores (confident TP).

        Returns the underlying pivot DataFrame (score_bucket × feature).
        """
        import matplotlib.pyplot as plt

        t = threshold if threshold is not None else self.threshold
        pos_mask = self._y == 1
        scores_pos = self._p[pos_mask]
        X_pos = self._X.iloc[pos_mask].copy()
        X_pos["_score"] = scores_pos

        if len(X_pos) < n_buckets:
            logger.warning("Too few positives (%d) for %d buckets.", len(X_pos), n_buckets)
            return pd.DataFrame()

        X_pos["_bucket"] = pd.qcut(
            X_pos["_score"], n_buckets, labels=[f"Q{i+1}" for i in range(n_buckets)],
            duplicates="drop",
        )

        feat_cols = [c for c in X_pos.columns if c not in ("_score", "_bucket")]
        pivot = X_pos.groupby("_bucket", observed=True)[feat_cols].mean()

        # Select top_n features by variance across buckets
        top_feats = pivot.var(axis=0).nlargest(top_n).index
        pivot_top = pivot[top_feats]

        # Normalise each feature to [0, 1] for heatmap readability
        pivot_norm = (pivot_top - pivot_top.min()) / (pivot_top.max() - pivot_top.min() + 1e-9)

        fig, ax = plt.subplots(figsize=(max(8, top_n * 0.5), max(3, n_buckets * 0.6)))
        im = ax.imshow(pivot_norm.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(top_feats)))
        ax.set_xticklabels(top_feats, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(pivot_norm)))
        ax.set_yticklabels(pivot_norm.index)
        ax.set_ylabel("Score bucket (Q1=low, Qn=high)")
        ax.set_title(
            f"Feature means по скор-бакетам  (только позитивный класс)\n"
            f"threshold={t:.4f}  —  нормировано в [0,1] по каждой фиче"
        )
        plt.colorbar(im, ax=ax, shrink=0.6, label="normalised mean")
        plt.tight_layout()
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)

        return pivot_top
