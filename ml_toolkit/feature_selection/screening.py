"""Многоступенчатый статистический пре-фильтр признаков (polars-native).

Стадии применяются последовательно; причина удаления — первый сработавший фильтр:

    1. high_null_rate    — доля пропусков > max_null_rate
    2. low_variance      — дисперсия < min_variance  (константы)
    3. quasi_constant    — доля доминирующего значения > max_quasi_constant_rate
    4. duplicate         — точный дубликат ранее встреченного признака (drop_duplicates=True)
    5. low_auc           — однофакторный ROC-AUC < min_univariate_auc
    6. low_mutual_info   — взаимная информация < min_mutual_info (если задан)
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

_NOT_FITTED = "FeatureScreener has not been fitted yet. Call fit() first."


def _arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """NaN-safe поэлементное сравнение двух массивов."""
    if a.shape != b.shape:
        return False
    try:
        a = a.astype(float, copy=False)
        b = b.astype(float, copy=False)
    except (ValueError, TypeError):
        return bool(np.array_equal(a, b))
    return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


class FeatureScreener:
    """Многоступенчатый статистический пре-фильтр признаков.

    Применяет фильтры последовательно; причина удаления — первый сработавший.
    Поддерживает sklearn-style API: fit / transform / fit_transform.

    Принимает ``pl.DataFrame`` / ``pl.Series``; возвращает ``pl.DataFrame``.
    Для работы с pandas используйте ``ml_toolkit.feature_screening.FeatureScreener``.

    Attributes (after fit):
        selected_features_: list[str]  — признаки, прошедшие все фильтры.

    Example::

        import polars as pl
        from ml_toolkit.feature_selection import FeatureScreener

        screener = FeatureScreener(min_univariate_auc=0.55, max_null_rate=0.9)
        screener.fit(X_train, y_train)
        print(screener.removal_summary())
        X_clean = screener.transform(X_train)
    """

    def __init__(
        self,
        max_null_rate: float = 0.95,
        min_variance: float = 1e-5,
        max_quasi_constant_rate: float = 0.95,
        min_univariate_auc: float = 0.52,
        auc_subsample: int | None = None,
        min_mutual_info: float | None = None,
        drop_duplicates: bool = False,
    ) -> None:
        self.max_null_rate = max_null_rate
        self.min_variance = min_variance
        self.max_quasi_constant_rate = max_quasi_constant_rate
        self.min_univariate_auc = min_univariate_auc
        self.auc_subsample = auc_subsample
        self.min_mutual_info = min_mutual_info
        self.drop_duplicates = drop_duplicates

        self._stats: pl.DataFrame | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X: pl.DataFrame, y: pl.Series | np.ndarray) -> "FeatureScreener":
        """Вычислить статистики и применить фильтры. Возвращает self."""
        y_arr = y.to_numpy() if isinstance(y, pl.Series) else np.asarray(y)

        # Normalize: float NaN → null (polars разделяет NaN и null в Float-колонках)
        float_cols = [c for c in X.columns if X[c].dtype in (pl.Float32, pl.Float64)]
        if float_cols:
            X = X.with_columns([pl.col(c).fill_nan(None) for c in float_cols])

        records: list[dict] = []

        for col_name in X.columns:
            s = X[col_name]
            n_total = len(s)
            n_null = s.null_count()
            n_non_null = n_total - n_null

            null_rate = n_null / n_total if n_total > 0 else 1.0

            non_null = s.drop_nulls()
            variance = float(non_null.cast(pl.Float64).var(ddof=0)) if n_non_null > 1 else 0.0

            if n_non_null > 0:
                vc = non_null.value_counts().sort("count", descending=True)
                quasi_constant_rate = float(vc[0, "count"]) / n_non_null
            else:
                quasi_constant_rate = 1.0

            records.append({
                "feature":             col_name,
                "null_rate":           null_rate,
                "variance":            variance,
                "quasi_constant_rate": quasi_constant_rate,
                "univariate_auc":      float("nan"),
                "mutual_info":         float("nan"),
                "kept":                True,
                "removed_by":          None,
            })

        # Стадии 1–3
        for r in records:
            if r["null_rate"] > self.max_null_rate:
                r["kept"] = False
                r["removed_by"] = "high_null_rate"
            elif r["variance"] < self.min_variance:
                r["kept"] = False
                r["removed_by"] = "low_variance"
            elif r["quasi_constant_rate"] > self.max_quasi_constant_rate:
                r["kept"] = False
                r["removed_by"] = "quasi_constant"

        # Стадия 4: дубликаты (только среди пока не удалённых)
        if self.drop_duplicates:
            surviving_idx = [i for i, r in enumerate(records) if r["kept"]]
            if len(surviving_idx) > 1:
                seen: list[np.ndarray] = []
                for i in surviving_idx:
                    arr = X[records[i]["feature"]].to_numpy()
                    if any(_arrays_equal(arr, s) for s in seen):
                        records[i]["kept"] = False
                        records[i]["removed_by"] = "duplicate"
                    else:
                        seen.append(arr)

        # AUC и MI вычисляются для всех признаков (для полноты отчёта)
        compute_mi = self.min_mutual_info is not None
        for r in records:
            r["univariate_auc"] = self._auc(X[r["feature"]], y_arr)
            if compute_mi:
                r["mutual_info"] = self._mutual_info(X[r["feature"]], y_arr)

        # Стадия 5: low_auc
        for r in records:
            if not r["kept"]:
                continue
            if r["univariate_auc"] < self.min_univariate_auc:
                r["kept"] = False
                r["removed_by"] = "low_auc"

        # Стадия 6: low_mutual_info
        if compute_mi:
            for r in records:
                if not r["kept"]:
                    continue
                mi = r["mutual_info"]
                if not np.isnan(mi) and mi < self.min_mutual_info:  # type: ignore[operator]
                    r["kept"] = False
                    r["removed_by"] = "low_mutual_info"

        self._stats = pl.DataFrame(
            records,
            schema={
                "feature":             pl.Utf8,
                "null_rate":           pl.Float64,
                "variance":            pl.Float64,
                "quasi_constant_rate": pl.Float64,
                "univariate_auc":      pl.Float64,
                "mutual_info":         pl.Float64,
                "kept":                pl.Boolean,
                "removed_by":          pl.Utf8,
            },
        )

        n_kept = int(self._stats["kept"].sum())
        logger.info("FeatureScreener: оставлено %d / %d признаков", n_kept, len(self._stats))
        return self

    @property
    def selected_features_(self) -> list[str]:
        if self._stats is None:
            raise RuntimeError(_NOT_FITTED)
        return self._stats.filter(pl.col("kept"))["feature"].to_list()

    def report(self) -> pl.DataFrame:
        """Полная таблица статистик и причин удаления.

        Columns:
            feature, null_rate, variance, quasi_constant_rate, univariate_auc,
            mutual_info, kept, removed_by
        """
        if self._stats is None:
            raise RuntimeError(_NOT_FITTED)
        return self._stats.clone()

    def removal_summary(self) -> pl.DataFrame:
        """Сводка: сколько признаков удалено каждым фильтром + итог.

        Returns:
            ``pl.DataFrame`` с колонками ``причина`` и ``признаков``.
            Последняя строка — ``'ИТОГО удалено'``.
        """
        if self._stats is None:
            raise RuntimeError(_NOT_FITTED)

        removed = self._stats.filter(~pl.col("kept"))
        n_removed = len(removed)

        if n_removed == 0:
            counts = pl.DataFrame({
                "причина":   pl.Series([], dtype=pl.Utf8),
                "признаков": pl.Series([], dtype=pl.Int64),
            })
        else:
            vc = removed["removed_by"].value_counts(sort=True)
            counts = (
                vc.rename({"removed_by": "причина", "count": "признаков"})
                .with_columns(pl.col("признаков").cast(pl.Int64))
            )

        total = pl.DataFrame({"причина": ["ИТОГО удалено"], "признаков": [n_removed]})
        return pl.concat([counts, total])

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Вернуть датасет только с отобранными признаками."""
        return X.select(self.selected_features_)

    def fit_transform(self, X: pl.DataFrame, y: pl.Series | np.ndarray) -> pl.DataFrame:
        return self.fit(X, y).transform(X)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _auc(self, col: pl.Series, y: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score

        not_null = col.is_not_null()
        if not_null.sum() < 2:
            return 0.5

        feat = col.filter(not_null).to_numpy().astype(float)
        target = y[not_null.to_numpy()]

        if len(np.unique(target)) < 2:
            return 0.5

        if self.auc_subsample is not None and len(feat) > self.auc_subsample:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(feat), size=self.auc_subsample, replace=False)
            feat, target = feat[idx], target[idx]
            if len(np.unique(target)) < 2:
                return 0.5

        try:
            auc = roc_auc_score(target, feat)
            return float(max(auc, 1.0 - auc))
        except ValueError:
            return 0.5

    def _mutual_info(self, col: pl.Series, y: np.ndarray) -> float:
        from sklearn.feature_selection import mutual_info_classif

        not_null = col.is_not_null()
        if not_null.sum() < 2:
            return 0.0

        feat = col.filter(not_null).to_numpy().reshape(-1, 1).astype(float)
        target = y[not_null.to_numpy()]

        try:
            return float(mutual_info_classif(feat, target, random_state=42)[0])
        except Exception:
            return 0.0
