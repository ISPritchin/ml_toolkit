"""Pandas-адаптер для обратной совместимости.

Основной код — ml_toolkit/feature_selection/screening.py (polars-native).
Этот шим принимает pd.DataFrame / pd.Series и возвращает pd.DataFrame,
конвертируя данные в polars и обратно прозрачно для вызывающей стороны.

report() возвращает pd.DataFrame с именами признаков в качестве индекса,
что соответствует исходному поведению до перехода на polars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from ml_toolkit.feature_selection.screening import FeatureScreener as _FeatureScreener

__all__ = ["FeatureScreener"]


def _to_pl_frame(X: pd.DataFrame) -> pl.DataFrame:
    """pd.DataFrame → pl.DataFrame; pandas NaN конвертируется в polars null."""
    result = pl.from_pandas(X)
    # polars сохраняет float NaN как NaN, а не null — нормализуем
    float_cols = [c for c in result.columns if result[c].dtype in (pl.Float32, pl.Float64)]
    if float_cols:
        result = result.with_columns([pl.col(c).fill_nan(None) for c in float_cols])
    return result


def _to_pl_y(y: np.ndarray | pd.Series) -> np.ndarray:
    return y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)


class FeatureScreener(_FeatureScreener):
    """FeatureScreener с pandas-совместимым интерфейсом.

    fit() и fit_transform() принимают pd.DataFrame + pd.Series / np.ndarray.
    transform() принимает pd.DataFrame и возвращает pd.DataFrame.
    report() возвращает pd.DataFrame с именем признака в качестве индекса.
    removal_summary() возвращает pd.DataFrame.

    Параметры идентичны оригиналу — см. ml_toolkit/feature_selection/screening.py.
    """

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "FeatureScreener":
        super().fit(_to_pl_frame(X), _to_pl_y(y))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return super().transform(_to_pl_frame(X)).to_pandas()

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def report(self) -> pd.DataFrame:
        """Полная таблица статистик с именем признака в качестве индекса."""
        return super().report().to_pandas().set_index("feature")

    def removal_summary(self) -> pd.DataFrame:
        """Сводка удалённых признаков по причинам + строка ИТОГО."""
        return super().removal_summary().to_pandas()
