"""Общая логика boosting-семейства (CatBoost/LightGBM/XGBoost + их рэнкеры).

Временное расположение — при физическом переносе адаптеров в подпакет
``ml_toolkit/models/_tabular/_boosting/`` этот файл переедет туда же как
``_common.py`` без изменения содержимого.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_residual(
    y_arr: np.ndarray, X: pd.DataFrame, baseline_col: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Возвращает (y_arr - baseline, baseline) для residual learning.

    ``y_arr`` — уже сконвертированный в numpy таргет (вызывающая сторона решает dtype:
    LightGBM использует ``y.values`` как есть, XGBoost — ``y.to_numpy(dtype=float)``).
    ``baseline`` — None, если ``baseline_col`` не задан или отсутствует в X (тогда
    возвращается (y_arr как есть, None)). Используется LightGBMRegressor и
    XGBoostRegressor — CatBoostRegressor residual learning делает иначе, через нативный
    Pool(baseline=...), и этот хелпер не переиспользует.
    """
    baseline = X[baseline_col].values if baseline_col and baseline_col in X.columns else None
    resid = y_arr - baseline if baseline is not None else y_arr
    return resid, baseline


def add_baseline(raw_pred: np.ndarray, baseline_values: np.ndarray | None) -> np.ndarray:
    """Прибавляет baseline обратно к сырому предсказанию модели (или возвращает raw_pred как есть)."""
    return raw_pred + baseline_values if baseline_values is not None else raw_pred
