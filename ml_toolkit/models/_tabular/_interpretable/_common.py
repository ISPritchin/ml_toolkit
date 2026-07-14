"""Общая логика interpretable-семейства (numeric-only препроцессинг: категориальные исключаются).

Временное расположение — при физическом переносе адаптеров в подпакет
``ml_toolkit/models/_tabular/_interpretable/`` этот файл переедет туда же как
``_common.py`` без изменения содержимого.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def numeric_features(selected_features: list[str], cat_features: list[str]) -> list[str]:
    """Список признаков из selected_features без категориальных — все adaptер'ы этой семьи принимают только числовые."""
    cat_set = set(cat_features)
    return [f for f in selected_features if f not in cat_set]


def make_impute_scale_pipeline() -> Pipeline:
    """Pipeline([SimpleImputer(median), StandardScaler]) — общий числовой препроцессинг interpretable-семейства."""
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])


def fit_impute_scale(
    X_train: pd.DataFrame, X_valid: pd.DataFrame | None, num_feats: list[str],
) -> tuple[np.ndarray, np.ndarray | None, SimpleImputer, StandardScaler]:
    """Fit-transform импутера+скейлера на train, transform на valid; возвращает и сами объекты для predict()."""
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(imputer.fit_transform(X_train[num_feats].to_numpy(dtype=float)))
    X_va = None
    if X_valid is not None:
        X_va = scaler.transform(imputer.transform(X_valid[num_feats].to_numpy(dtype=float)))
    return X_tr, X_va, imputer, scaler
