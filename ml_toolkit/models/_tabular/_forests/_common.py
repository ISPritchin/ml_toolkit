"""Общая логика forests-семейства (sklearn Pipeline([imputer, estimator]) + predict через Pipeline+cat_encoder).

Временное расположение — при физическом переносе адаптеров в подпакет
``ml_toolkit/models/_tabular/_forests/`` этот файл переедет туда же как
``_common.py`` без изменения содержимого.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import apply_cat_encoder


def make_impute_pipeline(estimator_cls: type, params: dict) -> Pipeline:
    """Pipeline([SimpleImputer(median), estimator_cls(**params)]) — общий NaN-паттерн forest-семейства."""
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('estimator', estimator_cls(**params))])


def predict_via_pipeline(self: BaseModel, X: pd.DataFrame) -> np.ndarray:
    """_predict_impl для адаптеров с self._model = Pipeline([imputer, estimator]) + cat_encoder."""
    X_enc = apply_cat_encoder(X, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
    return self._model.predict(X_enc[self.selected_features_])


def predict_proba_via_pipeline(self: BaseModel, X: pd.DataFrame) -> np.ndarray:
    """_predict_proba_impl (бинарный) для адаптеров с self._model = Pipeline([imputer, estimator]) + calibrator_."""
    X_enc = apply_cat_encoder(X, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
    raw = self._model.predict_proba(X_enc[self.selected_features_])[:, 1]
    return self.calibrator_.predict(raw) if self.calibrator_ is not None else raw
