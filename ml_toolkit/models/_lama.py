"""LightAutoML (LAMA) adapter.

LAMA управляет hyperparameter tuning внутри себя.
`n_optuna_trials` используется как таймаут: timeout = n_optuna_trials * 60 секунд.
Переопределить через model_settings['timeout'].

Regression: `fit_predict` возвращает in-sample предикты на train,
`predict` используется для valid / inference.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import fit_calibrator

logger = logging.getLogger(__name__)

_TARGET = '__lama_target__'


def _build_roles(cat_features: list[str], selected_features: list[str]) -> dict[str, str]:
    """Формирует словарь ролей для LightAutoML: таргет и категориальные признаки."""
    roles: dict[str, str] = {'target': _TARGET}
    for col in cat_features:
        if col in selected_features:
            roles[col] = 'category'
    return roles


# ── Классы (новый API) ────────────────────────────────────────────────────────

class LAMARegressor(BaseModel):
    """LightAutoML (TabularAutoML) для регрессии. LAMA самостоятельно управляет тюнингом.

    n_optuna_trials используется для расчёта таймаута (n_trials * 60 сек).
    model_settings['timeout'] переопределяет таймаут напрямую.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> LAMARegressor:
        try:
            from lightautoml.automl.presets.tabular_presets import TabularAutoML
            from lightautoml.tasks import Task
        except ImportError as err:
            raise ImportError('LightAutoML not installed. Run: pip install lightautoml') from err

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings

        baseline_col: str = ms.get('baseline_col', 'fee_nds_amount')
        timeout = int(ms.get('timeout', self.n_optuna_trials * 60))
        self._feats = list(dict.fromkeys([*self.selected_features_, baseline_col]))
        logger.info('[LAMA Reg] timeout=%ds, baseline=%s', timeout, baseline_col)

        automl = TabularAutoML(
            task=Task('reg', loss='mae', metric='mae'),
            timeout=timeout,
            reader_params={'cv': 5, 'random_state': 42},
        )

        train_df = X_train[self._feats].copy()
        train_df[_TARGET] = y_train.values

        automl.fit_predict(train_df, roles=_build_roles(self.cat_features_, self.selected_features_), verbose=0)
        self._model = automl

        self.train_pred_ = np.array(automl.predict(train_df).data[:, 0])
        self.best_params_ = {'timeout': timeout, 'task': 'reg', 'loss': 'mae', 'metric': 'mae', 'cv': 5}

        if X_valid is not None:
            valid_df = X_valid[self._feats].copy()
            self.valid_pred_ = np.array(automl.predict(valid_df).data[:, 0])
            logger.info('[LAMA Reg] Final MAE: %.3f', mean_absolute_error(y_valid, self.valid_pred_))
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        df = X[self._feats].copy()
        return np.array(self._model.predict(df).data[:, 0])


class LAMAClassifier(BaseModel):
    """LightAutoML (TabularAutoML) для бинарной классификации. LAMA самостоятельно управляет тюнингом.

    n_optuna_trials используется для расчёта таймаута. Вероятности калибруются изотонической регрессией.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> LAMAClassifier:
        try:
            from lightautoml.automl.presets.tabular_presets import TabularAutoML
            from lightautoml.tasks import Task
        except ImportError as err:
            raise ImportError('LightAutoML not installed. Run: pip install lightautoml') from err

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings

        timeout = int(ms.get('timeout', self.n_optuna_trials * 60))
        self._feats = self.selected_features_
        logger.info('[LAMA Cls] timeout=%ds', timeout)

        automl = TabularAutoML(
            task=Task('binary', loss='logloss', metric='auc'),
            timeout=timeout,
            reader_params={'cv': 5, 'random_state': 42},
        )

        train_df = X_train[self._feats].copy()
        train_df[_TARGET] = y_train.values if hasattr(y_train, 'values') else y_train

        automl.fit_predict(train_df, roles=_build_roles(self.cat_features_, self.selected_features_), verbose=0)
        self._model = automl

        self.train_pred_ = np.array(automl.predict(train_df).data[:, 0])
        self.best_params_ = {'timeout': timeout, 'task': 'binary', 'loss': 'logloss', 'metric': 'auc', 'cv': 5}

        if X_valid is not None:
            valid_df = X_valid[self._feats].copy()
            self.valid_pred_ = np.array(automl.predict(valid_df).data[:, 0])
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
            logger.info('[LAMA Cls] Final PR-AUC: %.3f', average_precision_score(y_valid, self.valid_pred_))
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        df = X[self._feats].copy()
        raw = np.array(self._model.predict(df).data[:, 0])
        return self.calibrator_.predict(raw) if self.calibrator_ is not None else raw


# ── Backward-compat functional wrappers ──────────────────────────────────────

def train_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_inference: pd.DataFrame,
    selected_features: list[str],
    cat_features: list[str],
    model_settings: dict[str, Any],
    n_optuna_trials: int,
    postprocess_fn: Callable[[pd.DataFrame, np.ndarray], np.ndarray] | None = None,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict]:
    model = LAMARegressor(n_optuna_trials=n_optuna_trials, model_settings=model_settings)
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    _pp = postprocess_fn or (lambda _X, p: p)
    train_pred = _pp(X_train, model.train_pred_)
    valid_pred = _pp(X_valid, model.valid_pred_)
    infer_pred = _pp(X_inference, model.predict(X_inference))
    logger.info('[LAMA Reg] Final MAE: %.3f', mean_absolute_error(y_valid, valid_pred))
    return model._model, train_pred, valid_pred, infer_pred, model.best_params_


def train_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_inference: pd.DataFrame,
    selected_features: list[str],
    cat_features: list[str],
    n_optuna_trials: int,
    model_settings: dict[str, Any] | None = None,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict]:
    model = LAMAClassifier(n_optuna_trials=n_optuna_trials, model_settings=model_settings or {})
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    infer_proba = model.predict_proba(X_inference)
    logger.info('[LAMA Cls] Final PR-AUC: %.3f', average_precision_score(y_valid, model.valid_pred_))
    return model._model, model.train_pred_, model.valid_pred_, infer_proba, model.best_params_


def make_predict_fn(model: Any, task: str, selected_features: list[str]) -> Any:
    """Возвращает callable (X → np.ndarray) для перменных важности через LAMA predict."""
    _m = model
    return lambda X: np.array(_m.predict(X).data[:, 0])
