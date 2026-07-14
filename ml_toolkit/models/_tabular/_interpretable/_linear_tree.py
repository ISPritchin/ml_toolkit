"""Linear Tree — дерево решений с линейными моделями в листьях.

LinearTreeRegressor: каждый лист содержит Ridge-регрессию вместо константы.
Сочетает интерпретируемость дерева с нелинейными разбиениями и локальной линейностью.
Optuna тюнит max_depth ∈ [2, 15] — охватывает как интерпретируемые (2–8) так и точные (9–15) деревья.

Пакет: linear-tree (pip install linear-tree)
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._tabular._interpretable._common import fit_impute_scale, numeric_features
from ml_toolkit.models._utils import (
    CLS_METRICS,
    REG_METRICS,
    fit_calibrator,
    make_study,
    resolve_metric_fn,
    resolve_timeout,
    set_optuna_verbosity,
)

logger = logging.getLogger(__name__)

_DEPTH_MIN, _DEPTH_MAX = 2, 15


# ── Классы (новый API) ────────────────────────────────────────────────────────

class LinearTreeRegressor(BaseModel):
    """LinearTreeRegressor с Ridge в листьях и подбором гиперпараметров через Optuna.

    Категориальные признаки исключаются. Хранит _imputer, _scaler, _num_feats_.
    params=None → Optuna; params=dict → прямое обучение без тюнинга.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> LinearTreeRegressor:
        try:
            from lineartree import LinearTreeRegressor as _LTR
            from sklearn.linear_model import Ridge
        except ImportError as exc:
            raise ImportError('Установи пакет: pip install linear-tree') from exc

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._num_feats_ = numeric_features(self.selected_features_, self.cat_features_)
        logger.info('[LINEAR_TREE Reg] features=%d', len(self._num_feats_))

        X_tr, X_va, self._imputer, self._scaler = fit_impute_scale(X_train, X_valid, self._num_feats_)
        y_tr = y_train.to_numpy(dtype=float)

        metric_fn, direction = resolve_metric_fn(ms, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS)

        if self.params is not None:
            self._model = _LTR(base_estimator=Ridge(), **self.params)
            self._model.fit(X_tr, y_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            y_va = y_valid.to_numpy(dtype=float)

            def objective(trial: optuna.Trial) -> float:
                params = {
                    'max_depth': trial.suggest_int('max_depth', _DEPTH_MIN, _DEPTH_MAX),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 100),
                    'criterion': trial.suggest_categorical('criterion', ['mse', 'mae']),
                }
                m = _LTR(base_estimator=Ridge(), **params)
                m.fit(X_tr, y_tr)
                return metric_fn(y_va, m.predict(X_va))

            study = make_study(direction, ms)
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = study.best_params
            logger.info('[LINEAR_TREE Reg] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = _LTR(base_estimator=Ridge(), **self.best_params_)
            self._model.fit(X_tr, y_tr)

        self.train_pred_ = self._model.predict(X_tr)
        if X_valid is not None:
            self.valid_pred_ = self._model.predict(X_va)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_t = self._scaler.transform(self._imputer.transform(X[self._num_feats_].to_numpy(dtype=float)))
        return np.asarray(self._model.predict(X_t))


class LinearTreeClassifier(BaseModel):
    """LinearTreeClassifier с LogisticRegression в листьях через Optuna.

    Категориальные признаки исключаются. Вероятности калибруются изотонической регрессией.
    params=None → Optuna; params=dict → прямое обучение без тюнинга.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> LinearTreeClassifier:
        try:
            from lineartree import LinearTreeClassifier as _LTC
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise ImportError('Установи пакет: pip install linear-tree') from exc

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._num_feats_ = numeric_features(self.selected_features_, self.cat_features_)
        logger.info('[LINEAR_TREE Cls] features=%d', len(self._num_feats_))

        X_tr, X_va, self._imputer, self._scaler = fit_impute_scale(X_train, X_valid, self._num_feats_)
        y_tr = y_train.to_numpy(dtype=int)

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)
        base = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=500, random_state=42)

        if self.params is not None:
            self._model = _LTC(base_estimator=base, **self.params)
            self._model.fit(X_tr, y_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            y_va = y_valid.to_numpy(dtype=int)

            def objective(trial: optuna.Trial) -> float:
                params = {
                    'max_depth': trial.suggest_int('max_depth', _DEPTH_MIN, _DEPTH_MAX),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 100),
                }
                m = _LTC(base_estimator=base, **params)
                m.fit(X_tr, y_tr)
                return metric_fn(y_va, m.predict_proba(X_va)[:, 1])

            study = make_study(direction, ms)
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = study.best_params
            logger.info('[LINEAR_TREE Cls] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = _LTC(base_estimator=base, **self.best_params_)
            self._model.fit(X_tr, y_tr)

        self.train_pred_ = self._model.predict_proba(X_tr)[:, 1]
        if X_valid is not None:
            self.valid_pred_ = self._model.predict_proba(X_va)[:, 1]
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_t = self._scaler.transform(self._imputer.transform(X[self._num_feats_].to_numpy(dtype=float)))
        raw = self._model.predict_proba(X_t)[:, 1]
        return self.calibrator_.predict(raw) if self.calibrator_ is not None else raw

