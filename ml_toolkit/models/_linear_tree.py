"""Linear Tree — дерево решений с линейными моделями в листьях.

LinearTreeRegressor: каждый лист содержит Ridge-регрессию вместо константы.
Сочетает интерпретируемость дерева с нелинейными разбиениями и локальной линейностью.
Optuna тюнит max_depth ∈ [2, 15] — охватывает как интерпретируемые (2–8) так и точные (9–15) деревья.

Пакет: linear-tree (pip install linear-tree)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import CLS_METRICS, REG_METRICS, calibrate_proba, fit_calibrator, resolve_metric_fn, resolve_timeout, set_optuna_verbosity

logger = logging.getLogger(__name__)

_DEPTH_MIN, _DEPTH_MAX = 2, 15


def _num_features(selected_features: list[str], cat_features: list[str]) -> list[str]:
    cat_set = set(cat_features)
    return [f for f in selected_features if f not in cat_set]


def _fit_prep(
    X_train: pd.DataFrame, X_valid: pd.DataFrame | None, num_feats: list[str],
) -> tuple[np.ndarray, np.ndarray | None, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(imputer.fit_transform(X_train[num_feats].to_numpy(dtype=float)))
    X_va = None
    if X_valid is not None:
        X_va = scaler.transform(imputer.transform(X_valid[num_feats].to_numpy(dtype=float)))
    return X_tr, X_va, imputer, scaler


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
    ) -> 'LinearTreeRegressor':
        try:
            from lineartree import LinearTreeRegressor as _LTR
            from sklearn.linear_model import Ridge
        except ImportError as exc:
            raise ImportError('Установи пакет: pip install linear-tree') from exc

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        set_optuna_verbosity(ms)

        self._num_feats_ = _num_features(self.selected_features_, self.cat_features_)
        logger.info('[LINEAR_TREE Reg] features=%d', len(self._num_feats_))

        X_tr, X_va, self._imputer, self._scaler = _fit_prep(X_train, X_valid, self._num_feats_)
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

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = study.best_params
            logger.info('[LINEAR_TREE Reg] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = _LTR(base_estimator=Ridge(), **self.best_params_)
            self._model.fit(X_tr, y_tr)

        self.train_pred_ = self._model.predict(X_tr)
        if X_valid is not None:
            self.valid_pred_ = self._model.predict(X_va)
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
    ) -> 'LinearTreeClassifier':
        try:
            from lineartree import LinearTreeClassifier as _LTC
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise ImportError('Установи пакет: pip install linear-tree') from exc

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        set_optuna_verbosity(ms)

        self._num_feats_ = _num_features(self.selected_features_, self.cat_features_)
        logger.info('[LINEAR_TREE Cls] features=%d', len(self._num_feats_))

        X_tr, X_va, self._imputer, self._scaler = _fit_prep(X_train, X_valid, self._num_feats_)
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

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = study.best_params
            logger.info('[LINEAR_TREE Cls] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = _LTC(base_estimator=base, **self.best_params_)
            self._model.fit(X_tr, y_tr)

        self.train_pred_ = self._model.predict_proba(X_tr)[:, 1]
        if X_valid is not None:
            self.valid_pred_ = self._model.predict_proba(X_va)[:, 1]
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_t = self._scaler.transform(self._imputer.transform(X[self._num_feats_].to_numpy(dtype=float)))
        raw = self._model.predict_proba(X_t)[:, 1]
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
    model = LinearTreeRegressor(n_optuna_trials=n_optuna_trials, model_settings=model_settings)
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    _pp = postprocess_fn or (lambda _X, p: p)
    train_pred = _pp(X_train, model.train_pred_)
    valid_pred = _pp(X_valid, model.valid_pred_)
    infer_pred = _pp(X_inference, model.predict(X_inference))
    logger.info('[LINEAR_TREE Reg] Final MAE: %.3f', mean_absolute_error(y_valid, valid_pred))
    return (model._model, model._imputer, model._scaler, model._num_feats_), train_pred, valid_pred, infer_pred, model.best_params_


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
    model = LinearTreeClassifier(n_optuna_trials=n_optuna_trials, model_settings=model_settings or {})
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    infer_proba = model.predict_proba(X_inference)
    logger.info('[LINEAR_TREE Cls] Final PR-AUC: %.3f', average_precision_score(y_valid, model.valid_pred_))
    return (model._model, model._imputer, model._scaler, model._num_feats_), model.train_pred_, model.valid_pred_, infer_proba, model.best_params_


def make_predict_fn(model: Any, task: str, selected_features: list[str]) -> Any:
    """Возвращает callable (X → np.ndarray) с препроцессингом (imputer + scaler) для SHAP."""
    import numpy as _np  # noqa: PLC0415
    _m, _imp, _sc, _nf = model
    if task == 'regression':
        return lambda X: _np.asarray(_m.predict(_sc.transform(_imp.transform(X[_nf].to_numpy(dtype=float)))))
    return lambda X: _m.predict_proba(_sc.transform(_imp.transform(X[_nf].to_numpy(dtype=float))))[:, 1]
