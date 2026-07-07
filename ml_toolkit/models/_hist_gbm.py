"""HistGradientBoosting (sklearn).

Gradient Boosting на гистограммах — аналог LightGBM внутри sklearn:
- NaN обрабатывает нативно (внутренний surrogate-сплит).
- Категориальные признаки поддерживает через categorical_features=.
- Feature importance: permutation (встроенный feature_importances_ — MDI-like).
- SHAP: TreeExplainer (через внутренние C-деревья).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import CLS_METRICS, REG_METRICS, calibrate_proba, encode_cat_features, fit_calibrator, resolve_metric_fn, resolve_timeout, set_optuna_verbosity

logger = logging.getLogger(__name__)

_SK_VERSION = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
_SUPPORTS_CAT = _SK_VERSION >= (1, 2)


def _cat_indices(selected: list[str], cat: list[str]) -> list[int]:
    cat_set = set(cat)
    return [i for i, f in enumerate(selected) if f in cat_set]


def _to_arrays(X_train, X_valid, X_inference, features):
    tr = X_train[features].to_numpy(dtype=float)
    va = X_valid[features].to_numpy(dtype=float)
    inf = X_inference[features].to_numpy(dtype=float)
    return tr, va, inf


def _suggest(trial: optuna.Trial) -> dict:
    return {
        'max_iter': trial.suggest_int('max_iter', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-6, 10.0, log=True),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 127),
        'random_state': 42,
    }


def _make_extra(selected: list[str], cat: list[str]) -> dict:
    if _SUPPORTS_CAT:
        cat_idx = _cat_indices(selected, cat)
        return {'categorical_features': cat_idx} if cat_idx else {}
    return {}


# ── Классы (новый API) ────────────────────────────────────────────────────────

class HistGBMRegressor(BaseModel):
    """HistGradientBoostingRegressor с автоматическим подбором гиперпараметров через Optuna.

    Нативная обработка NaN и категориальных признаков через индексы столбцов.
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
    ) -> 'HistGBMRegressor':
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        set_optuna_verbosity(ms)

        X_train, X_valid_enc, _, self.selected_features_ = encode_cat_features(
            X_train, X_valid if X_valid is not None else X_train,
            X_train, self.selected_features_, self.cat_features_, ms,
        )
        extra = _make_extra(self.selected_features_, [])

        X_tr = X_train[self.selected_features_].to_numpy(dtype=float)
        y_tr = y_train.to_numpy(dtype=float)

        metric_fn, direction = resolve_metric_fn(ms, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS)

        if self.params is not None:
            self._model = HistGradientBoostingRegressor(**self.params)
            self._model.fit(X_tr, y_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_va = X_valid_enc[self.selected_features_].to_numpy(dtype=float)
            y_va = y_valid.to_numpy(dtype=float)

            def objective(trial: optuna.Trial) -> float:
                m = HistGradientBoostingRegressor(loss='absolute_error', **_suggest(trial), **extra)
                m.fit(X_tr, y_tr)
                return metric_fn(y_va, m.predict(X_va))

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = {**study.best_params, 'random_state': 42, 'loss': 'absolute_error', **extra}
            logger.info('[HIST_GBM Reg] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = HistGradientBoostingRegressor(**self.best_params_)
            self._model.fit(X_tr, y_tr)

        self.train_pred_ = self._model.predict(X_tr)
        if X_valid is not None:
            X_va = X_valid_enc[self.selected_features_].to_numpy(dtype=float)
            self.valid_pred_ = self._model.predict(X_va)
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X[self.selected_features_].to_numpy(dtype=float))


class HistGBMClassifier(BaseModel):
    """HistGradientBoostingClassifier с автоматическим подбором гиперпараметров через Optuna.

    Вероятности инференса калибруются изотонической регрессией по валидационной выборке.
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
    ) -> 'HistGBMClassifier':
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        set_optuna_verbosity(ms)

        X_train, X_valid_enc, _, self.selected_features_ = encode_cat_features(
            X_train, X_valid if X_valid is not None else X_train,
            X_train, self.selected_features_, self.cat_features_, ms,
        )
        extra = _make_extra(self.selected_features_, [])

        X_tr = X_train[self.selected_features_].to_numpy(dtype=float)
        y_tr = y_train.to_numpy(dtype=int)

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

        if self.params is not None:
            self._model = HistGradientBoostingClassifier(**self.params)
            self._model.fit(X_tr, y_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_va = X_valid_enc[self.selected_features_].to_numpy(dtype=float)
            y_va = y_valid.to_numpy(dtype=int)

            def objective(trial: optuna.Trial) -> float:
                m = HistGradientBoostingClassifier(**_suggest(trial), **extra)
                m.fit(X_tr, y_tr)
                return metric_fn(y_va, m.predict_proba(X_va)[:, 1])

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = {**study.best_params, 'random_state': 42, **extra}
            logger.info('[HIST_GBM Cls] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = HistGradientBoostingClassifier(**self.best_params_)
            self._model.fit(X_tr, y_tr)

        self.train_pred_ = self._model.predict_proba(X_tr)[:, 1]
        if X_valid is not None:
            X_va = X_valid_enc[self.selected_features_].to_numpy(dtype=float)
            self.valid_pred_ = self._model.predict_proba(X_va)[:, 1]
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._model.predict_proba(X[self.selected_features_].to_numpy(dtype=float))[:, 1]
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
    model = HistGBMRegressor(n_optuna_trials=n_optuna_trials, model_settings=model_settings)
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    _pp = postprocess_fn or (lambda _X, p: p)
    train_pred = _pp(X_train, model.train_pred_)
    valid_pred = _pp(X_valid, model.valid_pred_)
    infer_pred = _pp(X_inference, model.predict(X_inference))
    logger.info('[HIST_GBM Reg] Final MAE: %.3f', mean_absolute_error(y_valid, valid_pred))
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
    model = HistGBMClassifier(n_optuna_trials=n_optuna_trials, model_settings=model_settings or {})
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    infer_proba = model.predict_proba(X_inference)
    logger.info('[HIST_GBM Cls] Final PR-AUC: %.3f', average_precision_score(y_valid, model.valid_pred_))
    return model._model, model.train_pred_, model.valid_pred_, infer_proba, model.best_params_


def make_predict_fn(model: Any, task: str, selected_features: list[str]) -> Any:
    """Возвращает callable (X → np.ndarray) для перменных важности через permutation."""
    _m, _feats = model, selected_features
    if task == 'regression':
        return lambda X: _m.predict(X[_feats].to_numpy(dtype=float))
    return lambda X: _m.predict_proba(X[_feats].to_numpy(dtype=float))[:, 1]
