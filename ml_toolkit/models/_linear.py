"""Линейные и регуляризованные модели.

Поддерживаемые имена (через model_settings['name']):
    ridge           — Ridge regression (L2)
    elasticnet      — ElasticNet (L1 + L2)
    huber           — HuberRegressor (robust, ε-insensitive MAE/MSE blend)
    tweedie         — TweedieRegressor (compound Poisson-Gamma; requires y > 0);
                      Optuna тюнит power ∈ [1.01, 2.99] — включает область Gamma (power=2)
    quantile        — QuantileRegressor(q=0.5) ≡ MAE minimiser
    bayesian_ridge  — BayesianRidge (self-tuning, Optuna не нужен)

Препроцессинг (единый для всех):
    Категориальные признаки исключаются. Числовые:
    SimpleImputer(median) → StandardScaler.

Классификатор (единый): LogisticRegression(saga) + изотоническая калибровка.

Возвращаемая «модель»: (sklearn_estimator, prep_pipeline, num_feature_names).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    LogisticRegression,
    QuantileRegressor,
    Ridge,
    TweedieRegressor,
)
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import CLS_METRICS, REG_METRICS, calibrate_proba, encode_cat_features, fit_calibrator, resolve_metric_fn, resolve_timeout, set_optuna_verbosity

logger = logging.getLogger(__name__)

_LINEAR_TYPE_NAMES = frozenset({'ridge', 'elasticnet', 'huber', 'tweedie', 'quantile', 'bayesian_ridge'})
_QUANTILE_MAX_TRAIN_ROWS = 20_000


def _make_preprocessor() -> Pipeline:
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])


def _make_regressor(name: str, params: dict) -> Any:
    if name == 'ridge':
        return Ridge(alpha=params.get('alpha', 1.0), fit_intercept=True)
    if name == 'elasticnet':
        return ElasticNet(alpha=params.get('alpha', 0.1), l1_ratio=params.get('l1_ratio', 0.5), max_iter=5000, fit_intercept=True)
    if name == 'huber':
        return HuberRegressor(epsilon=params.get('epsilon', 1.35), alpha=params.get('alpha', 0.0001), max_iter=500, fit_intercept=True)
    if name == 'tweedie':
        return TweedieRegressor(power=params.get('power', 1.5), alpha=params.get('alpha', 0.1), max_iter=500, fit_intercept=True)
    if name == 'quantile':
        return QuantileRegressor(quantile=0.5, alpha=params.get('alpha', 0.001), solver='highs', fit_intercept=True)
    if name == 'bayesian_ridge':
        return BayesianRidge(max_iter=500, fit_intercept=True)
    raise ValueError(f'Unknown linear regressor: {name!r}')


def _suggest_reg_params(name: str, trial: optuna.Trial) -> dict:
    if name == 'ridge':
        return {'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True)}
    if name == 'elasticnet':
        return {'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True), 'l1_ratio': trial.suggest_float('l1_ratio', 0.05, 0.95)}
    if name == 'huber':
        return {'epsilon': trial.suggest_float('epsilon', 1.05, 8.0), 'alpha': trial.suggest_float('alpha', 1e-5, 10.0, log=True)}
    if name == 'tweedie':
        return {'power': trial.suggest_float('power', 1.01, 2.99), 'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True)}
    if name == 'quantile':
        return {'alpha': trial.suggest_float('alpha', 1e-7, 1.0, log=True)}
    if name == 'bayesian_ridge':
        return {}
    raise ValueError(name)


def _clip_targets(name: str, y: np.ndarray) -> np.ndarray:
    if name == 'tweedie':
        return np.where(y <= 0, 1.0, y)
    return y


def _num_features(selected_features: list[str], cat_features: list[str]) -> list[str]:
    cat_set = set(cat_features)
    return [f for f in selected_features if f not in cat_set]


# ── Классы (новый API) ────────────────────────────────────────────────────────

class LinearRegressor(BaseModel):
    """Линейный регрессор с автоматическим подбором гиперпараметров через Optuna.

    Тип модели задаётся через model_settings['name'] ∈ {ridge, elasticnet, huber,
    tweedie, quantile, bayesian_ridge}. Категориальные признаки исключаются.
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
    ) -> 'LinearRegressor':
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        set_optuna_verbosity(ms)

        name = ms.get('name', 'ridge')
        if name not in _LINEAR_TYPE_NAMES:
            raise ValueError(f'Unknown linear regression type: {name!r}. Valid: {sorted(_LINEAR_TYPE_NAMES)}')

        baseline_col: str = ms.get('baseline_col', 'fee_nds_amount')
        X_train, X_valid_enc, _, self.selected_features_ = encode_cat_features(
            X_train, X_valid if X_valid is not None else X_train,
            X_train, self.selected_features_, self.cat_features_, ms,
        )

        cat_set = set(self.cat_features_)
        self._num_feats_ = [f for f in self.selected_features_ if f not in cat_set]
        if baseline_col not in self._num_feats_ and baseline_col in X_train.columns:
            self._num_feats_ = [baseline_col] + self._num_feats_

        self._prep = _make_preprocessor()
        X_tr_sc = self._prep.fit_transform(X_train[self._num_feats_].to_numpy(dtype=float))
        y_tr = _clip_targets(name, y_train.to_numpy(dtype=float))

        metric_fn, direction = resolve_metric_fn(ms, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS)

        if self.params is not None or name == 'bayesian_ridge':
            best_params = self.params or {}
            reg = _make_regressor(name, best_params)
            reg.fit(X_tr_sc, y_tr)
            self._model = reg
            self.best_params_ = best_params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_va_sc = self._prep.transform(X_valid_enc[self._num_feats_].to_numpy(dtype=float))
            y_va = y_valid.to_numpy(dtype=float)

            X_tr_opt, y_tr_opt = X_tr_sc, y_tr
            if name == 'quantile' and len(X_tr_sc) > _QUANTILE_MAX_TRAIN_ROWS:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X_tr_sc), size=_QUANTILE_MAX_TRAIN_ROWS, replace=False)
                X_tr_opt, y_tr_opt = X_tr_sc[idx], y_tr[idx]

            def objective(trial: optuna.Trial) -> float:
                reg = _make_regressor(name, _suggest_reg_params(name, trial))
                try:
                    reg.fit(X_tr_opt, y_tr_opt)
                    return metric_fn(y_va, np.nan_to_num(reg.predict(X_va_sc), nan=0.0))
                except Exception:
                    return float('inf') if direction == 'minimize' else -float('inf')

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=self.n_optuna_trials, timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = study.best_params
            logger.info('[%s Reg] Best score=%.4f params=%s', name.upper(), study.best_value, self.best_params_)

            self._model = _make_regressor(name, self.best_params_)
            self._model.fit(X_tr_sc, y_tr)

        self.train_pred_ = np.nan_to_num(self._model.predict(X_tr_sc), nan=0.0)
        if X_valid is not None:
            X_va_sc = self._prep.transform(X_valid_enc[self._num_feats_].to_numpy(dtype=float))
            self.valid_pred_ = np.nan_to_num(self._model.predict(X_va_sc), nan=0.0)
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_sc = self._prep.transform(X[self._num_feats_].to_numpy(dtype=float))
        return np.nan_to_num(self._model.predict(X_sc), nan=0.0, posinf=0.0, neginf=0.0)


class LinearClassifier(BaseModel):
    """LogisticRegression (ElasticNet) с QuantileTransformer-препроцессингом через Optuna.

    Используется для всех линейных адаптеров независимо от регрессионного типа.
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
    ) -> 'LinearClassifier':
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        set_optuna_verbosity(ms)

        X_train, X_valid_enc, _, self.selected_features_ = encode_cat_features(
            X_train, X_valid if X_valid is not None else X_train,
            X_train, self.selected_features_, self.cat_features_, ms,
        )

        cat_set = set(self.cat_features_)
        self._num_feats_ = [f for f in self.selected_features_ if f not in cat_set]

        self._prep = _make_preprocessor()
        X_tr_sc = self._prep.fit_transform(X_train[self._num_feats_].to_numpy(dtype=float))
        y_tr = y_train.to_numpy(dtype=int)

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

        if self.params is not None:
            self._model = LogisticRegression(**self.params, class_weight='balanced')
            self._model.fit(X_tr_sc, y_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_va_sc = self._prep.transform(X_valid_enc[self._num_feats_].to_numpy(dtype=float))
            y_va = y_valid.to_numpy(dtype=int)

            def objective(trial: optuna.Trial) -> float:
                C = trial.suggest_float('C', 1e-3, 1e2, log=True)
                l1_ratio = trial.suggest_categorical('l1_ratio', [0.0, 1.0])
                m = LogisticRegression(
                    C=C, l1_ratio=l1_ratio, solver='saga', max_iter=1000,
                    class_weight='balanced', random_state=42,
                )
                try:
                    m.fit(X_tr_sc, y_tr)
                    return metric_fn(y_va, m.predict_proba(X_va_sc)[:, 1])
                except Exception:
                    return -float('inf') if direction == 'maximize' else float('inf')

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = study.best_params
            logger.info('[Linear Cls] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = LogisticRegression(
                C=self.best_params_.get('C', 1.0), l1_ratio=self.best_params_.get('l1_ratio', 0.0),
                solver='saga', max_iter=2000, class_weight='balanced', random_state=42,
            )
            self._model.fit(X_tr_sc, y_tr)

        self.train_pred_ = self._model.predict_proba(X_tr_sc)[:, 1]
        if X_valid is not None:
            X_va_sc = self._prep.transform(X_valid_enc[self._num_feats_].to_numpy(dtype=float))
            self.valid_pred_ = self._model.predict_proba(X_va_sc)[:, 1]
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_sc = self._prep.transform(X[self._num_feats_].to_numpy(dtype=float))
        raw = self._model.predict_proba(X_sc)[:, 1]
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
    model = LinearRegressor(n_optuna_trials=n_optuna_trials, model_settings=model_settings)
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    _pp = postprocess_fn or (lambda _X, p: p)
    train_pred = _pp(X_train, model.train_pred_)
    valid_pred = _pp(X_valid, model.valid_pred_)
    infer_pred = _pp(X_inference, model.predict(X_inference))
    name = model_settings.get('name', 'ridge')
    logger.info('[%s Reg] Final MAE: %.3f', name.upper(), mean_absolute_error(y_valid, valid_pred))
    return (model._model, model._prep, model._num_feats_), train_pred, valid_pred, infer_pred, model.best_params_


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
    ms = model_settings or {}
    model = LinearClassifier(n_optuna_trials=n_optuna_trials, model_settings=ms)
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    infer_proba = model.predict_proba(X_inference)
    name = ms.get('name', 'ridge')
    logger.info('[%s Cls] Final PR-AUC: %.3f', name.upper(), average_precision_score(y_valid, model.valid_pred_))
    return (model._model, model._prep, model._num_feats_), model.train_pred_, model.valid_pred_, infer_proba, model.best_params_


def make_predict_fn(model: Any, task: str, selected_features: list[str]) -> None:
    """Линейные модели не поддерживают SHAP TreeExplainer; используется permutation importance."""
    return None
