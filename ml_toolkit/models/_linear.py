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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import (
    CLS_METRICS,
    REG_METRICS,
    apply_cat_encoder,
    build_cat_encoder,
    fit_calibrator,
    resolve_metric_fn,
    resolve_timeout,
    set_optuna_verbosity,
)

logger = logging.getLogger(__name__)

_LINEAR_TYPE_NAMES = frozenset({'ridge', 'elasticnet', 'huber', 'tweedie', 'quantile', 'bayesian_ridge'})
_QUANTILE_MAX_TRAIN_ROWS = 20_000


def _make_preprocessor() -> Pipeline:
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])


def _make_regressor(name: str, params: dict) -> Any:
    """Строит sklearn-регрессор: дефолты этого тулкита + любые явные params поверх них.

    Раньше явные params читались точечно через ``params.get('alpha', ...)`` — любой
    ключ вне жёстко заданного списка (например ``fit_intercept``, ``max_iter``, а для
    ``quantile`` — даже сам ``quantile``, всегда захардкоженный в 0.5) молча
    игнорировался. Дефолты-литералы ниже воспроизводят прежнее поведение при
    отсутствии ключа в ``params``, но ``**params`` теперь может переопределить любой
    параметр конструктора, а не только заранее перечисленные.
    """
    if name == 'ridge':
        return Ridge(**{'alpha': 1.0, 'fit_intercept': True, **params})
    if name == 'elasticnet':
        return ElasticNet(**{'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 5000, 'fit_intercept': True, **params})
    if name == 'huber':
        return HuberRegressor(**{'epsilon': 1.35, 'alpha': 0.0001, 'max_iter': 500, 'fit_intercept': True, **params})
    if name == 'tweedie':
        return TweedieRegressor(**{'power': 1.5, 'alpha': 0.1, 'max_iter': 500, 'fit_intercept': True, **params})
    if name == 'quantile':
        return QuantileRegressor(**{'quantile': 0.5, 'alpha': 0.001, 'solver': 'highs', 'fit_intercept': True, **params})
    if name == 'bayesian_ridge':
        return BayesianRidge(**{'max_iter': 500, 'fit_intercept': True, **params})
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
    params=None → Optuna; params=dict → прямое обучение без тюнинга — `params` может
    содержать любой валидный kwarg конструктора выбранного sklearn-регрессора (не
    только `alpha`/`l1_ratio`/`epsilon`/`power`), включая `quantile` для
    `name='quantile'` (по умолчанию 0.5 — медиана, если явно не переопределён).
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> LinearRegressor:
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        name = ms.get('name', 'ridge')
        if name not in _LINEAR_TYPE_NAMES:
            raise ValueError(f'Unknown linear regression type: {name!r}. Valid: {sorted(_LINEAR_TYPE_NAMES)}')

        baseline_col: str | None = ms.get('baseline_col')
        self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_, self.selected_features_ = \
            build_cat_encoder(X_train, self.selected_features_, self.cat_features_, ms)
        X_train = apply_cat_encoder(X_train, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        X_valid_enc = (
            apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            if X_valid is not None else None
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
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = apply_cat_encoder(X, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        X_sc = self._prep.transform(X_enc[self._num_feats_].to_numpy(dtype=float))
        return np.nan_to_num(self._model.predict(X_sc), nan=0.0, posinf=0.0, neginf=0.0)


class LinearClassifier(BaseModel):
    """LogisticRegression (ElasticNet) с QuantileTransformer-препроцессингом через Optuna.

    Используется для всех линейных адаптеров независимо от регрессионного типа.
    Категориальные признаки исключаются. Вероятности калибруются изотонической регрессией.
    params=None → Optuna; params=dict → прямое обучение без тюнинга. ``class_weight``
    по умолчанию `'balanced'`, но явный `class_weight` в `params` побеждает (не
    приводит к `TypeError` — `{'class_weight': 'balanced', **params}`).
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> LinearClassifier:
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_, self.selected_features_ = \
            build_cat_encoder(X_train, self.selected_features_, self.cat_features_, ms)
        X_train = apply_cat_encoder(X_train, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        X_valid_enc = (
            apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            if X_valid is not None else None
        )

        cat_set = set(self.cat_features_)
        self._num_feats_ = [f for f in self.selected_features_ if f not in cat_set]

        self._prep = _make_preprocessor()
        X_tr_sc = self._prep.fit_transform(X_train[self._num_feats_].to_numpy(dtype=float))
        y_tr = y_train.to_numpy(dtype=int)

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

        if self.params is not None:
            # class_weight по умолчанию 'balanced' (типичное дефолтное поведение для этого
            # тулкита, ориентированного на дисбаланс), но явный class_weight в params должен
            # побеждать, а не приводить к TypeError('multiple values for keyword argument').
            direct_params = {'class_weight': 'balanced', **self.params}
            self._model = LogisticRegression(**direct_params)
            self._model.fit(X_tr_sc, y_tr)
            self.best_params_ = direct_params
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
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = apply_cat_encoder(X, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        X_sc = self._prep.transform(X_enc[self._num_feats_].to_numpy(dtype=float))
        raw = self._model.predict_proba(X_sc)[:, 1]
        return self.calibrator_.predict(raw) if self.calibrator_ is not None else raw

