"""pyGAM — Generalized Additive Models через сплайновую регрессию.

LinearGAM строит модель f(x) = β₀ + f₁(x₁) + … + fₚ(xₚ), где каждый fᵢ — B-сплайн,
регуляризованный параметром lam. LogisticGAM — аналог с logit link-функцией.
Интерпретируемость через partial dependence (shape) plots для каждого признака.

Поддерживаемые имена (model_settings['name']): 'pygam'

Пакет: pygam (pip install pygam)
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import (
    CLS_METRICS,
    REG_METRICS,
    fit_calibrator,
    resolve_metric_fn,
    resolve_timeout,
    set_optuna_verbosity,
)

logger = logging.getLogger(__name__)


def _make_prep() -> Pipeline:
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])


def _num_features(selected_features: list[str], cat_features: list[str]) -> list[str]:
    cat_set = set(cat_features)
    return [f for f in selected_features if f not in cat_set]


# ── Классы (новый API) ────────────────────────────────────────────────────────

class PyGAMRegressor(BaseModel):
    """LinearGAM с подбором lam через Optuna.

    Категориальные признаки исключаются. Хранит _prep (импутер+скейлер) и _num_feats_.
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
    ) -> PyGAMRegressor:
        try:
            from pygam import LinearGAM
        except ImportError as exc:
            raise ImportError('Установи пакет: pip install pygam') from exc

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._num_feats_ = _num_features(self.selected_features_, self.cat_features_)
        logger.info('[PYGAM Reg] features=%d', len(self._num_feats_))

        self._prep = _make_prep()
        X_tr = self._prep.fit_transform(X_train[self._num_feats_].to_numpy(dtype=float))
        y_tr = y_train.to_numpy(dtype=float)

        metric_fn, direction = resolve_metric_fn(ms, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS)
        n_trials = min(self.n_optuna_trials, 10) if len(self._num_feats_) > 50 else self.n_optuna_trials

        if self.params is not None:
            self._model = LinearGAM(**self.params).fit(X_tr, y_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_va = self._prep.transform(X_valid[self._num_feats_].to_numpy(dtype=float))
            y_va = y_valid.to_numpy(dtype=float)

            def objective(trial: optuna.Trial) -> float:
                lam = trial.suggest_float('lam', 1e-5, 1e3, log=True)
                m = LinearGAM(lam=lam).fit(X_tr, y_tr)
                return metric_fn(y_va, m.predict(X_va))

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, n_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = study.best_params
            logger.info('[PYGAM Reg] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = LinearGAM(lam=self.best_params_.get('lam', 0.6)).fit(X_tr, y_tr)

        self.train_pred_ = self._model.predict(X_tr)
        if X_valid is not None:
            X_va = self._prep.transform(X_valid[self._num_feats_].to_numpy(dtype=float))
            self.valid_pred_ = self._model.predict(X_va)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(
            self._prep.transform(X[self._num_feats_].to_numpy(dtype=float))
        ))


class PyGAMClassifier(BaseModel):
    """LogisticGAM с подбором lam через Optuna. Вероятности калибруются изотонической регрессией.

    Категориальные признаки исключаются. params=None → Optuna; params=dict → прямое обучение.
    Важно: LogisticGAM.predict_proba() возвращает 1D-массив (не 2D как sklearn).
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> PyGAMClassifier:
        try:
            from pygam import LogisticGAM
        except ImportError as exc:
            raise ImportError('Установи пакет: pip install pygam') from exc

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._num_feats_ = _num_features(self.selected_features_, self.cat_features_)
        logger.info('[PYGAM Cls] features=%d', len(self._num_feats_))

        self._prep = _make_prep()
        X_tr = self._prep.fit_transform(X_train[self._num_feats_].to_numpy(dtype=float))
        y_tr = y_train.to_numpy(dtype=int)
        w_tr = compute_sample_weight('balanced', y_tr)

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)
        n_trials = min(self.n_optuna_trials, 10) if len(self._num_feats_) > 50 else self.n_optuna_trials

        if self.params is not None:
            self._model = LogisticGAM(**self.params).fit(X_tr, y_tr, weights=w_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_va = self._prep.transform(X_valid[self._num_feats_].to_numpy(dtype=float))
            y_va = y_valid.to_numpy(dtype=int)

            def objective(trial: optuna.Trial) -> float:
                lam = trial.suggest_float('lam', 1e-5, 1e3, log=True)
                m = LogisticGAM(lam=lam).fit(X_tr, y_tr, weights=w_tr)
                return metric_fn(y_va, m.predict_proba(X_va))

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, n_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = study.best_params
            logger.info('[PYGAM Cls] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = LogisticGAM(lam=self.best_params_.get('lam', 0.6)).fit(X_tr, y_tr, weights=w_tr)

        self.train_pred_ = np.asarray(self._model.predict_proba(X_tr))
        if X_valid is not None:
            X_va = self._prep.transform(X_valid[self._num_feats_].to_numpy(dtype=float))
            self.valid_pred_ = np.asarray(self._model.predict_proba(X_va))
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self._model.predict_proba(
            self._prep.transform(X[self._num_feats_].to_numpy(dtype=float))
        ))
        return self.calibrator_.predict(raw) if self.calibrator_ is not None else raw

