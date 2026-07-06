"""Quantile Random Forest.

Предсказывает квантили распределения таргета, а не только среднее.
Для регрессии: использует медиану (q=0.5), что оптимально для MAE.

Требует: pip install quantile-forest

Модель возвращается как Pipeline([imputer, estimator]) для нативной обработки NaN.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.pipeline import Pipeline

try:
    from quantile_forest import RandomForestQuantileRegressor
except ImportError as e:
    raise ImportError(
        'Quantile Random Forest requires quantile-forest package: pip install quantile-forest'
    ) from e

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import (
    CLS_METRICS, REG_METRICS, apply_cat_encoder, build_cat_encoder,
    calibrate_proba, fit_calibrator, resolve_metric_fn,
)

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

_MEDIAN_QUANTILE = 0.5


class _QuantileMedianWrapper:
    """Обёртка вокруг RandomForestQuantileRegressor для sklearn Pipeline-совместимости.

    Pipeline.predict() вызывает estimator.predict() без параметров. Эта обёртка
    подменяет predict() → predict(quantiles=0.5).
    """

    def __init__(self, **params):
        self._model = RandomForestQuantileRegressor(**params)

    def fit(self, X, y):
        """Обучает внутренний RandomForestQuantileRegressor и копирует feature_importances_."""
        self._model.fit(X, y)
        self.feature_importances_ = self._model.feature_importances_
        return self

    def predict(self, X):
        """Возвращает медианное предсказание (quantile=0.5) для совместимости с Pipeline.predict()."""
        return self._model.predict(X, quantiles=_MEDIAN_QUANTILE)

    def predict_quantiles(self, X, quantiles):
        """Возвращает предсказания для произвольных квантилей `quantiles`."""
        return self._model.predict(X, quantiles=quantiles)


def _make_reg_pipeline(params: dict) -> Pipeline:
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('estimator', _QuantileMedianWrapper(**params))])


def _suggest(trial: optuna.Trial) -> dict:
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=100),
        'max_depth': trial.suggest_int('max_depth', 4, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3]),
        'random_state': 42,
        'n_jobs': -1,
    }


# ── Классы (новый API) ────────────────────────────────────────────────────────

class QuantileForestRegressor(BaseModel):
    """QuantileForestRegressor — медианное предсказание QRF с подбором через Optuna.

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
    ) -> 'QuantileForestRegressor':
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings

        self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_, self.selected_features_ = \
            build_cat_encoder(X_train, self.selected_features_, self.cat_features_, ms)
        X_train_enc = apply_cat_encoder(X_train, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)

        Xtr = X_train_enc[self.selected_features_]
        y_tr = y_train.to_numpy(dtype=float)

        metric_fn, direction = resolve_metric_fn(ms, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS)

        if self.params is not None:
            self._model = _make_reg_pipeline(self.params)
            self._model.fit(Xtr, y_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            Xva = X_valid_enc[self.selected_features_]
            y_va = y_valid.to_numpy(dtype=float)

            def objective(trial: optuna.Trial) -> float:
                pipe = _make_reg_pipeline(_suggest(trial))
                pipe.fit(Xtr, y_tr)
                return metric_fn(y_va, pipe.predict(Xva))

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), show_progress_bar=False)
            self.best_params_ = {**study.best_params, 'random_state': 42, 'n_jobs': -1}
            logger.info('[QUANTILE_FOREST Reg] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = _make_reg_pipeline(self.best_params_)
            self._model.fit(Xtr, y_tr)

        self.train_pred_ = self._model.predict(Xtr)
        if X_valid is not None:
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            self.valid_pred_ = self._model.predict(X_valid_enc[self.selected_features_])
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = apply_cat_encoder(X, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        return self._model.predict(X_enc[self.selected_features_])


class QuantileForestClassifier(BaseModel):
    """QuantileForestClassifier — QRF-признаки (квантили) + LogisticRegression с подбором через Optuna.

    Хранит: _qrf (QRF), _clf (LogisticRegression), _imp (SimpleImputer).
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
    ) -> 'QuantileForestClassifier':
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings

        self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_, self.selected_features_ = \
            build_cat_encoder(X_train, self.selected_features_, self.cat_features_, ms)
        X_train_enc = apply_cat_encoder(X_train, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)

        self._imp = SimpleImputer(strategy='median')
        X_tr = self._imp.fit_transform(X_train_enc[self.selected_features_])
        y_tr = y_train.to_numpy(dtype=int)

        qrf_params = {'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 5,
                      'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1}
        self._qrf = RandomForestQuantileRegressor(**qrf_params)
        self._qrf.fit(X_tr, y_tr.astype(float))
        self._model = self._qrf  # for _check_fitted

        def _qrf_feats(X_arr: np.ndarray) -> np.ndarray:
            q25 = self._qrf.predict(X_arr, quantiles=0.25)
            q50 = self._qrf.predict(X_arr, quantiles=0.5)
            q75 = self._qrf.predict(X_arr, quantiles=0.75)
            return np.column_stack([q25, q50, q75, q75 - q25])

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

        F_tr = _qrf_feats(X_tr)

        if self.params is not None:
            self._clf = LogisticRegression(**self.params, max_iter=500, class_weight='balanced')
            self._clf.fit(F_tr, y_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            X_va = self._imp.transform(X_valid_enc[self.selected_features_])
            y_va = y_valid.to_numpy(dtype=int)
            F_va = _qrf_feats(X_va)

            def objective(trial: optuna.Trial) -> float:
                C = trial.suggest_float('C', 1e-3, 100.0, log=True)
                clf = LogisticRegression(C=C, max_iter=500, class_weight='balanced', random_state=42)
                clf.fit(F_tr, y_tr)
                return metric_fn(y_va, clf.predict_proba(F_va)[:, 1])

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), show_progress_bar=False)
            self.best_params_ = {**study.best_params, 'random_state': 42}
            logger.info('[QUANTILE_FOREST Cls] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._clf = LogisticRegression(**self.best_params_, max_iter=500, class_weight='balanced')
            self._clf.fit(F_tr, y_tr)

        self.train_pred_ = self._clf.predict_proba(F_tr)[:, 1]
        if X_valid is not None:
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            X_va = self._imp.transform(X_valid_enc[self.selected_features_])
            F_va = _qrf_feats(X_va)
            self.valid_pred_ = self._clf.predict_proba(F_va)[:, 1]
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = apply_cat_encoder(X, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        X_imp = self._imp.transform(X_enc[self.selected_features_])
        q25 = self._qrf.predict(X_imp, quantiles=0.25)
        q50 = self._qrf.predict(X_imp, quantiles=0.5)
        q75 = self._qrf.predict(X_imp, quantiles=0.75)
        F = np.column_stack([q25, q50, q75, q75 - q25])
        raw = self._clf.predict_proba(F)[:, 1]
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
    model = QuantileForestRegressor(n_optuna_trials=n_optuna_trials, model_settings=model_settings)
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    _pp = postprocess_fn or (lambda _X, p: p)
    train_pred = _pp(X_train, model.train_pred_)
    valid_pred = _pp(X_valid, model.valid_pred_)
    infer_pred = _pp(X_inference, model.predict(X_inference))
    logger.info('[QUANTILE_FOREST Reg] Final MAE: %.3f', mean_absolute_error(y_valid, valid_pred))
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
    model = QuantileForestClassifier(n_optuna_trials=n_optuna_trials, model_settings=model_settings or {})
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    infer_proba = model.predict_proba(X_inference)
    logger.info('[QUANTILE_FOREST Cls] Final PR-AUC: %.3f', average_precision_score(y_valid, model.valid_pred_))
    return (model._qrf, model._clf, model._imp), model.train_pred_, model.valid_pred_, infer_proba, model.best_params_


def make_predict_fn(model: Any, task: str, selected_features: list[str]) -> None:
    """QRF предоставляет feature_importances_ через Pipeline; predict_fn не нужна."""
    return None
