"""Random Forest и Extra Trees (sklearn).

Random Forest: bagging N деревьев на bootstrap-выборках, случайный subset из sqrt(p) признаков.
Extra Trees: дополнительная рандомизация — пороги сплитов тоже случайные, bootstrap опционален.

NaN: медианная импутация через SimpleImputer внутри Pipeline.
Модель возвращается как Pipeline([imputer, estimator]) — predict() принимает сырые DataFrames с NaN.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier as _SKETClassifier
from sklearn.ensemble import ExtraTreesRegressor as _SKETRegressor
from sklearn.ensemble import RandomForestClassifier as _SKRFClassifier
from sklearn.ensemble import RandomForestRegressor as _SKRFRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.pipeline import Pipeline

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import (
    CLS_METRICS, REG_METRICS, apply_cat_encoder, build_cat_encoder,
    calibrate_proba, fit_calibrator, resolve_metric_fn,
)

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

_REG_CLASSES = {'random_forest': _SKRFRegressor, 'extra_trees': _SKETRegressor}
_CLS_CLASSES = {'random_forest': _SKRFClassifier, 'extra_trees': _SKETClassifier}


def _make_pipeline(EstClass: type, params: dict) -> Pipeline:
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('estimator', EstClass(**params))])


def _suggest(trial: optuna.Trial) -> dict:
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=100),
        'max_depth': trial.suggest_int('max_depth', 4, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
        'random_state': 42,
        'n_jobs': -1,
    }


# ── Классы (новый API) ────────────────────────────────────────────────────────

class RandomForestRegressor(BaseModel):
    """RandomForestRegressor (Pipeline + SimpleImputer) с подбором гиперпараметров через Optuna.

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
    ) -> 'RandomForestRegressor':
        return _fit_forest_reg(self, _SKRFRegressor, 'random_forest', X_train, y_train, X_valid, y_valid, selected_features, cat_features)

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        return _predict_pipeline(self, X)


class RandomForestClassifier(BaseModel):
    """RandomForestClassifier с подбором через Optuna. Вероятности калибруются изотонической регрессией.

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
    ) -> 'RandomForestClassifier':
        return _fit_forest_cls(self, _SKRFClassifier, 'random_forest', X_train, y_train, X_valid, y_valid, selected_features, cat_features)

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        return _predict_proba_pipeline(self, X)


class ExtraTreesRegressor(BaseModel):
    """ExtraTreesRegressor (Pipeline + SimpleImputer) с подбором гиперпараметров через Optuna.

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
    ) -> 'ExtraTreesRegressor':
        return _fit_forest_reg(self, _SKETRegressor, 'extra_trees', X_train, y_train, X_valid, y_valid, selected_features, cat_features)

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        return _predict_pipeline(self, X)


class ExtraTreesClassifier(BaseModel):
    """ExtraTreesClassifier с подбором через Optuna. Вероятности калибруются изотонической регрессией.

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
    ) -> 'ExtraTreesClassifier':
        return _fit_forest_cls(self, _SKETClassifier, 'extra_trees', X_train, y_train, X_valid, y_valid, selected_features, cat_features)

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        return _predict_proba_pipeline(self, X)


# ── Вспомогательные функции для классов (избегают дублирования кода) ─────────

def _fit_forest_reg(
    self: BaseModel, EstClass: type, name: str,
    X_train, y_train, X_valid, y_valid, selected_features, cat_features,
) -> BaseModel:
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
        self._model = _make_pipeline(EstClass, self.params)
        self._model.fit(Xtr, y_tr)
        self.best_params_ = self.params
    else:
        if X_valid is None:
            raise ValueError('X_valid обязателен при params=None (режим Optuna)')
        X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        Xva = X_valid_enc[self.selected_features_]
        y_va = y_valid.to_numpy(dtype=float)

        def objective(trial: optuna.Trial) -> float:
            pipe = _make_pipeline(EstClass, _suggest(trial))
            pipe.fit(Xtr, y_tr)
            return metric_fn(y_va, pipe.predict(Xva))

        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=max(1, self.n_optuna_trials), show_progress_bar=False)
        self.best_params_ = {**study.best_params, 'random_state': 42, 'n_jobs': -1}
        logger.info('[%s Reg] Best score=%.4f params=%s', name.upper(), study.best_value, self.best_params_)

        self._model = _make_pipeline(EstClass, self.best_params_)
        self._model.fit(Xtr, y_tr)

    self.train_pred_ = self._model.predict(Xtr)
    if X_valid is not None:
        X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        self.valid_pred_ = self._model.predict(X_valid_enc[self.selected_features_])
    return self


def _fit_forest_cls(
    self: BaseModel, EstClass: type, name: str,
    X_train, y_train, X_valid, y_valid, selected_features, cat_features,
) -> BaseModel:
    X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
    self.selected_features_ = self._resolve_features(X_train, selected_features)
    self.cat_features_ = list(cat_features or [])
    ms = self.model_settings

    self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_, self.selected_features_ = \
        build_cat_encoder(X_train, self.selected_features_, self.cat_features_, ms)
    X_train_enc = apply_cat_encoder(X_train, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)

    Xtr = X_train_enc[self.selected_features_]
    y_tr = y_train.to_numpy(dtype=int)

    metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

    if self.params is not None:
        self._model = _make_pipeline(EstClass, {**self.params, 'class_weight': 'balanced'})
        self._model.fit(Xtr, y_tr)
        self.best_params_ = self.params
    else:
        if X_valid is None:
            raise ValueError('X_valid обязателен при params=None (режим Optuna)')
        X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        Xva = X_valid_enc[self.selected_features_]
        y_va = y_valid.to_numpy(dtype=int)

        def objective(trial: optuna.Trial) -> float:
            pipe = _make_pipeline(EstClass, {**_suggest(trial), 'class_weight': 'balanced'})
            pipe.fit(Xtr, y_tr)
            return metric_fn(y_va, pipe.predict_proba(Xva)[:, 1])

        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=max(1, self.n_optuna_trials), show_progress_bar=False)
        self.best_params_ = {**study.best_params, 'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'}
        logger.info('[%s Cls] Best score=%.4f params=%s', name.upper(), study.best_value, self.best_params_)

        self._model = _make_pipeline(EstClass, self.best_params_)
        self._model.fit(Xtr, y_tr)

    self.train_pred_ = self._model.predict_proba(Xtr)[:, 1]
    if X_valid is not None:
        X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        self.valid_pred_ = self._model.predict_proba(X_valid_enc[self.selected_features_])[:, 1]
        self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
    return self


def _predict_pipeline(self: BaseModel, X: pd.DataFrame) -> np.ndarray:
    X_enc = apply_cat_encoder(X, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
    return self._model.predict(X_enc[self.selected_features_])


def _predict_proba_pipeline(self: BaseModel, X: pd.DataFrame) -> np.ndarray:
    X_enc = apply_cat_encoder(X, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
    raw = self._model.predict_proba(X_enc[self.selected_features_])[:, 1]
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
    name = model_settings.get('name', 'random_forest')
    EstClass = _REG_CLASSES[name]
    ModelClass = RandomForestRegressor if name == 'random_forest' else ExtraTreesRegressor
    model = ModelClass(n_optuna_trials=n_optuna_trials, model_settings=model_settings)
    # Override internal class to use correct sklearn estimator
    model._est_class = EstClass  # stored for reference
    _fit_forest_reg(model, EstClass, name, X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    _pp = postprocess_fn or (lambda _X, p: p)
    train_pred = _pp(X_train, model.train_pred_)
    valid_pred = _pp(X_valid, model.valid_pred_)
    infer_pred = _pp(X_inference, model.predict(X_inference))
    logger.info('[%s Reg] Final MAE: %.3f', name.upper(), mean_absolute_error(y_valid, valid_pred))
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
    ms = model_settings or {}
    name = ms.get('name', 'random_forest')
    EstClass = _CLS_CLASSES[name]
    ModelClass = RandomForestClassifier if name == 'random_forest' else ExtraTreesClassifier
    model = ModelClass(n_optuna_trials=n_optuna_trials, model_settings=ms)
    _fit_forest_cls(model, EstClass, name, X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    infer_proba = model.predict_proba(X_inference)
    logger.info('[%s Cls] Final PR-AUC: %.3f', name.upper(), average_precision_score(y_valid, model.valid_pred_))
    return model._model, model.train_pred_, model.valid_pred_, infer_proba, model.best_params_


def make_predict_fn(model: Any, task: str, selected_features: list[str]) -> None:
    """sklearn Pipeline предоставляет feature_importances_ напрямую; predict_fn не нужна."""
    return None
