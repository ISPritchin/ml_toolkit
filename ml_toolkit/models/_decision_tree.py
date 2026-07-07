"""Одиночное дерево решений (sklearn) — максимально интерпретируемая модель.

Shallow tree (max_depth 2-8) предназначен для человекочитаемой визуализации:
можно экспортировать через sklearn.tree.export_text() или plot_tree().
NaN: SimpleImputer(median) внутри Pipeline.

Поддерживаемые имена (model_settings['name']): 'decision_tree'
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier as _SKDTClassifier
from sklearn.tree import DecisionTreeRegressor as _SKDTRegressor

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


def _make_reg_pipeline(params: dict) -> Pipeline:
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('estimator', _SKDTRegressor(**params))])


def _make_cls_pipeline(params: dict) -> Pipeline:
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('estimator', _SKDTClassifier(**params))])


def _suggest_reg(trial: optuna.Trial) -> dict:
    return {
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 100),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'criterion': trial.suggest_categorical('criterion', ['absolute_error', 'squared_error']),
        'random_state': 42,
    }


def _suggest_cls(trial: optuna.Trial) -> dict:
    return {
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 100),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'class_weight': 'balanced',
        'random_state': 42,
    }


# ── Классы (новый API) ────────────────────────────────────────────────────────

class DecisionTreeRegressor(BaseModel):
    """Мелкое дерево решений (Pipeline + SimpleImputer) с подбором глубины через Optuna.

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
    ) -> DecisionTreeRegressor:
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        set_optuna_verbosity(ms)

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
                pipe = _make_reg_pipeline(_suggest_reg(trial))
                pipe.fit(Xtr, y_tr)
                return metric_fn(y_va, pipe.predict(Xva))

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = {**study.best_params, 'random_state': 42}
            logger.info('[DECISION_TREE Reg] Best score=%.4f params=%s', study.best_value, self.best_params_)

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


class DecisionTreeClassifier(BaseModel):
    """Мелкое дерево классификации (Pipeline + SimpleImputer) с подбором через Optuna.

    class_weight='balanced'. Вероятности калибруются изотонической регрессией.
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
    ) -> DecisionTreeClassifier:
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        set_optuna_verbosity(ms)

        self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_, self.selected_features_ = \
            build_cat_encoder(X_train, self.selected_features_, self.cat_features_, ms)
        X_train_enc = apply_cat_encoder(X_train, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)

        Xtr = X_train_enc[self.selected_features_]
        y_tr = y_train.to_numpy(dtype=int)

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

        if self.params is not None:
            self._model = _make_cls_pipeline(self.params)
            self._model.fit(Xtr, y_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            Xva = X_valid_enc[self.selected_features_]
            y_va = y_valid.to_numpy(dtype=int)

            def objective(trial: optuna.Trial) -> float:
                pipe = _make_cls_pipeline(_suggest_cls(trial))
                pipe.fit(Xtr, y_tr)
                return metric_fn(y_va, pipe.predict_proba(Xva)[:, 1])

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = {**study.best_params, 'class_weight': 'balanced', 'random_state': 42}
            logger.info('[DECISION_TREE Cls] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = _make_cls_pipeline(self.best_params_)
            self._model.fit(Xtr, y_tr)

        self.train_pred_ = self._model.predict_proba(Xtr)[:, 1]
        if X_valid is not None:
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            self.valid_pred_ = self._model.predict_proba(X_valid_enc[self.selected_features_])[:, 1]
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
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
    model = DecisionTreeRegressor(n_optuna_trials=n_optuna_trials, model_settings=model_settings)
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    _pp = postprocess_fn or (lambda _X, p: p)
    train_pred = _pp(X_train, model.train_pred_)
    valid_pred = _pp(X_valid, model.valid_pred_)
    infer_pred = _pp(X_inference, model.predict(X_inference))
    logger.info('[DECISION_TREE Reg] Final MAE: %.3f', mean_absolute_error(y_valid, valid_pred))
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
    model = DecisionTreeClassifier(n_optuna_trials=n_optuna_trials, model_settings=model_settings or {})
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    infer_proba = model.predict_proba(X_inference)
    logger.info('[DECISION_TREE Cls] Final PR-AUC: %.3f', average_precision_score(y_valid, model.valid_pred_))
    return model._model, model.train_pred_, model.valid_pred_, infer_proba, model.best_params_


def make_predict_fn(model: Any, task: str, selected_features: list[str]) -> None:
    """Sklearn DecisionTree предоставляет feature_importances_ напрямую; predict_fn не нужна."""
    return
