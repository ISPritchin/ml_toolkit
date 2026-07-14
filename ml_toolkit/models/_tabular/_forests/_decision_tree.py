"""Одиночное дерево решений (sklearn) — максимально интерпретируемая модель.

Shallow tree (max_depth 2-8) предназначен для человекочитаемой визуализации:
можно экспортировать через sklearn.tree.export_text() или plot_tree().
NaN: SimpleImputer(median) внутри Pipeline.

Поддерживаемые имена (model_settings['name']): 'decision_tree'
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as _SKDTClassifier
from sklearn.tree import DecisionTreeRegressor as _SKDTRegressor

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._tabular._forests._common import (
    make_impute_pipeline,
    predict_via_pipeline,
    predict_proba_via_pipeline,
)
from ml_toolkit.models._utils import (
    CLS_METRICS,
    REG_METRICS,
    apply_cat_encoder,
    build_cat_encoder,
    fit_calibrator,
    make_study,
    resolve_metric_fn,
    resolve_timeout,
    set_optuna_verbosity,
)

logger = logging.getLogger(__name__)


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
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_, self.selected_features_ = \
            build_cat_encoder(X_train, self.selected_features_, self.cat_features_, ms)
        X_train_enc = apply_cat_encoder(X_train, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)

        Xtr = X_train_enc[self.selected_features_]
        y_tr = y_train.to_numpy(dtype=float)

        metric_fn, direction = resolve_metric_fn(ms, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS)

        if self.params is not None:
            self._model = make_impute_pipeline(_SKDTRegressor, self.params)
            self._model.fit(Xtr, y_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            Xva = X_valid_enc[self.selected_features_]
            y_va = y_valid.to_numpy(dtype=float)

            def objective(trial: optuna.Trial) -> float:
                pipe = make_impute_pipeline(_SKDTRegressor, _suggest_reg(trial))
                pipe.fit(Xtr, y_tr)
                return metric_fn(y_va, pipe.predict(Xva))

            study = make_study(direction, ms)
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = {**study.best_params, 'random_state': 42}
            logger.info('[DECISION_TREE Reg] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = make_impute_pipeline(_SKDTRegressor, self.best_params_)
            self._model.fit(Xtr, y_tr)

        self.train_pred_ = self._model.predict(Xtr)
        if X_valid is not None:
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            self.valid_pred_ = self._model.predict(X_valid_enc[self.selected_features_])
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        return predict_via_pipeline(self, X)


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
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_, self.selected_features_ = \
            build_cat_encoder(X_train, self.selected_features_, self.cat_features_, ms)
        X_train_enc = apply_cat_encoder(X_train, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)

        Xtr = X_train_enc[self.selected_features_]
        y_tr = y_train.to_numpy(dtype=int)

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

        if self.params is not None:
            self._model = make_impute_pipeline(_SKDTClassifier, self.params)
            self._model.fit(Xtr, y_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            Xva = X_valid_enc[self.selected_features_]
            y_va = y_valid.to_numpy(dtype=int)

            def objective(trial: optuna.Trial) -> float:
                pipe = make_impute_pipeline(_SKDTClassifier, _suggest_cls(trial))
                pipe.fit(Xtr, y_tr)
                return metric_fn(y_va, pipe.predict_proba(Xva)[:, 1])

            study = make_study(direction, ms)
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = {**study.best_params, 'class_weight': 'balanced', 'random_state': 42}
            logger.info('[DECISION_TREE Cls] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = make_impute_pipeline(_SKDTClassifier, self.best_params_)
            self._model.fit(Xtr, y_tr)

        self.train_pred_ = self._model.predict_proba(Xtr)[:, 1]
        if X_valid is not None:
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            self.valid_pred_ = self._model.predict_proba(X_valid_enc[self.selected_features_])[:, 1]
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        return predict_proba_via_pipeline(self, X)

