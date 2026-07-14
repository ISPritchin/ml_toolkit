"""Random Forest и Extra Trees (sklearn).

Random Forest: bagging N деревьев на bootstrap-выборках, случайный subset из sqrt(p) признаков.
Extra Trees: дополнительная рандомизация — пороги сплитов тоже случайные, bootstrap опционален.

NaN: медианная импутация через SimpleImputer внутри Pipeline.
Модель возвращается как Pipeline([imputer, estimator]) — predict() принимает сырые DataFrames с NaN.
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier as _SKETClassifier
from sklearn.ensemble import ExtraTreesRegressor as _SKETRegressor
from sklearn.ensemble import RandomForestClassifier as _SKRFClassifier
from sklearn.ensemble import RandomForestRegressor as _SKRFRegressor

from ml_toolkit.models._base import BaseModel, XInput, YInput
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

_REG_CLASSES = {'random_forest': _SKRFRegressor, 'extra_trees': _SKETRegressor}
_CLS_CLASSES = {'random_forest': _SKRFClassifier, 'extra_trees': _SKETClassifier}


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
    ) -> RandomForestRegressor:
        return _fit_forest_reg(self, _SKRFRegressor, 'random_forest', X_train, y_train, X_valid, y_valid, selected_features, cat_features)

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        return predict_via_pipeline(self, X)


class RandomForestClassifier(BaseModel):
    """RandomForestClassifier с подбором через Optuna. Вероятности калибруются изотонической регрессией.

    params=None → Optuna; params=dict → прямое обучение без тюнинга. ``class_weight``
    по умолчанию `'balanced'` в обеих ветках, но явный `class_weight` в `params`
    побеждает (не молча отбрасывается — `{'class_weight': 'balanced', **params}`).
    `cat_encoder`/`cat_features` — как у `RandomForestRegressor`.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> RandomForestClassifier:
        return _fit_forest_cls(self, _SKRFClassifier, 'random_forest', X_train, y_train, X_valid, y_valid, selected_features, cat_features)

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        return predict_proba_via_pipeline(self, X)


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
    ) -> ExtraTreesRegressor:
        return _fit_forest_reg(self, _SKETRegressor, 'extra_trees', X_train, y_train, X_valid, y_valid, selected_features, cat_features)

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        return predict_via_pipeline(self, X)


class ExtraTreesClassifier(BaseModel):
    """ExtraTreesClassifier с подбором через Optuna. Вероятности калибруются изотонической регрессией.

    params=None → Optuna; params=dict → прямое обучение без тюнинга. ``class_weight``
    по умолчанию `'balanced'` в обеих ветках, но явный `class_weight` в `params`
    побеждает (см. `RandomForestClassifier`).
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> ExtraTreesClassifier:
        return _fit_forest_cls(self, _SKETClassifier, 'extra_trees', X_train, y_train, X_valid, y_valid, selected_features, cat_features)

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        return predict_proba_via_pipeline(self, X)


# ── Вспомогательные функции для классов (избегают дублирования кода) ─────────

def _fit_forest_reg(
    self: BaseModel, EstClass: type, name: str,
    X_train: XInput, y_train: YInput, X_valid: XInput | None, y_valid: YInput | None,
    selected_features: list[str] | None, cat_features: list[str] | None,
) -> BaseModel:
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
        self._model = make_impute_pipeline(EstClass, self.params)
        self._model.fit(Xtr, y_tr)
        self.best_params_ = self.params
    else:
        if X_valid is None:
            raise ValueError('X_valid обязателен при params=None (режим Optuna)')
        X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        Xva = X_valid_enc[self.selected_features_]
        y_va = y_valid.to_numpy(dtype=float)

        def objective(trial: optuna.Trial) -> float:
            pipe = make_impute_pipeline(EstClass, _suggest(trial))
            pipe.fit(Xtr, y_tr)
            return metric_fn(y_va, pipe.predict(Xva))

        study = make_study(direction, ms)
        study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
        self.best_params_ = {**study.best_params, 'random_state': 42, 'n_jobs': -1}
        logger.info('[%s Reg] Best score=%.4f params=%s', name.upper(), study.best_value, self.best_params_)

        self._model = make_impute_pipeline(EstClass, self.best_params_)
        self._model.fit(Xtr, y_tr)

    self.train_pred_ = self._model.predict(Xtr)
    if X_valid is not None:
        X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        self.valid_pred_ = self._model.predict(X_valid_enc[self.selected_features_])
    optuna.logging.set_verbosity(_optuna_prev_verbosity)
    return self


def _fit_forest_cls(
    self: BaseModel, EstClass: type, name: str,
    X_train: XInput, y_train: YInput, X_valid: XInput | None, y_valid: YInput | None,
    selected_features: list[str] | None, cat_features: list[str] | None,
) -> BaseModel:
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
        # class_weight по умолчанию 'balanced', но явный class_weight в params должен
        # побеждать — раньше {**self.params, 'class_weight': 'balanced'} молча отбрасывал
        # выбор пользователя (последний ключ в dict-literal побеждает).
        direct_params = {'class_weight': 'balanced', **self.params}
        self._model = make_impute_pipeline(EstClass, direct_params)
        self._model.fit(Xtr, y_tr)
        self.best_params_ = direct_params
    else:
        if X_valid is None:
            raise ValueError('X_valid обязателен при params=None (режим Optuna)')
        X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        Xva = X_valid_enc[self.selected_features_]
        y_va = y_valid.to_numpy(dtype=int)

        def objective(trial: optuna.Trial) -> float:
            pipe = make_impute_pipeline(EstClass, {**_suggest(trial), 'class_weight': 'balanced'})
            pipe.fit(Xtr, y_tr)
            return metric_fn(y_va, pipe.predict_proba(Xva)[:, 1])

        study = make_study(direction, ms)
        study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
        self.best_params_ = {**study.best_params, 'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'}
        logger.info('[%s Cls] Best score=%.4f params=%s', name.upper(), study.best_value, self.best_params_)

        self._model = make_impute_pipeline(EstClass, self.best_params_)
        self._model.fit(Xtr, y_tr)

    self.train_pred_ = self._model.predict_proba(Xtr)[:, 1]
    if X_valid is not None:
        X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        self.valid_pred_ = self._model.predict_proba(X_valid_enc[self.selected_features_])[:, 1]
        self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
    optuna.logging.set_verbosity(_optuna_prev_verbosity)
    return self

