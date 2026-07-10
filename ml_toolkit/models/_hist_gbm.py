"""HistGradientBoosting (sklearn).

Gradient Boosting на гистограммах — аналог LightGBM внутри sklearn:
- NaN обрабатывает нативно (внутренний surrogate-сплит).
- Категориальные признаки поддерживает через categorical_features=.
- Feature importance: permutation (встроенный feature_importances_ — MDI-like).
- SHAP: TreeExplainer (через внутренние C-деревья).
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

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

    Нативная обработка NaN и категориальных признаков через индексы столбцов
    (`categorical_features=`, sklearn >= 1.2) — индексы вычисляются по `cat_features`
    и применяются в обеих ветках, `params=None` и `params={...}` (явный `categorical_features`
    в `params` побеждает над вычисленным). params=None → Optuna; params=dict → прямое
    обучение без тюнинга.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> HistGBMRegressor:
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_, self.selected_features_ = \
            build_cat_encoder(X_train, self.selected_features_, self.cat_features_, ms)
        X_train_enc = apply_cat_encoder(X_train, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        extra = _make_extra(self.selected_features_, self._cat_in_sel_)

        X_tr = X_train_enc[self.selected_features_].to_numpy(dtype=float)
        y_tr = y_train.to_numpy(dtype=float)

        metric_fn, direction = resolve_metric_fn(ms, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS)

        if self.params is not None:
            # extra (categorical_features) должен применяться и здесь, а не только в
            # Optuna-ветке ниже — иначе cat_features молча теряют нативную categorical-
            # обработку HistGBM при явных params. self.params (если содержит свой
            # categorical_features) побеждает над вычисленным extra.
            direct_params = {**extra, **self.params}
            self._model = HistGradientBoostingRegressor(**direct_params)
            self._model.fit(X_tr, y_tr)
            self.best_params_ = direct_params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
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
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            X_va = X_valid_enc[self.selected_features_].to_numpy(dtype=float)
            self.valid_pred_ = self._model.predict(X_va)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = apply_cat_encoder(X, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        return self._model.predict(X_enc[self.selected_features_].to_numpy(dtype=float))


class HistGBMClassifier(BaseModel):
    """HistGradientBoostingClassifier с автоматическим подбором гиперпараметров через Optuna.

    Категориальные признаки — как у `HistGBMRegressor` (индексы применяются в обеих ветках).
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
    ) -> HistGBMClassifier:
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_, self.selected_features_ = \
            build_cat_encoder(X_train, self.selected_features_, self.cat_features_, ms)
        X_train_enc = apply_cat_encoder(X_train, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        extra = _make_extra(self.selected_features_, self._cat_in_sel_)

        X_tr = X_train_enc[self.selected_features_].to_numpy(dtype=float)
        y_tr = y_train.to_numpy(dtype=int)

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

        if self.params is not None:
            direct_params = {**extra, **self.params}
            self._model = HistGradientBoostingClassifier(**direct_params)
            self._model.fit(X_tr, y_tr)
            self.best_params_ = direct_params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
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
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            X_va = X_valid_enc[self.selected_features_].to_numpy(dtype=float)
            self.valid_pred_ = self._model.predict_proba(X_va)[:, 1]
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = apply_cat_encoder(X, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        raw = self._model.predict_proba(X_enc[self.selected_features_].to_numpy(dtype=float))[:, 1]
        return self.calibrator_.predict(raw) if self.calibrator_ is not None else raw

