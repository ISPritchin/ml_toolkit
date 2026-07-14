"""Explainable Boosting Machine (EBM) — интерпретируемый GAM с попарными взаимодействиями.

EBM строит Generalized Additive Model через cyclic gradient boosting: каждая итерация
обновляет одну shape function за раз. Попарные взаимодействия — отдельные 2D shape functions.
Интерпретируемость через shape plots для каждого признака и каждой пары.
Точность сопоставима с GBDT на средних датасетах.

Поддерживаемые имена (model_settings['name']): 'ebm'

Пакет: interpret (pip install interpret)
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._tabular._interpretable._common import numeric_features
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


def _ebm_suggest(trial: optuna.Trial) -> dict:
    return {
        'max_bins': trial.suggest_int('max_bins', 32, 512, step=32),
        'interactions': trial.suggest_int('interactions', 0, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_rounds': trial.suggest_int('max_rounds', 1000, 10000, step=1000),
        'random_state': 42,
    }


# ── Классы (новый API) ────────────────────────────────────────────────────────

class EBMRegressor(BaseModel):
    """ExplainableBoostingRegressor с автоматическим подбором гиперпараметров через Optuna.

    Категориальные признаки исключаются; принимает только числовые.
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
    ) -> EBMRegressor:
        try:
            from interpret.glassbox import ExplainableBoostingRegressor
        except ImportError as exc:
            raise ImportError('Установи пакет: pip install interpret') from exc

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
        self._num_feats_ = numeric_features(self.selected_features_, self.cat_features_)

        Xtr = X_train[self._num_feats_]
        y_tr = y_train.to_numpy(dtype=float)

        metric_fn, direction = resolve_metric_fn(ms, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS)

        if self.params is not None:
            self._model = ExplainableBoostingRegressor(**self.params)
            self._model.fit(Xtr, y_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            Xva = X_valid_enc[self._num_feats_]
            y_va = y_valid.to_numpy(dtype=float)

            def objective(trial: optuna.Trial) -> float:
                m = ExplainableBoostingRegressor(**_ebm_suggest(trial))
                m.fit(Xtr, y_tr)
                return metric_fn(y_va, m.predict(Xva))

            study = make_study(direction, ms)
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = {**study.best_params, 'random_state': 42}
            logger.info('[EBM Reg] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = ExplainableBoostingRegressor(**self.best_params_)
            self._model.fit(Xtr, y_tr)

        self.train_pred_ = self._model.predict(Xtr)
        if X_valid is not None:
            self.valid_pred_ = self._model.predict(X_valid_enc[self._num_feats_])
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = apply_cat_encoder(X, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        return np.asarray(self._model.predict(X_enc[self._num_feats_]))


class EBMClassifier(BaseModel):
    """ExplainableBoostingClassifier с автоматическим подбором гиперпараметров через Optuna.

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
    ) -> EBMClassifier:
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
        except ImportError as exc:
            raise ImportError('Установи пакет: pip install interpret') from exc

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
        self._num_feats_ = numeric_features(self.selected_features_, self.cat_features_)

        Xtr = X_train[self._num_feats_]
        y_tr = y_train.to_numpy(dtype=int)
        sw_tr = compute_sample_weight('balanced', y_tr)

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

        if self.params is not None:
            self._model = ExplainableBoostingClassifier(**self.params)
            self._model.fit(Xtr, y_tr, sample_weight=sw_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            Xva = X_valid_enc[self._num_feats_]
            y_va = y_valid.to_numpy(dtype=int)

            def objective(trial: optuna.Trial) -> float:
                m = ExplainableBoostingClassifier(**_ebm_suggest(trial))
                m.fit(Xtr, y_tr, sample_weight=sw_tr)
                return metric_fn(y_va, m.predict_proba(Xva)[:, 1])

            study = make_study(direction, ms)
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = {**study.best_params, 'random_state': 42}
            logger.info('[EBM Cls] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = ExplainableBoostingClassifier(**self.best_params_)
            self._model.fit(Xtr, y_tr, sample_weight=sw_tr)

        self.train_pred_ = self._model.predict_proba(Xtr)[:, 1]
        if X_valid is not None:
            self.valid_pred_ = self._model.predict_proba(X_valid_enc[self._num_feats_])[:, 1]
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = apply_cat_encoder(X, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        raw = np.asarray(self._model.predict_proba(X_enc[self._num_feats_])[:, 1])
        return self.calibrator_.predict(raw) if self.calibrator_ is not None else raw

