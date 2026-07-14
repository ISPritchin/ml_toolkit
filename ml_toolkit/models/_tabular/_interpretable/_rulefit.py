"""RuleFit — ансамбль правил из деревьев + ElasticNet (Friedman & Popescu, 2008).

Алгоритм: обучает ансамбль деревьев → извлекает булевы правила (пути в деревьях) →
применяет ElasticNet на расширенном пространстве {исходные признаки + правила}.
Результат: человекочитаемые условия с весами, объясняющими вклад каждого правила.

Поддерживаемые имена (model_settings['name']): 'rulefit'

Пакет: imodels (pip install imodels)
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._tabular._interpretable._common import make_impute_scale_pipeline, numeric_features
from ml_toolkit.models._utils import (
    CLS_METRICS,
    REG_METRICS,
    fit_calibrator,
    make_study,
    resolve_metric_fn,
    resolve_timeout,
    set_optuna_verbosity,
)

logger = logging.getLogger(__name__)


# ── Классы (новый API) ────────────────────────────────────────────────────────

class RuleFitRegressor(BaseModel):
    """RuleFitRegressor с подбором max_rules и tree_size через Optuna.

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
    ) -> RuleFitRegressor:
        try:
            from imodels import RuleFitRegressor as _RFReg
        except ImportError as exc:
            raise ImportError('Установи пакет: pip install imodels') from exc

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._num_feats_ = numeric_features(self.selected_features_, self.cat_features_)
        logger.info('[RULEFIT Reg] features=%d', len(self._num_feats_))

        self._prep = make_impute_scale_pipeline()
        X_tr = self._prep.fit_transform(X_train[self._num_feats_].to_numpy(dtype=float))
        y_tr = y_train.to_numpy(dtype=float)

        metric_fn, direction = resolve_metric_fn(ms, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS)

        if self.params is not None:
            self._model = _RFReg(**self.params)
            self._model.fit(X_tr, y_tr, feature_names=self._num_feats_)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_va = self._prep.transform(X_valid[self._num_feats_].to_numpy(dtype=float))
            y_va = y_valid.to_numpy(dtype=float)

            def objective(trial: optuna.Trial) -> float:
                params = {
                    'max_rules': trial.suggest_int('max_rules', 50, 500, step=50),
                    'tree_size': trial.suggest_int('tree_size', 2, 8),
                    'random_state': 42,
                }
                m = _RFReg(**params)
                m.fit(X_tr, y_tr, feature_names=self._num_feats_)
                return metric_fn(y_va, m.predict(X_va))

            study = make_study(direction, ms)
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = {**study.best_params, 'random_state': 42}
            logger.info('[RULEFIT Reg] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = _RFReg(**self.best_params_)
            self._model.fit(X_tr, y_tr, feature_names=self._num_feats_)

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


class RuleFitClassifier(BaseModel):
    """RuleFitClassifier с подбором max_rules и tree_size через Optuna.

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
    ) -> RuleFitClassifier:
        try:
            from imodels import RuleFitClassifier as _RFCls
        except ImportError as exc:
            raise ImportError('Установи пакет: pip install imodels') from exc

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._num_feats_ = numeric_features(self.selected_features_, self.cat_features_)
        logger.info('[RULEFIT Cls] features=%d', len(self._num_feats_))

        self._prep = make_impute_scale_pipeline()
        X_tr = self._prep.fit_transform(X_train[self._num_feats_].to_numpy(dtype=float))
        y_tr = y_train.to_numpy(dtype=int)

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

        if self.params is not None:
            self._model = _RFCls(**self.params)
            self._model.fit(X_tr, y_tr, feature_names=self._num_feats_)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_va = self._prep.transform(X_valid[self._num_feats_].to_numpy(dtype=float))
            y_va = y_valid.to_numpy(dtype=int)

            def objective(trial: optuna.Trial) -> float:
                params = {
                    'max_rules': trial.suggest_int('max_rules', 50, 500, step=50),
                    'tree_size': trial.suggest_int('tree_size', 2, 8),
                    'random_state': 42,
                }
                m = _RFCls(**params)
                m.fit(X_tr, y_tr, feature_names=self._num_feats_)
                return metric_fn(y_va, m.predict_proba(X_va)[:, 1])

            study = make_study(direction, ms)
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), show_progress_bar=False)
            self.best_params_ = {**study.best_params, 'random_state': 42}
            logger.info('[RULEFIT Cls] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = _RFCls(**self.best_params_)
            self._model.fit(X_tr, y_tr, feature_names=self._num_feats_)

        self.train_pred_ = self._model.predict_proba(X_tr)[:, 1]
        if X_valid is not None:
            X_va = self._prep.transform(X_valid[self._num_feats_].to_numpy(dtype=float))
            self.valid_pred_ = self._model.predict_proba(X_va)[:, 1]
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._model.predict_proba(
            self._prep.transform(X[self._num_feats_].to_numpy(dtype=float))
        )[:, 1]
        return self.calibrator_.predict(raw) if self.calibrator_ is not None else raw

