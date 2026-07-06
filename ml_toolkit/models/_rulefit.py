"""RuleFit — ансамбль правил из деревьев + ElasticNet (Friedman & Popescu, 2008).

Алгоритм: обучает ансамбль деревьев → извлекает булевы правила (пути в деревьях) →
применяет ElasticNet на расширенном пространстве {исходные признаки + правила}.
Результат: человекочитаемые условия с весами, объясняющими вклад каждого правила.

Поддерживаемые имена (model_settings['name']): 'rulefit'

Пакет: imodels (pip install imodels)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import CLS_METRICS, REG_METRICS, calibrate_proba, fit_calibrator, resolve_metric_fn

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_prep() -> Pipeline:
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])


def _num_features(selected_features: list[str], cat_features: list[str]) -> list[str]:
    cat_set = set(cat_features)
    return [f for f in selected_features if f not in cat_set]


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
    ) -> 'RuleFitRegressor':
        try:
            from imodels import RuleFitRegressor as _RFReg
        except ImportError as exc:
            raise ImportError('Установи пакет: pip install imodels') from exc

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings

        self._num_feats_ = _num_features(self.selected_features_, self.cat_features_)
        logger.info('[RULEFIT Reg] features=%d', len(self._num_feats_))

        self._prep = _make_prep()
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

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), show_progress_bar=False)
            self.best_params_ = {**study.best_params, 'random_state': 42}
            logger.info('[RULEFIT Reg] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = _RFReg(**self.best_params_)
            self._model.fit(X_tr, y_tr, feature_names=self._num_feats_)

        self.train_pred_ = self._model.predict(X_tr)
        if X_valid is not None:
            X_va = self._prep.transform(X_valid[self._num_feats_].to_numpy(dtype=float))
            self.valid_pred_ = self._model.predict(X_va)
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
    ) -> 'RuleFitClassifier':
        try:
            from imodels import RuleFitClassifier as _RFCls
        except ImportError as exc:
            raise ImportError('Установи пакет: pip install imodels') from exc

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings

        self._num_feats_ = _num_features(self.selected_features_, self.cat_features_)
        logger.info('[RULEFIT Cls] features=%d', len(self._num_feats_))

        self._prep = _make_prep()
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

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
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
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._model.predict_proba(
            self._prep.transform(X[self._num_feats_].to_numpy(dtype=float))
        )[:, 1]
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
    model = RuleFitRegressor(n_optuna_trials=n_optuna_trials, model_settings=model_settings)
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    _pp = postprocess_fn or (lambda _X, p: p)
    train_pred = _pp(X_train, model.train_pred_)
    valid_pred = _pp(X_valid, model.valid_pred_)
    infer_pred = _pp(X_inference, model.predict(X_inference))
    logger.info('[RULEFIT Reg] Final MAE: %.3f', mean_absolute_error(y_valid, valid_pred))
    return (model._model, model._prep, model._num_feats_), train_pred, valid_pred, infer_pred, model.best_params_


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
    model = RuleFitClassifier(n_optuna_trials=n_optuna_trials, model_settings=model_settings or {})
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    infer_proba = model.predict_proba(X_inference)
    logger.info('[RULEFIT Cls] Final PR-AUC: %.3f', average_precision_score(y_valid, model.valid_pred_))
    return (model._model, model._prep, model._num_feats_), model.train_pred_, model.valid_pred_, infer_proba, model.best_params_


def make_predict_fn(model: Any, task: str, selected_features: list[str]) -> Any:
    """Возвращает callable (X → np.ndarray) с QuantileTransformer-препроцессингом для permutation importance."""
    import numpy as _np  # noqa: PLC0415
    _m, _prep, _nf = model
    if task == 'regression':
        return lambda X: _np.asarray(_m.predict(_prep.transform(X[_nf].to_numpy(dtype=float))))
    return lambda X: _np.asarray(_m.predict_proba(_prep.transform(X[_nf].to_numpy(dtype=float))))[:, 1]
