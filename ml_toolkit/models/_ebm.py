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
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import CLS_METRICS, REG_METRICS, calibrate_proba, encode_cat_features, fit_calibrator, resolve_metric_fn

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _num_features(selected_features: list[str], cat_features: list[str]) -> list[str]:
    cat_set = set(cat_features)
    return [f for f in selected_features if f not in cat_set]


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
    ) -> 'EBMRegressor':
        try:
            from interpret.glassbox import ExplainableBoostingRegressor
        except ImportError as exc:
            raise ImportError('Установи пакет: pip install interpret') from exc

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings

        X_train, X_valid_enc, _, self.selected_features_ = encode_cat_features(
            X_train, X_valid if X_valid is not None else X_train,
            X_train, self.selected_features_, self.cat_features_, ms,
        )
        self._num_feats_ = _num_features(self.selected_features_, self.cat_features_)

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

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), show_progress_bar=False)
            self.best_params_ = {**study.best_params, 'random_state': 42}
            logger.info('[EBM Reg] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = ExplainableBoostingRegressor(**self.best_params_)
            self._model.fit(Xtr, y_tr)

        self.train_pred_ = self._model.predict(Xtr)
        if X_valid is not None:
            self.valid_pred_ = self._model.predict(X_valid_enc[self._num_feats_])
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(X[self._num_feats_]))


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
    ) -> 'EBMClassifier':
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
        except ImportError as exc:
            raise ImportError('Установи пакет: pip install interpret') from exc

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings

        X_train, X_valid_enc, _, self.selected_features_ = encode_cat_features(
            X_train, X_valid if X_valid is not None else X_train,
            X_train, self.selected_features_, self.cat_features_, ms,
        )
        self._num_feats_ = _num_features(self.selected_features_, self.cat_features_)

        Xtr = X_train[self._num_feats_]
        y_tr = y_train.to_numpy(dtype=int)

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

        if self.params is not None:
            self._model = ExplainableBoostingClassifier(**self.params)
            self._model.fit(Xtr, y_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            Xva = X_valid_enc[self._num_feats_]
            y_va = y_valid.to_numpy(dtype=int)

            def objective(trial: optuna.Trial) -> float:
                m = ExplainableBoostingClassifier(**_ebm_suggest(trial))
                m.fit(Xtr, y_tr)
                return metric_fn(y_va, m.predict_proba(Xva)[:, 1])

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), show_progress_bar=False)
            self.best_params_ = {**study.best_params, 'random_state': 42}
            logger.info('[EBM Cls] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = ExplainableBoostingClassifier(**self.best_params_)
            self._model.fit(Xtr, y_tr)

        self.train_pred_ = self._model.predict_proba(Xtr)[:, 1]
        if X_valid is not None:
            self.valid_pred_ = self._model.predict_proba(X_valid_enc[self._num_feats_])[:, 1]
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self._model.predict_proba(X[self._num_feats_])[:, 1])
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
    model = EBMRegressor(n_optuna_trials=n_optuna_trials, model_settings=model_settings)
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    _pp = postprocess_fn or (lambda _X, p: p)
    train_pred = _pp(X_train, model.train_pred_)
    valid_pred = _pp(X_valid, model.valid_pred_)
    infer_pred = _pp(X_inference, model.predict(X_inference))
    logger.info('[EBM Reg] Final MAE: %.3f', mean_absolute_error(y_valid, valid_pred))
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
    model = EBMClassifier(n_optuna_trials=n_optuna_trials, model_settings=model_settings or {})
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    infer_proba = model.predict_proba(X_inference)
    logger.info('[EBM Cls] Final PR-AUC: %.3f', average_precision_score(y_valid, model.valid_pred_))
    return model._model, model.train_pred_, model.valid_pred_, infer_proba, model.best_params_


def make_predict_fn(model: Any, task: str, selected_features: list[str]) -> Any:
    """Возвращает callable (X → np.ndarray) для permutation importance через EBM predict."""
    _m = model
    _feats = list(getattr(model, 'feature_names_in_', None) or selected_features)
    if task == 'regression':
        return lambda X: np.asarray(_m.predict(X[_feats]))
    return lambda X: np.asarray(_m.predict_proba(X[_feats])[:, 1])
