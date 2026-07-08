# ruff: noqa: N806
"""XGBoost Ranker адаптер — оптимизирует ранжирование через rank:ndcg / rank:pairwise.

Весь датасет подаётся как одна группа (qid=zeros). При бинарных метках (0/1)
rank:ndcg оптимизирует NDCG со всей выборкой как единым запросом, что при
равнозначных объектах эквивалентно оптимизации AUC-подобной метрики.
Скоры калибруются изотонической регрессией на валидационной выборке.

Objective выбирается через model_settings['rank_objective']:
    'rank:ndcg'     — листовая оптимизация NDCG (по умолч.).
    'rank:pairwise' — попарные потери (быстрее, чуть хуже качества).
    'rank:map'      — оптимизация MAP (Mean Average Precision).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import (
    CLS_METRICS,
    fit_calibrator,
    make_xgb_pruning_callback,
    resolve_metric_fn,
    resolve_pruner,
    resolve_timeout,
    set_optuna_verbosity,
)

logger = logging.getLogger(__name__)


def _qid(n: int, group_size: int | None = None) -> np.ndarray:
    """Возвращает массив qid (group id) для XGBoost.

    group_size=None → одна группа (все нули).
    group_size=k    → последовательные группы: [0,0,...,1,1,...,2,...].
    """
    if group_size is None or group_size <= 0 or group_size >= n:
        return np.zeros(n, dtype=np.int32)
    return (np.arange(n, dtype=np.int32) // group_size)


def _to_float(df: pd.DataFrame) -> pd.DataFrame:
    """XGBoost не поддерживает category/object dtype — конвертируем в float."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.name == 'category':
            out[col] = out[col].cat.codes.astype(float)
        elif out[col].dtype == object:
            out[col] = out[col].astype('category').cat.codes.astype(float)
    return out


class XGBoostRanker(BaseModel):
    """XGBoost ранжировщик для бинарной классификации.

    Вся выборка — одна группа; оптимизирует NDCG между позитивами и негативами.
    После обучения скоры калибруются изотонической регрессией на валидации.

    Примеры::

        model = XGBoostRanker(n_optuna_trials=30,
                               model_settings={'rank_objective': 'rank:ndcg'})
        model.fit(X_train, y_train, X_valid, y_valid)
        scores = model.predict_proba(X_new)

        model = XGBoostRanker(params={'n_estimators': 500, 'max_depth': 5,
                                       'learning_rate': 0.05, 'objective': 'rank:ndcg'})
        model.fit(X_train, y_train, X_valid, y_valid)
    """

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any | None = None,
        y_valid: Any | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> XGBoostRanker:
        try:
            import xgboost as xgb
        except ImportError as err:
            raise ImportError('XGBoost не установлен. Выполните: pip install xgboost') from err

        import optuna
        _optuna_prev_verbosity = set_optuna_verbosity(self.model_settings)
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = cat_features or []

        group_size: int | None = self.model_settings.get('group_size', 2000)

        Xtr = _to_float(X_train[self.selected_features_])
        ytr = y_train.values.astype(int)
        qid_tr = _qid(len(Xtr), group_size)

        Xva = yva = qid_va = None
        if X_valid is not None and y_valid is not None:
            Xva = _to_float(X_valid[self.selected_features_])
            yva = y_valid.values.astype(int)
            qid_va = _qid(len(Xva), group_size)

        if self.params is None:
            if Xva is None:
                raise ValueError('X_valid и y_valid обязательны при params=None (Optuna)')
            self._model, self.best_params_ = self._fit_with_optuna(
                xgb, Xtr, ytr, qid_tr, Xva, yva, qid_va,
            )
        else:
            self._model, self.best_params_ = self._fit_direct(
                xgb, Xtr, ytr, qid_tr, Xva, yva, qid_va,
            )

        raw_tr = self._model.predict(Xtr)
        if Xva is not None:
            raw_va = self._model.predict(Xva)
            self.calibrator_ = fit_calibrator(raw_va, yva)
            self.train_pred_ = self.calibrator_.predict(raw_tr)
            self.valid_pred_ = self.calibrator_.predict(raw_va)
            logger.info('[XGB Ranker] Final PR-AUC: %.3f', average_precision_score(yva, self.valid_pred_))
        else:
            lo, hi = raw_tr.min(), raw_tr.max()
            self.train_pred_ = (raw_tr - lo) / (hi - lo + 1e-12)

        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _fit_with_optuna(self, xgb, Xtr, ytr, qid_tr, Xva, yva, qid_va):
        import optuna

        rank_obj = self.model_settings.get('rank_objective', 'rank:ndcg')
        metric_fn, direction = resolve_metric_fn(
            self.model_settings, 'cls_metric', average_precision_score, 'maximize', CLS_METRICS,
        )

        def objective(trial: optuna.Trial) -> float:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 2000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'objective': rank_obj,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0,
                'early_stopping_rounds': 100,
            }
            m = xgb.XGBRanker(**params)
            m.fit(
                Xtr, ytr, qid=qid_tr,
                eval_set=[(Xva, yva)], eval_qid=[qid_va],
                verbose=False, callbacks=[make_xgb_pruning_callback(trial)],
            )
            cal = fit_calibrator(m.predict(Xva), yva)
            return metric_fn(yva, cal.predict(m.predict(Xva)))

        ms = self.model_settings
        logger.info('[XGB Ranker] Optuna: %d trials, objective=%s', self.n_optuna_trials, rank_obj)
        study = optuna.create_study(
            direction=direction, sampler=optuna.samplers.TPESampler(seed=42), pruner=resolve_pruner(ms),
        )
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=resolve_timeout(ms), show_progress_bar=False)

        best_params = {
            **study.best_params,
            'objective': rank_obj,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            'early_stopping_rounds': 100,
        }
        logger.info('[XGB Ranker] Best score=%.4f', study.best_value)
        model = xgb.XGBRanker(**best_params)
        model.fit(
            Xtr, ytr, qid=qid_tr,
            eval_set=[(Xva, yva)], eval_qid=[qid_va],
            verbose=False,
        )
        return model, best_params

    def _fit_direct(self, xgb, Xtr, ytr, qid_tr, Xva, yva, qid_va):
        params = dict(self.params)
        model = xgb.XGBRanker(**params)
        if Xva is not None:
            model.fit(
                Xtr, ytr, qid=qid_tr,
                eval_set=[(Xva, yva)], eval_qid=[qid_va],
                verbose=False,
            )
        else:
            model.fit(Xtr, ytr, qid=qid_tr)
        return model, params

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        Xp = _to_float(X[self.selected_features_])
        raw = self._model.predict(Xp)
        if self.calibrator_ is not None:
            return self.calibrator_.predict(raw)
        lo, hi = raw.min(), raw.max()
        return (raw - lo) / (hi - lo + 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Backward-совместимые функции
# ─────────────────────────────────────────────────────────────────────────────

def train_regression(*args, **kwargs):
    raise NotImplementedError(
        "XGBoostRanker не поддерживает регрессию. "
        "Для регрессии используйте 'xgboost'."
    )


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
    model = XGBoostRanker(n_optuna_trials=n_optuna_trials, model_settings=model_settings or {})
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    infer_scores = model.predict_proba(X_inference)
    return model._model, model.train_pred_, model.valid_pred_, infer_scores, model.best_params_


def make_predict_fn(model: Any, task: str, selected_features: list[str]) -> None:
    return None
