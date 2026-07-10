# ruff: noqa: N806
"""CatBoost Ranker адаптер — оптимизирует ранжирование через YetiRank / QuerySoftMax.

Весь датасет подаётся как одна группа (group_id=zeros). YetiRank использует
стохастическую аппроксимацию NDCG-градиентов; QuerySoftMax — мягкий аналог
Winner-Takes-All для группы. При бинарных метках (0/1) и единственной группе
оба функционально близки к оптимизации AUC.
Скоры калибруются изотонической регрессией на валидационной выборке.

Objective выбирается через model_settings['rank_objective']:
    'YetiRank'       — стохастическая аппроксимация NDCG (по умолч.).
    'YetiRankPairwise' — попарный вариант YetiRank (быстрее).
    'QuerySoftMax'   — мягкий softmax внутри группы.
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
    make_catboost_pruning_callback,
    resolve_metric_fn,
    resolve_pruner,
    resolve_timeout,
    set_optuna_verbosity,
)

logger = logging.getLogger(__name__)


def _import_catboost():
    try:
        from catboost import CatBoostRanker as _ranker
        from catboost import Pool as _pool
        return _ranker, _pool
    except ImportError as err:
        raise ImportError('CatBoost не установлен. Выполните: pip install catboost') from err


def _make_group_ids(n: int, group_size: int | None) -> np.ndarray:
    """Создаёт массив group_id: одна группа при group_size=None, иначе нарезает батчами."""
    if group_size is None or group_size <= 0 or group_size >= n:
        return np.zeros(n, dtype=np.int32)
    return (np.arange(n, dtype=np.int32) // group_size)


def _make_rank_pool(
    Pool,
    X: pd.DataFrame,
    y: np.ndarray | None,
    cat_features: list[str],
    group_size: int | None = None,
):
    group_id = _make_group_ids(len(X), group_size)
    return Pool(X, label=y, cat_features=cat_features, group_id=group_id)


class CatBoostRanker(BaseModel):
    """CatBoost ранжировщик для бинарной классификации.

    Вся выборка — одна группа; YetiRank оптимизирует NDCG-аппроксимацию
    между позитивами и негативами. Нативная поддержка категориальных признаков.
    Скоры калибруются изотонической регрессией на валидации.

    Примеры::

        model = CatBoostRanker(n_optuna_trials=30,
                                model_settings={'rank_objective': 'YetiRank'})
        model.fit(X_train, y_train, X_valid, y_valid,
                  cat_features=['region', 'industry'])
        scores = model.predict_proba(X_new)

        model = CatBoostRanker(params={'iterations': 700, 'max_depth': 5,
                                        'loss_function': 'YetiRank'})
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
    ) -> CatBoostRanker:
        _CB_Ranker, Pool = _import_catboost()

        import optuna
        _optuna_prev_verbosity = set_optuna_verbosity(self.model_settings)
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = cat_features or []
        cat_in_sel = [c for c in self.cat_features_ if c in self.selected_features_]

        group_size: int | None = self.model_settings.get('group_size', 2000)

        Xtr = X_train[self.selected_features_]
        ytr = y_train.values.astype(int)
        tr_pool = _make_rank_pool(Pool, Xtr, ytr, cat_in_sel, group_size)

        va_pool = Xva = yva = None
        if X_valid is not None and y_valid is not None:
            Xva = X_valid[self.selected_features_]
            yva = y_valid.values.astype(int)
            va_pool = _make_rank_pool(Pool, Xva, yva, cat_in_sel, group_size)

        if self.params is None:
            if va_pool is None:
                raise ValueError('X_valid и y_valid обязательны при params=None (Optuna)')
            self._model, self.best_params_ = self._fit_with_optuna(
                _CB_Ranker, Pool, tr_pool, va_pool, Xva, yva, cat_in_sel, group_size,
            )
        else:
            self._model, self.best_params_ = self._fit_direct(
                _CB_Ranker, tr_pool, va_pool,
            )

        raw_tr = self._model.predict(_make_rank_pool(Pool, Xtr, None, cat_in_sel, group_size))
        if va_pool is not None:
            raw_va = self._model.predict(_make_rank_pool(Pool, Xva, None, cat_in_sel, group_size))
            self.calibrator_ = fit_calibrator(raw_va, yva)
            self.train_pred_ = self.calibrator_.predict(raw_tr)
            self.valid_pred_ = self.calibrator_.predict(raw_va)
            logger.info('[CB Ranker] Final PR-AUC: %.3f', average_precision_score(yva, self.valid_pred_))
        else:
            lo, hi = raw_tr.min(), raw_tr.max()
            self.train_pred_ = (raw_tr - lo) / (hi - lo + 1e-12)

        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _fit_with_optuna(self, _CB_Ranker, Pool, tr_pool, va_pool, Xva, yva, cat_in_sel, group_size):
        import optuna

        rank_obj = self.model_settings.get('rank_objective', 'YetiRank')
        metric_fn, direction = resolve_metric_fn(
            self.model_settings, 'cls_metric', average_precision_score, 'maximize', CLS_METRICS,
        )

        def objective(trial: optuna.Trial) -> float:
            params = {
                'iterations': trial.suggest_int('iterations', 300, 2000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 2.0),
                'loss_function': rank_obj,
                'random_seed': 42,
                'verbose': False,
                'early_stopping_rounds': 100,
            }
            m = _CB_Ranker(**params)
            pruning_callback = make_catboost_pruning_callback(trial)
            m.fit(tr_pool, eval_set=va_pool, callbacks=[pruning_callback])
            if pruning_callback.pruned:
                raise optuna.TrialPruned(f'Trial pruned (best iteration {m.get_best_iteration()}).')
            raw_va = m.predict(_make_rank_pool(Pool, Xva, None, cat_in_sel, group_size))
            cal = fit_calibrator(raw_va, yva)
            return metric_fn(yva, cal.predict(raw_va))

        ms = self.model_settings
        logger.info('[CB Ranker] Optuna: %d trials, objective=%s', self.n_optuna_trials, rank_obj)
        study = optuna.create_study(
            direction=direction, sampler=optuna.samplers.TPESampler(seed=42), pruner=resolve_pruner(ms),
        )
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=resolve_timeout(ms), show_progress_bar=False)

        best_params = {
            **study.best_params,
            'loss_function': rank_obj,
            'random_seed': 42,
            'verbose': False,
            'early_stopping_rounds': 100,
        }
        logger.info('[CB Ranker] Best score=%.4f', study.best_value)
        model = _CB_Ranker(**best_params)
        model.fit(tr_pool, eval_set=va_pool)
        return model, best_params

    def _fit_direct(self, _CB_Ranker, tr_pool, va_pool):
        model = _CB_Ranker(**self.params)
        model.fit(tr_pool, eval_set=va_pool)
        return model, dict(self.params)

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        try:
            from catboost import Pool
        except ImportError as err:
            raise ImportError('CatBoost не установлен') from err
        cat_in_sel = [c for c in self.cat_features_ if c in self.selected_features_]
        group_size: int | None = self.model_settings.get('group_size', 2000)
        Xp = X[self.selected_features_]
        pool = _make_rank_pool(Pool, Xp, None, cat_in_sel, group_size)
        raw = self._model.predict(pool)
        if self.calibrator_ is not None:
            return self.calibrator_.predict(raw)
        lo, hi = raw.min(), raw.max()
        return (raw - lo) / (hi - lo + 1e-12)

