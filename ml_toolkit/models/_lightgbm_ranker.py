# ruff: noqa: N806
"""LightGBM Ranker адаптер — оптимизирует ранжирование через LambdaMART / XE-NDCG.

При бинарных метках (0/1) LambdaRank оптимизирует попарные предпочтения
позитив > негатив, что функционально эквивалентно оптимизации AUC.
Скоры калибруются изотонической регрессией на валидации → [0, 1].

Размер группы — model_settings['group_size'] (по умолч. 2000):
    LightGBM ограничивает одну группу ≤ 10 000 строк. При больших датасетах
    обучающий набор случайным образом разбивается на группы заданного размера.
    Каждая группа — независимая задача ранжирования: модель учится ставить
    позитивы выше негативов внутри группы, что аппроксимирует глобальный AUC.
    group_size=None → одна группа (только для датасетов до ~10K строк).

Objective — model_settings['rank_objective']:
    'lambdarank'   — LambdaMART (по умолч.): NDCG-взвешенные попарные потери.
    'rank_xendcg'  — XE-NDCG: мягкая дифференцируемая аппроксимация NDCG.
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
    make_lgb_pruning_callback,
    prep_cat_features,
    resolve_metric_fn,
    resolve_pruner,
    resolve_timeout,
    set_optuna_verbosity,
)

logger = logging.getLogger(__name__)

_prep = prep_cat_features

_LGB_MAX_GROUP = 10_000  # жёсткий лимит LightGBM на размер одной группы


def _make_groups(n: int, group_size: int | None) -> list[int]:
    """Разбивает n строк на группы заданного размера.

    При group_size=None или group_size >= n возвращает одну группу [n].
    Последняя группа вбирает остаток (может быть меньше group_size).
    """
    if group_size is None or group_size <= 0 or group_size >= n:
        return [n]
    n_full, rem = divmod(n, group_size)
    return [group_size] * n_full + ([rem] if rem > 0 else [])


def _shuffle_for_groups(
    X: pd.DataFrame, y: np.ndarray, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    """Перемешивает строки, чтобы в каждой группе был примерный баланс классов."""
    idx = rng.permutation(len(X))
    return X.iloc[idx].reset_index(drop=True), y[idx]


class LightGBMRanker(BaseModel):
    """LightGBM ранжировщик для бинарной классификации.

    Вся выборка передаётся как одна группа, поэтому модель учится ставить
    позитивы выше негативов глобально (≈ оптимизация AUC).
    После обучения скоры калибруются изотонической регрессией на валидации.

    Примеры::

        model = LightGBMRanker(n_optuna_trials=30,
                                model_settings={'rank_objective': 'lambdarank'})
        model.fit(X_train, y_train, X_valid, y_valid)
        scores = model.predict_proba(X_new)   # калиброванные вероятности

        model = LightGBMRanker(params={'n_estimators': 500, 'num_leaves': 64,
                                        'learning_rate': 0.05, 'objective': 'lambdarank'})
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
    ) -> LightGBMRanker:
        try:
            import lightgbm as lgb
        except ImportError as err:
            raise ImportError('LightGBM не установлен. Выполните: pip install lightgbm') from err

        import optuna
        _optuna_prev_verbosity = set_optuna_verbosity(self.model_settings)
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = cat_features or []
        cat_in_sel = [c for c in self.cat_features_ if c in self.selected_features_]

        group_size: int | None = self.model_settings.get('group_size', 2000)
        rng = np.random.default_rng(42)

        Xtr = _prep(X_train, self.selected_features_, self.cat_features_)
        ytr = y_train.values.astype(int)
        Xtr, ytr = _shuffle_for_groups(Xtr, ytr, rng)
        tr_groups = _make_groups(len(Xtr), group_size)

        Xva = yva = va_groups = None
        if X_valid is not None and y_valid is not None:
            Xva = _prep(X_valid, self.selected_features_, self.cat_features_)
            yva = y_valid.values.astype(int)
            Xva, yva = _shuffle_for_groups(Xva, yva, rng)
            va_groups = _make_groups(len(Xva), group_size)

        if self.params is None:
            if Xva is None:
                raise ValueError('X_valid и y_valid обязательны при params=None (Optuna)')
            self._model, self.best_params_ = self._fit_with_optuna(
                lgb, Xtr, ytr, tr_groups, Xva, yva, va_groups, cat_in_sel,
            )
        else:
            self._model, self.best_params_ = self._fit_direct(
                lgb, Xtr, ytr, tr_groups, Xva, yva, va_groups, cat_in_sel,
            )

        raw_tr = self._model.predict(Xtr)
        if Xva is not None:
            raw_va = self._model.predict(Xva)
            self.calibrator_ = fit_calibrator(raw_va, yva)
            self.train_pred_ = self.calibrator_.predict(raw_tr)
            self.valid_pred_ = self.calibrator_.predict(raw_va)
            logger.info('[LGB Ranker] Final PR-AUC: %.3f', average_precision_score(yva, self.valid_pred_))
        else:
            lo, hi = raw_tr.min(), raw_tr.max()
            self.train_pred_ = (raw_tr - lo) / (hi - lo + 1e-12)

        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _fit_with_optuna(self, lgb, Xtr, ytr, tr_groups, Xva, yva, va_groups, cat_in_sel):
        import optuna

        rank_obj = self.model_settings.get('rank_objective', 'lambdarank')
        metric_fn, direction = resolve_metric_fn(
            self.model_settings, 'cls_metric', average_precision_score, 'maximize', CLS_METRICS,
        )
        callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]

        def objective(trial: optuna.Trial) -> float:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 2000, step=100),
                'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'objective': rank_obj,
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1,
            }
            m = lgb.LGBMRanker(**params)
            m.fit(
                Xtr, ytr, group=tr_groups,
                eval_set=[(Xva, yva)], eval_group=[va_groups],
                categorical_feature=cat_in_sel or 'auto',
                callbacks=[*callbacks, make_lgb_pruning_callback(trial)],
            )
            cal = fit_calibrator(m.predict(Xva), yva)
            return metric_fn(yva, cal.predict(m.predict(Xva)))

        ms = self.model_settings
        logger.info(
            '[LGB Ranker] Optuna: %d trials, objective=%s, n_groups_train=%d',
            self.n_optuna_trials, rank_obj, len(tr_groups),
        )
        study = optuna.create_study(
            direction=direction, sampler=optuna.samplers.TPESampler(seed=42), pruner=resolve_pruner(ms),
        )
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=resolve_timeout(ms), show_progress_bar=False)

        best_params = {
            **study.best_params,
            'objective': rank_obj,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1,
        }
        logger.info('[LGB Ranker] Best score=%.4f', study.best_value)
        model = lgb.LGBMRanker(**best_params)
        model.fit(
            Xtr, ytr, group=tr_groups,
            eval_set=[(Xva, yva)], eval_group=[va_groups],
            categorical_feature=cat_in_sel or 'auto',
            callbacks=callbacks,
        )
        return model, best_params

    def _fit_direct(self, lgb, Xtr, ytr, tr_groups, Xva, yva, va_groups, cat_in_sel):
        model = lgb.LGBMRanker(**self.params)
        callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
        logger.debug('[LGB Ranker] fit direct, n_groups=%d', len(tr_groups))
        if Xva is not None:
            model.fit(
                Xtr, ytr, group=tr_groups,
                eval_set=[(Xva, yva)], eval_group=[va_groups],
                categorical_feature=cat_in_sel or 'auto',
                callbacks=callbacks,
            )
        else:
            model.fit(Xtr, ytr, group=tr_groups, categorical_feature=cat_in_sel or 'auto')
        return model, dict(self.params)

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        Xp = _prep(X, self.selected_features_, self.cat_features_)
        raw = self._model.predict(Xp)
        if self.calibrator_ is not None:
            return self.calibrator_.predict(raw)
        lo, hi = raw.min(), raw.max()
        return (raw - lo) / (hi - lo + 1e-12)

