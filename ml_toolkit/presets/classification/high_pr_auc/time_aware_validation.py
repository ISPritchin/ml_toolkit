"""TimeAwareValidationClassifier: расширяющееся окно по ts_key с purge/embargo.

Закрывает методологический риск [M2] из plan.txt (см. корень репозитория):
«Нигде нет группового/временного сплита. Для панельных данных (клиент × месяц)
случайный или единственный (one-shot) train/valid/test сплит даёт шумную,
невоспроизводимую оценку и подбор гиперпараметров».

Вместо одного train/valid/test cutoff — n_windows последовательных окон:
сортированные уникальные периоды (по ts_key) делятся на n_windows + 1 блок,
первый блок — только для train, каждый следующий блок по очереди становится
validation-окном, а train для него — все периоды до этого блока, за вычетом
embargo_periods периодов непосредственно перед ним (purge/embargo — типовая
защита от утечки, когда метка присваивается с лагом относительно события,
как в last_full_month_before_it-подобных пайплайнах: события у самой границы
train/val могут быть размечены по данным, которых val ещё не видит, но train
уже частично «видит будущее» относительно них без embargo).

Архитектура (base) тюнится Optuna ОДИН раз на последнем (самом полном) окне —
дешёвый proxy-тюнинг, как в EasyEnsembleClassifier/WeightedBaggingByRecency —
и переиспользуется во всех n_windows окнах, вместо n_optuna_trials * n_windows
полных тюнингов.

Атрибуты после fit:
  estimators_       — список обученных моделей, по одной на окно
  window_scores_    — val PR-AUC каждого окна
  window_bounds_    — границы train/val каждого окна (для диагностики/графиков)
  oof_score_        — PR-AUC по объединённым out-of-window предсказаниям всех
                       окон — честная, не привязанная к одному-единственному
                       валидационному месяцу оценка
  final_estimator_  — модель последнего (самого свежего) окна — то, что
                       предполагается деплоить; predict_proba() использует её
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.presets.classification._base import BasePreset
from ml_toolkit.presets.classification._optuna_utils import (
    CatBoostPruningCallback,
    catboost_arch_space,
    make_pruner,
)
from ml_toolkit.presets.classification._time_utils import compute_periods

logger = logging.getLogger(__name__)

_DEFAULT_LGB_PARAMS: dict[str, Any] = {
    'n_estimators': 500,
    'max_depth': 5,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'n_jobs': -1,
}

_DEFAULT_CBT_PARAMS: dict[str, Any] = {
    'iterations': 500,
    'max_depth': 5,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'early_stopping_rounds': 80,
    'loss_function': 'Logloss',
    'eval_metric': 'PRAUC',
    'verbose': 0,
}


class TimeAwareValidationClassifier(BasePreset):
    """Walk-forward (expanding window + purge/embargo) валидация и обучение по ts_key.

    Parameters
    ----------
    n_windows:
        Число последовательных validation-окон (рекомендуется 3–8; больше
        окон — точнее оценка дисперсии, но каждое окно тренируется на меньшей
        истории).
    embargo_periods:
        Число периодов, исключаемых из train непосредственно перед каждым
        validation-окном (purge/embargo). 0 — без зазора (используйте только
        если метка гарантированно не зависит от будущего относительно своего
        периода).
    period_unit:
        Pandas frequency alias ('M', 'W', 'D', ...) для бинования datetime
        `ts_key` в периоды. Игнорируется для числового `ts_key`.
    base:
        'catboost' (по умолчанию) или 'lightgbm'.
    base_params:
        Гиперпараметры базовой модели. None → дефолтные для выбранного base.
        Игнорируется, если n_optuna_trials > 0.
    n_optuna_trials:
        Если > 0, архитектура (одна на все n_windows окон) подбирается через
        Optuna по val PR-AUC на последнем (самом полном) окне.
    param_space, optuna_timeout, optuna_verbose, optuna_pruner, random_seed,
    cat_features, selected_features:
        Как в EasyEnsembleClassifier.

    Атрибуты после fit — см. докстринг модуля.

    Пример::

        model = TimeAwareValidationClassifier(n_windows=5, embargo_periods=1)
        model.fit(X, y, ts_key=X['REPORT_DATE'], selected_features=[...])
        print(model.window_scores_, model.oof_score_)
        proba = model.predict_proba(X_test)

    """

    def __init__(
        self,
        n_windows: int = 5,
        embargo_periods: int = 1,
        period_unit: str = 'M',
        base: str = 'catboost',
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        param_space: Callable[[Any], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        optuna_pruner: str | object | None = 'none',
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(params=base_params, n_optuna_trials=n_optuna_trials)
        if base not in ('lightgbm', 'catboost'):
            raise ValueError(f"base должен быть 'lightgbm' или 'catboost', получено {base!r}")
        if n_windows < 2:
            raise ValueError(f'n_windows должен быть >= 2, получено {n_windows}')
        if embargo_periods < 0:
            raise ValueError(f'embargo_periods должен быть >= 0, получено {embargo_periods}')
        self.n_windows = n_windows
        self.embargo_periods = embargo_periods
        self.period_unit = period_unit
        self.base = base
        self.base_params = base_params
        self.param_space = param_space
        self.optuna_timeout = optuna_timeout
        self.optuna_verbose = optuna_verbose
        self.optuna_pruner = optuna_pruner
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.estimators_: list[Any] = []
        self.window_scores_: list[float] = []
        self.window_bounds_: list[dict[str, Any]] = []
        self.oof_score_: float = 0.0
        self.final_estimator_: Any = None

    # ── Обучение одной модели ────────────────────────────────────────────────

    def _fit_one_lgb(self, X_tr, y_tr, X_va, y_va, params: dict[str, Any] | None = None) -> Any:
        import lightgbm as lgb

        p = {**(params or self.base_params or _DEFAULT_LGB_PARAMS), 'random_state': self.random_seed}
        model = lgb.LGBMClassifier(**p)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        return model

    def _fit_one_cbt(self, X_tr, y_tr, X_va, y_va, params: dict[str, Any] | None = None) -> Any:
        from catboost import CatBoostClassifier, Pool

        p = {**(params or self.base_params or _DEFAULT_CBT_PARAMS), 'random_seed': self.random_seed}
        model = CatBoostClassifier(**p)
        tr_pool = Pool(X_tr, y_tr, cat_features=self.cat_features_)
        va_pool = Pool(X_va, y_va, cat_features=self.cat_features_)
        model.fit(tr_pool, eval_set=va_pool, verbose=False)
        return model

    def _predict_one(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        if self.base == 'lightgbm':
            return model.predict_proba(X)[:, 1]
        from catboost import Pool
        return model.predict_proba(Pool(X, cat_features=self.cat_features_))[:, 1]

    # ── Optuna (один раз, на последнем окне) ─────────────────────────────────

    def _tune_cbt(self, X_tr, y_tr, X_va, y_va) -> dict[str, Any]:
        from catboost import CatBoostClassifier, Pool
        import optuna

        _optuna_prev_verbosity = optuna.logging.get_verbosity()
        if not self.optuna_verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        tr_pool = Pool(X_tr, y_tr, cat_features=self.cat_features_)
        va_pool = Pool(X_va, y_va, cat_features=self.cat_features_)

        def objective(trial: optuna.Trial) -> float:
            tunable = self.param_space(trial) if self.param_space is not None else catboost_arch_space(trial)
            params = {
                'loss_function': 'Logloss',
                'eval_metric': 'PRAUC',
                'early_stopping_rounds': 80,
                'random_seed': self.random_seed,
                'verbose': 0,
                **tunable,
            }
            trial.set_user_attr('cb_params', params)
            m = CatBoostClassifier(**params)
            if params.get('task_type') == 'GPU':
                m.fit(tr_pool, eval_set=va_pool, verbose=False)
            else:
                pruning_cb = CatBoostPruningCallback(trial, params['eval_metric'])
                m.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=[pruning_cb])
                pruning_cb.check_pruned()
            p = m.predict_proba(va_pool)[:, 1]
            return float(average_precision_score(y_va, p))

        logger.info('[TimeAwareValidation] Optuna (catboost): %d trials (последнее окно)', self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed),
                                    pruner=make_pruner(self.optuna_pruner))
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return dict(study.best_trial.user_attrs['cb_params'])

    def _tune_lgb(self, X_tr, y_tr, X_va, y_va) -> dict[str, Any]:
        import lightgbm as lgb
        import optuna

        _optuna_prev_verbosity = optuna.logging.get_verbosity()
        if not self.optuna_verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _default_space(trial: optuna.Trial) -> dict[str, Any]:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            }

        def objective(trial: optuna.Trial) -> float:
            tunable = self.param_space(trial) if self.param_space is not None else _default_space(trial)
            params = {'random_state': self.random_seed, 'verbose': -1, 'n_jobs': -1, **tunable}
            trial.set_user_attr('cb_params', params)
            m = lgb.LGBMClassifier(**params)
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            p = m.predict_proba(X_va)[:, 1]
            return float(average_precision_score(y_va, p))

        logger.info('[TimeAwareValidation] Optuna (lightgbm): %d trials (последнее окно)', self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed))
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return dict(study.best_trial.user_attrs['cb_params'])

    # ── Границы окон ──────────────────────────────────────────────────────────

    def _build_windows(self, periods: np.ndarray) -> list[dict[str, np.ndarray]]:
        uniq = np.unique(periods)
        if len(uniq) < self.n_windows + 1:
            raise ValueError(
                f'Недостаточно уникальных периодов ({len(uniq)}) для n_windows={self.n_windows} '
                f'(нужно минимум {self.n_windows + 1}: один блок только под train + по одному на окно)'
            )
        blocks = np.array_split(uniq, self.n_windows + 1)

        windows = []
        for i in range(self.n_windows):
            val_periods = blocks[i + 1]
            val_start = val_periods.min()
            val_mask = np.isin(periods, val_periods)
            train_mask = periods < (val_start - self.embargo_periods)

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            windows.append({'train_idx': train_idx, 'val_idx': val_idx, 'periods': periods})
        return windows

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: Any,
        y: Any,
        ts_key: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> TimeAwareValidationClassifier:
        X, y, _, _ = self._coerce_inputs(X, y, None, None)
        feats = self._resolve_features(X, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features if cat_features is not None else self.cat_features

        y_arr = y.values
        X_feats = X[feats]

        ts_series = pd.Series(np.asarray(ts_key)).reset_index(drop=True)
        if len(ts_series) != len(X_feats):
            raise ValueError(
                f'ts_key должен быть той же длины, что X: {len(ts_series)} != {len(X_feats)}'
            )
        periods = compute_periods(ts_series, self.period_unit)
        windows = self._build_windows(periods)

        logger.info(
            '[TimeAwareValidation] n_windows=%d  embargo_periods=%d  base=%s  n_rows=%d',
            self.n_windows, self.embargo_periods, self.base, len(X_feats),
        )

        tuned_params = None
        if self.n_optuna_trials > 0:
            last = windows[-1]
            if len(np.unique(y_arr[last['train_idx']])) < 2 or len(np.unique(y_arr[last['val_idx']])) < 2:
                raise ValueError('Последнее окно вырождено (один класс) — тюнинг Optuna невозможен')
            X_tr_last, y_tr_last = X_feats.iloc[last['train_idx']], y_arr[last['train_idx']]
            X_va_last, y_va_last = X_feats.iloc[last['val_idx']], y_arr[last['val_idx']]
            tuned_params = (
                self._tune_lgb(X_tr_last, y_tr_last, X_va_last, y_va_last) if self.base == 'lightgbm'
                else self._tune_cbt(X_tr_last, y_tr_last, X_va_last, y_va_last)
            )

        self.estimators_ = []
        self.window_scores_ = []
        self.window_bounds_ = []
        oof_pred = np.full(len(X_feats), np.nan)

        for i, w in enumerate(windows):
            train_idx, val_idx = w['train_idx'], w['val_idx']
            if len(train_idx) == 0 or len(val_idx) == 0:
                logger.warning('[TimeAwareValidation] окно %d/%d пустое (train=%d, val=%d) — пропущено',
                               i + 1, self.n_windows, len(train_idx), len(val_idx))
                continue
            if len(np.unique(y_arr[train_idx])) < 2 or len(np.unique(y_arr[val_idx])) < 2:
                logger.warning('[TimeAwareValidation] окно %d/%d вырождено (один класс) — пропущено',
                               i + 1, self.n_windows)
                continue

            X_tr, y_tr = X_feats.iloc[train_idx], y_arr[train_idx]
            X_va, y_va = X_feats.iloc[val_idx], y_arr[val_idx]

            model = (
                self._fit_one_lgb(X_tr, y_tr, X_va, y_va, tuned_params) if self.base == 'lightgbm'
                else self._fit_one_cbt(X_tr, y_tr, X_va, y_va, tuned_params)
            )
            va_score = self._predict_one(model, X_va)
            ap = float(average_precision_score(y_va, va_score))

            self.estimators_.append(model)
            self.window_scores_.append(ap)
            self.window_bounds_.append({
                'window': i,
                'train_start': float(periods[train_idx].min()), 'train_end': float(periods[train_idx].max()),
                'val_start': float(periods[val_idx].min()), 'val_end': float(periods[val_idx].max()),
                'n_train': len(train_idx), 'n_val': len(val_idx),
            })
            oof_pred[val_idx] = va_score
            self.final_estimator_ = model
            logger.info('[TimeAwareValidation] окно %d/%d  n_train=%d  n_val=%d  val PR-AUC=%.4f',
                        i + 1, self.n_windows, len(train_idx), len(val_idx), ap)

        if self.final_estimator_ is None:
            raise ValueError('Все окна вырождены или пусты — ни одной модели не обучено')

        oof_mask = ~np.isnan(oof_pred)
        self.oof_score_ = float(average_precision_score(y_arr[oof_mask], oof_pred[oof_mask]))
        logger.info('[TimeAwareValidation] oof PR-AUC=%.4f (по %d/%d строкам)  mean window PR-AUC=%.4f',
                    self.oof_score_, int(oof_mask.sum()), len(X_feats), float(np.mean(self.window_scores_)))

        self.valid_pred_ = oof_pred[oof_mask]
        self.train_pred_ = self._predict_one(self.final_estimator_, X_feats)

        self.best_params_ = {
            'n_windows': len(self.estimators_),
            'embargo_periods': self.embargo_periods,
            'base': self.base,
            'base_params': tuned_params or (self.base_params or (
                _DEFAULT_LGB_PARAMS if self.base == 'lightgbm' else _DEFAULT_CBT_PARAMS
            )),
        }
        self._model = True
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_feats = X[self.selected_features_]
        return self._predict_one(self.final_estimator_, X_feats)
