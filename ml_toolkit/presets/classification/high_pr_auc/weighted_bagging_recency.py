"""WeightedBaggingByRecency: бэггинг с вероятностью попадания строки в бутстрэп,
убывающей экспоненциально с давностью (по ts_key).

Компромисс между двумя крайностями:
  - полный рефит на всей истории — старые паттерны наравне со свежими разбавляют
    сигнал, если поведение дрейфует;
  - жёсткое скользящее окно (TemporalEnsembleClassifier/RollingRefitPolicy) —
    старые паттерны выбрасываются совсем, хотя иногда возвращаются (сезонность).

Здесь ничего не выбрасывается: каждая строка участвует в каждом бутстрэпе с
вероятностью, пропорциональной 0.5 ** (age_periods / halflife_periods), где
age_periods — возраст строки в периодах (по умолчанию месяцах) относительно
самого свежего периода в train. N estimators обучаются на N независимых
взвешенных бутстрэпах, финальный скор — среднее нормированных рангов (как в
EasyEnsembleClassifier).
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.models._utils import fit_rank_reference, rank_transform
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


def _recency_weights(periods: np.ndarray, halflife_periods: float) -> np.ndarray:
    """0.5 ** (age / halflife_periods), нормировано в вероятностное распределение.

    age=0 (самый свежий период в train) -> вес 1.0 до нормировки; чем дальше
    период от максимума, тем экспоненциально меньше шанс строки попасть в
    очередной бутстрэп-сэмпл.
    """
    age = periods.max() - periods
    w = 0.5 ** (age / halflife_periods)
    return w / w.sum()


class WeightedBaggingByRecency(BasePreset):
    """Бэггинг с экспоненциально убывающей по давности вероятностью сэмплирования.

    Parameters
    ----------
    n_estimators:
        Количество базовых моделей (рекомендуется 10–20).
    halflife_periods:
        Период полураспада веса свежести, в единицах `period_unit`
        (рекомендуется 3–12 месяцев в зависимости от скорости дрейфа).
    period_unit:
        Pandas frequency alias ('M', 'W', 'D', ...) для бинования datetime-
        подобного `ts_key` в периоды. Игнорируется, если `ts_key` уже
        числовой (тогда halflife_periods — в тех же единицах, что сам ts_key).
    sample_frac:
        Доля от n_train строк в каждом бутстрэп-сэмпле (с возвращением —
        вес влияет только на вероятность отбора, не запрещает повторы).
    base:
        'catboost' (по умолчанию) или 'lightgbm'.
    base_params:
        Гиперпараметры базовой модели. None → дефолтные для выбранного base.
        Игнорируется, если n_optuna_trials > 0.
    n_optuna_trials:
        Если > 0, общая архитектура (одна на всех estimator'ов) подбирается через
        Optuna по val PR-AUC на одном представительном взвешенном бутстрэпе (того
        же размера, что и остальные estimator'ы).
    param_space:
        Кастомная функция `f(trial) -> dict` — search space для Optuna вместо
        дефолтного (под выбранный `base`). Может как включать только часть
        тюнящихся параметров (недостающие подставляются дефолтами), так и
        переопределять любой из них. Действует только при n_optuna_trials > 0.
        None → дефолтный search space.
    optuna_timeout:
        Ограничение по времени (сек) на весь Optuna-поиск. None — без ограничения.
    optuna_verbose:
        Если True — не глушит логи Optuna. Если False (по умолчанию) —
        форсирует WARNING на время поиска.
    optuna_pruner:
        None/строковый алиас ('median'/'hyperband'/'percentile'/
        'successive_halving'/'none')/готовый optuna.pruners.BasePruner —
        см. ml_toolkit.models model_settings.md. 'none' (по умолчанию) —
        прунинг выключен.
    random_seed:
        Начальное зерно. Каждый estimator получает независимый (SeedSequence.spawn)
        генератор. Также сид Optuna sampler'а.

    Атрибуты после fit::

        estimators_        — список обученных базовых моделей
        estimator_scores_  — val PR-AUC каждого estimator
        ensemble_score_    — val PR-AUC финального ансамбля

    Пример::

        model = WeightedBaggingByRecency(n_estimators=10, halflife_periods=6)
        model.fit(X_train, y_train, X_valid, y_valid, ts_key=X_train_dates)
        proba = model.predict_proba(X_test)

    """

    def __init__(
        self,
        n_estimators: int = 10,
        halflife_periods: float = 6.0,
        period_unit: str = 'M',
        sample_frac: float = 1.0,
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
        if halflife_periods <= 0:
            raise ValueError(f'halflife_periods должен быть > 0, получено {halflife_periods}')
        if not 0.0 < sample_frac <= 2.0:
            raise ValueError(f'sample_frac должен быть в (0, 2], получено {sample_frac}')
        self.n_estimators = n_estimators
        self.halflife_periods = halflife_periods
        self.period_unit = period_unit
        self.sample_frac = sample_frac
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
        self.estimator_scores_: list[float] = []
        self.ensemble_score_: float = 0.0
        self._rank_refs_: list[np.ndarray] = []

    # ── Обучение одного estimator ──────────────────────────────────────────────

    def _fit_one_lgb(
        self,
        X_sub: pd.DataFrame,
        y_sub: np.ndarray,
        X_va: pd.DataFrame,
        y_va: np.ndarray,
        seed: int,
        params: dict[str, Any] | None = None,
    ) -> Any:
        import lightgbm as lgb

        p = {**(params or self.base_params or _DEFAULT_LGB_PARAMS), 'random_state': seed}
        model = lgb.LGBMClassifier(**p)
        model.fit(
            X_sub, y_sub,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        return model

    def _fit_one_cbt(
        self,
        X_sub: pd.DataFrame,
        y_sub: np.ndarray,
        X_va: pd.DataFrame,
        y_va: np.ndarray,
        seed: int,
        params: dict[str, Any] | None = None,
    ) -> Any:
        from catboost import CatBoostClassifier, Pool

        p = {**(params or self.base_params or _DEFAULT_CBT_PARAMS), 'random_seed': seed}
        model = CatBoostClassifier(**p)
        tr_pool = Pool(X_sub, y_sub, cat_features=self.cat_features_)
        va_pool = Pool(X_va, y_va, cat_features=self.cat_features_)
        model.fit(tr_pool, eval_set=va_pool, verbose=False)
        return model

    def _predict_one(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        if self.base == 'lightgbm':
            return model.predict_proba(X)[:, 1]
        from catboost import Pool
        return model.predict_proba(Pool(X, cat_features=self.cat_features_))[:, 1]

    def _tune_cbt(self, X_sub: pd.DataFrame, y_sub: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray) -> dict[str, Any]:
        from catboost import CatBoostClassifier, Pool
        import optuna

        _optuna_prev_verbosity = optuna.logging.get_verbosity()
        if not self.optuna_verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        tr_pool = Pool(X_sub, y_sub, cat_features=self.cat_features_)
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
                # CatBoost GPU не поддерживает user-defined callbacks — прунинг
                # для GPU-trial'ов недоступен, trial всегда доучивается до конца.
                m.fit(tr_pool, eval_set=va_pool, verbose=False)
            else:
                pruning_cb = CatBoostPruningCallback(trial, params['eval_metric'])
                m.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=[pruning_cb])
                pruning_cb.check_pruned()
            p = m.predict_proba(va_pool)[:, 1]
            return float(average_precision_score(y_va, p))

        logger.info('[WeightedBaggingByRecency] Optuna (catboost): %d trials', self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed),
                                    pruner=make_pruner(self.optuna_pruner))
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return dict(study.best_trial.user_attrs['cb_params'])

    def _tune_lgb(self, X_sub: pd.DataFrame, y_sub: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray) -> dict[str, Any]:
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
                X_sub, y_sub,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            p = m.predict_proba(X_va)[:, 1]
            return float(average_precision_score(y_va, p))

        logger.info('[WeightedBaggingByRecency] Optuna (lightgbm): %d trials', self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed))
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return dict(study.best_trial.user_attrs['cb_params'])

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        ts_key: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> WeightedBaggingByRecency:
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features if cat_features is not None else self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values
        X_tr_feats = X_train[feats]
        X_va_feats = X_valid[feats]

        ts_series = pd.Series(np.asarray(ts_key)).reset_index(drop=True)
        if len(ts_series) != len(X_tr_feats):
            raise ValueError(
                f'ts_key должен быть той же длины, что X_train: {len(ts_series)} != {len(X_tr_feats)}'
            )
        periods = compute_periods(ts_series, self.period_unit)
        weights = _recency_weights(periods, self.halflife_periods)

        n_train = len(X_tr_feats)
        n_sample = max(1, int(round(self.sample_frac * n_train)))

        logger.info(
            '[WeightedBaggingByRecency] n_estimators=%d  halflife_periods=%.1f  '
            'n_sample/estimator=%d/%d  base=%s',
            self.n_estimators, self.halflife_periods, n_sample, n_train, self.base,
        )

        # SeedSequence.spawn даёт статистически независимые генераторы из одного
        # random_seed — подвыборка тюнинга и подвыборка estimator'а i=0 иначе
        # рискуют совпасть (см. EasyEnsembleClassifier, где default_rng(seed) и
        # default_rng(seed + 0) были одним и тем же генератором).
        tune_seed_seq, *estimator_seed_seqs = np.random.SeedSequence(self.random_seed).spawn(
            self.n_estimators + 1
        )

        tuned_params = None
        if self.n_optuna_trials > 0:
            rng0 = np.random.default_rng(tune_seed_seq)
            sample_idx0 = rng0.choice(n_train, size=n_sample, replace=True, p=weights)
            X_sub0 = X_tr_feats.iloc[sample_idx0].reset_index(drop=True)
            y_sub0 = y_tr[sample_idx0]
            tuned_params = (
                self._tune_lgb(X_sub0, y_sub0, X_va_feats, y_va) if self.base == 'lightgbm'
                else self._tune_cbt(X_sub0, y_sub0, X_va_feats, y_va)
            )

        self.estimators_ = []
        self.estimator_scores_ = []
        va_raw_scores: list[np.ndarray] = []

        for i in range(self.n_estimators):
            rng = np.random.default_rng(estimator_seed_seqs[i])
            sample_idx = rng.choice(n_train, size=n_sample, replace=True, p=weights)

            X_sub = X_tr_feats.iloc[sample_idx].reset_index(drop=True)
            y_sub = y_tr[sample_idx]

            seed = self.random_seed + i
            if self.base == 'lightgbm':
                model = self._fit_one_lgb(X_sub, y_sub, X_va_feats, y_va, seed, tuned_params)
            else:
                model = self._fit_one_cbt(X_sub, y_sub, X_va_feats, y_va, seed, tuned_params)

            va_score = self._predict_one(model, X_va_feats)
            ap = float(average_precision_score(y_va, va_score))
            self.estimators_.append(model)
            self.estimator_scores_.append(ap)
            va_raw_scores.append(va_score)
            logger.info('[WeightedBaggingByRecency] estimator %2d/%d  val PR-AUC=%.4f',
                        i + 1, self.n_estimators, ap)

        # Референсы rank-нормализации — train-скоры каждого estimator; predict_proba
        # использует их же, поэтому скор объекта не зависит от состава батча.
        tr_raw_scores = [self._predict_one(m, X_tr_feats) for m in self.estimators_]
        self._rank_refs_ = [fit_rank_reference(s) for s in tr_raw_scores]

        va_ensemble = np.stack(
            [rank_transform(s, ref) for s, ref in zip(va_raw_scores, self._rank_refs_)],
            axis=1,
        ).mean(axis=1)
        self.ensemble_score_ = float(average_precision_score(y_va, va_ensemble))
        logger.info('[WeightedBaggingByRecency] ensemble val PR-AUC=%.4f  (mean single=%.4f)',
                    self.ensemble_score_, float(np.mean(self.estimator_scores_)))

        self.valid_pred_ = va_ensemble
        self.train_pred_ = np.stack(
            [rank_transform(s, ref) for s, ref in zip(tr_raw_scores, self._rank_refs_)],
            axis=1,
        ).mean(axis=1)

        self.best_params_ = {
            'n_estimators': self.n_estimators,
            'halflife_periods': self.halflife_periods,
            'sample_frac': self.sample_frac,
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
        rank_matrix = [
            rank_transform(self._predict_one(m, X_feats), ref)
            for m, ref in zip(self.estimators_, self._rank_refs_)
        ]
        return np.stack(rank_matrix, axis=1).mean(axis=1)
