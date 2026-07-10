"""MonotonicConstrainedClassifier: бустинг с монотонными ограничениями из доменного знания.

Одна модель (CatBoost или LightGBM), которой явно запрещено давать
контринтуитивный скор по части признаков: если по знанию предметной области
скор обязан не убывать (или не возрастать) с ростом значения признака —
например, «больше оборот → не ниже скор» — monotone_constraints жёстко
проводит эту связь через все сплиты дерева, а не полагается на то, что её
выучит сам бустинг на шумных данных.

Отличие от обычного CatBoostClassifier/LightGBMClassifier — не в самом
факте передачи monotone_constraints (это нативный параметр обеих
библиотек), а в том, что здесь ограничения задаются по ИМЕНИ признака
(dict), а не по позиционному индексу — не нужно вручную синхронизировать
порядок с list(X.columns) и пересобирать его при каждом изменении
selected_features.
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


def _lgb_constraints_list(monotone_constraints: dict[str, int], feats: list[str]) -> list[int]:
    """dict {feature: ±1} -> список, выровненный по порядку feats (LightGBM не
    принимает dict — только позиционный список, где отсутствующий признак = 0).
    """
    return [int(monotone_constraints.get(f, 0)) for f in feats]


class MonotonicConstrainedClassifier(BasePreset):
    """CatBoost/LightGBM с монотонными ограничениями по доменному знанию.

    Parameters
    ----------
    monotone_constraints:
        `{имя_признака: ±1}`. +1 — скор не должен убывать с ростом признака,
        -1 — не должен возрастать. Признаки, не упомянутые в словаре, остаются
        без ограничения (0). Ключи должны быть подмножеством `selected_features`
        (после `_resolve_features`) — иначе `ValueError` при `fit()`, а не
        молчаливое игнорирование опечатки в имени признака.
    base:
        'lightgbm' (по умолчанию) или 'catboost'.
    base_params:
        Гиперпараметры базовой модели. None → дефолтные для выбранного base.
        Игнорируется, если n_optuna_trials > 0.
    n_optuna_trials:
        Если > 0, архитектура подбирается через Optuna по val PR-AUC.
        monotone_constraints при этом фиксированы — не тюнятся (это
        доменное ограничение, а не гиперпараметр качества).
    param_space:
        Кастомная функция `f(trial) -> dict` — search space для Optuna вместо
        дефолтного (под выбранный `base`). Может как включать только часть
        тюнящихся параметров (недостающие подставляются дефолтами), так и
        переопределять любой из них. `monotone_constraints` в param_space не
        участвует — задаётся только конструкторским параметром. Действует
        только при n_optuna_trials > 0. None → дефолтный search space.
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
        Зерно базовой модели и Optuna sampler'а.

    Атрибуты после fit::

        model_score_ — val PR-AUC

    Пример::

        model = MonotonicConstrainedClassifier(
            monotone_constraints={'trans_sum__level': 1, 'inactive_streak': -1},
        )
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=[...])

    """

    def __init__(
        self,
        monotone_constraints: dict[str, int],
        base: str = 'lightgbm',
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
        if not monotone_constraints:
            raise ValueError('monotone_constraints не может быть пустым')
        if any(v not in (-1, 0, 1) for v in monotone_constraints.values()):
            raise ValueError(f'monotone_constraints значения должны быть в {{-1, 0, 1}}, получено {monotone_constraints}')
        self.monotone_constraints = monotone_constraints
        self.base = base
        self.base_params = base_params
        self.param_space = param_space
        self.n_optuna_trials = n_optuna_trials
        self.optuna_timeout = optuna_timeout
        self.optuna_verbose = optuna_verbose
        self.optuna_pruner = optuna_pruner
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.model_score_: float = 0.0

    # ── Optuna ────────────────────────────────────────────────────────────────

    def _tune_cbt(self, tr_pool: Any, va_pool: Any, y_va: np.ndarray) -> dict[str, Any]:
        from catboost import CatBoostClassifier
        import optuna

        _optuna_prev_verbosity = optuna.logging.get_verbosity()
        if not self.optuna_verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            tunable = self.param_space(trial) if self.param_space is not None else catboost_arch_space(trial)
            params = {
                'loss_function': 'Logloss',
                'eval_metric': 'PRAUC',
                'early_stopping_rounds': 80,
                'random_seed': self.random_seed,
                'verbose': 0,
                'monotone_constraints': self.monotone_constraints_,
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

        logger.info('[MonotonicConstrained] Optuna (catboost): %d trials', self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed),
                                    pruner=make_pruner(self.optuna_pruner))
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return dict(study.best_trial.user_attrs['cb_params'])

    def _tune_lgb(self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray) -> dict[str, Any]:
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
            params = {
                'random_state': self.random_seed, 'verbose': -1, 'n_jobs': -1,
                'monotone_constraints': self.monotone_constraints_,
                **tunable,
            }
            trial.set_user_attr('cb_params', params)
            m = lgb.LGBMClassifier(**params)
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            p = m.predict_proba(X_va)[:, 1]
            return float(average_precision_score(y_va, p))

        logger.info('[MonotonicConstrained] Optuna (lightgbm): %d trials', self.n_optuna_trials)
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
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> MonotonicConstrainedClassifier:
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features if cat_features is not None else self.cat_features

        unknown = set(self.monotone_constraints) - set(feats)
        if unknown:
            raise ValueError(
                f'monotone_constraints ссылается на признаки не из selected_features: {sorted(unknown)}'
            )
        # LightGBM не поддерживает monotone_constraints вместе с cat_features
        # среди ограниченных колонок — направление сплита по категории не
        # определено. CatBoost dict-формат допускает и категориальные ключи,
        # но здесь для единообразия обеих веток требуем то же самое.
        constrained_cat = set(self.monotone_constraints) & set(self.cat_features_)
        if constrained_cat:
            raise ValueError(
                f'monotone_constraints не может включать категориальные признаки: {sorted(constrained_cat)}'
            )

        if self.base == 'catboost':
            self.monotone_constraints_ = dict(self.monotone_constraints)
        else:
            self.monotone_constraints_ = _lgb_constraints_list(self.monotone_constraints, feats)

        y_tr = y_train.values
        y_va = y_valid.values
        X_tr_feats = X_train[feats]
        X_va_feats = X_valid[feats]

        logger.info(
            '[MonotonicConstrained] base=%s  n_constrained=%d/%d',
            self.base, len(self.monotone_constraints), len(feats),
        )

        if self.n_optuna_trials > 0:
            if self.base == 'catboost':
                from catboost import Pool
                tr_pool = Pool(X_tr_feats, y_tr, cat_features=self.cat_features_)
                va_pool = Pool(X_va_feats, y_va, cat_features=self.cat_features_)
                params = self._tune_cbt(tr_pool, va_pool, y_va)
            else:
                params = self._tune_lgb(X_tr_feats, y_tr, X_va_feats, y_va)
        else:
            params = dict(self.base_params or (
                _DEFAULT_LGB_PARAMS if self.base == 'lightgbm' else _DEFAULT_CBT_PARAMS
            ))
            params['monotone_constraints'] = self.monotone_constraints_

        if self.base == 'catboost':
            from catboost import CatBoostClassifier, Pool
            params = {**params, 'random_seed': self.random_seed}
            self._model = CatBoostClassifier(**params)
            tr_pool = Pool(X_tr_feats, y_tr, cat_features=self.cat_features_)
            va_pool = Pool(X_va_feats, y_va, cat_features=self.cat_features_)
            self._model.fit(tr_pool, eval_set=va_pool, verbose=False)
            self.train_pred_ = self._model.predict_proba(Pool(X_tr_feats, cat_features=self.cat_features_))[:, 1]
            self.valid_pred_ = self._model.predict_proba(va_pool)[:, 1]
        else:
            import lightgbm as lgb
            params = {**params, 'random_state': self.random_seed}
            self._model = lgb.LGBMClassifier(**params)
            self._model.fit(
                X_tr_feats, y_tr,
                eval_set=[(X_va_feats, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            self.train_pred_ = self._model.predict_proba(X_tr_feats)[:, 1]
            self.valid_pred_ = self._model.predict_proba(X_va_feats)[:, 1]

        self.model_score_ = float(average_precision_score(y_va, self.valid_pred_))
        logger.info('[MonotonicConstrained] val PR-AUC=%.4f', self.model_score_)

        self.best_params_ = {'base': self.base, 'model_params': params}
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_feats = X[self.selected_features_]
        if self.base == 'lightgbm':
            return self._model.predict_proba(X_feats)[:, 1]
        from catboost import Pool
        return self._model.predict_proba(Pool(X_feats, cat_features=self.cat_features_))[:, 1]
