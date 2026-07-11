"""TrimmedLossRegressor: self-paced итеративное обучение с исключением top-q% остатков.

Least-Trimmed-Squares-style схема: грубые ошибки записи в таргете (опечатки,
единицы измерения перепутаны и т.п.) нельзя вычистить заранее правилами, но их
можно распознать по устойчиво большому остатку после нескольких раундов
обучения. Каждый раунд:

1. Обучаем CatBoost на текущем активном подмножестве train (раунд 0 — все строки).
2. Предсказываем на ВСЕЙ обучающей выборке (не только активной — исключённая на
   предыдущем раунде строка могла перестать быть выбросом после того, как модель
   стала точнее в её окрестности, поэтому активное множество пересчитывается с
   нуля от последней модели, а не накопительно).
3. Новое активное множество — (1 - trim_frac) строк с наименьшим |остатком|.
4. Повторяем n_rounds раз; лучший по val-метрике раунд сохраняется как финальная модель.

Архитектура CatBoost тюнится Optuna один раз в раунде 0 (на полных данных) и
используется во всех последующих раундах — как в HardNegativeMiner (см.
ml_toolkit/presets/classification/high_pr_auc/hard_negative_mining.py), тюнить
архитектуру заново на каждом раунде избыточно и просто медленнее.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ml_toolkit.models._base import XInput, YInput
from ml_toolkit.presets.regression._base import BasePreset
from ml_toolkit.presets.regression._optuna_utils import (
    CatBoostPruningCallback,
    catboost_arch_space,
    make_pruner,
)

if TYPE_CHECKING:
    from catboost import CatBoostRegressor, Pool

logger = logging.getLogger(__name__)

_DEFAULT_BASE_PARAMS: dict[str, Any] = {
    'iterations': 700,
    'max_depth': 5,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'min_data_in_leaf': 10,
    'early_stopping_rounds': 100,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 0,
}


class TrimmedLossRegressor(BasePreset):
    """Итеративный self-paced trimming (Least Trimmed Squares-стиль) с CatBoost.

    Parameters
    ----------
    trim_frac:
        Доля строк с наибольшим |остатком|, исключаемых из активного множества
        на каждом раунде (кроме последнего).
    n_rounds:
        Число раундов обучения.
    base_params:
        Параметры CatBoost. Если None — используются дефолтные.
    n_optuna_trials:
        Если > 0, параметры раунда 0 ищутся через Optuna (по MAE на val),
        последующие раунды используют найденные параметры.
    param_space:
        Кастомная функция `f(trial) -> dict` — переопределяет search space для
        Optuna в раунде 0 (те же ключи, что у catboost_arch_space, плюс
        loss_function/eval_metric/early_stopping_rounds/random_seed/verbose).
        Действует только при n_optuna_trials > 0.
    optuna_verbose:
        Если True — не глушит логи Optuna.
    optuna_pruner:
        None/строковый алиас ('median'/'hyperband'/'percentile'/
        'successive_halving'/'none')/готовый optuna.pruners.BasePruner —
        см. ml_toolkit.models model_settings.md. 'none' (по умолчанию) —
        прунинг выключен.
    random_seed:
        Зерно CatBoost и Optuna sampler'а.

    Атрибуты после fit::

        mae_per_round_    — val MAE по раундам
        active_frac_per_round_ — доля активных train-строк по раундам

    Пример::

        model = TrimmedLossRegressor(trim_frac=0.05, n_rounds=3)
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=[...])
        pred = model.predict(X_test)
        print(model.mae_per_round_)

    """

    def __init__(
        self,
        trim_frac: float = 0.05,
        n_rounds: int = 3,
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
        if not 0.0 < trim_frac < 0.5:
            raise ValueError(f'trim_frac должен быть в (0, 0.5), получено {trim_frac}')
        if n_rounds < 1:
            raise ValueError('n_rounds должен быть >= 1')
        self.trim_frac = trim_frac
        self.n_rounds = n_rounds
        self.base_params = base_params
        self.param_space = param_space
        self.optuna_timeout = optuna_timeout
        self.optuna_verbose = optuna_verbose
        self.optuna_pruner = optuna_pruner
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []
        self.mae_per_round_: list[float] = []
        self.active_frac_per_round_: list[float] = []
        self.models_: list = []

    # ── Optuna (раунд 0) ────────────────────────────────────────────────────

    def _fit_round0_optuna(
        self, tr_pool: Pool, va_pool: Pool, y_va: np.ndarray,
    ) -> tuple[CatBoostRegressor, dict]:
        from catboost import CatBoostRegressor
        import optuna

        _optuna_prev_verbosity = optuna.logging.get_verbosity()
        if not self.optuna_verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            custom = self.param_space(trial) if self.param_space is not None else {}
            params = {
                **catboost_arch_space(trial, custom),
                'loss_function': custom.get('loss_function', 'RMSE'),
                'eval_metric': custom.get('eval_metric', 'RMSE'),
                'early_stopping_rounds': custom.get('early_stopping_rounds', 100),
                'random_seed': custom.get('random_seed', self.random_seed),
                'verbose': custom.get('verbose', 0),
            }
            trial.set_user_attr('cb_params', params)
            pruning_cb = CatBoostPruningCallback(trial, params['eval_metric'])
            m = CatBoostRegressor(**params)
            m.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=[pruning_cb])
            pruning_cb.check_pruned()
            p = m.predict(va_pool)
            return float(mean_absolute_error(y_va, p))

        logger.info('[TrimmedLoss] Optuna round 0: %d trials', self.n_optuna_trials)
        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed),
                                    pruner=make_pruner(self.optuna_pruner))
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        best = dict(study.best_trial.user_attrs['cb_params'])
        m = CatBoostRegressor(**best)
        m.fit(tr_pool, eval_set=va_pool, verbose=False)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return m, best

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: XInput,
        y_train: YInput,
        X_valid: XInput,
        y_valid: YInput,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> TrimmedLossRegressor:
        from catboost import CatBoostRegressor, Pool

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values
        n = len(y_tr)
        n_trim = max(1, round(n * self.trim_frac))

        va_pool = Pool(X_valid[feats], y_va, cat_features=self.cat_features_)
        active = np.ones(n, dtype=bool)

        fixed_params = {**(self.base_params or _DEFAULT_BASE_PARAMS), 'random_seed': self.random_seed}
        best_mae = np.inf
        best_model = None
        self.models_ = []
        self.mae_per_round_ = []
        self.active_frac_per_round_ = []

        for r in range(self.n_rounds):
            tr_pool = Pool(X_train.loc[active, feats], y_tr[active], cat_features=self.cat_features_)

            if r == 0 and self.n_optuna_trials > 0:
                model, fixed_params = self._fit_round0_optuna(tr_pool, va_pool, y_va)
                self.best_params_ = fixed_params
            else:
                model = CatBoostRegressor(**fixed_params)
                model.fit(tr_pool, eval_set=va_pool, verbose=False)

            va_p = model.predict(va_pool)
            mae = float(mean_absolute_error(y_va, va_p))
            self.mae_per_round_.append(mae)
            self.active_frac_per_round_.append(float(active.mean()))
            self.models_.append(model)
            logger.info(
                '[TrimmedLoss] Раунд %d/%d  val MAE=%.4f  active=%.1f%%',
                r + 1, self.n_rounds, mae, active.mean() * 100,
            )

            if mae < best_mae:
                best_mae = mae
                best_model = model

            if r < self.n_rounds - 1:
                full_pred = model.predict(Pool(X_train[feats], cat_features=self.cat_features_))
                residuals = np.abs(y_tr - full_pred)
                keep_idx = np.argsort(residuals)[: n - n_trim]
                active = np.zeros(n, dtype=bool)
                active[keep_idx] = True

        self._model = best_model
        if self.best_params_ is None:
            self.best_params_ = fixed_params

        best_round = int(np.argmin(self.mae_per_round_)) + 1
        logger.info('[TrimmedLoss] Лучший раунд %d  val MAE=%.4f', best_round, best_mae)

        self.valid_pred_ = self._model.predict(va_pool)
        self.train_pred_ = self._model.predict(Pool(X_train[feats], cat_features=self.cat_features_))
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        pool = Pool(X[self.selected_features_], cat_features=self.cat_features_)
        return self._model.predict(pool)
