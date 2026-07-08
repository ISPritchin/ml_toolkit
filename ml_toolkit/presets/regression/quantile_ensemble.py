"""QuantileEnsembleRegressor: согласованный набор квантильных CatBoost-моделей.

Каждый квантиль (p05...p95 по умолчанию) — отдельная CatBoostRegressor модель
со встроенным `Quantile:alpha=q`, обученная независимо (в т.ч. с собственным
Optuna-поиском архитектуры, если n_optuna_trials > 0 — objective для каждой
модели — pinball loss на её собственном q, чтобы архитектура подбиралась под
то, что реально предсказывается, а не под общую MAE).

Независимо обученные квантильные модели не гарантируют монотонность:
p75-модель на конкретной строке может предсказать МЕНЬШЕ, чем p50-модель —
физически бессмысленно (квантильная функция по определению неубывающая).
non_crossing=True включает rearrangement-поправку (Chernozhukov et al., 2010,
"Quantile and Probability Curves Without Crossing"): предсказания всех
квантилей на строке пересортировываются по возрастанию — простейшая
состоятельная оценка, устраняющая crossing без переобучения моделей.

predict() возвращает точечный прогноз — квантиль, ближайший к 0.5 (после
non-crossing поправки); полный профиль — predict_quantiles().
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import numpy as np
import pandas as pd

from ml_toolkit.models._base import _to_pandas
from ml_toolkit.models._utils import quantile_loss
from ml_toolkit.presets.regression._base import BasePreset
from ml_toolkit.presets.regression._optuna_utils import (
    CatBoostPruningCallback,
    catboost_arch_space,
    make_pruner,
)

logger = logging.getLogger(__name__)

_DEFAULT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

_DEFAULT_ARCH_PARAMS: dict[str, Any] = {
    'iterations': 700,
    'max_depth': 5,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'min_data_in_leaf': 10,
    'early_stopping_rounds': 100,
}


class QuantileEnsembleRegressor(BasePreset):
    """Ансамбль независимых квантильных CatBoost-моделей с non-crossing поправкой.

    Parameters
    ----------
    quantiles:
        Список квантилей ∈ (0, 1). По умолчанию (0.05, 0.25, 0.5, 0.75, 0.95).
    non_crossing:
        Применять ли rearrangement-поправку (см. докстринг модуля) к
        predict_quantiles()/predict().
    base_params:
        Параметры CatBoost (без loss_function/eval_metric — задаются
        автоматически на квантиль) для прямого режима (n_optuna_trials == 0).
    n_optuna_trials:
        Число Optuna trials НА КАЖДЫЙ квантиль (независимая архитектура на
        квантиль). 0 → прямой режим с base_params для всех квантилей.
    param_space / optuna_timeout / optuna_verbose / random_seed:
        См. другие Optuna-пресеты пакета.

    Атрибуты после fit::

        models_                    — {quantile: обученная CatBoostRegressor}
        best_params_per_quantile_  — {quantile: dict параметров}

    Пример::

        model = QuantileEnsembleRegressor(quantiles=[0.1, 0.5, 0.9], n_optuna_trials=20)
        model.fit(X_train, y_train, X_valid, y_valid)
        median_pred = model.predict(X_test)
        profile = model.predict_quantiles(X_test)   # DataFrame, колонки — квантили

    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        non_crossing: bool = True,
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        param_space: Callable[[Any], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(params=base_params, n_optuna_trials=n_optuna_trials)
        qs = sorted(quantiles) if quantiles else sorted(_DEFAULT_QUANTILES)
        if len(qs) != len(set(qs)):
            raise ValueError(f'quantiles содержит дубликаты: {qs}')
        for q in qs:
            if not 0.0 < q < 1.0:
                raise ValueError(f'Каждый quantile должен быть в (0, 1), получено {q}')
        self.quantiles = qs
        self.non_crossing = non_crossing
        self.base_params = base_params
        self.param_space = param_space
        self.optuna_timeout = optuna_timeout
        self.optuna_verbose = optuna_verbose
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.models_: dict[float, Any] = {}
        self.best_params_per_quantile_: dict[float, dict] = {}
        self._median_q = min(qs, key=lambda x: abs(x - 0.5))

    # ── обучение одной квантильной модели ───────────────────────────────────

    def _tune_one_quantile(self, tr_pool, va_pool, y_va, q):
        import optuna
        from catboost import CatBoostRegressor

        if not self.optuna_verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        loss_name = f'Quantile:alpha={q}'
        esr = _DEFAULT_ARCH_PARAMS['early_stopping_rounds']

        def objective(trial: optuna.Trial) -> float:
            custom = self.param_space(trial) if self.param_space is not None else {}
            params = {
                **catboost_arch_space(trial, custom),
                'loss_function': loss_name,
                'eval_metric': loss_name,
                'early_stopping_rounds': custom.get('early_stopping_rounds', esr),
                'random_seed': custom.get('random_seed', self.random_seed),
                'verbose': custom.get('verbose', 0),
            }
            trial.set_user_attr('cb_params', params)
            pruning_cb = CatBoostPruningCallback(trial, loss_name)
            m = CatBoostRegressor(**params)
            m.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=[pruning_cb])
            pruning_cb.check_pruned()
            p = m.predict(va_pool)
            return quantile_loss(y_va, p, q=q)

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed),
                                    pruner=make_pruner())
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        best = dict(study.best_trial.user_attrs['cb_params'])
        model = CatBoostRegressor(**best)
        model.fit(tr_pool, eval_set=va_pool, verbose=False)
        return model, best

    def _fit_one_quantile(self, tr_pool, va_pool, y_va, q):
        from catboost import CatBoostRegressor

        if self.n_optuna_trials > 0:
            return self._tune_one_quantile(tr_pool, va_pool, y_va, q)
        loss_name = f'Quantile:alpha={q}'
        params = {
            **(self.base_params or _DEFAULT_ARCH_PARAMS),
            'loss_function': loss_name,
            'eval_metric': loss_name,
            'random_seed': self.random_seed,
            'verbose': 0,
        }
        model = CatBoostRegressor(**params)
        model.fit(tr_pool, eval_set=va_pool, verbose=False)
        return model, params

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> QuantileEnsembleRegressor:
        from catboost import Pool

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values
        tr_pool = Pool(X_train[feats], y_tr, cat_features=self.cat_features_)
        va_pool = Pool(X_valid[feats], y_va, cat_features=self.cat_features_)

        self.models_ = {}
        self.best_params_per_quantile_ = {}
        for q in self.quantiles:
            model, best = self._fit_one_quantile(tr_pool, va_pool, y_va, q)
            self.models_[q] = model
            self.best_params_per_quantile_[q] = best
            logger.info('[QuantileEnsemble] q=%.3f  pinball=%.4f', q,
                       quantile_loss(y_va, model.predict(va_pool), q=q))

        self._model = self.models_
        self.best_params_ = self.best_params_per_quantile_

        self.train_pred_ = self.predict_quantiles(X_train)[self._median_q].values
        self.valid_pred_ = self.predict_quantiles(X_valid)[self._median_q].values
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict_quantiles(self, X: Any) -> pd.DataFrame:
        """Полный квантильный профиль (после non-crossing поправки, если включена)."""
        self._check_fitted()
        from catboost import Pool

        Xp = _to_pandas(X)
        pool = Pool(Xp[self.selected_features_], cat_features=self.cat_features_)
        preds = np.column_stack([self.models_[q].predict(pool) for q in self.quantiles])
        if self.non_crossing:
            preds = np.sort(preds, axis=1)
        return pd.DataFrame(preds, columns=self.quantiles, index=Xp.index)

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_quantiles(X)[self._median_q].values
