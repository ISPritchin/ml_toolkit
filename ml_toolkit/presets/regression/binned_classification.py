"""RegressionByBinnedClassification: биннинг таргета → мультикласс → декодирование E[y|bin].

Сложное мультимодальное распределение таргета (несколько кластеров значений с
разной плотностью) размазывается MSE/MAE-регрессией между модами — точечный
прогноз тянется к «среднему между пиками», которое само по себе почти никогда
не встречается в данных. Здесь таргет бинуется на n_bins интервалов, CatBoost
учится как MultiClass-классификатор предсказывать бин, а итоговое значение —
ожидание по предсказанному распределению вероятностей бинов:
E[y|x] ≈ Σ_k P(bin=k|x) * repr_k, где repr_k — mean/median y_train внутри бина k.
Такой probability-weighted decode уже частично учитывает упорядоченность бинов
(в отличие от простого argmax), не требуя отдельной ordinal-модели.

binning='quantile' (равное число обучающих строк на бин, устойчиво к
скошенности) или 'uniform' (равная ширина интервала). Совпадающие квантильные
границы (частые повторяющиеся значения таргета) схлопываются — реальное число
бинов (`n_bins_actual_`) может быть меньше n_bins.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ml_toolkit.presets.regression._base import BasePreset
from ml_toolkit.presets.regression._optuna_utils import (
    CatBoostPruningCallback,
    catboost_arch_space,
    make_pruner,
)

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    'iterations': 700, 'max_depth': 6, 'learning_rate': 0.05,
    'l2_leaf_reg': 3.0, 'early_stopping_rounds': 100,
}


class RegressionByBinnedClassification(BasePreset):
    """Регрессия через биннинг таргета + CatBoost MultiClass + expected-value decode.

    Parameters
    ----------
    n_bins:
        Число бинов таргета (запрошенное — реальное после схлопывания
        совпадающих границ доступно как `n_bins_actual_` после fit()).
    binning:
        'quantile' (по умолчанию, равное число строк на бин) или 'uniform'
        (равная ширина интервала).
    decode:
        'mean' (по умолчанию) или 'median' — представитель бина repr_k,
        используемый в E[y|x] = Σ_k P(bin=k|x) * repr_k.
    base_params:
        Параметры CatBoost для прямого режима (n_optuna_trials == 0).
    n_optuna_trials:
        Число Optuna trials, тюнящих архитектуру CatBoost. Trial отбирается
        по MAE декодированного прогноза на валидации (не по точности
        классификации бина — итоговая метрика этого пресета непрерывная).
    param_space / optuna_timeout / optuna_verbose / random_seed:
        См. другие Optuna-пресеты пакета.

    Атрибуты после fit::

        bin_edges_      — границы бинов (len = n_bins_actual_ + 1)
        bin_repr_       — representативное значение на бин (len = n_bins_actual_)
        n_bins_actual_  — реальное число бинов после схлопывания дубликатов

    Пример::

        model = RegressionByBinnedClassification(n_bins=32, decode='median')
        model.fit(X_train, y_train, X_valid, y_valid)
        pred = model.predict(X_test)

    """

    def __init__(
        self,
        n_bins: int = 32,
        binning: str = 'quantile',
        decode: str = 'mean',
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
        if n_bins < 2:
            raise ValueError(f'n_bins должен быть >= 2, получено {n_bins}')
        if binning not in ('quantile', 'uniform'):
            raise ValueError(f"binning должен быть 'quantile' или 'uniform', получено {binning!r}")
        if decode not in ('mean', 'median'):
            raise ValueError(f"decode должен быть 'mean' или 'median', получено {decode!r}")
        self.n_bins = n_bins
        self.binning = binning
        self.decode = decode
        self.base_params = base_params
        self.param_space = param_space
        self.optuna_timeout = optuna_timeout
        self.optuna_verbose = optuna_verbose
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.bin_edges_: np.ndarray | None = None
        self.bin_repr_: np.ndarray | None = None
        self.n_bins_actual_: int | None = None

    # ── биннинг ──────────────────────────────────────────────────────────────

    def _compute_edges(self, y: np.ndarray) -> np.ndarray:
        if self.binning == 'quantile':
            edges = np.quantile(y, np.linspace(0.0, 1.0, self.n_bins + 1))
        else:
            edges = np.linspace(y.min(), y.max(), self.n_bins + 1)
        edges = np.unique(edges)
        if len(edges) < 3:
            raise ValueError(
                f'Таргет слишком вырожден для {self.n_bins} бинов (получилось {len(edges) - 1} '
                f'после схлопывания дубликатов границ) — уменьшите n_bins.'
            )
        return edges

    def _assign_bins(self, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
        n_bins_actual = len(edges) - 1
        labels = np.searchsorted(edges, y, side='right') - 1
        return np.clip(labels, 0, n_bins_actual - 1)

    def _decode(self, proba: np.ndarray) -> np.ndarray:
        return proba @ self.bin_repr_

    # ── обучение ─────────────────────────────────────────────────────────────

    def _tune(self, tr_pool, va_pool, y_va):
        import optuna
        from catboost import CatBoostClassifier

        if not self.optuna_verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        esr = _DEFAULT_PARAMS['early_stopping_rounds']

        def objective(trial: optuna.Trial) -> float:
            custom = self.param_space(trial) if self.param_space is not None else {}
            params = {
                **catboost_arch_space(trial, custom),
                'loss_function': 'MultiClass',
                'eval_metric': 'MultiClass',
                'classes_count': self.n_bins_actual_,
                # Дефолтный bootstrap_type для MultiClass — Bayesian, не поддерживающий
                # 'subsample' (см. ml_toolkit/models/_catboost.py::_default_cls_param_space).
                'bootstrap_type': custom.get('bootstrap_type', 'Bernoulli'),
                'early_stopping_rounds': custom.get('early_stopping_rounds', esr),
                'random_seed': custom.get('random_seed', self.random_seed),
                'verbose': custom.get('verbose', 0),
            }
            trial.set_user_attr('cb_params', params)
            pruning_cb = CatBoostPruningCallback(trial, 'MultiClass')
            m = CatBoostClassifier(**params)
            m.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=[pruning_cb])
            pruning_cb.check_pruned()
            decoded = self._decode(m.predict_proba(va_pool))
            return float(mean_absolute_error(y_va, decoded))

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed),
                                    pruner=make_pruner())
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        best = dict(study.best_trial.user_attrs['cb_params'])
        model = CatBoostClassifier(**best)
        model.fit(tr_pool, eval_set=va_pool, verbose=False)
        return model, best

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> RegressionByBinnedClassification:
        from catboost import CatBoostClassifier, Pool

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values

        edges = self._compute_edges(y_tr)
        self.bin_edges_ = edges
        self.n_bins_actual_ = len(edges) - 1
        bin_tr = self._assign_bins(y_tr, edges)
        bin_va = self._assign_bins(y_va, edges)

        repr_vals = np.empty(self.n_bins_actual_)
        for k in range(self.n_bins_actual_):
            mask = bin_tr == k
            if mask.any():
                repr_vals[k] = float(np.mean(y_tr[mask]) if self.decode == 'mean' else np.median(y_tr[mask]))
            else:
                repr_vals[k] = 0.5 * (edges[k] + edges[k + 1])
        self.bin_repr_ = repr_vals

        tr_pool = Pool(X_train[feats], bin_tr, cat_features=self.cat_features_)
        va_pool = Pool(X_valid[feats], bin_va, cat_features=self.cat_features_)

        if self.n_optuna_trials > 0:
            self._model, self.best_params_ = self._tune(tr_pool, va_pool, y_va)
        else:
            params = {
                **(self.base_params or _DEFAULT_PARAMS),
                'loss_function': 'MultiClass', 'eval_metric': 'MultiClass',
                'classes_count': self.n_bins_actual_, 'random_seed': self.random_seed, 'verbose': 0,
            }
            model = CatBoostClassifier(**params)
            model.fit(tr_pool, eval_set=va_pool, verbose=False)
            self._model = model
            self.best_params_ = params

        tr_pred_pool = Pool(X_train[feats], cat_features=self.cat_features_)
        self.train_pred_ = self._decode(self._model.predict_proba(tr_pred_pool))
        self.valid_pred_ = self._decode(self._model.predict_proba(va_pool))
        mae = float(mean_absolute_error(y_va, self.valid_pred_))
        logger.info('[BinnedClassification] n_bins_actual=%d  val MAE=%.4f', self.n_bins_actual_, mae)
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        pool = Pool(X[self.selected_features_], cat_features=self.cat_features_)
        return self._decode(self._model.predict_proba(pool))
