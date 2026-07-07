"""FeatureBaggingEnsemble: ансамбль CatBoost на случайных подпространствах признаков.

Каждый из n_estimators обучается на своём случайном подмножестве признаков
(feature_frac от общего числа), а не на подвыборке строк — при сотнях
коррелированных инженерных признаков (типичный результат feature generation)
деревья одного бустинга склонны залипать на одном и том же ведущем подмножестве;
разные подпространства признаков заставляют модели видеть разные сигналы.

Итоговый скор — среднее нормированных рангов (как в EasyEnsembleClassifier):
модели обучены на разных подпространствах и потенциально разных по масштабу
вероятностях, ранговая нормализация делает усреднение устойчивым.
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

logger = logging.getLogger(__name__)

_DEFAULT_BASE_PARAMS: dict[str, Any] = {
    'iterations': 500,
    'max_depth': 5,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'min_data_in_leaf': 10,
    'early_stopping_rounds': 80,
    'loss_function': 'Logloss',
    'eval_metric': 'PRAUC',
    'verbose': 0,
}


class FeatureBaggingEnsemble(BasePreset):
    """Ансамбль CatBoost на случайных подпространствах признаков.

    Parameters
    ----------
    n_estimators:
        Число базовых моделей.
    feature_frac:
        Доля признаков (от selected_features), случайно выбираемых для каждой
        модели без возврата внутри одной модели (рекомендуется 0.4–0.8).
    base_params:
        Параметры CatBoost, общие для всех моделей. None → дефолтные.
        Игнорируется, если n_optuna_trials > 0.
    n_optuna_trials:
        Если > 0, общая архитектура (одна на всех членов ансамбля) подбирается
        через Optuna по val PR-AUC на полном наборе признаков (до разбиения на
        подпространства).
    param_space:
        Кастомная функция `f(trial) -> dict` — search space для Optuna вместо
        дефолтного. Может как включать только часть тюнящихся параметров
        (недостающие из loss_function/eval_metric/early_stopping_rounds/
        random_seed/verbose подставляются дефолтами), так и переопределять
        любой из них, включая loss_function/eval_metric — то, что вернула
        param_space, имеет приоритет над дефолтами. Действует только при
        n_optuna_trials > 0. None → дефолтный search space.
    optuna_timeout:
        Ограничение по времени (сек) на весь Optuna-поиск. None — без ограничения.
    optuna_verbose:
        Если True — не глушит логи Optuna. Если False (по умолчанию) —
        форсирует WARNING на время поиска.
    random_seed:
        Начальное зерно. Модель i получает seed + i — и для выбора
        подпространства признаков, и для самого CatBoost. Также сид Optuna sampler'а.

    Атрибуты после fit::

        estimators_         — список обученных базовых моделей
        feature_subsets_     — список подмножеств признаков, по одному на модель
        estimator_scores_   — val PR-AUC каждой модели
        ensemble_score_      — val PR-AUC ансамбля

    Пример::

        model = FeatureBaggingEnsemble(n_estimators=15, feature_frac=0.6)
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=[...])
        proba = model.predict_proba(X_test)

    """

    def __init__(
        self,
        n_estimators: int = 15,
        feature_frac: float = 0.6,
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        param_space: Callable[[Any], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        if not 0.0 < feature_frac <= 1.0:
            raise ValueError(f'feature_frac должен быть в (0, 1], получено {feature_frac}')
        super().__init__(params=base_params, n_optuna_trials=n_optuna_trials)
        self.n_estimators = n_estimators
        self.feature_frac = feature_frac
        self.base_params = base_params
        self.param_space = param_space
        self.optuna_timeout = optuna_timeout
        self.optuna_verbose = optuna_verbose
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.estimators_: list[Any] = []
        self.feature_subsets_: list[list[str]] = []
        self.estimator_scores_: list[float] = []
        self.ensemble_score_: float = 0.0
        self._rank_refs_: list[np.ndarray] = []

    def _tune(self, tr_pool: Any, va_pool: Any, y_va: np.ndarray) -> dict[str, Any]:
        from catboost import CatBoostClassifier
        import optuna

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
                **tunable,
            }
            trial.set_user_attr('cb_params', params)
            pruning_cb = CatBoostPruningCallback(trial, params['eval_metric'])
            m = CatBoostClassifier(**params)
            m.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=[pruning_cb])
            pruning_cb.check_pruned()
            p = m.predict_proba(va_pool)[:, 1]
            return float(average_precision_score(y_va, p))

        logger.info('[FeatureBagging] Optuna: %d trials (общая архитектура, полный набор фичей)',
                    self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed),
                                    pruner=make_pruner())
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        return dict(study.best_trial.user_attrs['cb_params'])

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> FeatureBaggingEnsemble:
        from catboost import CatBoostClassifier, Pool

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values
        n_sub = max(1, int(round(self.feature_frac * len(feats))))
        if self.n_optuna_trials > 0:
            full_tr_pool = Pool(X_train[feats], y_tr, cat_features=self.cat_features_)
            full_va_pool = Pool(X_valid[feats], y_va, cat_features=self.cat_features_)
            base_params = self._tune(full_tr_pool, full_va_pool, y_va)
        else:
            base_params = self.base_params or _DEFAULT_BASE_PARAMS

        logger.info(
            '[FeatureBagging] n_estimators=%d  feature_frac=%.2f  n_feats/estimator=%d/%d',
            self.n_estimators, self.feature_frac, n_sub, len(feats),
        )

        self.estimators_ = []
        self.feature_subsets_ = []
        self.estimator_scores_ = []
        tr_raw_scores: list[np.ndarray] = []
        va_raw_scores: list[np.ndarray] = []

        for i in range(self.n_estimators):
            rng = np.random.default_rng(self.random_seed + i)
            idx = rng.choice(len(feats), size=n_sub, replace=False)
            subset = [feats[j] for j in sorted(idx)]
            cat_sub = [c for c in self.cat_features_ if c in subset]

            tr_pool = Pool(X_train[subset], y_tr, cat_features=cat_sub)
            va_pool = Pool(X_valid[subset], y_va, cat_features=cat_sub)

            model = CatBoostClassifier(**{**base_params, 'random_seed': self.random_seed + i})
            model.fit(tr_pool, eval_set=va_pool, verbose=False)

            va_p = model.predict_proba(va_pool)[:, 1]
            tr_p = model.predict_proba(tr_pool)[:, 1]
            ap = float(average_precision_score(y_va, va_p))

            self.estimators_.append(model)
            self.feature_subsets_.append(subset)
            self.estimator_scores_.append(ap)
            tr_raw_scores.append(tr_p)
            va_raw_scores.append(va_p)
            logger.info('[FeatureBagging] estimator %2d/%d  val PR-AUC=%.4f',
                        i + 1, self.n_estimators, ap)

        # Референсы rank-нормализации — train-скоры каждой модели; predict_proba
        # использует их же, поэтому скор объекта не зависит от состава батча.
        self._rank_refs_ = [fit_rank_reference(s) for s in tr_raw_scores]

        va_ensemble = np.stack(
            [rank_transform(s, ref) for s, ref in zip(va_raw_scores, self._rank_refs_)],
            axis=1,
        ).mean(axis=1)
        self.ensemble_score_ = float(average_precision_score(y_va, va_ensemble))
        logger.info('[FeatureBagging] ensemble val PR-AUC=%.4f  (mean single=%.4f)',
                    self.ensemble_score_, float(np.mean(self.estimator_scores_)))

        self.valid_pred_ = va_ensemble
        self.train_pred_ = np.stack(
            [rank_transform(s, ref) for s, ref in zip(tr_raw_scores, self._rank_refs_)],
            axis=1,
        ).mean(axis=1)

        self.best_params_ = {
            'n_estimators': self.n_estimators,
            'feature_frac': self.feature_frac,
            'base_params': base_params,
        }
        self._model = True  # sentinel for _check_fitted
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        rank_matrix = []
        for model, subset, ref in zip(self.estimators_, self.feature_subsets_, self._rank_refs_):
            cat_sub = [c for c in self.cat_features_ if c in subset]
            pool = Pool(X[subset], cat_features=cat_sub)
            rank_matrix.append(rank_transform(model.predict_proba(pool)[:, 1], ref))
        return np.stack(rank_matrix, axis=1).mean(axis=1)
