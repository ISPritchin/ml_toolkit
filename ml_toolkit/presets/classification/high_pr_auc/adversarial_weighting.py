"""AdversarialValidationWeighting: importance weighting train-примеров вместо

удаления признаков (см. DriftRobustClassifier/044 — тот выбрасывает дрейфующие
КОЛОНКИ целиком; здесь колонки остаются, а переweight'иваются СТРОКИ train).

Механика:
  1. Обучаем adversarial-классификатор train(label=0) vs valid(label=1) на
     honest 70/30 сплите — тот же диагностический AUC, что и в
     AdversarialDriftFilter, но здесь используется только как метрика
     доверия к весам (адеварсариальный AUC~0.5 → веса будут ~1 для всех).
  2. Отдельно обучаем ВТОРОЙ adversarial-классификатор на ПОЛНОМ объединении
     train+valid (без сплита) — он даёт наиболее информативную propensity-
     оценку p(valid|x) для собственно взвешивания (в отличие от diagnostic-
     модели, которой специально урезан train, чтобы честно оценить AUC).
  3. weight(x) = p(valid|x) / p(train|x) = odds — доля train-строк, "похожих"
     на valid, получает вес > 1, непохожих — вес < 1. Клип в clip_weights
     защищает от единичных экстремальных весов, разрушающих эффективный
     размер выборки; после клипа веса нормализуются к среднему 1.0.
  4. Финальная модель обучается на train с этими весами (sample_weight),
     val — без изменений (там веса не нужны, дрейф был по train/valid, вес
     valid-строк, оценённых относительно train, всегда был бы 1 по построению).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ml_toolkit.presets.classification._base import BasePreset
from ml_toolkit.presets.classification._optuna_utils import (
    CatBoostPruningCallback,
    catboost_arch_space,
    make_pruner,
)

logger = logging.getLogger(__name__)

_ADV_PARAMS: dict[str, Any] = {
    'iterations': 300, 'max_depth': 4, 'learning_rate': 0.05,
    'loss_function': 'Logloss', 'eval_metric': 'AUC',
    'early_stopping_rounds': 30, 'verbose': 0,
}

_DEFAULT_PARAMS: dict[str, Any] = {
    'iterations': 600,
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


class AdversarialValidationWeighting(BasePreset):
    """CatBoost с весами train-примеров ∝ p(valid|x)/p(train|x).

    Parameters
    ----------
    clip_weights:
        (min, max) — диапазон клипа весов до нормализации к среднему 1.0.
    base_params:
        Параметры финальной CatBoost-модели. None → дефолтные. Игнорируется,
        если n_optuna_trials > 0.
    n_optuna_trials:
        Если > 0, параметры финальной модели подбираются через Optuna (по val
        PR-AUC, с MedianPruner) вместо base_params/дефолтных.
    param_space:
        Кастомная функция `f(trial) -> dict` — search space для Optuna вместо
        дефолтного. Может как включать только часть тюнящихся параметров
        (недостающие из loss_function/eval_metric/early_stopping_rounds/
        random_seed/verbose подставляются дефолтами), так и переопределять
        любой из них, включая loss_function/eval_metric (например, свой лосс,
        зафиксированный или перебираемый через trial.suggest_categorical) —
        то, что вернула param_space, имеет приоритет над дефолтами. Действует
        только при n_optuna_trials > 0. None → дефолтный search space.
    optuna_timeout:
        Ограничение по времени (сек) на весь Optuna-поиск. None — без ограничения.
    optuna_verbose:
        Если True — не глушит логи Optuna (по умолчанию сам Optuna обычно
        выводит INFO на каждый trial). Если False (по умолчанию) — форсирует
        WARNING на время поиска, вне зависимости от того, что было выставлено
        снаружи через optuna.logging.set_verbosity().
    random_seed:
        Зерно adversarial-классификаторов, финальной модели и Optuna sampler'а.

    Атрибуты после fit::

        adversarial_auc_   — honest AUC diagnostic-модели (train vs valid, 70/30)
        weights_           — итоговые (клипнутые, нормализованные) веса train-строк
        weight_stats_      — {min, max, mean_before_norm, effective_sample_size_ratio}

    Пример::

        model = AdversarialValidationWeighting(clip_weights=(0.2, 5.0))
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
        print(f"adversarial AUC={model.adversarial_auc_:.3f}")
    """

    def __init__(
        self,
        clip_weights: tuple[float, float] = (0.2, 5.0),
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
        if clip_weights[0] <= 0 or clip_weights[1] <= clip_weights[0]:
            raise ValueError(f'clip_weights должен быть (min>0, max>min), получено {clip_weights}')
        self.clip_weights = clip_weights
        self.base_params = base_params
        self.param_space = param_space
        self.optuna_timeout = optuna_timeout
        self.optuna_verbose = optuna_verbose
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.adversarial_auc_: float = 0.5
        self.weights_: np.ndarray = np.array([])
        self.weight_stats_: dict[str, float] = {}

    def _fit_adversarial(self, X: pd.DataFrame, y: np.ndarray, cat_features: list[str]) -> Any:
        from catboost import CatBoostClassifier, Pool
        m = CatBoostClassifier(**{**_ADV_PARAMS, 'random_seed': self.random_seed})
        m.fit(Pool(X, y, cat_features=cat_features), verbose=False)
        return m

    def _tune(self, tr_pool: Any, va_pool: Any, y_va: np.ndarray) -> dict[str, Any]:
        import optuna
        from catboost import CatBoostClassifier

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

        logger.info('[AdvWeighting] Optuna: %d trials', self.n_optuna_trials)
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
    ) -> 'AdversarialValidationWeighting':
        from catboost import CatBoostClassifier, Pool

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        X_combined = pd.concat(
            [X_train[feats].reset_index(drop=True), X_valid[feats].reset_index(drop=True)],
            ignore_index=True,
        )
        y_combined = np.array([0] * len(X_train) + [1] * len(X_valid))

        # ── Диагностика: honest AUC на held-out сплите ───────────────────────
        X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
            X_combined, y_combined, test_size=0.30, stratify=y_combined, random_state=self.random_seed,
        )
        diag_model = self._fit_adversarial(X_tr_d, y_tr_d, self.cat_features_)
        diag_proba = diag_model.predict_proba(Pool(X_te_d, cat_features=self.cat_features_))[:, 1]
        self.adversarial_auc_ = float(roc_auc_score(y_te_d, diag_proba))
        logger.info('[AdvWeighting] adversarial AUC (honest, 70/30)=%.4f', self.adversarial_auc_)
        if self.adversarial_auc_ < 0.55:
            logger.info('[AdvWeighting] AUC~0.5 — значимого дрейфа не обнаружено, веса будут близки к 1.0')

        # ── Модель для собственно весов: на полном train+valid ──────────────
        weighting_model = self._fit_adversarial(X_combined, y_combined, self.cat_features_)
        p_valid = weighting_model.predict_proba(
            Pool(X_train[feats], cat_features=self.cat_features_)
        )[:, 1]
        eps = 1e-4
        p_valid = np.clip(p_valid, eps, 1.0 - eps)
        raw_weights = p_valid / (1.0 - p_valid)

        lo, hi = self.clip_weights
        clipped = np.clip(raw_weights, lo, hi)
        mean_before_norm = float(clipped.mean())
        weights = clipped / mean_before_norm

        self.weights_ = weights
        ess_ratio = float((weights.sum() ** 2) / (len(weights) * np.sum(weights ** 2)))
        self.weight_stats_ = {
            'min': float(weights.min()), 'max': float(weights.max()),
            'mean_before_norm': mean_before_norm, 'effective_sample_size_ratio': ess_ratio,
        }
        logger.info('[AdvWeighting] веса: min=%.3f max=%.3f  ESS/n=%.3f',
                    self.weight_stats_['min'], self.weight_stats_['max'], ess_ratio)

        tr_pool = Pool(X_train[feats], y_train.values, cat_features=self.cat_features_, weight=weights)
        va_pool = Pool(X_valid[feats], y_valid.values, cat_features=self.cat_features_)
        if self.n_optuna_trials > 0:
            params = self._tune(tr_pool, va_pool, y_valid.values)
        else:
            params = {**(self.base_params or _DEFAULT_PARAMS), 'random_seed': self.random_seed}
        self._model = CatBoostClassifier(**params)
        self._model.fit(tr_pool, eval_set=va_pool, verbose=False)
        self.best_params_ = params

        self.train_pred_ = self._model.predict_proba(tr_pool)[:, 1]
        self.valid_pred_ = self._model.predict_proba(va_pool)[:, 1]
        val_pr_auc = float(average_precision_score(y_valid.values, self.valid_pred_))
        logger.info('[AdvWeighting] val PR-AUC=%.4f', val_pr_auc)
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        pool = Pool(X[self.selected_features_], cat_features=self.cat_features_)
        return self._model.predict_proba(pool)[:, 1]
