"""HardNegativeMiner: итеративный Hard Negative Mining с CatBoost.

Идея: после каждого раунда обучения находим «трудные» негативы — истинно-нулевые
наблюдения, которым модель ошибочно присваивает высокую вероятность. В следующем
раунде они получают повышенный sample_weight, направляя градиент сильнее на эти примеры.

По val PR-AUC отслеживается лучший раунд — итоговая модель берётся оттуда.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.models._utils import fit_calibrator
from ml_toolkit.presets.classification._base import BasePreset
from ml_toolkit.presets.classification._optuna_utils import CatBoostPruningCallback, make_pruner

logger = logging.getLogger(__name__)

_DEFAULT_BASE_PARAMS: dict[str, Any] = {
    'iterations': 700,
    'max_depth': 5,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'min_data_in_leaf': 10,
    'early_stopping_rounds': 100,
    'loss_function': 'Logloss',
    'eval_metric': 'PRAUC',
    'random_seed': 42,
    'verbose': 0,
}


class HardNegativeMiner(BasePreset):
    """Итеративный Hard Negative Mining.

    Каждый раунд:
    1. Обучаем CatBoost с текущими весами (раунд 0 — равные веса).
    2. Находим «трудные» негативы: y=0, но proba > percentile(hard_percentile) по негативам.
    3. Умножаем их вес на hard_weight.
    4. Повторяем n_rounds раундов.

    Лучшая модель (по val PR-AUC) сохраняется в self._model.

    Parameters
    ----------
    n_rounds:
        Число итераций HNM.
    hard_percentile:
        Перцентиль предсказаний среди негативов; negatives выше — hard negatives.
        0.80 = топ 20% негативов по вероятности.
    hard_weight:
        Множитель веса для hard negatives в следующем раунде.
    base_params:
        Параметры CatBoost. Если None — используются дефолтные.
    n_optuna_trials:
        Если > 0, параметры первого раунда ищутся через Optuna (P@K), последующие
        раунды используют найденные параметры.
    calibrate:
        Применять ли изотоническую калибровку к valid_pred_.
    random_seed:
        Зерно CatBoost и Optuna sampler'а.

    Пример::

        model = HardNegativeMiner(n_rounds=3, hard_percentile=0.80, hard_weight=4.0)
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=[...])
        proba = model.predict_proba(X_test)
        print(model.pr_auc_per_round_)
    """

    def __init__(
        self,
        n_rounds: int = 3,
        hard_percentile: float = 0.80,
        hard_weight: float = 4.0,
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        optuna_timeout: int | None = None,
        calibrate: bool = True,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ):
        super().__init__(params=base_params, n_optuna_trials=n_optuna_trials)
        self.optuna_timeout = optuna_timeout
        self.n_rounds = n_rounds
        self.hard_percentile = hard_percentile
        self.hard_weight = hard_weight
        self.base_params = base_params
        self.calibrate = calibrate
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []
        self.pr_auc_per_round_: list[float] = []
        self.models_: list = []

    # ── Optuna (раунд 0) ────────────────────────────────────────────────────

    def _fit_round0_optuna(self, tr_pool, va_pool, y_va):
        import optuna
        from catboost import CatBoostClassifier

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                'iterations': trial.suggest_int('iterations', 300, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 20.0, log=True),
                'loss_function': 'Logloss',
                'eval_metric': 'PRAUC',
                'early_stopping_rounds': 100,
                'random_seed': self.random_seed,
                'verbose': 0,
            }
            pruning_cb = CatBoostPruningCallback(trial, 'PRAUC')
            m = CatBoostClassifier(**params)
            m.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=[pruning_cb])
            pruning_cb.check_pruned()
            p = m.predict_proba(va_pool)[:, 1]
            return float(average_precision_score(y_va, p))

        logger.info('[HNM] Optuna round 0: %d trials', self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed),
                                    pruner=make_pruner())
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        best = {
            **study.best_params,
            'loss_function': 'Logloss', 'eval_metric': 'PRAUC',
            'early_stopping_rounds': 100, 'random_seed': self.random_seed, 'verbose': 0,
        }
        m = CatBoostClassifier(**best)
        m.fit(tr_pool, eval_set=va_pool, verbose=False)
        return m, best

    # ── Обновление весов ────────────────────────────────────────────────────

    def _update_weights(
        self,
        y_tr: np.ndarray,
        train_proba: np.ndarray,
        current_weights: np.ndarray,
    ) -> np.ndarray:
        neg_mask = y_tr == 0
        neg_probas = train_proba[neg_mask]
        if len(neg_probas) == 0:
            return current_weights
        threshold = float(np.percentile(neg_probas, self.hard_percentile * 100))
        hard_neg_mask = neg_mask & (train_proba >= threshold)

        new_weights = current_weights.copy()
        new_weights[hard_neg_mask] = new_weights[hard_neg_mask] * self.hard_weight
        # Нормируем чтобы средний вес оставался = 1
        new_weights = new_weights / new_weights.mean()

        n_hard = int(hard_neg_mask.sum())
        logger.info('[HNM] Hard negatives: %d (%.1f%% негативов)  threshold=%.4f',
                    n_hard, n_hard / max(neg_mask.sum(), 1) * 100, threshold)
        return new_weights

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> 'HardNegativeMiner':
        from catboost import CatBoostClassifier, Pool

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values

        sample_weights = np.ones(len(y_tr))
        va_pool = Pool(X_valid[feats], y_va, cat_features=self.cat_features_)

        fixed_params = {**(self.base_params or _DEFAULT_BASE_PARAMS), 'random_seed': self.random_seed}
        best_auc = -1.0
        best_model = None
        self.models_ = []
        self.pr_auc_per_round_ = []

        for r in range(self.n_rounds):
            tr_pool = Pool(
                X_train[feats], y_tr,
                cat_features=self.cat_features_,
                weight=sample_weights,
            )

            if r == 0 and self.n_optuna_trials > 0:
                # В раунде 0 все веса равны 1.0, поэтому Pool без weight эквивалентен
                tr_pool_unweighted = Pool(X_train[feats], y_tr, cat_features=self.cat_features_)
                model, fixed_params = self._fit_round0_optuna(tr_pool_unweighted, va_pool, y_va)
                self.best_params_ = fixed_params
            else:
                model = CatBoostClassifier(**fixed_params)
                model.fit(tr_pool, eval_set=va_pool, verbose=False)

            va_p = model.predict_proba(va_pool)[:, 1]
            auc = float(average_precision_score(y_va, va_p))
            self.pr_auc_per_round_.append(auc)
            self.models_.append(model)
            logger.info('[HNM] Раунд %d/%d  val PR-AUC=%.4f', r + 1, self.n_rounds, auc)

            if auc > best_auc:
                best_auc = auc
                best_model = model

            if r < self.n_rounds - 1:
                train_proba = model.predict_proba(
                    Pool(X_train[feats], cat_features=self.cat_features_)
                )[:, 1]
                sample_weights = self._update_weights(y_tr, train_proba, sample_weights)

        self._model = best_model
        if self.best_params_ is None:
            self.best_params_ = fixed_params

        best_round = int(np.argmax(self.pr_auc_per_round_)) + 1
        logger.info('[HNM] Лучший раунд %d  PR-AUC=%.4f', best_round, best_auc)

        raw_va = self._model.predict_proba(va_pool)[:, 1]
        if self.calibrate:
            self.calibrator_ = fit_calibrator(raw_va, y_va)
            self.valid_pred_ = self.calibrator_.predict(raw_va)
        else:
            self.valid_pred_ = raw_va

        self.train_pred_ = self._model.predict_proba(
            Pool(X_train[feats], cat_features=self.cat_features_)
        )[:, 1]
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        pool = Pool(X[self.selected_features_], cat_features=self.cat_features_)
        raw = self._model.predict_proba(pool)[:, 1]
        if self.calibrate and self.calibrator_ is not None:
            return self.calibrator_.predict(raw)
        return raw
