"""PrecisionAtKClassifier: Optuna оптимизирует гиперпараметры CatBoost под precision@K,
где K — доля наблюдений (например, 0.10 = топ 10% выборки).

Дополнительно тюнирует scale_pos_weight и majority_fraction совместно с архитектурными
параметрами, что нужно для задач с сильным дисбалансом классов и фиксированным cutoff.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.models._utils import fit_calibrator, precision_at_k
from ml_toolkit.presets.classification._base import BasePreset
from ml_toolkit.presets.classification._optuna_utils import CatBoostPruningCallback, make_pruner

logger = logging.getLogger(__name__)


class PrecisionAtKClassifier(BasePreset):
    """CatBoost, оптимизированный под Precision@K (K — доля наблюдений).

    Optuna совместно ищет:
    - архитектурные параметры (iterations, max_depth, lr, …)
    - majority_fraction (урезание мажоритарного класса в trial-выборке)

    Финальная модель воспроизводит лучший триал точно: обучается на той же
    подвыборке мажоритарного класса (тот же seed) с теми же параметрами,
    поэтому best_precision_at_k_ соответствует возвращаемой модели.

    После fit доступны:
    - valid_pred_       — откалиброванные вероятности на val
    - train_pred_       — вероятности на train
    - best_params_      — лучшие параметры Optuna
    - best_precision_at_k_ — P@K на val лучшего trial

    Parameters
    ----------
    k_fraction:
        Доля наблюдений для precision@K (например, 0.10 = топ 10%).
    n_optuna_trials:
        Число trials Optuna.
    calibrate:
        Применять ли изотоническую калибровку к выходным вероятностям.
    random_seed:
        Зерно CatBoost, Optuna sampler'а и подвыборки мажоритарного класса.

    Пример::

        model = PrecisionAtKClassifier(k_fraction=0.05, n_optuna_trials=60)
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=[...])
        proba = model.predict_proba(X_test)
        print(f"val P@5%: {model.best_precision_at_k_:.4f}")
    """

    def __init__(
        self,
        k_fraction: float = 0.10,
        n_optuna_trials: int = 50,
        optuna_timeout: int | None = None,
        calibrate: bool = True,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ):
        if not 0.0 < k_fraction <= 1.0:
            raise ValueError(f"k_fraction должен быть в (0, 1], получено {k_fraction}")
        super().__init__(params=None, n_optuna_trials=n_optuna_trials)
        self.optuna_timeout = optuna_timeout
        self.k_fraction = k_fraction
        self.calibrate = calibrate
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []
        self.best_precision_at_k_: float = 0.0

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> 'PrecisionAtKClassifier':
        import optuna
        from catboost import CatBoostClassifier, Pool

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values

        classes, counts = np.unique(y_tr, return_counts=True)
        maj_pos = int(np.argmax(counts))
        majority_cls = classes[maj_pos]
        minority_cls = classes[1 - maj_pos]
        majority_idx = np.where(y_tr == majority_cls)[0]
        minority_idx = np.where(y_tr == minority_cls)[0]
        n_majority = int(counts[maj_pos])

        va_pool = Pool(X_valid[feats], y_va, cat_features=self.cat_features_)

        for cls, cnt in zip(classes, counts):
            logger.info('[P@K] Класс %s: %d (%.1f%%)', cls, cnt, cnt / len(y_tr) * 100)
        logger.info('[P@K] k_fraction=%.3f → топ %d наблюдений val',
                    self.k_fraction, max(1, int(len(y_va) * self.k_fraction)))

        def objective(trial: optuna.Trial) -> float:
            majority_fraction = trial.suggest_float('majority_fraction', 0.05, 1.0)
            n_keep = max(1, int(n_majority * majority_fraction))
            rng = np.random.default_rng(self.random_seed + trial.number)
            sampled_maj = rng.choice(majority_idx, size=n_keep, replace=False)
            idx = np.sort(np.concatenate([minority_idx, sampled_maj]))

            params = {
                'iterations': trial.suggest_int('iterations', 300, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
                'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'rsm': trial.suggest_float('rsm', 0.3, 1.0),
                'loss_function': 'Logloss',
                'eval_metric': 'PRAUC',
                'verbose': 0,
                'early_stopping_rounds': 100,
                'random_seed': self.random_seed,
            }

            trial_pool = Pool(
                X_train[feats].iloc[idx], y_tr[idx], cat_features=self.cat_features_
            )
            pruning_cb = CatBoostPruningCallback(trial, 'PRAUC')
            m = CatBoostClassifier(**params)
            m.fit(trial_pool, eval_set=va_pool, verbose=False, callbacks=[pruning_cb])
            pruning_cb.check_pruned()
            proba = m.predict_proba(va_pool)[:, 1]
            return precision_at_k(y_va, proba, k=self.k_fraction)

        logger.info('[P@K] Optuna: %d trials', self.n_optuna_trials)
        study = optuna.create_study(
            direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            pruner=make_pruner(),
        )
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        self.best_precision_at_k_ = float(study.best_value)
        logger.info('[P@K] Лучший P@K=%.4f  params=%s', study.best_value, study.best_params)

        # Финальная модель точно воспроизводит лучший триал: та же подвыборка
        # мажоритарного класса (тот же seed = random_seed + номер триала) и те же
        # параметры. Пересчёт majority_fraction в scale_pos_weight некорректен:
        # undersampling и reweighting для деревьев не эквивалентны, и метрика
        # best_precision_at_k_ была бы измерена не у возвращаемой модели.
        best = dict(study.best_params)
        majority_fraction = best.pop('majority_fraction')
        n_keep = max(1, int(n_majority * majority_fraction))
        rng = np.random.default_rng(self.random_seed + study.best_trial.number)
        sampled_maj = rng.choice(majority_idx, size=n_keep, replace=False)
        final_idx = np.sort(np.concatenate([minority_idx, sampled_maj]))

        self.best_params_ = {
            **best,
            'majority_fraction': majority_fraction,
            'loss_function': 'Logloss',
            'eval_metric': 'PRAUC',
            'early_stopping_rounds': 100,
            'random_seed': self.random_seed,
            'verbose': 0,
        }
        fit_params = {k: v for k, v in self.best_params_.items() if k != 'majority_fraction'}

        tr_pool = Pool(
            X_train[feats].iloc[final_idx], y_tr[final_idx], cat_features=self.cat_features_
        )
        self._model = CatBoostClassifier(**fit_params)
        self._model.fit(tr_pool, eval_set=va_pool, verbose=False)

        self.train_pred_ = self._model.predict_proba(
            Pool(X_train[feats], cat_features=self.cat_features_)
        )[:, 1]

        raw_va = self._model.predict_proba(va_pool)[:, 1]
        if self.calibrate:
            self.calibrator_ = fit_calibrator(raw_va, y_va)
            self.valid_pred_ = self.calibrator_.predict(raw_va)
        else:
            self.valid_pred_ = raw_va

        logger.info('[P@K] Финал  P@K=%.4f  PR-AUC=%.4f',
                    precision_at_k(y_va, self.valid_pred_, k=self.k_fraction),
                    average_precision_score(y_va, self.valid_pred_))
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        pool = Pool(X[self.selected_features_], cat_features=self.cat_features_)
        raw = self._model.predict_proba(pool)[:, 1]
        if self.calibrate and self.calibrator_ is not None:
            return self.calibrator_.predict(raw)
        return raw
