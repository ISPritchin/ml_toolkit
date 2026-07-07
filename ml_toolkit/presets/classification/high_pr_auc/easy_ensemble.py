"""EasyEnsembleClassifier: ансамбль с undersampling-diversification.

Каждый из N estimators обучается на всех позитивах + случайном срезе негативов
(1 : neg_ratio), что создаёт N разнообразных взглядов на пространство негативов.
Финальный скор — среднее нормированных рангов (rank averaging), устойчивое к
разному масштабу вероятностей между estimators.

Отличие от SubsampleStacking:
  - Нет мета-слоя, нет риска утечки через OOB.
  - Подвыборка только негативов, позитивы — всегда полностью.
  - Поддерживает LightGBM и CatBoost как базовые модели.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.models._utils import fit_rank_reference, rank_transform
from ml_toolkit.presets.classification._base import BasePreset
from ml_toolkit.presets.classification._optuna_utils import CatBoostPruningCallback, make_pruner

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


class EasyEnsembleClassifier(BasePreset):
    """Ансамбль с undersampling-diversification для экстремального дисбаланса.

    Parameters
    ----------
    n_estimators:
        Количество базовых моделей (рекомендуется 10–20).
    neg_ratio:
        Негативов на один позитив в каждом train-подмножестве (рекомендуется 5–20).
        Если в train негативов меньше, чем neg_ratio * n_pos, берутся все негативы.
    base:
        'lightgbm' (по умолчанию) или 'catboost'.
    base_params:
        Гиперпараметры базовой модели. None → дефолтные для выбранного base.
        Игнорируется, если n_optuna_trials > 0.
    n_optuna_trials:
        Если > 0, общая архитектура (одна на всех estimator'ов) подбирается через
        Optuna по val PR-AUC на одном представительном под-бэге (undersampled как
        и остальные estimator'ы).
    optuna_timeout:
        Ограничение по времени (сек) на весь Optuna-поиск. None — без ограничения.
    random_seed:
        Начальное зерно. Каждый estimator получает seed + i. Также сид Optuna sampler'а.

    Атрибуты после fit::

        estimators_        — список обученных базовых моделей
        estimator_scores_  — val PR-AUC каждого estimator
        ensemble_score_    — val PR-AUC финального ансамбля
    """

    def __init__(
        self,
        n_estimators: int = 10,
        neg_ratio: int = 10,
        base: str = 'catboost',
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        optuna_timeout: int | None = None,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(params=base_params, n_optuna_trials=n_optuna_trials)
        if base not in ('lightgbm', 'catboost'):
            raise ValueError(f"base должен быть 'lightgbm' или 'catboost', получено {base!r}")
        self.n_estimators = n_estimators
        self.neg_ratio = neg_ratio
        self.base = base
        self.base_params = base_params
        self.optuna_timeout = optuna_timeout
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
        import optuna
        from catboost import CatBoostClassifier, Pool

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        tr_pool = Pool(X_sub, y_sub, cat_features=self.cat_features_)
        va_pool = Pool(X_va, y_va, cat_features=self.cat_features_)

        def objective(trial: optuna.Trial) -> float:
            params = {
                'iterations': trial.suggest_int('iterations', 300, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
                'loss_function': 'Logloss',
                'eval_metric': 'PRAUC',
                'early_stopping_rounds': 80,
                'random_seed': self.random_seed,
                'verbose': 0,
            }
            pruning_cb = CatBoostPruningCallback(trial, 'PRAUC')
            m = CatBoostClassifier(**params)
            m.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=[pruning_cb])
            pruning_cb.check_pruned()
            p = m.predict_proba(va_pool)[:, 1]
            return float(average_precision_score(y_va, p))

        logger.info('[EasyEnsemble] Optuna (catboost): %d trials', self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed),
                                    pruner=make_pruner())
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        return {
            **study.best_params,
            'loss_function': 'Logloss', 'eval_metric': 'PRAUC',
            'early_stopping_rounds': 80, 'random_seed': self.random_seed, 'verbose': 0,
        }

    def _tune_lgb(self, X_sub: pd.DataFrame, y_sub: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray) -> dict[str, Any]:
        import optuna
        import lightgbm as lgb

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'random_state': self.random_seed,
                'verbose': -1,
                'n_jobs': -1,
            }
            m = lgb.LGBMClassifier(**params)
            m.fit(
                X_sub, y_sub,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            p = m.predict_proba(X_va)[:, 1]
            return float(average_precision_score(y_va, p))

        logger.info('[EasyEnsemble] Optuna (lightgbm): %d trials', self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed))
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        return {**study.best_params, 'random_state': self.random_seed, 'verbose': -1, 'n_jobs': -1}

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> 'EasyEnsembleClassifier':
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values
        X_tr_feats = X_train[feats]
        X_va_feats = X_valid[feats]

        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        n_pos = len(pos_idx)
        n_neg_sample = min(len(neg_idx), self.neg_ratio * n_pos)

        logger.info(
            '[EasyEnsemble] n_estimators=%d  neg_ratio=%d  n_pos=%d  n_neg/estimator=%d  base=%s',
            self.n_estimators, self.neg_ratio, n_pos, n_neg_sample, self.base,
        )

        tuned_params = None
        if self.n_optuna_trials > 0:
            rng0 = np.random.default_rng(self.random_seed)
            neg_sample0 = rng0.choice(neg_idx, size=n_neg_sample, replace=False)
            sample_idx0 = np.concatenate([pos_idx, neg_sample0])
            rng0.shuffle(sample_idx0)
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
            rng = np.random.default_rng(self.random_seed + i)
            neg_sample = rng.choice(neg_idx, size=n_neg_sample, replace=False)
            sample_idx = np.concatenate([pos_idx, neg_sample])
            rng.shuffle(sample_idx)

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
            logger.info('[EasyEnsemble] estimator %2d/%d  val PR-AUC=%.4f', i + 1, self.n_estimators, ap)

        # Референсы rank-нормализации — train-скоры каждого estimator; predict_proba
        # использует их же, поэтому скор объекта не зависит от состава батча.
        tr_raw_scores = [self._predict_one(m, X_tr_feats) for m in self.estimators_]
        self._rank_refs_ = [fit_rank_reference(s) for s in tr_raw_scores]

        va_ensemble = np.stack(
            [rank_transform(s, ref) for s, ref in zip(va_raw_scores, self._rank_refs_)],
            axis=1,
        ).mean(axis=1)
        self.ensemble_score_ = float(average_precision_score(y_va, va_ensemble))
        logger.info('[EasyEnsemble] ensemble val PR-AUC=%.4f  (mean single=%.4f)',
                    self.ensemble_score_, float(np.mean(self.estimator_scores_)))

        self.valid_pred_ = va_ensemble
        self.train_pred_ = np.stack(
            [rank_transform(s, ref) for s, ref in zip(tr_raw_scores, self._rank_refs_)],
            axis=1,
        ).mean(axis=1)

        self.best_params_ = {
            'n_estimators': self.n_estimators,
            'neg_ratio': self.neg_ratio,
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
