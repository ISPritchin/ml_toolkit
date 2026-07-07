"""SubsampleStacking: стекинг CatBoost-моделей с K-fold OOF мета-матрицей.

Архитектура:
  1. Для каждого из N конфигов базовой модели строятся out-of-fold предсказания:
     модель обучается на K-1 фолдах (внутри фолда — stratified подвыборка
     subsample_rate для diversity) и предсказывает held-out фолд. В итоге каждая
     строка train имеет предсказания ВСЕХ N конфигов, полученные честно
     out-of-fold — без импутации пропусков.
  2. Мета-модель обучается на полной OOF-матрице (n_train × n_base_models).
  3. Финальная базовая модель каждого конфига обучается на stratified
     подвыборке полного train и используется для val/inference предсказаний.
  4. Val набор используется исключительно для оценки и early stopping,
     не для обучения мета-модели.
  5. train_pred_ = предсказания мета-модели на OOF-матрице; valid_pred_ =
     предсказания мета-модели (корректные, т.к. мета-модель не видела val).

Мета-модели:
  'logistic'  — LogisticRegression(C=1.0) на OOF предсказаниях каждой базовой модели.
  'weighted'  — softmax-веса, оптимизированные scipy.optimize.minimize по BCE
                (гладкий surrogate; прямой PR-AUC кусочно-постоянен и не даёт
                градиента для L-BFGS-B).
  'catboost'  — мини CatBoost (iterations=200, max_depth=3) на OOF предсказаниях.
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


def _default_base_configs(n: int) -> list[dict]:
    """N разнообразных конфига для базовых моделей."""
    pool = [
        {'scale_pos_weight': 1.0, 'max_depth': 4, 'learning_rate': 0.05, 'random_seed': 42},
        {'scale_pos_weight': 3.0, 'max_depth': 5, 'learning_rate': 0.05, 'random_seed': 123},
        {'scale_pos_weight': 6.0, 'max_depth': 6, 'learning_rate': 0.03, 'random_seed': 456},
        {'scale_pos_weight': 2.0, 'max_depth': 4, 'learning_rate': 0.10, 'random_seed': 789},
        {'scale_pos_weight': 1.0, 'max_depth': 6, 'learning_rate': 0.03, 'random_seed': 321},
        {'scale_pos_weight': 4.0, 'max_depth': 5, 'learning_rate': 0.07, 'random_seed': 654},
        {'scale_pos_weight': 1.5, 'max_depth': 3, 'learning_rate': 0.15, 'random_seed': 987},
        {'scale_pos_weight': 8.0, 'max_depth': 7, 'learning_rate': 0.02, 'random_seed': 111},
    ]
    return [pool[i % len(pool)] for i in range(n)]


_SHARED_PARAMS: dict[str, Any] = {
    'iterations': 600,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'min_data_in_leaf': 10,
    'early_stopping_rounds': 80,
    'loss_function': 'Logloss',
    'eval_metric': 'PRAUC',
    'verbose': 0,
}


class SubsampleStacking(BasePreset):
    """Стекинг CatBoost с K-fold OOF мета-обучением.

    Parameters
    ----------
    n_base_models:
        Число базовых моделей (конфигов).
    subsample_rate:
        Доля строк, используемая при обучении каждой модели (stratified, без
        замены) — и внутри OOF-фолдов, и для финальных базовых моделей.
    n_folds:
        Число фолдов для построения OOF мета-матрицы.
    base_configs:
        Список dict — специфичные параметры каждой базовой модели, мёрджатся
        с _SHARED_PARAMS. None → авто-разнообразные конфиги.
    n_optuna_trials:
        Если > 0, общая часть архитектуры (_SHARED_PARAMS: iterations,
        l2_leaf_reg, subsample, min_data_in_leaf) подбирается через Optuna по
        val PR-AUC на одном представительном конфиге, вместо дефолтных
        _SHARED_PARAMS. per-конфиг diversity-параметры (max_depth,
        learning_rate, scale_pos_weight, random_seed из base_configs) не
        затрагиваются — тюнится только общая для всех конфигов часть.
    optuna_timeout:
        Ограничение по времени (сек) на весь Optuna-поиск. None — без ограничения.
    meta:
        Мета-модель: 'logistic', 'weighted', 'catboost'.
    calibrate:
        Применять ли изотоническую калибровку к финальным предсказаниям.
    random_seed:
        Зерно stratified-подвыборки, StratifiedKFold и мета-модели. Отдельные
        базовые конфиги (base_configs) намеренно используют разные seed'ы —
        это разнообразие внутри ансамбля, а не несогласованность.

    Атрибуты после fit::

        base_models_        — список финальных CatBoost моделей (по одной на конфиг)
        meta_model_         — обученная мета-модель
        oob_pr_aucs_        — PR-AUC OOF-предсказаний каждого конфига на train
        valid_pr_auc_       — PR-AUC ансамбля на val

    Пример::

        model = SubsampleStacking(n_base_models=6, meta='logistic')
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=[...])
        proba = model.predict_proba(X_test)
    """

    def __init__(
        self,
        n_base_models: int = 5,
        subsample_rate: float = 0.75,
        n_folds: int = 5,
        base_configs: list[dict] | None = None,
        n_optuna_trials: int = 0,
        optuna_timeout: int | None = None,
        meta: str = 'logistic',
        calibrate: bool = True,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ):
        if not 0.0 < subsample_rate < 1.0:
            raise ValueError(f"subsample_rate должен быть в (0, 1), получено {subsample_rate}")
        if n_folds < 2:
            raise ValueError(f"n_folds должен быть >= 2, получено {n_folds}")
        if meta not in ('logistic', 'weighted', 'catboost'):
            raise ValueError(f"meta должен быть 'logistic', 'weighted' или 'catboost'")
        super().__init__(params=None, n_optuna_trials=n_optuna_trials)
        self.n_base_models = n_base_models
        self.subsample_rate = subsample_rate
        self.n_folds = n_folds
        self.base_configs = base_configs
        self.optuna_timeout = optuna_timeout
        self.meta = meta
        self.calibrate = calibrate
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.base_models_: list = []
        self.meta_model_: Any = None
        self.oob_pr_aucs_: list[float] = []
        self.valid_pr_auc_: float = 0.0

    # ── Мета-модели ─────────────────────────────────────────────────────────

    def _fit_meta_logistic(self, X_meta: np.ndarray, y_meta: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=self.random_seed)
        m.fit(X_meta, y_meta)
        return m

    def _fit_meta_weighted(self, X_meta: np.ndarray, y_meta: np.ndarray):
        from scipy.optimize import minimize

        n = X_meta.shape[1]
        y = y_meta.astype(float)
        eps = 1e-7

        # BCE — гладкий выпуклый surrogate; прямой PR-AUC кусочно-постоянен
        # по весам, и градиентный L-BFGS-B на нём не сдвигается со старта.
        def neg_log_likelihood(raw_w):
            w = np.exp(raw_w) / np.exp(raw_w).sum()   # softmax → сумма = 1
            blend = np.clip(X_meta @ w, eps, 1.0 - eps)
            return -float(np.mean(y * np.log(blend) + (1.0 - y) * np.log(1.0 - blend)))

        res = minimize(neg_log_likelihood, np.zeros(n), method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-12})
        raw_w = res.x
        weights = np.exp(raw_w) / np.exp(raw_w).sum()
        logger.info('[Stacking] Мета-веса (BCE): %s  OOF PR-AUC blend=%.4f',
                    np.round(weights, 3),
                    average_precision_score(y_meta, X_meta @ weights))
        return weights  # просто np.ndarray

    def _fit_meta_catboost(self, X_meta: np.ndarray, y_meta: np.ndarray):
        from catboost import CatBoostClassifier, Pool
        params = {
            'iterations': 200, 'max_depth': 3, 'learning_rate': 0.05,
            'loss_function': 'Logloss', 'eval_metric': 'PRAUC',
            'random_seed': self.random_seed, 'verbose': 0,
        }
        m = CatBoostClassifier(**params)
        m.fit(Pool(X_meta, y_meta))
        return m

    def _meta_predict(self, X_meta: np.ndarray) -> np.ndarray:
        if self.meta == 'logistic':
            return self.meta_model_.predict_proba(X_meta)[:, 1]
        if self.meta == 'weighted':
            return X_meta @ self.meta_model_          # meta_model_ = weights array
        if self.meta == 'catboost':
            from catboost import Pool
            return self.meta_model_.predict_proba(Pool(X_meta))[:, 1]
        raise ValueError(self.meta)

    # ── Optuna (общая часть архитектуры) ─────────────────────────────────────

    def _tune_shared_params(
        self, X_tr_feats: pd.DataFrame, y_tr: np.ndarray, va_pool_full: Any, y_va: np.ndarray,
    ) -> dict[str, Any]:
        import optuna
        from catboost import CatBoostClassifier, Pool

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        rng = np.random.default_rng(self.random_seed)
        sub_idx = self._stratified_subsample(np.arange(len(y_tr)), y_tr, rng)
        tr_pool = Pool(X_tr_feats.iloc[sub_idx], y_tr[sub_idx], cat_features=self.cat_features_)

        def objective(trial: optuna.Trial) -> float:
            params = {
                'iterations': trial.suggest_int('iterations', 300, 1000, step=100),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
                'early_stopping_rounds': 80,
                'loss_function': 'Logloss',
                'eval_metric': 'PRAUC',
                'random_seed': self.random_seed,
                'verbose': 0,
            }
            pruning_cb = CatBoostPruningCallback(trial, 'PRAUC')
            m = CatBoostClassifier(**params)
            m.fit(tr_pool, eval_set=va_pool_full, verbose=False, callbacks=[pruning_cb])
            pruning_cb.check_pruned()
            p = m.predict_proba(va_pool_full)[:, 1]
            return float(average_precision_score(y_va, p))

        logger.info('[Stacking] Optuna: %d trials (общая часть архитектуры для всех конфигов)',
                    self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed),
                                    pruner=make_pruner())
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        best = study.best_params
        return {
            'iterations': best['iterations'], 'l2_leaf_reg': best['l2_leaf_reg'],
            'subsample': best['subsample'], 'min_data_in_leaf': best['min_data_in_leaf'],
            'early_stopping_rounds': 80, 'loss_function': 'Logloss', 'eval_metric': 'PRAUC', 'verbose': 0,
        }

    # ── Вспомогательные ─────────────────────────────────────────────────────

    def _stratified_subsample(
        self, idx_pool: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Stratified подвыборка subsample_rate внутри idx_pool, без замены."""
        parts = []
        for cls in np.unique(y[idx_pool]):
            cls_idx = idx_pool[y[idx_pool] == cls]
            n_cls = max(1, int(len(cls_idx) * self.subsample_rate))
            parts.append(rng.choice(cls_idx, size=n_cls, replace=False))
        return np.sort(np.concatenate(parts))

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> 'SubsampleStacking':
        from catboost import CatBoostClassifier, Pool
        from sklearn.model_selection import StratifiedKFold

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values
        n_train = len(y_tr)
        rng = np.random.default_rng(self.random_seed)

        configs = self.base_configs or _default_base_configs(self.n_base_models)
        if len(configs) < self.n_base_models:
            configs = (configs * (self.n_base_models // len(configs) + 1))[:self.n_base_models]

        min_class = int(min(np.bincount(y_tr.astype(int))))
        if min_class < 2:
            raise ValueError(
                f'Для OOF-стекинга нужно >= 2 примеров каждого класса в train, '
                f'минимальный класс содержит {min_class}.'
            )
        n_splits = min(self.n_folds, min_class)
        if n_splits < self.n_folds:
            logger.warning('[Stacking] n_folds снижен %d → %d (мало позитивов в train)',
                           self.n_folds, n_splits)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        folds = list(skf.split(np.zeros(n_train), y_tr))

        X_tr_feats = X_train[feats]
        oof_matrix = np.zeros((n_train, self.n_base_models))
        va_matrix = np.zeros((len(y_va), self.n_base_models))

        self.base_models_ = []
        self.oob_pr_aucs_ = []
        va_pool_full = Pool(X_valid[feats], y_va, cat_features=self.cat_features_)

        shared_params = (
            self._tune_shared_params(X_tr_feats, y_tr, va_pool_full, y_va)
            if self.n_optuna_trials > 0 else _SHARED_PARAMS
        )

        for i, cfg in enumerate(configs[:self.n_base_models]):
            params = {**shared_params, **cfg}
            logger.info('[Stacking] Конфиг %d/%d  cfg=%s', i + 1, self.n_base_models, cfg)

            # OOF: обучение на K-1 фолдах (со stratified подвыборкой внутри),
            # предсказание held-out фолда → полная мета-матрица без импутации.
            for tr_idx_f, te_idx_f in folds:
                sub_idx = self._stratified_subsample(tr_idx_f, y_tr, rng)
                fold_pool = Pool(
                    X_tr_feats.iloc[sub_idx], y_tr[sub_idx], cat_features=self.cat_features_
                )
                m_f = CatBoostClassifier(**params)
                m_f.fit(fold_pool, eval_set=va_pool_full, verbose=False)
                oof_matrix[te_idx_f, i] = m_f.predict_proba(
                    Pool(X_tr_feats.iloc[te_idx_f], cat_features=self.cat_features_)
                )[:, 1]

            oof_auc = float(average_precision_score(y_tr, oof_matrix[:, i]))
            self.oob_pr_aucs_.append(oof_auc)
            logger.info('[Stacking] Конфиг %d  OOF PR-AUC=%.4f', i + 1, oof_auc)

            # Финальная базовая модель конфига — на подвыборке полного train;
            # используется для val- и inference-предсказаний.
            sub_idx = self._stratified_subsample(np.arange(n_train), y_tr, rng)
            m = CatBoostClassifier(**params)
            m.fit(
                Pool(X_tr_feats.iloc[sub_idx], y_tr[sub_idx], cat_features=self.cat_features_),
                eval_set=va_pool_full, verbose=False,
            )
            va_matrix[:, i] = m.predict_proba(va_pool_full)[:, 1]
            self.base_models_.append(m)

        # ── Обучение мета-модели на полной OOF-матрице ───────────────────────
        if self.meta == 'logistic':
            self.meta_model_ = self._fit_meta_logistic(oof_matrix, y_tr)
        elif self.meta == 'weighted':
            self.meta_model_ = self._fit_meta_weighted(oof_matrix, y_tr)
        elif self.meta == 'catboost':
            self.meta_model_ = self._fit_meta_catboost(oof_matrix, y_tr)

        # ── Итоговые предсказания ────────────────────────────────────────────
        # valid_pred_ — честные (мета-модель не видела val)
        raw_va = self._meta_predict(va_matrix)
        if self.calibrate:
            self.calibrator_ = fit_calibrator(raw_va, y_va)
            self.valid_pred_ = self.calibrator_.predict(raw_va)
        else:
            self.valid_pred_ = raw_va

        self.train_pred_ = self._meta_predict(oof_matrix)

        self.valid_pr_auc_ = float(average_precision_score(y_va, self.valid_pred_))
        self.best_params_ = {'meta': self.meta, 'n_base_models': self.n_base_models,
                             'subsample_rate': self.subsample_rate, 'n_folds': n_splits,
                             'shared_params': shared_params}
        self._model = True

        logger.info('[Stacking] val PR-AUC=%.4f  OOF PR-AUCs=%s',
                    self.valid_pr_auc_, [f'{x:.3f}' for x in self.oob_pr_aucs_])
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        n = self.n_base_models
        preds = np.zeros((len(X), n))
        for i, m in enumerate(self.base_models_):
            pool = Pool(X[self.selected_features_], cat_features=self.cat_features_)
            preds[:, i] = m.predict_proba(pool)[:, 1]
        raw = self._meta_predict(preds)
        if self.calibrate and self.calibrator_ is not None:
            return self.calibrator_.predict(raw)
        return raw
