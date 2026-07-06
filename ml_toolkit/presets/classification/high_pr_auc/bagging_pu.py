"""BaggingPUClassifier: PU-бэггинг (Mordelet & Vert, 2014).

Каждый из n_estimators обучается на всех позитивах P против bootstrap-выборки
(с возвратом) из "негативов" U (которые в PU-постановке на деле unlabeled — U
почти наверняка содержит незамеченные позитивы). Ключевая идея — out-of-bag
(OOB) оценка: скор точки u из U усредняется ТОЛЬКО по тем estimator'ам, в
bootstrap-выборку которых u НЕ попала. Если усреднять по всем estimator'ам
(включая те, что видели u как "негатив" при обучении), скор был бы
оптимистично занижен именно там, где U ошибочно, — OOB убирает это смещение.

Отличие от EasyEnsembleClassifier: тот подвыбирает НЕГАТИВЫ ratio-к-позитивам
для diversity при обычном (не PU) дисбалансе и усредняет по всем моделям без
OOB-разбора; здесь semantics другая — U trактуется как зашумлённый (не
достоверный) класс, и OOB — не опция для устойчивости, а необходимость метода.

Когда Элкан-Ното (PULearningClassifier/029) нестабилен: c оценивается по
малому числу val-позитивов и шумно; PU-бэггинг не оценивает никакой skalar c,
а прямо переусредняет предсказания — более устойчиво при малых выборках.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.presets.classification._base import BasePreset

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    'iterations': 300,
    'max_depth': 5,
    'learning_rate': 0.08,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'min_data_in_leaf': 5,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': 0,
}


class BaggingPUClassifier(BasePreset):
    """PU-бэггинг с out-of-bag усреднением (Mordelet & Vert, 2014).

    Parameters
    ----------
    n_estimators:
        Число базовых моделей (рекомендуется 20-50 — OOB-покрытие каждой
        точки U растёт с числом estimator'ов).
    u_sample_size:
        Размер bootstrap-выборки из U на один estimator. None → равен числу
        позитивов в train (сбалансированный P/U на каждой итерации).
    base_params:
        Параметры CatBoost. None → дефолтные (не PRAUC-eval_metric — на
        подвыборке из нескольких десятков объектов PRAUC/early stopping
        нестабильны, используется фиксированное iterations без eval_set).
    random_seed:
        Базовое зерно; estimator i получает random_seed + i.

    Атрибуты после fit::

        oob_coverage_    — доля примеров U, получивших хотя бы одну OOB-оценку
        train_pu_pr_auc_ — PR-AUC OOB-скоров (позитивы) / in-bag среднего (U) на train

    Пример::

        model = BaggingPUClassifier(n_estimators=30)
        model.fit(X_train, y_train, X_valid, y_valid)
    """

    def __init__(
        self,
        n_estimators: int = 30,
        u_sample_size: int | None = None,
        base_params: dict[str, Any] | None = None,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(params=None, n_optuna_trials=0)
        if n_estimators < 2:
            raise ValueError(f'n_estimators должен быть >= 2, получено {n_estimators}')
        self.n_estimators = n_estimators
        self.u_sample_size = u_sample_size
        self.base_params = base_params
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.estimators_: list[Any] = []
        self.oob_coverage_: float = 0.0
        self.train_pu_pr_auc_: float = 0.0

    def _fit_one(self, X: pd.DataFrame, y: np.ndarray, seed: int) -> Any:
        from catboost import CatBoostClassifier, Pool

        params = {**(self.base_params or _DEFAULT_PARAMS), 'random_seed': seed}
        m = CatBoostClassifier(**params)
        m.fit(Pool(X, y, cat_features=self.cat_features_), verbose=False)
        return m

    def _predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        return model.predict_proba(Pool(X, cat_features=self.cat_features_))[:, 1]

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> 'BaggingPUClassifier':
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        X_tr = X_train[feats].reset_index(drop=True)
        y_tr = y_train.values
        pos_idx = np.where(y_tr == 1)[0]
        u_idx = np.where(y_tr == 0)[0]
        n_pos, n_u = len(pos_idx), len(u_idx)
        u_size = self.u_sample_size or n_pos

        logger.info('[BaggingPU] n_estimators=%d  n_pos=%d  n_u=%d  u_sample_size=%d',
                    self.n_estimators, n_pos, n_u, u_size)

        oob_sum = np.zeros(n_u)
        oob_count = np.zeros(n_u)
        self.estimators_ = []

        for i in range(self.n_estimators):
            rng = np.random.default_rng(self.random_seed + i)
            boot_local = rng.integers(0, n_u, size=u_size)  # bootstrap с возвратом, локальные индексы внутри U
            in_bag_local = np.unique(boot_local)
            oob_local_mask = np.ones(n_u, dtype=bool)
            oob_local_mask[in_bag_local] = False

            sample_idx = np.concatenate([pos_idx, u_idx[boot_local]])
            model = self._fit_one(X_tr.iloc[sample_idx], y_tr[sample_idx], self.random_seed + i)
            self.estimators_.append(model)

            if oob_local_mask.any():
                oob_global_idx = u_idx[oob_local_mask]
                proba_oob = self._predict(model, X_tr.iloc[oob_global_idx])
                oob_sum[oob_local_mask] += proba_oob
                oob_count[oob_local_mask] += 1

        self.oob_coverage_ = float((oob_count > 0).mean())
        if self.oob_coverage_ < 1.0:
            logger.warning(
                '[BaggingPU] OOB-покрытие=%.1f%% — часть точек U не получила ни одной '
                'OOB-оценки (увеличьте n_estimators или u_sample_size)',
                100.0 * self.oob_coverage_,
            )

        # Точки U без OOB-оценки — fallback на среднее по всем estimator'ам (in-bag, смещённое).
        no_oob = oob_count == 0
        u_scores = np.zeros(n_u)
        has_oob = ~no_oob
        u_scores[has_oob] = oob_sum[has_oob] / oob_count[has_oob]
        if no_oob.any():
            all_proba = np.stack([self._predict(m, X_tr.iloc[u_idx[no_oob]]) for m in self.estimators_], axis=1)
            u_scores[no_oob] = all_proba.mean(axis=1)

        # Позитивы: среднее по всем estimator'ам (все видели P каждый раз).
        pos_scores = np.stack([self._predict(m, X_tr.iloc[pos_idx]) for m in self.estimators_], axis=1).mean(axis=1)

        train_scores = np.empty(len(y_tr))
        train_scores[pos_idx] = pos_scores
        train_scores[u_idx] = u_scores
        self.train_pu_pr_auc_ = float(average_precision_score(y_tr, train_scores))
        self.train_pred_ = train_scores

        X_va = X_valid[feats]
        va_scores = np.stack([self._predict(m, X_va) for m in self.estimators_], axis=1).mean(axis=1)
        self.valid_pred_ = va_scores
        val_pr_auc = float(average_precision_score(y_valid.values, va_scores))

        self.best_params_ = {'n_estimators': self.n_estimators, 'u_sample_size': u_size}
        self._model = True
        logger.info('[BaggingPU] OOB coverage=%.1f%%  train PU PR-AUC=%.4f  val PR-AUC=%.4f',
                    100.0 * self.oob_coverage_, self.train_pu_pr_auc_, val_pr_auc)
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_feats = X[self.selected_features_]
        return np.stack([self._predict(m, X_feats) for m in self.estimators_], axis=1).mean(axis=1)
