"""SnapshotEnsembleClassifier: ансамбль из чекпоинтов одного бустинга.

Обучается один CatBoost на train (с early stopping на val), после чего берутся
«снэпшоты» — предсказания по префиксам деревьев на долях snapshot_fracs от
итогового числа деревьев. CatBoost умеет резать предсказания по диапазону
деревьев через ntree_end в predict_proba без переобучения, так что снэпшоты
не стоят дополнительных fit().

Снэпшоты усредняются простым средним: в отличие от разнородных ансамблей (разные
seed'ы / подпространства признаков, см. FeatureBaggingEnsemble, EasyEnsembleClassifier),
все снэпшоты — префиксы одного и того же бустинга, поэтому их вероятности уже
в одном масштабе, и ранговая нормализация не нужна.

Когда: бюджет на обучение — одна модель, а разнообразие всё же хочется получить
почти бесплатно.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.models._utils import fit_calibrator
from ml_toolkit.presets.classification._base import BasePreset

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
    'verbose': 0,
}


class SnapshotEnsembleClassifier(BasePreset):
    """Ансамбль из чекпоинтов (по числу деревьев) одного бустинга CatBoost.

    Parameters
    ----------
    snapshot_fracs:
        Доли от итогового числа деревьев модели (``tree_count_`` после fit,
        учитывает early stopping), на которых берётся срез предсказаний.
        По умолчанию [0.4, 0.6, 0.8, 1.0]. Совпадающие после округления доли
        схлопываются в один снэпшот (не задваиваются в среднем).
    base_params:
        Параметры единственного обучаемого CatBoost. None → дефолтные.
    calibrate:
        Применять ли изотоническую калибровку к итоговым (усреднённым по
        снэпшотам) вероятностям.
    random_seed:
        Зерно CatBoost.

    Атрибуты после fit::

        tree_counts_      — фактическое число деревьев на каждом снэпшоте
        snapshot_scores_  — val PR-AUC каждого снэпшота отдельно (диагностика)
        ensemble_score_   — val PR-AUC усреднённого по снэпшотам ансамбля (до калибровки)

    Пример::

        model = SnapshotEnsembleClassifier(snapshot_fracs=[0.5, 0.75, 1.0])
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=[...])
        proba = model.predict_proba(X_test)
    """

    def __init__(
        self,
        snapshot_fracs: list[float] | None = None,
        base_params: dict[str, Any] | None = None,
        calibrate: bool = True,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        fracs = snapshot_fracs or [0.4, 0.6, 0.8, 1.0]
        if not fracs:
            raise ValueError("snapshot_fracs не может быть пустым.")
        if any(not 0.0 < f <= 1.0 for f in fracs):
            raise ValueError(f"Все snapshot_fracs должны быть в (0, 1], получено {fracs}")
        super().__init__(params=None, n_optuna_trials=0)
        self.snapshot_fracs = fracs
        self.base_params = base_params
        self.calibrate = calibrate
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.tree_counts_: list[int] = []
        self.snapshot_scores_: list[float] = []
        self.ensemble_score_: float = 0.0

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> 'SnapshotEnsembleClassifier':
        from catboost import CatBoostClassifier, Pool

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values

        tr_pool = Pool(X_train[feats], y_tr, cat_features=self.cat_features_)
        va_pool = Pool(X_valid[feats], y_va, cat_features=self.cat_features_)

        params = {**(self.base_params or _DEFAULT_BASE_PARAMS), 'random_seed': self.random_seed}
        self._model = CatBoostClassifier(**params)
        self._model.fit(tr_pool, eval_set=va_pool, verbose=False)

        total_trees = self._model.tree_count_
        self.tree_counts_ = sorted({max(1, int(round(f * total_trees))) for f in self.snapshot_fracs})
        logger.info('[Snapshot] total_trees=%d  snapshot_fracs=%s  tree_counts=%s',
                    total_trees, self.snapshot_fracs, self.tree_counts_)

        va_snaps = [
            self._model.predict_proba(va_pool, ntree_start=0, ntree_end=nt)[:, 1]
            for nt in self.tree_counts_
        ]
        self.snapshot_scores_ = [float(average_precision_score(y_va, s)) for s in va_snaps]
        for nt, ap in zip(self.tree_counts_, self.snapshot_scores_):
            logger.info('[Snapshot] ntree_end=%d  val PR-AUC=%.4f', nt, ap)

        raw_va = np.mean(va_snaps, axis=0)
        self.ensemble_score_ = float(average_precision_score(y_va, raw_va))
        logger.info('[Snapshot] ensemble val PR-AUC=%.4f  (mean single snapshot=%.4f)',
                    self.ensemble_score_, float(np.mean(self.snapshot_scores_)))

        tr_snaps = [
            self._model.predict_proba(tr_pool, ntree_start=0, ntree_end=nt)[:, 1]
            for nt in self.tree_counts_
        ]
        raw_tr = np.mean(tr_snaps, axis=0)

        if self.calibrate:
            self.calibrator_ = fit_calibrator(raw_va, y_va)
            self.valid_pred_ = self.calibrator_.predict(raw_va)
            self.train_pred_ = self.calibrator_.predict(raw_tr)
        else:
            self.valid_pred_ = raw_va
            self.train_pred_ = raw_tr

        self.best_params_ = {'snapshot_fracs': self.snapshot_fracs, 'tree_counts': self.tree_counts_}
        return self

    def _snapshot_proba(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        pool = Pool(X[self.selected_features_], cat_features=self.cat_features_)
        snaps = [
            self._model.predict_proba(pool, ntree_start=0, ntree_end=nt)[:, 1]
            for nt in self.tree_counts_
        ]
        return np.mean(snaps, axis=0)

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._snapshot_proba(X)
        if self.calibrate and self.calibrator_ is not None:
            return self.calibrator_.predict(raw)
        return raw
