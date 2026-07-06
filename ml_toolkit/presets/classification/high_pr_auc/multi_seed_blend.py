"""MultiSeedBlend: одна конфигурация CatBoost, K случайных зёрен, rank-avg blend.

Самое дешёвое снижение дисперсии из всех пресетов этого пакета — никакой
diversity кроме random_seed (в отличие от EasyEnsembleClassifier — там ещё и
подвыборка негативов, или SubsampleStacking/HeterogeneousStacking — там ещё и
разные конфиги/алгоритмы). Не решает никакой конкретной проблемы дисбаланса
или шума — просто усредняет K независимых обучений одной и той же модели,
чтобы убрать шум инициализации/порядка данных перед тем, как вообще сравнивать
пресеты между собой (нестабильно сравнивать A против B, если сам A шумит
сильнее, чем разница между A и B).

Rank-avg (не среднее сырых вероятностей) — та же логика и та же утилита
(fit_rank_reference/rank_transform), что и в EasyEnsembleClassifier: масштаб
вероятностей между независимо обученными моделями может слегка плавать,
усреднение рангов устойчивее прямого усреднения вероятностей.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.models._utils import fit_rank_reference, rank_transform
from ml_toolkit.presets.classification._base import BasePreset

logger = logging.getLogger(__name__)

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


class MultiSeedBlend(BasePreset):
    """CatBoost с одним конфигом, K сидов, rank-avg blend.

    Parameters
    ----------
    n_seeds:
        Число независимых обучений (рекомендуется 5-10; отдача убывает
        быстро после ~10 — дисперсия одного сида делится на sqrt(n_seeds)).
    base_params:
        Параметры CatBoost (без random_seed — задаётся автоматически на
        каждый прогон). None → дефолтные.
    random_seed:
        Базовое зерно; прогон i получает random_seed + i.

    Атрибуты после fit::

        seed_scores_    — val PR-AUC каждого отдельного сида
        blend_score_    — val PR-AUC итогового rank-avg blend'а

    Пример::

        model = MultiSeedBlend(n_seeds=7)
        model.fit(X_train, y_train, X_valid, y_valid)
        print(f"single mean={np.mean(model.seed_scores_):.4f}  blend={model.blend_score_:.4f}")
    """

    def __init__(
        self,
        n_seeds: int = 7,
        base_params: dict[str, Any] | None = None,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(params=None, n_optuna_trials=0)
        if n_seeds < 2:
            raise ValueError(f'n_seeds должен быть >= 2, получено {n_seeds}')
        self.n_seeds = n_seeds
        self.base_params = base_params
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.models_: list[Any] = []
        self.seed_scores_: list[float] = []
        self.blend_score_: float = 0.0
        self._rank_refs_: list[np.ndarray] = []

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> 'MultiSeedBlend':
        from catboost import CatBoostClassifier, Pool

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

        self.models_ = []
        self.seed_scores_ = []
        va_raw_scores: list[np.ndarray] = []

        for i in range(self.n_seeds):
            seed = self.random_seed + i
            params = {**(self.base_params or _DEFAULT_PARAMS), 'random_seed': seed}
            tr_pool = Pool(X_tr_feats, y_tr, cat_features=self.cat_features_)
            va_pool = Pool(X_va_feats, y_va, cat_features=self.cat_features_)
            m = CatBoostClassifier(**params)
            m.fit(tr_pool, eval_set=va_pool, verbose=False)

            va_score = m.predict_proba(va_pool)[:, 1]
            ap = float(average_precision_score(y_va, va_score))
            self.models_.append(m)
            self.seed_scores_.append(ap)
            va_raw_scores.append(va_score)
            logger.info('[MultiSeedBlend] seed %d/%d (seed=%d)  val PR-AUC=%.4f',
                        i + 1, self.n_seeds, seed, ap)

        tr_raw_scores = [
            m.predict_proba(Pool(X_tr_feats, cat_features=self.cat_features_))[:, 1] for m in self.models_
        ]
        self._rank_refs_ = [fit_rank_reference(s) for s in tr_raw_scores]

        va_blend = np.stack(
            [rank_transform(s, ref) for s, ref in zip(va_raw_scores, self._rank_refs_)], axis=1,
        ).mean(axis=1)
        self.blend_score_ = float(average_precision_score(y_va, va_blend))
        logger.info('[MultiSeedBlend] single seed mean PR-AUC=%.4f (std=%.4f)  blend PR-AUC=%.4f',
                    float(np.mean(self.seed_scores_)), float(np.std(self.seed_scores_)), self.blend_score_)

        self.valid_pred_ = va_blend
        self.train_pred_ = np.stack(
            [rank_transform(s, ref) for s, ref in zip(tr_raw_scores, self._rank_refs_)], axis=1,
        ).mean(axis=1)
        self.best_params_ = {'n_seeds': self.n_seeds}
        self._model = True
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool

        X_feats = X[self.selected_features_]
        pool = Pool(X_feats, cat_features=self.cat_features_)
        raw_scores = [m.predict_proba(pool)[:, 1] for m in self.models_]
        return np.stack(
            [rank_transform(s, ref) for s, ref in zip(raw_scores, self._rank_refs_)], axis=1,
        ).mean(axis=1)
