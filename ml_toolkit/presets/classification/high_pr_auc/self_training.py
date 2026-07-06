"""SelfTrainingBooster: итеративный semi-supervised через pseudo-labeling.

Логика:
  1. Обучить модель на оригинальных train (labeled data).
  2. Предсказать на «негативах» train (которые могут содержать незамеченных позитивов).
  3. Негативы с высоким score → добавить как pseudo-positives с пониженным весом.
  4. Переобучить на расширенной выборке → повторить N раз.

Отличие от PULearningClassifier:
  PUL исправляет статистику вероятностей через c = P(s=1|y=1).
  SelfTraining итеративно пополняет обучающую выборку — работает как
  label propagation в пространстве признаков.

Когда имеет смысл:
  - Сегментация «Крупные» менялась со временем; часть клиентов в train
    ещё не попала в сегмент, хотя по поведению уже соответствует.
  - Time split: позитивы появляются постепенно, train content подрастает в val.

Управление порогом (threshold):
  None (default) → автоматически: 5-й персентиль val-позитивов.
    Это означает: «считай pseudo-positive тех, кто лучше 95% настоящих val-позитивов».
    Консервативно, но надёжно.
  float → абсолютный порог в пространстве вероятностей.
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
    'iterations': 600,
    'max_depth': 5,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'min_data_in_leaf': 10,
    'early_stopping_rounds': 80,
    'loss_function': 'Logloss',
    'eval_metric': 'PRAUC',
    'random_seed': 42,
    'verbose': 0,
}


class SelfTrainingBooster(BasePreset):
    """Итеративный pseudo-labeling поверх CatBoost.

    Parameters
    ----------
    n_rounds:
        Число раундов self-training (рекомендуется 2–5).
    threshold:
        Порог score для добавления pseudo-positives.
        None → 5-й персентиль score на val-позитивах.
        float → абсолютный порог (например, 0.6).
    pseudo_weight:
        Вес pseudo-labeled примеров (0 < pseudo_weight ≤ 1).
        Реальные позитивы/негативы имеют вес 1.0.
    max_pseudo_ratio:
        Максимальное число pseudo-positives как кратное числу реальных позитивов.
        Защита от лавинного добавления «мусора» в поздних раундах.
    base_params:
        Параметры CatBoost. None → дефолтные.
    random_seed:
        Зерно CatBoost.

    Атрибуты после fit::

        round_scores_     — val PR-AUC после каждого раунда
        pseudo_added_     — количество pseudo-positives в каждом раунде
        threshold_used_   — реально использованный порог

    Пример::

        model = SelfTrainingBooster(n_rounds=3, pseudo_weight=0.3)
        model.fit(X_train, y_train, X_valid, y_valid)
    """

    def __init__(
        self,
        n_rounds: int = 3,
        threshold: float | None = None,
        pseudo_weight: float = 0.3,
        max_pseudo_ratio: float = 2.0,
        base_params: dict[str, Any] | None = None,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(params=None, n_optuna_trials=0)
        self.n_rounds = n_rounds
        self.threshold = threshold
        self.pseudo_weight = pseudo_weight
        self.max_pseudo_ratio = max_pseudo_ratio
        self.base_params = base_params
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.round_scores_: list[float] = []
        self.pseudo_added_: list[int] = []
        self.threshold_used_: float = 0.0

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def _train_model(
        self,
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
        w_tr: np.ndarray,
        X_va: pd.DataFrame,
        y_va: np.ndarray,
    ) -> Any:
        from catboost import CatBoostClassifier, Pool

        params = {**(self.base_params or _DEFAULT_PARAMS), 'random_seed': self.random_seed}
        model = CatBoostClassifier(**params)
        tr_pool = Pool(X_tr, y_tr, cat_features=self.cat_features_, weight=w_tr)
        va_pool = Pool(X_va, y_va, cat_features=self.cat_features_)
        model.fit(tr_pool, eval_set=va_pool, verbose=False)
        return model

    def _predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool

        pool = Pool(X, cat_features=self.cat_features_)
        return model.predict_proba(pool)[:, 1]

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> 'SelfTrainingBooster':
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values.copy()
        y_va = y_valid.values
        X_tr_feats = X_train[feats].reset_index(drop=True)
        X_va_feats = X_valid[feats]

        n_pos_orig = int(y_tr.sum())
        max_pseudo = int(self.max_pseudo_ratio * n_pos_orig)
        logger.info(
            '[SelfTraining] n_rounds=%d  n_pos_orig=%d  max_pseudo=%d  pseudo_weight=%.2f',
            self.n_rounds, n_pos_orig, max_pseudo, self.pseudo_weight,
        )

        # Исходные веса: 1.0 для всех
        w_tr = np.ones(len(y_tr), dtype=np.float64)

        # Маска уже добавленных pseudo-positives (изначально нет)
        is_pseudo = np.zeros(len(y_tr), dtype=bool)

        self.round_scores_ = []
        self.pseudo_added_ = []

        # ── Раунд 0: обучение на оригинальных данных ─────────────────────────
        logger.info('[SelfTraining] Раунд 0 / %d (baseline)', self.n_rounds)
        model = self._train_model(X_tr_feats, y_tr, w_tr, X_va_feats, y_va)
        va_proba = self._predict(model, X_va_feats)
        pr_auc = float(average_precision_score(y_va, va_proba))
        self.round_scores_.append(pr_auc)
        logger.info('[SelfTraining] Раунд 0  val PR-AUC=%.4f', pr_auc)

        # Определяем порог: auto → 5-й персентиль val-позитивов
        if self.threshold is None:
            pos_val_scores = va_proba[y_va == 1]
            if len(pos_val_scores) == 0:
                self.threshold_used_ = 0.5
            else:
                self.threshold_used_ = float(np.percentile(pos_val_scores, 5))
        else:
            self.threshold_used_ = float(self.threshold)

        logger.info('[SelfTraining] Порог pseudo-positive: %.4f', self.threshold_used_)

        # ── Раунды 1..n ──────────────────────────────────────────────────────
        total_pseudo = 0
        for round_idx in range(1, self.n_rounds + 1):
            # Предсказываем на оригинальных негативах (не pseudo)
            neg_orig_mask = (y_tr == 0) & ~is_pseudo
            if not neg_orig_mask.any():
                logger.info('[SelfTraining] Нет оригинальных негативов — остановка')
                break

            neg_orig_idx = np.where(neg_orig_mask)[0]
            tr_proba_neg = self._predict(model, X_tr_feats.iloc[neg_orig_idx])

            # Кандидаты в pseudo-positive
            above_thresh = tr_proba_neg > self.threshold_used_
            n_candidates = int(above_thresh.sum())
            n_to_add = min(n_candidates, max(0, max_pseudo - total_pseudo))

            if n_to_add == 0:
                logger.info(
                    '[SelfTraining] Раунд %d: 0 кандидатов выше порога (%.4f) — остановка',
                    round_idx, self.threshold_used_,
                )
                self.pseudo_added_.append(0)
                break

            # Берём топ-n_to_add по score (наиболее уверенные)
            cand_global_idx = neg_orig_idx[above_thresh]
            cand_scores = tr_proba_neg[above_thresh]
            top_local = np.argsort(cand_scores)[-n_to_add:]
            pseudo_global_idx = cand_global_idx[top_local]

            y_tr[pseudo_global_idx] = 1
            w_tr[pseudo_global_idx] = self.pseudo_weight
            is_pseudo[pseudo_global_idx] = True
            total_pseudo += n_to_add

            logger.info(
                '[SelfTraining] Раунд %d: добавлено %d pseudo-positives (total=%d)',
                round_idx, n_to_add, total_pseudo,
            )
            self.pseudo_added_.append(n_to_add)

            model = self._train_model(X_tr_feats, y_tr, w_tr, X_va_feats, y_va)
            va_proba = self._predict(model, X_va_feats)
            pr_auc = float(average_precision_score(y_va, va_proba))
            self.round_scores_.append(pr_auc)
            logger.info('[SelfTraining] Раунд %d  val PR-AUC=%.4f', round_idx, pr_auc)

        self._model = model
        self.valid_pred_ = self._predict(model, X_va_feats)
        from catboost import Pool
        self.train_pred_ = model.predict_proba(
            Pool(X_tr_feats, y_tr, cat_features=self.cat_features_, weight=w_tr)
        )[:, 1]

        self.best_params_ = {
            'n_rounds': self.n_rounds,
            'threshold': self.threshold_used_,
            'pseudo_weight': self.pseudo_weight,
            'total_pseudo_added': total_pseudo,
        }
        logger.info(
            '[SelfTraining] Итог: val PR-AUC=%.4f (baseline=%.4f, delta=%.4f)  '
            'pseudo_added=%s',
            self.round_scores_[-1], self.round_scores_[0],
            self.round_scores_[-1] - self.round_scores_[0],
            self.pseudo_added_,
        )
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict(self._model, X[self.selected_features_])
