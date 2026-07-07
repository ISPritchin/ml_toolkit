"""LambdaRankClassifier: прямая оптимизация MAP через LightGBM LambdaRank.

Все остальные пресеты оптимизируют surrogate-loss (logloss, focal) и надеются,
что PR-AUC вырастет. LambdaRank вычисляет градиенты напрямую из изменений
ранговой метрики при перестановке пар (positive, negative).

Почему это важно при < 1%:
  - Logloss присваивает штраф за каждую неверную вероятность;
    MAP штрафует только за неправильный ПОРЯДОК позитивов среди негативов.
  - При 100 позитивах из 10 000 — правильный порядок позитивов важнее
    точной калибровки вероятностей.

LightGBM lambdarank трактует весь датасет как один «запрос» (одну группу).
Это даёт корректный MAP по всей обучающей выборке без искусственного разбиения.

Выходные скоры не являются вероятностями (нет сигмоиды), но монотонно связаны
с вероятностью принадлежности к классу 1. Нормализуются в [0, 1] через ранги.

Требует LightGBM >= 4.0 (уже в зависимостях проекта).
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
    'objective': 'lambdarank',
    'metric': 'map',
    'label_gain': [0, 1],
    'num_leaves': 63,
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 800,
    'min_child_samples': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
}


class LambdaRankClassifier(BasePreset):
    """LightGBM LambdaRank: прямая оптимизация MAP для ранжирования позитивов.

    Parameters
    ----------
    base_params:
        Параметры LightGBM. None → дефолтные. Ключ 'objective' всегда 'lambdarank'.
    eval_at:
        Позиции для MAP@k (e.g. [10, 20, 50]). None → MAP по всем.
    early_stopping_rounds:
        Число раундов без улучшения MAP до остановки.
    truncation_level:
        lambdarank_truncation_level LightGBM: сколько топ-позиций участвует в
        градиентах пар. None → дефолт LightGBM (30). Поскольку весь датасет —
        одна query-группа, дефолт означает фокус градиента на топ-30 объектов;
        для полного охвата пар ставьте значение порядка размера train.
    random_seed:
        Зерно. Переопределяет 'random_state' в base_params.

    Атрибуты после fit::

        map_train_  — average precision (= MAP при бинарных метках) на train
        map_valid_  — average precision на валидационной выборке

    Пример::

        model = LambdaRankClassifier()
        model.fit(X_train, y_train, X_valid, y_valid)
        # predict_proba возвращает позицию скора в train-распределении, [0, 1]
        proba = model.predict_proba(X_test)

    """

    def __init__(
        self,
        base_params: dict[str, Any] | None = None,
        eval_at: list[int] | None = None,
        early_stopping_rounds: int = 80,
        truncation_level: int | None = None,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(params=None, n_optuna_trials=0)
        self.base_params = base_params
        self.eval_at = eval_at
        self.early_stopping_rounds = early_stopping_rounds
        self.truncation_level = truncation_level
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.map_train_: float = 0.0
        self.map_valid_: float = 0.0
        self._rank_ref_: np.ndarray | None = None

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> LambdaRankClassifier:
        import lightgbm as lgb

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values.astype(int)
        y_va = y_valid.values.astype(int)
        X_tr_feats = X_train[feats]
        X_va_feats = X_valid[feats]

        n_pos_tr = int(y_tr.sum())
        n_pos_va = int(y_va.sum())
        logger.info(
            '[LambdaRank] n_train=%d (pos=%d, %.2f%%)  n_val=%d (pos=%d, %.2f%%)',
            len(y_tr), n_pos_tr, 100 * n_pos_tr / len(y_tr),
            len(y_va), n_pos_va, 100 * n_pos_va / len(y_va),
        )

        params = {**(_DEFAULT_PARAMS.copy()), **(self.base_params or {})}
        params['objective'] = 'lambdarank'
        params['metric'] = 'map'
        params['label_gain'] = [0, 1]
        params['random_state'] = self.random_seed

        if self.eval_at:
            params['eval_at'] = self.eval_at
        if self.truncation_level is not None:
            params['lambdarank_truncation_level'] = self.truncation_level

        train_set = lgb.Dataset(
            X_tr_feats, label=y_tr,
            group=[len(y_tr)],
            categorical_feature=self.cat_features_ or 'auto',
        )
        valid_set = lgb.Dataset(
            X_va_feats, label=y_va,
            group=[len(y_va)],
            reference=train_set,
        )

        n_estimators = params.pop('n_estimators', 800)
        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ]

        self._model = lgb.train(
            params,
            train_set,
            num_boost_round=n_estimators,
            valid_sets=[valid_set],
            callbacks=callbacks,
        )

        raw_tr = self._model.predict(X_tr_feats)
        raw_va = self._model.predict(X_va_feats)

        # Референс rank-нормализации — train-скоры; predict_proba интерполирует
        # по нему, а не ранжирует внутри батча (скор не зависит от состава батча).
        self._rank_ref_ = fit_rank_reference(raw_tr)

        self.train_pred_ = rank_transform(raw_tr, self._rank_ref_)
        self.valid_pred_ = rank_transform(raw_va, self._rank_ref_)

        self.map_train_ = float(average_precision_score(y_tr, raw_tr))
        self.map_valid_ = float(average_precision_score(y_va, raw_va))
        logger.info(
            '[LambdaRank] MAP train=%.4f  val=%.4f  (rank-norm PR-AUC val=%.4f)',
            self.map_train_, self.map_valid_,
            float(average_precision_score(y_va, self.valid_pred_)),
        )

        self.best_params_ = {'n_estimators': self._model.num_trees(), **params}
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._model.predict(X[self.selected_features_])
        return rank_transform(raw, self._rank_ref_)
