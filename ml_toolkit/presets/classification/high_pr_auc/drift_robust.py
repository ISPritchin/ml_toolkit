"""DriftRobustClassifier: AdversarialDriftFilter + переобучение + PSI-отчёт.

Связывает уже существующий ml_toolkit.feature_selection.AdversarialDriftFilter
(итеративно удаляет признаки, по которым train и valid легко различимы
классификатором) с обучением модели: признаки с drift просто не видны модели
вообще, а не down-weight'ятся или иначе компенсируются (для этого — см.
AdversarialValidationWeighting/045, которая переweight'ивает СТРОКИ, а не
выбрасывает КОЛОНКИ — полезно, когда дрейфующие признаки слишком ценные, чтобы
их терять).

compute_psi даёт быстрый (без обучения adversarial-модели) параллельный
отчёт — используется здесь просто как диагностика, PSI не участвует в отборе
признаков (это делает AdversarialDriftFilter через AUC, более чувствительный
к многомерному/нелинейному смещению, чем одномерный PSI).

base_preset — любой уже сконструированный (не обученный) объект с интерфейсом
BasePreset (fit(X_train, y_train, X_valid, y_valid, selected_features=,
cat_features=) + predict_proba(X)); None → внутренний обычный CatBoost.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.feature_selection.drift_filter import AdversarialDriftFilter, compute_psi
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


class DriftRobustClassifier(BasePreset):
    """Обучение только на признаках, устойчивых к train/valid drift.

    Parameters
    ----------
    target_auc:
        Целевой adversarial AUC для AdversarialDriftFilter (0.55 — мягкий
        порог, 0.50 — максимальный).
    base_preset:
        Необученный объект с интерфейсом BasePreset для финального обучения.
        None → внутренний CatBoost с base_params.
    base_params:
        Параметры внутреннего CatBoost (игнорируется, если задан base_preset).
    random_seed:
        Зерно AdversarialDriftFilter и внутреннего CatBoost.

    Атрибуты после fit::

        removed_features_        — признаки, удалённые из-за drift
        adversarial_auc_history_ — adversarial AUC на каждой итерации фильтра
        psi_report_              — DataFrame compute_psi по ВСЕМ исходным признакам

    Пример::

        model = DriftRobustClassifier(target_auc=0.55)
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
        print(model.removed_features_)
    """

    def __init__(
        self,
        target_auc: float = 0.55,
        base_preset: Any = None,
        base_params: dict[str, Any] | None = None,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(params=None, n_optuna_trials=0)
        self.target_auc = target_auc
        self.base_preset = base_preset
        self.base_params = base_params
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.removed_features_: list[str] = []
        self.adversarial_auc_history_: list[float] = []
        self.psi_report_: pd.DataFrame | None = None
        self._drift_filter: AdversarialDriftFilter | None = None

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> 'DriftRobustClassifier':
        from catboost import CatBoostClassifier, Pool

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.cat_features_ = cat_features or self.cat_features

        self.psi_report_ = compute_psi(X_train[feats], X_valid[feats])
        high_drift = self.psi_report_[self.psi_report_['drift_level'] == 'high']
        if len(high_drift) > 0:
            logger.info('[DriftRobust] PSI: %d признаков с высоким смещением (PSI>0.25): %s',
                        len(high_drift), high_drift['feature'].tolist())

        self._drift_filter = AdversarialDriftFilter(
            target_auc=self.target_auc, cat_features=self.cat_features_, random_seed=self.random_seed,
        )
        self._drift_filter.fit(X_train[feats], X_valid[feats])
        self.selected_features_ = self._drift_filter.selected_features_
        self.removed_features_ = self._drift_filter.removed_features_
        self.adversarial_auc_history_ = self._drift_filter.adversarial_auc_history_

        logger.info('[DriftRobust] Удалено %d/%d признаков из-за drift: %s',
                    len(self.removed_features_), len(feats), self.removed_features_)

        clean_cat_features = [c for c in self.cat_features_ if c in self.selected_features_]

        if self.base_preset is not None:
            self.base_preset.fit(
                X_train, y_train, X_valid, y_valid,
                selected_features=self.selected_features_, cat_features=clean_cat_features,
            )
            self._model = self.base_preset
            self.train_pred_ = self.base_preset.train_pred_
            self.valid_pred_ = self.base_preset.valid_pred_
            self.best_params_ = getattr(self.base_preset, 'best_params_', {})
        else:
            params = {**(self.base_params or _DEFAULT_PARAMS), 'random_seed': self.random_seed}
            tr_pool = Pool(X_train[self.selected_features_], y_train.values, cat_features=clean_cat_features)
            va_pool = Pool(X_valid[self.selected_features_], y_valid.values, cat_features=clean_cat_features)
            self._model = CatBoostClassifier(**params)
            self._model.fit(tr_pool, eval_set=va_pool, verbose=False)
            self.best_params_ = params
            self.train_pred_ = self._model.predict_proba(tr_pool)[:, 1]
            self.valid_pred_ = self._model.predict_proba(va_pool)[:, 1]

        val_pr_auc = float(average_precision_score(y_valid.values, self.valid_pred_))
        logger.info('[DriftRobust] val PR-AUC (на drift-clean признаках)=%.4f', val_pr_auc)
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        if self.base_preset is not None:
            return self.base_preset.predict_proba(X)
        from catboost import Pool
        clean_cat_features = [c for c in self.cat_features_ if c in self.selected_features_]
        pool = Pool(X[self.selected_features_], cat_features=clean_cat_features)
        return self._model.predict_proba(pool)[:, 1]
