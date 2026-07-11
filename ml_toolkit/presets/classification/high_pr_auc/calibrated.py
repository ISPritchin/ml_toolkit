"""CalibratedWrapper: пост-хок калибровка вероятностей любого пресета.

При undersampling-е (1:k → 1:1 в train) вероятности систематически завышены:
модель «думает», что позитивов 50%, а их на самом деле < 1%. Калибровка
возвращает вероятности в правильный диапазон.

PR-AUC напрямую зависит от качества вероятностей, в отличие от ROC-AUC:
если после CalibratedWrapper PR-AUC заметно вырос — bottleneck был в калибровке,
а не в дискриминационной способности модели.

Два метода:
  'isotonic' — IsotonicRegression (монотонная, нелинейная; требует ≥ 50 val-позитивов)
  'platt'    — LogisticRegression (сигмоида; устойчива при малом числе позитивов)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.models._base import XInput, YInput
from ml_toolkit.models._utils import fit_calibrator
from ml_toolkit.presets.classification._base import BasePreset

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class CalibratedWrapper(BasePreset):
    """Обёртка над любым BasePreset с пост-хок калибровкой на валидации.

    Parameters
    ----------
    base_preset:
        Любой экземпляр BasePreset (EasyEnsembleClassifier, TwoStageCascade и т.д.).
        Должен быть необученным — fit() будет вызван внутри CalibratedWrapper.fit().
    method:
        'isotonic' (по умолчанию) или 'platt'.

    Атрибуты после fit::

        base_             — обученный base_preset
        calibrator_       — обученный калибратор (IsotonicRegression или LogisticRegression)
        raw_pr_auc_       — PR-AUC base_preset до калибровки
        calibrated_pr_auc_ — PR-AUC после калибровки

    Пример::

        from ml_toolkit.presets.classification.high_pr_auc.easy_ensemble import EasyEnsembleClassifier
        from ml_toolkit.presets.classification.high_pr_auc.calibrated import CalibratedWrapper

        model = CalibratedWrapper(EasyEnsembleClassifier(n_estimators=10), method='isotonic')
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=[...])
        proba = model.predict_proba(X_test)

    """

    def __init__(
        self,
        base_preset: BasePreset,
        method: str = 'isotonic',
    ) -> None:
        super().__init__(params=None, n_optuna_trials=0)
        if method not in ('isotonic', 'platt'):
            raise ValueError(f"method должен быть 'isotonic' или 'platt', получено {method!r}")
        self.base_preset = base_preset
        self.method = method

        self.base_: BasePreset | None = None
        self.raw_pr_auc_: float = 0.0
        self.calibrated_pr_auc_: float = 0.0

    def _fit_platt(self, raw_va: np.ndarray, y_va: np.ndarray) -> LogisticRegression:
        from sklearn.linear_model import LogisticRegression

        cal = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        cal.fit(raw_va.reshape(-1, 1), y_va)
        return cal

    def _apply_calibrator(self, raw: np.ndarray) -> np.ndarray:
        if self.method == 'isotonic':
            return self.calibrator_.predict(raw)
        # platt
        return self.calibrator_.predict_proba(raw.reshape(-1, 1))[:, 1]

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: XInput,
        y_train: YInput,
        X_valid: XInput,
        y_valid: YInput,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> CalibratedWrapper:
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        y_va = y_valid.values

        logger.info('[Calibrated] Обучаем base_preset: %s', type(self.base_preset).__name__)
        self.base_preset.fit(
            X_train, y_train, X_valid, y_valid,
            selected_features=selected_features,
            cat_features=cat_features,
        )
        self.base_ = self.base_preset

        # Берём сырые вероятности с валидации (уже вычислены в base_.valid_pred_)
        raw_va = (
            self.base_.valid_pred_
            if self.base_.valid_pred_ is not None
            else self.base_.predict_proba(X_valid)
        )
        self.raw_pr_auc_ = float(average_precision_score(y_va, raw_va))

        n_pos = int(y_va.sum())
        if self.method == 'isotonic' and n_pos < 20:
            logger.warning(
                '[Calibrated] Метод isotonic требует ≥ 20 val-позитивов, найдено %d. '
                'Рассмотрите method="platt".',
                n_pos,
            )

        if self.method == 'isotonic':
            self.calibrator_ = fit_calibrator(raw_va, y_va)
        else:
            self.calibrator_ = self._fit_platt(raw_va, y_va)

        calibrated_va = self._apply_calibrator(raw_va)
        self.calibrated_pr_auc_ = float(average_precision_score(y_va, calibrated_va))

        logger.info(
            '[Calibrated] method=%s  raw PR-AUC=%.4f  calibrated PR-AUC=%.4f  Δ=%.4f',
            self.method, self.raw_pr_auc_, self.calibrated_pr_auc_,
            self.calibrated_pr_auc_ - self.raw_pr_auc_,
        )

        self.valid_pred_ = calibrated_va
        if self.base_.train_pred_ is not None:
            self.train_pred_ = self._apply_calibrator(self.base_.train_pred_)

        # Проксируем метаданные от base
        self.selected_features_ = self.base_.selected_features_
        self.cat_features_ = self.base_.cat_features_
        self.best_params_ = {
            'base': type(self.base_).__name__,
            'method': self.method,
            **(self.base_.best_params_ or {}),
        }
        self._model = True
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.base_.predict_proba(X)
        return self._apply_calibrator(raw)
