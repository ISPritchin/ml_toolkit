"""ConformalRegressionWrapper: split-conformal интервалы поверх любого регрессора с гарантией покрытия.

Оборачивает любую модель с интерфейсом BaseModel (fit(X_train, y_train, X_valid,
y_valid, ...) / predict(X)) — CatBoostRegressor из ml_toolkit.models, любой
пресет этого пакета и т.д. — и добавляет предиктивные интервалы с конечновыборочной
гарантией покрытия (1 - alpha) БЕЗ предположений о распределении остатков
(Vovk et al., split conformal prediction), в отличие от NGBoostPreset, где
интервал — следствие конкретной параметрической формы распределения.

Калибровка использует X_valid/y_valid как calibration-выборку — тот же приём,
что и CalibratedWrapper (ml_toolkit/presets/classification/high_pr_auc/calibrated.py):
X_valid уже не участвует в обучении базовой модели напрямую (используется только
для early stopping/Optuna внутри base_regressor.fit()), так что повторное
использование как calibration set — стандартное практическое допущение, а не
утечка целевой переменной в веса модели.

Два способа посчитать nonconformity score::

    score='absolute'    — s_i = |y_i - pred_i|; интервал одинаковой ширины
                           для всех x (не учитывает гетероскедастичность).
    score='normalized'  — s_i = |y_i - pred_i| / sigma_hat(x_i), где sigma_hat —
                           вспомогательная модель, предсказывающая |остаток|
                           базовой модели по X_train (learned difficulty);
                           интервал сужается там, где модель увереннее.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ml_toolkit.models._base import BaseModel, XInput, YInput, _to_pandas
from ml_toolkit.presets.regression._base import BasePreset

if TYPE_CHECKING:
    from catboost import CatBoostRegressor

_SIGMA_FLOOR = 1e-3


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Конечновыборочная поправка split-conformal: квантиль уровня.

    ceil((n+1)*(1-alpha))/n (а не «наивный» (1-alpha)-квантиль) — то, что
    реально гарантирует финитно-выборочное покрытие (Vovk et al., Lei et al. 2018).
    """
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method='higher'))


class ConformalRegressionWrapper(BasePreset):
    """Split-conformal обёртка с гарантией покрытия поверх любого регрессора.

    Parameters
    ----------
    base_regressor:
        Необученный объект с интерфейсом BaseModel (CatBoostRegressor из
        ml_toolkit.models, любой пресет этого пакета и т.п.). fit() будет
        вызван внутри ConformalRegressionWrapper.fit().
    alpha:
        Целевой уровень значимости — интервал покрывает истинное значение с
        вероятностью >= (1 - alpha) (при обменности калибровочной выборки).
    score:
        'absolute' (по умолчанию) или 'normalized' (см. докстринг модуля).

    Атрибуты после fit::

        base_        — обученный base_regressor
        q_hat_       — калиброванный порог nonconformity score (при self.alpha)
        sigma_model_ — вспомогательная модель |остаток| (только score='normalized')

    Пример::

        from ml_toolkit.models import CatBoostRegressor

        model = ConformalRegressionWrapper(CatBoostRegressor(n_optuna_trials=30), alpha=0.1)
        model.fit(X_train, y_train, X_valid, y_valid)
        pred = model.predict(X_test)
        lower, upper = model.predict_interval(X_test)

    """

    def __init__(
        self,
        base_regressor: BaseModel,
        alpha: float = 0.1,
        score: str = 'absolute',
    ) -> None:
        super().__init__(params=None, n_optuna_trials=0)
        if score not in ('absolute', 'normalized'):
            raise ValueError(f"score должен быть 'absolute' или 'normalized', получено {score!r}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f'alpha должен быть в (0, 1), получено {alpha}')
        self.base_regressor = base_regressor
        self.alpha = alpha
        self.score = score

        self.base_: Any = None
        self.q_hat_: float = 0.0
        self.sigma_model_: Any = None
        self._calib_scores_: np.ndarray | None = None
        self._calib_sigma_: np.ndarray | None = None

    def _fit_sigma_model(
        self, X_train: pd.DataFrame, feats: list[str], abs_resid_train: np.ndarray,
    ) -> CatBoostRegressor:
        from catboost import CatBoostRegressor, Pool

        model = CatBoostRegressor(iterations=300, depth=4, loss_function='MAE', verbose=0, random_seed=42)
        model.fit(Pool(X_train[feats], abs_resid_train, cat_features=self.cat_features_))
        return model

    def _sigma(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        pred = self.sigma_model_.predict(Pool(X[self.selected_features_], cat_features=self.cat_features_))
        return np.maximum(pred, _SIGMA_FLOOR)

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: XInput,
        y_train: YInput,
        X_valid: XInput,
        y_valid: YInput,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> ConformalRegressionWrapper:
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        y_va = y_valid.values

        self.base_regressor.fit(
            X_train, y_train, X_valid, y_valid,
            selected_features=selected_features, cat_features=cat_features,
        )
        self.base_ = self.base_regressor
        self.selected_features_ = self.base_.selected_features_
        self.cat_features_ = self.base_.cat_features_

        raw_va = (
            self.base_.valid_pred_
            if self.base_.valid_pred_ is not None
            else self.base_.predict(X_valid)
        )
        abs_resid_va = np.abs(y_va - raw_va)

        if self.score == 'normalized':
            raw_tr = (
                self.base_.train_pred_
                if self.base_.train_pred_ is not None
                else self.base_.predict(X_train)
            )
            abs_resid_tr = np.abs(y_train.values - raw_tr)
            self.sigma_model_ = self._fit_sigma_model(X_train, self.selected_features_, abs_resid_tr)
            self._calib_sigma_ = self._sigma(X_valid)
            scores = abs_resid_va / self._calib_sigma_
        else:
            scores = abs_resid_va

        self._calib_scores_ = scores
        self.q_hat_ = _conformal_quantile(scores, self.alpha)

        self.best_params_ = {
            'base': type(self.base_).__name__, 'score': self.score,
            'alpha': self.alpha, 'q_hat': self.q_hat_,
        }
        self._model = True
        self.valid_pred_ = raw_va
        self.train_pred_ = self.base_.train_pred_
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict_interval(self, X: XInput, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Предиктивный интервал уровня (1 - alpha). alpha=None → self.alpha конструктора.

        (переиспользует калибровочные scores без повторного обучения — порог
        q_hat пересчитывается на лету для произвольного alpha дёшево).
        """
        self._check_fitted()
        a = self.alpha if alpha is None else alpha
        if not 0.0 < a < 1.0:
            raise ValueError(f'alpha должен быть в (0, 1), получено {a}')
        q_hat = self.q_hat_ if a == self.alpha else _conformal_quantile(self._calib_scores_, a)

        Xp = _to_pandas(X)
        raw = self.base_.predict(Xp)
        if self.score == 'normalized':
            width = q_hat * self._sigma(Xp)
        else:
            width = q_hat
        return raw - width, raw + width

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_.predict(X)
