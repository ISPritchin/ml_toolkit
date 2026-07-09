"""AsymmetricCostRegressor: разная цена пере- и недо-прогноза (newsvendor-подобные задачи).

Запасы/лимиты/квоты: недопрогноз (дефицит) и перепрогноз (излишек) обычно стоят
бизнесу по-разному — CatBoost, обученный на симметричном MSE/MAE, оптимизирует
не ту величину, которой в итоге измеряют качество прогноза.

over_cost/under_cost — фиксированные бизнес-параметры, не тюнящиеся Optuna (как
и в RelativeErrorRegressor: их «оптимальное» значение по валидации не имеет
смысла — это вход, а не гиперпараметр). Optuna (если включена) тюнит только
архитектуру CatBoost, а trial отбирается по самой линейной asymmetric-cost
метрике (не по MAE/pinball внутреннего лосса) — независимо от выбранного
`loss`, чтобы обе ветки сравнивались по одному и тому же критерию, которым
реально измеряется бизнес-качество.

Два режима::

    loss='pinball'   — встроенный CatBoost `Quantile:alpha=q`, q = under_cost /
                        (over_cost + under_cost) (нативная C++ реализация).
    loss='asym_mse'   — кастомный Python AsymmetricMSELoss (квадратичный,
                        сильнее давит на большие отклонения в дорогую сторону).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ml_toolkit.presets.regression._custom_loss_base import (
    _CustomLossRegressorBase,
    _LossSpec,
)
from ml_toolkit.presets.regression._losses import AsymmetricMSELoss


def _asymmetric_cost(y_true: np.ndarray, y_pred: np.ndarray, over_cost: float, under_cost: float) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    over = np.maximum(y_pred - y_true, 0.0)   # перепрогноз: pred > y
    under = np.maximum(y_true - y_pred, 0.0)  # недопрогноз: pred < y
    return float(np.mean(over_cost * over + under_cost * under))


class AsymmetricCostRegressor(_CustomLossRegressorBase):
    """CatBoost с асимметричной ценой ошибки (pinball или asymmetric MSE).

    Parameters
    ----------
    loss:
        'pinball' (по умолчанию, встроенный CatBoost `Quantile:alpha=q`) или
        'asym_mse' (кастомный AsymmetricMSELoss).
    over_cost / under_cost:
        Цена перепрогноза (pred > y) / недопрогноза (pred < y). Фиксированные
        бизнес-параметры — не тюнятся Optuna (см. докстринг модуля).
    base_params:
        Параметры CatBoost для прямого режима (n_optuna_trials == 0).
    n_optuna_trials:
        Число Optuna trials, тюнящих только архитектуру CatBoost. Trial
        отбирается по линейной asymmetric-cost метрике (не по MAE).
    param_space / optuna_timeout / optuna_verbose / optuna_pruner / random_seed:
        См. другие Optuna-пресеты пакета.

    Пример::

        model = AsymmetricCostRegressor(loss='pinball', over_cost=1.0, under_cost=3.0)
        model.fit(X_train, y_train, X_valid, y_valid)
        pred = model.predict(X_test)

    """

    _direction = 'minimize'

    def __init__(
        self,
        loss: str = 'pinball',
        over_cost: float = 1.0,
        under_cost: float = 1.0,
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        param_space: Callable[[Any], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        optuna_pruner: str | Any | None = 'none',
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        if loss not in ('pinball', 'asym_mse'):
            raise ValueError(f"loss должен быть 'pinball' или 'asym_mse', получено {loss!r}")
        if over_cost <= 0 or under_cost <= 0:
            raise ValueError('over_cost и under_cost должны быть положительными')

        if loss == 'pinball':
            q = under_cost / (over_cost + under_cost)
            self._loss_spec = _LossSpec(
                name='AsymmetricCost[pinball]', param_bounds={},
                loss_function=lambda _p, q=q: f'Quantile:alpha={q}',
            )
        else:
            self._loss_spec = _LossSpec(
                name='AsymmetricCost[asym_mse]', param_bounds={}, loss_cls=AsymmetricMSELoss,
            )

        super().__init__(
            loss_params={},
            base_params=base_params,
            n_optuna_trials=n_optuna_trials,
            param_space=param_space,
            optuna_timeout=optuna_timeout,
            optuna_verbose=optuna_verbose,
            optuna_pruner=optuna_pruner,
            random_seed=random_seed,
            cat_features=cat_features,
            selected_features=selected_features,
        )
        self.loss = loss
        self.over_cost = over_cost
        self.under_cost = under_cost

    def _build_loss(self, loss_params: dict[str, float], *, tr_pool: Any) -> Any:
        if self.loss == 'asym_mse':
            return AsymmetricMSELoss(over_cost=self.over_cost, under_cost=self.under_cost)
        return self._loss_spec.loss_function(loss_params)

    def _trial_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return _asymmetric_cost(y_true, y_pred, self.over_cost, self.under_cost)
