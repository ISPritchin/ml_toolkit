"""QuantileHuberRegressor: CatBoost с квантильным Huber-лоссом (сглаженный pinball).

Чистый pinball loss (используемый, например, встроенным CatBoost `Quantile:alpha=q`)
имеет излом в r=0 — вблизи медианы/квантиля градиент постоянен по модулю и не
затухает, что даёт шумные, дёргающиеся обновления листьев на зашумлённых данных.
QuantileHuberLoss (см. _losses.py) сглаживает этот излом квадратичным участком
шириной `kappa` вокруг нуля (как в distributional RL, QR-DQN), сохраняя
асимметрию pinball за пределами этого участка.

quantile — не тюнящийся Optuna параметр: он определяет, ЧТО именно предсказывает
модель (квантиль q, а не медиану) — это решение постановки задачи, а не выбор,
который стоит доверять валидационному score (Optuna тривиально «выиграла» бы,
просто утаскивая quantile к вырожденному значению, минимизирующему pinball на
конкретной выборке). kappa, наоборот, — чисто техническая настройка сглаживания
без смысловой нагрузки, поэтому тюнится вместе с архитектурой CatBoost.
Objective — pinball loss на фиксированном quantile (ml_toolkit.models._utils.
quantile_loss), а не MAE — иначе отбор trial не соответствовал бы тому, что
реально требуется от модели.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ml_toolkit.models._utils import quantile_loss
from ml_toolkit.presets.regression._custom_loss_base import (
    _CustomLossRegressorBase,
    _LossSpec,
)
from ml_toolkit.presets.regression._losses import QuantileHuberLoss


class QuantileHuberRegressor(_CustomLossRegressorBase):
    """CatBoost с квантильным Huber-лоссом.

    Parameters
    ----------
    quantile:
        Целевой квантиль ∈ (0, 1). Фиксированный параметр постановки задачи —
        не тюнится Optuna (см. докстринг модуля).
    kappa:
        Ширина квадратичного (сглаживающего) участка вокруг r=0. Тюнится
        Optuna при n_optuna_trials > 0.
    base_params:
        Параметры CatBoost для прямого режима (n_optuna_trials == 0).
    n_optuna_trials:
        Число Optuna trials, тюнящих kappa + архитектуру CatBoost. Trial
        отбирается по pinball loss на `quantile` (не по MAE).
    param_space / optuna_timeout / optuna_verbose / random_seed:
        См. другие Optuna-пресеты пакета.

    Пример::

        model = QuantileHuberRegressor(quantile=0.9, kappa=1.0, n_optuna_trials=30)
        model.fit(X_train, y_train, X_valid, y_valid)
        p90 = model.predict(X_test)

    """

    _loss_spec = _LossSpec(
        name='QuantileHuber',
        param_bounds={'kappa': (0.01, 5.0)},
        loss_cls=QuantileHuberLoss,
    )
    _direction = 'minimize'

    def __init__(
        self,
        quantile: float = 0.5,
        kappa: float = 1.0,
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        param_space: Callable[[Any], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError(f'quantile должен быть в (0, 1), получено {quantile}')
        super().__init__(
            loss_params={'kappa': kappa},
            base_params=base_params,
            n_optuna_trials=n_optuna_trials,
            param_space=param_space,
            optuna_timeout=optuna_timeout,
            optuna_verbose=optuna_verbose,
            random_seed=random_seed,
            cat_features=cat_features,
            selected_features=selected_features,
        )
        self.quantile = quantile
        self.kappa = kappa

    def _build_loss(self, loss_params: dict[str, float], *, tr_pool: Any) -> Any:
        return QuantileHuberLoss(quantile=self.quantile, kappa=loss_params['kappa'])

    def _trial_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return quantile_loss(y_true, y_pred, q=self.quantile)
