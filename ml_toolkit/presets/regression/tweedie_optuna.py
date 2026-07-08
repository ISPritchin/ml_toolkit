"""TweedieOptunaRegressor: CatBoost со встроенным Tweedie-лоссом, variance power подбирается Optuna.

Tweedie-распределение (compound Poisson-Gamma) — стандартная модель для
неотрицательного таргета с массой в нуле (обороты, страховые премии, объём
потреблённых услуг): часть наблюдений — точный 0 (событие не произошло),
остальные — непрерывно распределены на положительной полуоси. `variance_power`
p ∈ (1, 2) определяет форму: p → 1 приближается к Poisson (счётные величины),
p → 2 — к Gamma (строго положительная непрерывная тяжесть); правильное p
заранее неизвестно и зависит от того, насколько «дискретна» смесь в данных —
поэтому тюнится Optuna, а не фиксируется руками. Использует встроенный
`Tweedie:variance_power=X` CatBoost (нативная C++ реализация).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ml_toolkit.presets.regression._custom_loss_base import (
    _CustomLossRegressorBase,
    _LossSpec,
)


class TweedieOptunaRegressor(_CustomLossRegressorBase):
    """CatBoost со встроенным Tweedie-лоссом (`Tweedie:variance_power=X`).

    Требует неотрицательный таргет (Tweedie не определён для y < 0) — fit()
    поднимает ValueError, если в y_train/y_valid есть отрицательные значения.

    Parameters
    ----------
    power:
        Начальное/дефолтное значение variance power для прямого режима и
        точка-якорь для первого Optuna trial; при n_optuna_trials > 0 реально
        подбирается в диапазоне (1.01, 1.99).
    base_params:
        Параметры CatBoost для прямого режима (n_optuna_trials == 0).
    n_optuna_trials:
        Число Optuna trials, тюнящих power + архитектуру CatBoost. Trial
        отбирается по MAE на валидации.
    param_space / optuna_timeout / optuna_verbose / random_seed:
        См. другие Optuna-пресеты пакета.

    Пример::

        model = TweedieOptunaRegressor(n_optuna_trials=30)
        model.fit(X_train, y_train, X_valid, y_valid)
        pred = model.predict(X_test)

    """

    _loss_spec = _LossSpec(
        name='TweedieOptuna',
        param_bounds={'power': (1.01, 1.99)},
        loss_function=lambda p: f"Tweedie:variance_power={p['power']}",
    )

    def __init__(
        self,
        power: float = 1.5,
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        param_space: Callable[[Any], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        if not 1.0 < power < 2.0:
            raise ValueError(f'power должен быть в (1, 2), получено {power}')
        super().__init__(
            loss_params={'power': power},
            base_params=base_params,
            n_optuna_trials=n_optuna_trials,
            param_space=param_space,
            optuna_timeout=optuna_timeout,
            optuna_verbose=optuna_verbose,
            random_seed=random_seed,
            cat_features=cat_features,
            selected_features=selected_features,
        )
        self.power = power

    def fit(self, X_train, y_train, X_valid, y_valid, selected_features=None, cat_features=None):
        for name, y in (('y_train', y_train), ('y_valid', y_valid)):
            if (np.asarray(y) < 0).any():
                raise ValueError(f'TweedieOptunaRegressor требует неотрицательный таргет: {name} содержит y < 0')
        return super().fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
