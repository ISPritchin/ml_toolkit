"""HuberOptunaRegressor: CatBoost со встроенным Huber-лоссом, delta подбирается Optuna по MAE.

Huber ведёт себя как MSE при |остаток| <= delta и как MAE за пределами —
устойчив к выбросам в таргете без полного отказа от квадратичной чувствительности
около нуля (в отличие от чистого MAE). `delta` — точка перехода, зависит от
масштаба шума в конкретных данных и плохо угадывается руками, поэтому тюнится
Optuna вместе с архитектурой. Использует встроенный `Huber:delta=X` CatBoost
(нативная C++ реализация, а не Python calc_ders_range — быстрее и без разницы
в leaf_estimation_method, см. каверзу в тестах RelativeErrorRegressor).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ml_toolkit.presets.regression._custom_loss_base import (
    _CustomLossRegressorBase,
    _LossSpec,
)

if TYPE_CHECKING:
    from optuna.pruners import BasePruner


class HuberOptunaRegressor(_CustomLossRegressorBase):
    """CatBoost со встроенным Huber-лоссом (`Huber:delta=X`).

    Parameters
    ----------
    delta:
        Точка перехода MSE → MAE. Начальное/дефолтное значение для прямого
        режима и точка-якорь для первого Optuna trial; при n_optuna_trials > 0
        реально подбирается в диапазоне (0.01, 10.0).
    base_params:
        Параметры CatBoost для прямого режима (n_optuna_trials == 0).
    n_optuna_trials:
        Число Optuna trials, тюнящих delta + архитектуру CatBoost. Trial
        отбирается по MAE на валидации.
    param_space / optuna_timeout / optuna_verbose / optuna_pruner / random_seed:
        См. другие Optuna-пресеты пакета.

    Пример::

        model = HuberOptunaRegressor(n_optuna_trials=30)
        model.fit(X_train, y_train, X_valid, y_valid)
        pred = model.predict(X_test)

    """

    _loss_spec = _LossSpec(
        name='HuberOptuna',
        param_bounds={'delta': (0.01, 10.0)},
        loss_function=lambda p: f"Huber:delta={p['delta']}",
    )

    def __init__(
        self,
        delta: float = 1.0,
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        param_space: Callable[[Any], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        optuna_pruner: str | BasePruner | None = 'none',
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        if delta <= 0:
            raise ValueError('delta должен быть положительным')
        super().__init__(
            loss_params={'delta': delta},
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
        self.delta = delta
