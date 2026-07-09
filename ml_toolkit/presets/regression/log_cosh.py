"""LogCoshRegressor: CatBoost с гладким робастным log-cosh лоссом.

L(r) = log(cosh(r)) ведёт себя как 0.5*r^2 при малых |r| и как |r| - log(2) при
больших — тот же переход MSE → MAE, что у Huber, но гладкий (без излома и без
параметра delta, который надо подбирать). «Мягкий Huber» для случаев, когда
достаточно устойчивости к выбросам, но не хочется вводить лишний
гиперпараметр (см. HuberOptunaRegressor, где delta тюнится явно).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ml_toolkit.presets.regression._custom_loss_base import (
    _CustomLossRegressorBase,
    _LossSpec,
)
from ml_toolkit.presets.regression._losses import LogCoshLoss


class LogCoshRegressor(_CustomLossRegressorBase):
    """CatBoost с log-cosh лоссом (кастомный Python calc_ders_range, без параметров).

    Parameters
    ----------
    base_params:
        Параметры CatBoost для прямого режима (n_optuna_trials == 0).
    n_optuna_trials:
        Число Optuna trials, тюнящих архитектуру CatBoost (у лосса нет
        собственных параметров). Trial отбирается по MAE на валидации.
    param_space / optuna_timeout / optuna_verbose / optuna_pruner / random_seed:
        См. другие Optuna-пресеты пакета.

    Пример::

        model = LogCoshRegressor(n_optuna_trials=30)
        model.fit(X_train, y_train, X_valid, y_valid)
        pred = model.predict(X_test)

    """

    _loss_spec = _LossSpec(name='LogCosh', param_bounds={}, loss_cls=LogCoshLoss)

    def __init__(
        self,
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
