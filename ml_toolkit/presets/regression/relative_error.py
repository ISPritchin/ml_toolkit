"""RelativeErrorRegressor: CatBoost, оптимизирующий относительную ошибку (MAPE/SMAPE/WAPE).

Когда бизнес оценивает модель в процентах отклонения (типичный сценарий для
прогнозов выручки/оборотов разного масштаба клиентов), обучение на RMSE/MAE
недооптимизирует именно ту метрику, по которой модель будут судить — крупные
клиенты доминируют в абсолютной ошибке, а относительная точность на мелких
может быть сколь угодно плохой. Здесь CatBoost учится напрямую на relative-loss
через кастомный calc_ders_range (ml_toolkit.presets.regression._losses.RelativeErrorLoss).

metric/denom_floor — не тюнящиеся Optuna-параметры (выбор метрики — решение
пользователя о том, что оптимизировать, а не гиперпараметр с "оптимальным"
значением по валидации), поэтому `_loss_spec.param_bounds` пуст: Optuna (если
включена) тюнит только архитектуру CatBoost, а trial отбирается по самой
relative-метрике на валидации (не по MAE, как в остальных пресетах движка) —
иначе выбор архитектуры не соответствовал бы тому, что реально оптимизируется.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ml_toolkit.presets.regression._custom_loss_base import (
    _CalcDersRangeLoss,
    _CustomLossRegressorBase,
    _LossSpec,
)
from ml_toolkit.presets.regression._losses import RelativeErrorLoss

if TYPE_CHECKING:
    from catboost import Pool
    from optuna.pruners import BasePruner


def _relative_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str, floor: float) -> float:
    e = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    if metric == 'mape':
        d = np.maximum(np.abs(y_true), floor)
        return float(np.mean(e / d))
    if metric == 'wape':
        d = max(float(np.mean(np.abs(y_true))), floor)
        return float(np.mean(e) / d)
    # smape
    d = np.abs(y_true) + np.abs(y_pred) + floor
    return float(np.mean(2.0 * e / d))


class RelativeErrorRegressor(_CustomLossRegressorBase):
    """CatBoost с кастомным лоссом на относительной ошибке.

    Parameters
    ----------
    metric:
        'wape' (по умолчанию) | 'mape' | 'smape'. Per-row поверхность лосса —
        см. докстринг RelativeErrorLoss в _losses.py. WAPE использует глобальный
        denom = mean(|y_train|) (честнее приближает sum|e|/sum|y|, чем per-row
        MAPE-подобная поверхность).
    denom_floor:
        Нижняя граница знаменателя — защита от деления на ~0 при y около нуля.
    base_params:
        Параметры CatBoost для прямого режима (n_optuna_trials == 0).
    n_optuna_trials:
        Число Optuna trials, тюнящих только архитектуру CatBoost (у лосса нет
        тюнящихся параметров). Trial отбирается по значению `metric` на
        валидации (не по MAE).
    param_space / optuna_timeout / optuna_verbose / optuna_pruner / random_seed:
        См. другие Optuna-пресеты пакета.

    Пример::

        model = RelativeErrorRegressor(metric='wape', denom_floor=1.0, n_optuna_trials=30)
        model.fit(X_train, y_train, X_valid, y_valid)
        pred = model.predict(X_test)

    """

    _loss_spec = _LossSpec(name='RelativeError', param_bounds={}, loss_cls=RelativeErrorLoss)
    _direction = 'minimize'

    def __init__(
        self,
        metric: str = 'wape',
        denom_floor: float = 1.0,
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
        if metric not in ('mape', 'smape', 'wape'):
            raise ValueError(f"metric должен быть 'mape'/'smape'/'wape', получено {metric!r}")
        self.metric = metric
        self.denom_floor = denom_floor

    def _build_loss(self, loss_params: dict[str, float], *, tr_pool: Pool) -> _CalcDersRangeLoss:
        # loss_params игнорируется — metric/denom_floor не тюнятся Optuna, у
        # лосса нет записи в _loss_spec.param_bounds (см. докстринг класса).
        loss = RelativeErrorLoss(metric=self.metric, denom_floor=self.denom_floor)
        if self.metric == 'wape':
            y_tr = np.asarray(tr_pool.get_label(), dtype=np.float64)
            loss.global_denom = max(float(np.mean(np.abs(y_tr))), self.denom_floor)
        return loss

    def _trial_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return _relative_score(y_true, y_pred, self.metric, self.denom_floor)
