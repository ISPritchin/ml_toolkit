"""LogitNormLossClassifier: CatBoost + LogitNormLoss (Wei et al., 2022).

Нормализация логитов (делением на их L2-норму, масштабированную temperature)
перед softmax CE — противодействует переуверенности, характерной для головы
длиннохвостого мультикласса.

Когда использовать: калибровка страдает именно из-за длиннохвостого
мультикласса (для бинарной калибровки — см. TemperatureScalingWrapper/014,
HistogramBinningCalibrator/209).

fit/tune/predict реализованы в _CustomLossClassifierMulticlassBase — этот файл
только объявляет _loss_spec (класс лосса + границы Optuna-поиска) и
именованные kwargs; в отличие от Equalization/BalancedSoftmax, LogitNormLoss
не использует частоты классов, поэтому _make_loss не переопределяется.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ml_toolkit.losses import LogitNormLoss as _LogitNormLoss
from ml_toolkit.presets.classification.multiclass_imbalance._custom_loss_base import (
    _CustomLossClassifierMulticlassBase,
    _MulticlassLossSpec,
)

if TYPE_CHECKING:
    import optuna
    from optuna.pruners import BasePruner


class LogitNormLossClassifier(_CustomLossClassifierMulticlassBase):
    """CatBoost с LogitNorm Loss вместо стандартного MultiClass CE.

    Parameters
    ----------
    temperature:
        Температура нормализации. Меньше temperature → сильнее нормализация
        → менее уверенные вероятности. Рекомендуется 0.01-0.1.
    base_params:
        Параметры CatBoost (без loss_function/classes_count — задаются автоматически).
    n_optuna_trials:
        Число Optuna-триалов для подбора temperature и гиперпараметров CatBoost.
        0 → использовать base_params напрямую.
    param_space:
        Кастомная функция `f(trial) -> dict` — переопределяет search space для
        Optuna (и лосса, и архитектуры CatBoost, в одном пространстве). Любой
        отсутствующий в возвращённом словаре ключ (temperature или
        iterations/max_depth/learning_rate/l2_leaf_reg/subsample/
        min_data_in_leaf) тюнится дефолтным способом — можно переопределить
        как ни одного параметра (default space целиком), так и часть, так и
        все параметры сразу (и лосса, и модели). Действует только при
        n_optuna_trials > 0. None → дефолтный search space.
    optuna_timeout:
        Ограничение по времени (сек) на весь Optuna-поиск. None — без ограничения.
    optuna_verbose:
        Если True — не глушит логи Optuna. Если False (по умолчанию) —
        форсирует WARNING на время поиска.
    optuna_pruner:
        None/строковый алиас ('median'/'hyperband'/'percentile'/
        'successive_halving'/'none')/готовый optuna.pruners.BasePruner —
        см. ml_toolkit.models model_settings.md. 'none' (по умолчанию) —
        прунинг выключен.
    random_seed:
        Зерно CatBoost и Optuna sampler'а.

    Пример::

        model = LogitNormLossClassifier(temperature=0.04)
        model.fit(X_train, y_train, X_valid, y_valid)

    """

    _loss_spec = _MulticlassLossSpec(
        loss_cls=_LogitNormLoss,
        param_bounds={'temperature': (0.01, 1.0)},
        name='LogitNormLoss',
    )

    def __init__(
        self,
        temperature: float = 0.04,
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        param_space: Callable[[optuna.Trial], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        optuna_pruner: str | BasePruner | None = 'none',
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(
            loss_params={'temperature': temperature},
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
        self.temperature = temperature
