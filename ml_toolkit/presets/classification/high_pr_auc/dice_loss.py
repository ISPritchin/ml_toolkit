"""DiceLossClassifier: CatBoost + DiceLoss (batch Dice/soft-F1 index).

Dice — частный случай TverskyLoss при alpha=beta=0.5 (см. TverskyLossClassifier
для произвольного FP/FN трейдоффа); здесь FP и FN штрафуются одинаково — прямая
мягкая аппроксимация F1.

Когда использовать: бизнес-метрика — F1/Dice-подобная, а не PR-AUC, и нет
причины смещать FP/FN трейдофф в одну сторону (иначе — TverskyLossClassifier).

fit/tune/predict реализованы в _CustomLossClassifierBase — этот файл только
объявляет _loss_spec (класс лосса + границы Optuna-поиска) и именованные kwargs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ml_toolkit.losses import DiceLoss as _DiceLoss
from ml_toolkit.presets.classification.high_pr_auc._custom_loss_base import (
    _CustomLossClassifierBase,
    _LossSpec,
)


class DiceLossClassifier(_CustomLossClassifierBase):
    """CatBoost с Dice Loss вместо стандартного Logloss.

    Parameters
    ----------
    smooth:
        Коэффициент сглаживания для численной устойчивости.
    base_params:
        Параметры CatBoost (без loss_function — задаётся автоматически).
    n_optuna_trials:
        Число Optuna-триалов для подбора smooth и гиперпараметров CatBoost.
        0 → использовать base_params напрямую.
    param_space:
        Кастомная функция `f(trial) -> dict` — переопределяет search space для
        Optuna (и лосса, и архитектуры CatBoost, в одном пространстве). Любой
        отсутствующий в возвращённом словаре ключ (smooth или iterations/
        max_depth/learning_rate/l2_leaf_reg/subsample/min_data_in_leaf)
        тюнится дефолтным способом — можно переопределить как ни одного
        параметра (default space целиком), так и часть, так и все параметры
        сразу (и лосса, и модели). Действует только при n_optuna_trials > 0.
        None → дефолтный search space.
    optuna_timeout:
        Ограничение по времени (сек) на весь Optuna-поиск. None — без ограничения.
    optuna_verbose:
        Если True — не глушит логи Optuna. Если False (по умолчанию) —
        форсирует WARNING на время поиска.
    random_seed:
        Зерно CatBoost и Optuna sampler'а.

    Пример::

        model = DiceLossClassifier(smooth=1.0)
        model.fit(X_train, y_train, X_valid, y_valid)

    """

    _loss_spec = _LossSpec(
        loss_cls=_DiceLoss,
        param_bounds={'smooth': (0.1, 5.0)},
        name='DiceLoss',
    )

    def __init__(
        self,
        smooth: float = 1.0,
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        param_space: Callable[[Any], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(
            loss_params={'smooth': smooth},
            base_params=base_params,
            n_optuna_trials=n_optuna_trials,
            param_space=param_space,
            optuna_timeout=optuna_timeout,
            optuna_verbose=optuna_verbose,
            random_seed=random_seed,
            cat_features=cat_features,
            selected_features=selected_features,
        )
        self.smooth = smooth
