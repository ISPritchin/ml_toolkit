"""GHMLossClassifier: CatBoost + GHMLoss (Gradient Harmonizing Mechanism).

GHM подавляет и лёгкие негативы, и выбросы-аутлайеры (в отличие от Focal Loss,
который давит только лёгкие примеры) — взвешивая каждый пример обратно
пропорционально плотности градиента в его окрестности.

Когда использовать: Focal Loss (см. FocalLossClassifier) недостаточно давит
"неудобные" выбросы (шумные метки, редкие аномальные объекты).

fit/tune/predict реализованы в _CustomLossClassifierBase — этот файл только
объявляет _loss_spec (класс лосса + границы Optuna-поиска) и именованные kwargs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ml_toolkit.losses import GHMLoss as _GHMLoss
from ml_toolkit.presets.classification.high_pr_auc._custom_loss_base import (
    _CustomLossClassifierBase,
    _LossSpec,
)


class GHMLossClassifier(_CustomLossClassifierBase):
    """CatBoost с Gradient Harmonizing Mechanism Loss вместо стандартного Logloss.

    Parameters
    ----------
    bins:
        Число интервалов гистограммы плотности градиента.
    momentum:
        EMA-коэффициент для сглаживания плотности градиента между итерациями
        бустинга (0 → без сглаживания).
    base_params:
        Параметры CatBoost (без loss_function — задаётся автоматически).
    n_optuna_trials:
        Число Optuna-триалов для подбора bins, momentum и гиперпараметров CatBoost.
        0 → использовать base_params напрямую.
    param_space:
        Кастомная функция `f(trial) -> dict` — переопределяет search space для
        Optuna (и лосса, и архитектуры CatBoost, в одном пространстве). Любой
        отсутствующий в возвращённом словаре ключ (bins/momentum или
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
    random_seed:
        Зерно CatBoost и Optuna sampler'а.

    Пример::

        model = GHMLossClassifier(bins=30, momentum=0.75)
        model.fit(X_train, y_train, X_valid, y_valid)
    """

    _loss_spec = _LossSpec(
        loss_cls=_GHMLoss,
        param_bounds={'bins': (10, 50), 'momentum': (0.0, 0.9)},
        name='GHMLoss',
    )

    def __init__(
        self,
        bins: int = 30,
        momentum: float = 0.75,
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
            loss_params={'bins': bins, 'momentum': momentum},
            base_params=base_params,
            n_optuna_trials=n_optuna_trials,
            param_space=param_space,
            optuna_timeout=optuna_timeout,
            optuna_verbose=optuna_verbose,
            random_seed=random_seed,
            cat_features=cat_features,
            selected_features=selected_features,
        )
        self.bins = bins
        self.momentum = momentum

    def _make_loss(self, loss_params: dict[str, float], *, tr_pool: Any, arch_params: dict) -> _GHMLoss:
        return _GHMLoss(bins=int(loss_params['bins']), momentum=loss_params['momentum'])
