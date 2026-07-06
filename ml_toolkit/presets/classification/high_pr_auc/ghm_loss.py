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
        optuna_timeout: int | None = None,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(
            loss_params={'bins': bins, 'momentum': momentum},
            base_params=base_params,
            n_optuna_trials=n_optuna_trials,
            optuna_timeout=optuna_timeout,
            random_seed=random_seed,
            cat_features=cat_features,
            selected_features=selected_features,
        )
        self.bins = bins
        self.momentum = momentum

    def _make_loss(self, loss_params: dict[str, float], *, tr_pool: Any, arch_params: dict) -> _GHMLoss:
        return _GHMLoss(bins=int(loss_params['bins']), momentum=loss_params['momentum'])
