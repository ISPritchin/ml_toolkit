"""FocalLossClassifier: одиночный CatBoost с FocalLoss из ml_toolkit.losses.

FL = -alpha_t * (1-p_t)^gamma * log(p_t) — единый gamma на все примеры (в
отличие от AsymmetricLossClassifier, где γ+/γ- разные для позитивов/негативов).

Когда использовать: нужен focal без ансамбля (см. BoostedEnsemble, где FocalLoss
используется как один из loss_configs) — быстрый одномодельный бейзлайн для
дисбаланса.

fit/tune/predict реализованы в _CustomLossClassifierBase — этот файл только
объявляет _loss_spec (класс лосса + границы Optuna-поиска) и именованные kwargs.
"""

from __future__ import annotations

from typing import Any

from ml_toolkit.losses import FocalLoss as _FocalLoss
from ml_toolkit.presets.classification.high_pr_auc._custom_loss_base import (
    _CustomLossClassifierBase,
    _LossSpec,
)


class FocalLossClassifier(_CustomLossClassifierBase):
    """CatBoost с Focal Loss вместо стандартного Logloss.

    Parameters
    ----------
    gamma:
        Фокусирующий параметр (>= 1). Чем выше, тем сильнее подавляются
        «лёгкие» примеры.
    alpha:
        Вес класса 1 (позитивы). 1-alpha — вес класса 0.
    base_params:
        Параметры CatBoost (без loss_function — задаётся автоматически).
    n_optuna_trials:
        Число Optuna-триалов для подбора gamma, alpha и гиперпараметров CatBoost.
        0 → использовать base_params напрямую.
    random_seed:
        Зерно CatBoost и Optuna sampler'а.

    Пример::

        model = FocalLossClassifier(gamma=2.0, alpha=0.25)
        model.fit(X_train, y_train, X_valid, y_valid)
    """

    _loss_spec = _LossSpec(
        loss_cls=_FocalLoss,
        param_bounds={'gamma': (1.0, 5.0), 'alpha': (0.05, 0.95)},
        name='FocalLoss',
    )

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        optuna_timeout: int | None = None,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(
            loss_params={'gamma': gamma, 'alpha': alpha},
            base_params=base_params,
            n_optuna_trials=n_optuna_trials,
            optuna_timeout=optuna_timeout,
            random_seed=random_seed,
            cat_features=cat_features,
            selected_features=selected_features,
        )
        self.gamma = gamma
        self.alpha = alpha
