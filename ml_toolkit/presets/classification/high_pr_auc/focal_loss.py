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

from collections.abc import Callable
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
    param_space:
        Кастомная функция `f(trial) -> dict` — переопределяет search space для
        Optuna (и лосса, и архитектуры CatBoost, в одном пространстве). Любой
        отсутствующий в возвращённом словаре ключ (gamma/alpha или
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
        param_space: Callable[[Any], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        optuna_pruner: str | Any | None = 'none',
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(
            loss_params={'gamma': gamma, 'alpha': alpha},
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
        self.gamma = gamma
        self.alpha = alpha
