"""TverskyLossClassifier: CatBoost + TverskyLoss (batch Tversky index).

TI = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth); L = 1 - TI.
alpha > beta → штрафуем FP сильнее → выше precision.
alpha < beta → штрафуем FN сильнее → выше recall.

Когда использовать: recall существенно дороже precision (или наоборот), и это
надо зашить прямо в градиент, а не подбирать порогом после обучения (см.
ThresholdMovingCV для порогового подхода).

fit/tune/predict реализованы в _CustomLossClassifierBase — этот файл только
объявляет _loss_spec (класс лосса + границы Optuna-поиска) и именованные kwargs.
"""

from __future__ import annotations

from typing import Any

from ml_toolkit.losses import TverskyLoss as _TverskyLoss
from ml_toolkit.presets.classification.high_pr_auc._custom_loss_base import (
    _CustomLossClassifierBase,
    _LossSpec,
)


class TverskyLossClassifier(_CustomLossClassifierBase):
    """CatBoost с Tversky Loss вместо стандартного Logloss.

    Parameters
    ----------
    alpha:
        Вес ложноположительных (FP). Меньше alpha → выше recall.
    beta:
        Вес ложноотрицательных (FN). Больше beta → выше recall.
    base_params:
        Параметры CatBoost (без loss_function — задаётся автоматически).
    n_optuna_trials:
        Число Optuna-триалов для подбора alpha, beta и гиперпараметров CatBoost.
        0 → использовать base_params напрямую.
    random_seed:
        Зерно CatBoost и Optuna sampler'а.

    Пример::

        model = TverskyLossClassifier(alpha=0.3, beta=0.7)  # штраф FN сильнее — выше recall
        model.fit(X_train, y_train, X_valid, y_valid)
    """

    _loss_spec = _LossSpec(
        loss_cls=_TverskyLoss,
        param_bounds={'alpha': (0.05, 0.95), 'beta': (0.05, 0.95)},
        name='TverskyLoss',
    )

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        optuna_timeout: int | None = None,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(
            loss_params={'alpha': alpha, 'beta': beta},
            base_params=base_params,
            n_optuna_trials=n_optuna_trials,
            optuna_timeout=optuna_timeout,
            random_seed=random_seed,
            cat_features=cat_features,
            selected_features=selected_features,
        )
        self.alpha = alpha
        self.beta = beta
