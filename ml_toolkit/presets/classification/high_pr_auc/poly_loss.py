"""PolyLossClassifier: CatBoost + PolyLoss (Poly-1 расширение CE, Leng et al. 2022).

L = CE + eps1*(1-p_t) — линейное расширение бинарного CE. eps1 > 0 усиливает
фокус на трудных примерах (похоже на Focal Loss, но дешевле — один линейный
член вместо степенного); eps1 < 0 — акцент на уверенных; eps1=0 — обычный CE.

Когда использовать: дешёвая альтернатива FocalLossClassifier; часто чуть лучше
обычного CE на дисбалансе почти без дополнительной цены.

fit/tune/predict реализованы в _CustomLossClassifierBase — этот файл только
объявляет _loss_spec (класс лосса + границы Optuna-поиска) и именованные kwargs.
"""

from __future__ import annotations

from typing import Any

from ml_toolkit.losses import PolyLoss as _PolyLoss
from ml_toolkit.presets.classification.high_pr_auc._custom_loss_base import (
    _CustomLossClassifierBase,
    _LossSpec,
)


class PolyLossClassifier(_CustomLossClassifierBase):
    """CatBoost с Poly-1 Loss вместо стандартного Logloss.

    Parameters
    ----------
    eps1:
        Коэффициент линейного члена. Рекомендуется 1.0–3.0 при дисбалансе;
        отрицательные значения смещают акцент на уверенные примеры.
    base_params:
        Параметры CatBoost (без loss_function — задаётся автоматически).
    n_optuna_trials:
        Число Optuna-триалов для подбора eps1 и гиперпараметров CatBoost.
        0 → использовать base_params напрямую.
    random_seed:
        Зерно CatBoost и Optuna sampler'а.

    Пример::

        model = PolyLossClassifier(eps1=2.0)
        model.fit(X_train, y_train, X_valid, y_valid)
    """

    _loss_spec = _LossSpec(
        loss_cls=_PolyLoss,
        param_bounds={'eps1': (-1.0, 5.0)},
        name='PolyLoss',
    )

    def __init__(
        self,
        eps1: float = 2.0,
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        optuna_timeout: int | None = None,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(
            loss_params={'eps1': eps1},
            base_params=base_params,
            n_optuna_trials=n_optuna_trials,
            optuna_timeout=optuna_timeout,
            random_seed=random_seed,
            cat_features=cat_features,
            selected_features=selected_features,
        )
        self.eps1 = eps1
