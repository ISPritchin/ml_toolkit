"""AsymmetricPolyLossClassifier: CatBoost + ASL с Poly-1 поправкой.

Комбинация AsymmetricLossClassifier (gamma_pos/gamma_neg/prob_margin) и
PolyLossClassifier (eps1) в одном лоссе — ещё одна степень свободы у лёгких
примеров поверх уже настроенного ASL.

Когда использовать: ASL уже используется (см. AsymmetricLossClassifier), но
нужна дополнительная настройка через линейный Poly-1 член.

fit/tune/predict реализованы в _CustomLossClassifierBase — этот файл только
объявляет _loss_spec (класс лосса + границы Optuna-поиска) и именованные kwargs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ml_toolkit.losses import AsymmetricPolyLoss as _AsymmetricPolyLoss
from ml_toolkit.presets.classification.high_pr_auc._custom_loss_base import (
    _CustomLossClassifierBase,
    _LossSpec,
)


class AsymmetricPolyLossClassifier(_CustomLossClassifierBase):
    """CatBoost с ASL + Poly-1 Loss вместо стандартного Logloss.

    Parameters
    ----------
    gamma_pos:
        Фокусирующий параметр ASL для позитивов (рекомендуется 0-1).
    gamma_neg:
        Фокусирующий параметр ASL для негативов (рекомендуется 2-6).
    prob_margin:
        Порог обрезки вероятностей негативов ASL (рекомендуется 0.0-0.1).
    eps1:
        Коэффициент линейного Poly-1 члена. Рекомендуется 1.0-3.0.
    base_params:
        Параметры CatBoost (без loss_function — задаётся автоматически).
    n_optuna_trials:
        Число Optuna-триалов для подбора gamma_pos, gamma_neg, prob_margin,
        eps1 и гиперпараметров CatBoost. 0 → использовать base_params напрямую.
    param_space:
        Кастомная функция `f(trial) -> dict` — переопределяет search space для
        Optuna (и лосса, и архитектуры CatBoost, в одном пространстве). Любой
        отсутствующий в возвращённом словаре ключ (gamma_pos/gamma_neg/
        prob_margin/eps1 или iterations/max_depth/learning_rate/l2_leaf_reg/
        subsample/min_data_in_leaf) тюнится дефолтным способом — можно
        переопределить как ни одного параметра (default space целиком), так и
        часть, так и все параметры сразу (и лосса, и модели). Действует
        только при n_optuna_trials > 0. None → дефолтный search space.
    optuna_timeout:
        Ограничение по времени (сек) на весь Optuna-поиск. None — без ограничения.
    optuna_verbose:
        Если True — не глушит логи Optuna. Если False (по умолчанию) —
        форсирует WARNING на время поиска.
    random_seed:
        Зерно CatBoost и Optuna sampler'а.

    Пример::

        model = AsymmetricPolyLossClassifier(gamma_pos=0, gamma_neg=4, eps1=2.0)
        model.fit(X_train, y_train, X_valid, y_valid)
    """

    _loss_spec = _LossSpec(
        loss_cls=_AsymmetricPolyLoss,
        param_bounds={
            'gamma_pos': (0.0, 2.0),
            'gamma_neg': (1.0, 8.0),
            'prob_margin': (0.0, 0.2),
            'eps1': (-2.0, 5.0),
        },
        name='AsymmetricPolyLoss',
    )

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        prob_margin: float = 0.05,
        eps1: float = 2.0,
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
            loss_params={
                'gamma_pos': gamma_pos,
                'gamma_neg': gamma_neg,
                'prob_margin': prob_margin,
                'eps1': eps1,
            },
            base_params=base_params,
            n_optuna_trials=n_optuna_trials,
            param_space=param_space,
            optuna_timeout=optuna_timeout,
            optuna_verbose=optuna_verbose,
            random_seed=random_seed,
            cat_features=cat_features,
            selected_features=selected_features,
        )
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.prob_margin = prob_margin
        self.eps1 = eps1
