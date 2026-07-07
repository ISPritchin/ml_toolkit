"""InfluenceBalancedLossClassifier: CatBoost + Influence-Balanced (IB) Loss.

По-сэмпловая (не по-классовая, как ClassBalancedWeightClassifier/006)
альтернатива: примеры с большим |p-y| (доминирующие в агрегированном
градиенте) подавляются множителем, помимо стандартного class-balanced веса.

Когда использовать: дисбаланс внутри класса — часть позитивов лёгкие
дубликаты, часть редкие паттерны, и по-классового веса недостаточно.

fit/tune/predict реализованы в _CustomLossClassifierBase — этот файл
переопределяет только _make_loss (InfluenceBalancedLoss, как и LDAMLoss,
нуждается в n_pos/n_neg из train, доступных только внутри _fit_model, не в
момент __init__).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ml_toolkit.losses import InfluenceBalancedLoss as _InfluenceBalancedLoss
from ml_toolkit.presets.classification.high_pr_auc._custom_loss_base import (
    _CustomLossClassifierBase,
    _LossSpec,
)


class InfluenceBalancedLossClassifier(_CustomLossClassifierBase):
    """CatBoost с Influence-Balanced Loss вместо стандартного Logloss.

    Parameters
    ----------
    alpha:
        Сила подавления примеров с большим influence (|p-y|). Рекомендуется
        порядка 100-1000.
    beta:
        Коэффициент effective number of samples (Cui et al., 2019) для
        per-class части веса. Не тюнится Optuna — фиксированный гиперпараметр.
    base_params:
        Параметры CatBoost (без loss_function — задаётся автоматически).
    n_optuna_trials:
        Число Optuna-триалов для подбора alpha и гиперпараметров CatBoost.
        0 → использовать base_params напрямую.
    param_space:
        Кастомная функция `f(trial) -> dict` — переопределяет search space для
        Optuna (и лосса, и архитектуры CatBoost, в одном пространстве). Любой
        отсутствующий в возвращённом словаре ключ (alpha или iterations/
        max_depth/learning_rate/l2_leaf_reg/subsample/min_data_in_leaf)
        тюнится дефолтным способом — можно переопределить как ни одного
        параметра (default space целиком), так и часть, так и все параметры
        сразу (и лосса, и модели). beta в param_space не участвует — это
        фиксированный конструкторский гиперпараметр, не тюнится Optuna вообще.
        Действует только при n_optuna_trials > 0. None → дефолтный search space.
    optuna_timeout:
        Ограничение по времени (сек) на весь Optuna-поиск. None — без ограничения.
    optuna_verbose:
        Если True — не глушит логи Optuna. Если False (по умолчанию) —
        форсирует WARNING на время поиска.
    random_seed:
        Зерно CatBoost и Optuna sampler'а.

    Пример::

        model = InfluenceBalancedLossClassifier(alpha=1000.0)
        model.fit(X_train, y_train, X_valid, y_valid)

    """

    _loss_spec = _LossSpec(
        loss_cls=_InfluenceBalancedLoss,
        param_bounds={'alpha': (10.0, 5000.0)},
        name='InfluenceBalancedLoss',
    )

    def __init__(
        self,
        alpha: float = 1000.0,
        beta: float = 0.9999,
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
            loss_params={'alpha': alpha},
            base_params=base_params,
            n_optuna_trials=n_optuna_trials,
            param_space=param_space,
            optuna_timeout=optuna_timeout,
            optuna_verbose=optuna_verbose,
            random_seed=random_seed,
            cat_features=cat_features,
            selected_features=selected_features,
        )
        self.alpha = alpha
        self.beta = beta

    def _make_loss(
        self, loss_params: dict[str, float], *, tr_pool: Any, arch_params: dict
    ) -> _InfluenceBalancedLoss:
        y_tr = np.asarray(tr_pool.get_label())
        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        return _InfluenceBalancedLoss(
            n_pos=n_pos,
            n_neg=n_neg,
            alpha=loss_params['alpha'],
            beta=self.beta,
        )
