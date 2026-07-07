"""LDAMClassifier: CatBoost + LDAM loss (Label-Distribution-Aware Margin) + Deferred Re-Weighting.

LDAM (Cao et al., 2019): миноритарный класс получает больший обязательный
margin от границы решения — Δ_j = C / n_j^{1/4}, чем меньше n_j, тем больше
Δ_j. Deferred Re-Weighting (DRW): первые reweight_epoch_frac * iterations
итераций обучение идёт с равными весами (только margin), затем включаются
веса по effective number of samples (Cui et al., 2019) — эмпирически лучше,
чем reweight с первой итерации.

Когда использовать: экстремальный дисбаланс, Focal/ASL не дали прироста.

fit/tune/predict реализованы в _CustomLossClassifierBase — этот файл
переопределяет только _make_loss (LDAMLoss, в отличие от Focal/Tversky/Poly/
Asymmetric, нуждается в n_pos/n_neg из train и в фактическом числе итераций
модели — оба доступны только внутри _fit_model, не в момент __init__).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ml_toolkit.losses import LDAMLoss as _LDAMLoss
from ml_toolkit.presets.classification.high_pr_auc._custom_loss_base import (
    _CustomLossClassifierBase,
    _LossSpec,
)


class LDAMClassifier(_CustomLossClassifierBase):
    """CatBoost с LDAM + Deferred Re-Weighting loss вместо стандартного Logloss.

    Parameters
    ----------
    max_margin:
        Максимальный margin C среди классов (рекомендуется 0.1–1.0).
    reweight_epoch_frac:
        Доля итераций, после которой включается DRW-переweighting (рекомендуется 0.5–0.95).
    beta:
        Коэффициент effective number of samples для DRW-весов (Cui et al., 2019).
        Не тюнится Optuna — фиксированный гиперпараметр.
    base_params:
        Параметры CatBoost (без loss_function — задаётся автоматически).
    n_optuna_trials:
        Число Optuna-триалов для подбора max_margin, reweight_epoch_frac и
        гиперпараметров CatBoost. 0 → использовать base_params напрямую.
    param_space:
        Кастомная функция `f(trial) -> dict` — переопределяет search space для
        Optuna (и лосса, и архитектуры CatBoost, в одном пространстве). Любой
        отсутствующий в возвращённом словаре ключ (max_margin/
        reweight_epoch_frac или iterations/max_depth/learning_rate/
        l2_leaf_reg/subsample/min_data_in_leaf) тюнится дефолтным способом —
        можно переопределить как ни одного параметра (default space целиком),
        так и часть, так и все параметры сразу (и лосса, и модели). beta в
        param_space не участвует — фиксированный конструкторский
        гиперпараметр, не тюнится Optuna вообще. Действует только при
        n_optuna_trials > 0. None → дефолтный search space.
    optuna_timeout:
        Ограничение по времени (сек) на весь Optuna-поиск. None — без ограничения.
    optuna_verbose:
        Если True — не глушит логи Optuna. Если False (по умолчанию) —
        форсирует WARNING на время поиска.
    random_seed:
        Зерно CatBoost и Optuna sampler'а.

    Пример::

        model = LDAMClassifier(max_margin=0.5, reweight_epoch_frac=0.8)
        model.fit(X_train, y_train, X_valid, y_valid)

    """

    _loss_spec = _LossSpec(
        loss_cls=_LDAMLoss,
        param_bounds={'max_margin': (0.1, 1.0), 'reweight_epoch_frac': (0.5, 0.95)},
        name='LDAM',
    )

    def __init__(
        self,
        max_margin: float = 0.5,
        reweight_epoch_frac: float = 0.8,
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
            loss_params={'max_margin': max_margin, 'reweight_epoch_frac': reweight_epoch_frac},
            base_params=base_params,
            n_optuna_trials=n_optuna_trials,
            param_space=param_space,
            optuna_timeout=optuna_timeout,
            optuna_verbose=optuna_verbose,
            random_seed=random_seed,
            cat_features=cat_features,
            selected_features=selected_features,
        )
        self.max_margin = max_margin
        self.reweight_epoch_frac = reweight_epoch_frac
        self.beta = beta

    def _make_loss(self, loss_params: dict[str, float], *, tr_pool: Any, arch_params: dict) -> _LDAMLoss:
        y_tr = np.asarray(tr_pool.get_label())
        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        return _LDAMLoss(
            n_pos=n_pos,
            n_neg=n_neg,
            max_margin=loss_params['max_margin'],
            reweight_epoch_frac=loss_params['reweight_epoch_frac'],
            n_total_iterations=arch_params['iterations'],
            beta=self.beta,
        )
