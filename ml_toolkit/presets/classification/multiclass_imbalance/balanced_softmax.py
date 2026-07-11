"""BalancedSoftmaxClassifier: CatBoost + BalancedSoftmaxLoss (Ren et al., 2020).

Training-time аналог пост-хок logit adjustment (см. LogitAdjustmentClassifier/
005): сдвиг softmax на log(class_prior) встроен в сам CE во время обучения,
а не применяется поверх уже обученной модели.

Когда использовать: мультикласс с известными и стабильными частотами классов
(при нестабильных/эволюционирующих частотах пост-хок LogitAdjustment гибче,
т.к. не требует переобучения при смене prior).

fit/tune/predict реализованы в _CustomLossClassifierMulticlassBase — этот файл
переопределяет только _make_loss (BalancedSoftmaxLoss нуждается в частотах
классов из train, доступных только внутри _fit_model, не в момент __init__).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ml_toolkit.losses import BalancedSoftmaxLoss as _BalancedSoftmaxLoss
from ml_toolkit.presets.classification.multiclass_imbalance._custom_loss_base import (
    _CustomLossClassifierMulticlassBase,
    _MulticlassLossSpec,
)

if TYPE_CHECKING:
    from catboost import Pool
    import optuna
    from optuna.pruners import BasePruner


class BalancedSoftmaxClassifier(_CustomLossClassifierMulticlassBase):
    """CatBoost с Balanced Softmax Loss вместо стандартного MultiClass CE.

    Parameters
    ----------
    tau:
        Сила сдвига логитов на log(class_prior). tau=1.0 — полная поправка
        (как в оригинальной статье), tau=0 — обычный softmax CE.
    base_params:
        Параметры CatBoost (без loss_function/classes_count — задаются автоматически).
    n_optuna_trials:
        Число Optuna-триалов для подбора tau и гиперпараметров CatBoost.
        0 → использовать base_params напрямую.
    param_space:
        Кастомная функция `f(trial) -> dict` — переопределяет search space для
        Optuna (и лосса, и архитектуры CatBoost, в одном пространстве). Любой
        отсутствующий в возвращённом словаре ключ (tau или iterations/
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
    optuna_pruner:
        None/строковый алиас ('median'/'hyperband'/'percentile'/
        'successive_halving'/'none')/готовый optuna.pruners.BasePruner —
        см. ml_toolkit.models model_settings.md. 'none' (по умолчанию) —
        прунинг выключен.
    random_seed:
        Зерно CatBoost и Optuna sampler'а.

    Пример::

        model = BalancedSoftmaxClassifier(tau=1.0)
        model.fit(X_train, y_train, X_valid, y_valid)

    """

    _loss_spec = _MulticlassLossSpec(
        loss_cls=_BalancedSoftmaxLoss,
        param_bounds={'tau': (0.1, 2.0)},
        name='BalancedSoftmax',
    )

    def __init__(
        self,
        tau: float = 1.0,
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        param_space: Callable[[optuna.Trial], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        optuna_pruner: str | BasePruner | None = 'none',
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(
            loss_params={'tau': tau},
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
        self.tau = tau

    def _make_loss(
        self, loss_params: dict[str, float], *, tr_pool: Pool, arch_params: dict, n_classes: int
    ) -> _BalancedSoftmaxLoss:
        y_tr = np.asarray(tr_pool.get_label()).astype(int)
        class_counts = np.bincount(y_tr, minlength=n_classes)
        return _BalancedSoftmaxLoss(class_counts=class_counts, tau=loss_params['tau'])
