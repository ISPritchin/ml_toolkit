"""EqualizationLossClassifier: CatBoost + EqualizationLoss (Seesaw/EQLv2-style).

Мультиклассовый (не бинарный, как AsymmetricLossClassifier/003-009) лосс для
длиннохвостого дисбаланса: подавляет градиент от головных классов на редкие
внутри батча через mitigation (по частоте класса) + compensation (по
предсказанной уверенности) множители.

Когда использовать: мультиклассовая классификация с сильно неравномерным
хвостом классов (не бинарный дисбаланс — для него см. high_pr_auc/).

fit/tune/predict реализованы в _CustomLossClassifierMulticlassBase — этот файл
переопределяет только _make_loss (EqualizationLoss нуждается в частотах
классов из train, доступных только внутри _fit_model, не в момент __init__).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ml_toolkit.losses import EqualizationLoss as _EqualizationLoss
from ml_toolkit.presets.classification.multiclass_imbalance._custom_loss_base import (
    _CustomLossClassifierMulticlassBase,
    _MulticlassLossSpec,
)


class EqualizationLossClassifier(_CustomLossClassifierMulticlassBase):
    """CatBoost с Equalization/Seesaw Loss вместо стандартного MultiClass CE.

    Parameters
    ----------
    lambda_:
        EMA-момент для сглаживания средней предсказанной вероятности класса.
    seesaw_p:
        Степень mitigation-множителя (сильнее подавляет частые негативы).
    seesaw_q:
        Степень compensation-множителя (сильнее подавляет уверенные ошибки).
    base_params:
        Параметры CatBoost (без loss_function/classes_count — задаются автоматически).
    n_optuna_trials:
        Число Optuna-триалов для подбора lambda_, seesaw_p, seesaw_q и
        гиперпараметров CatBoost. 0 → использовать base_params напрямую.
    param_space:
        Кастомная функция `f(trial) -> dict` — переопределяет search space для
        Optuna (и лосса, и архитектуры CatBoost, в одном пространстве). Любой
        отсутствующий в возвращённом словаре ключ (lambda_/seesaw_p/seesaw_q
        или iterations/max_depth/learning_rate/l2_leaf_reg/subsample/
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

        model = EqualizationLossClassifier(seesaw_p=0.8, seesaw_q=2.0)
        model.fit(X_train, y_train, X_valid, y_valid)

    """

    _loss_spec = _MulticlassLossSpec(
        loss_cls=_EqualizationLoss,
        param_bounds={'lambda_': (0.5, 0.99), 'seesaw_p': (0.2, 2.0), 'seesaw_q': (0.5, 4.0)},
        name='EqualizationLoss',
    )

    def __init__(
        self,
        lambda_: float = 0.9,
        seesaw_p: float = 0.8,
        seesaw_q: float = 2.0,
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
            loss_params={'lambda_': lambda_, 'seesaw_p': seesaw_p, 'seesaw_q': seesaw_q},
            base_params=base_params,
            n_optuna_trials=n_optuna_trials,
            param_space=param_space,
            optuna_timeout=optuna_timeout,
            optuna_verbose=optuna_verbose,
            random_seed=random_seed,
            cat_features=cat_features,
            selected_features=selected_features,
        )
        self.lambda_ = lambda_
        self.seesaw_p = seesaw_p
        self.seesaw_q = seesaw_q

    def _make_loss(
        self, loss_params: dict[str, float], *, tr_pool: Any, arch_params: dict, n_classes: int
    ) -> _EqualizationLoss:
        y_tr = np.asarray(tr_pool.get_label()).astype(int)
        class_counts = np.bincount(y_tr, minlength=n_classes)
        return _EqualizationLoss(
            class_counts=class_counts,
            lambda_=loss_params['lambda_'],
            seesaw_p=loss_params['seesaw_p'],
            seesaw_q=loss_params['seesaw_q'],
        )
