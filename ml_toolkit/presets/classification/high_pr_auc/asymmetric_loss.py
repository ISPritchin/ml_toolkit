"""AsymmetricLossClassifier: Asymmetric Loss (ASL) для экстремального дисбаланса.

ASL (Ridnik et al., 2021) — эволюция Focal Loss специально для мультилейбл задач
и бинарного экстремального дисбаланса. Два ключевых параметра:

  gamma_pos (γ+, default=0):
    Фокусировка на трудных позитивах. 0 = не фокусируем (стандартный CE для pos).
    Малое значение (0–1) обычно лучше: позитивов мало, штрафить «лёгкие» нет смысла.

  gamma_neg (γ-, default=4):
    Фокусировка на трудных негативах (подавление лёгких). Обычно 2–6.
    Большое γ- → сильнее игнорируем высококонфидентные негативы.

  prob_margin (m, default=0.05):
    Вероятностный сдвиг для негативов: p_s = max(p - m, 0).
    Все негативы с вероятностью < m полностью исключаются из градиента.
    m > 0 помогает при шумных метках (некоторые «негативы» — незамеченные позитивы).

Отличие от FocalLoss в BoostedEnsemble:
  - BoostedEnsemble: один γ на все примеры.
  - ASL: γ+ для позитивов (маленький), γ- для негативов (большой) + margin срез.
  - Значительно более агрессивное подавление уверенных негативов.

fit/tune/predict реализованы в _CustomLossClassifierBase — этот файл только
объявляет _loss_spec (класс лосса + границы Optuna-поиска) и именованные kwargs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ml_toolkit.losses import AsymmetricLoss as _AsymmetricLoss
from ml_toolkit.presets.classification.high_pr_auc._custom_loss_base import (
    _CustomLossClassifierBase,
    _LossSpec,
)


class AsymmetricLossClassifier(_CustomLossClassifierBase):
    """CatBoost с Asymmetric Loss (ASL) вместо стандартного Logloss.

    Parameters
    ----------
    gamma_pos:
        Фокусирующий параметр для позитивов (рекомендуется 0–1).
    gamma_neg:
        Фокусирующий параметр для негативов (рекомендуется 2–6).
    prob_margin:
        Порог обрезки вероятностей негативов; негативы с p < prob_margin
        полностью исключаются из градиента (рекомендуется 0.0–0.1).
    base_params:
        Параметры CatBoost (без loss_function — задаётся автоматически).
    n_optuna_trials:
        Число Optuna-триалов для подбора gamma_pos, gamma_neg, prob_margin, и гиперпарам.
        0 → использовать base_params напрямую.
    param_space:
        Кастомная функция `f(trial) -> dict` — переопределяет search space для
        Optuna (и лосса, и архитектуры CatBoost, в одном пространстве). Любой
        отсутствующий в возвращённом словаре ключ (gamma_pos/gamma_neg/
        prob_margin или iterations/max_depth/learning_rate/l2_leaf_reg/
        subsample/min_data_in_leaf) тюнится дефолтным способом — можно
        переопределить как ни одного параметра (default space целиком), так и
        часть, так и все параметры сразу (и лосса, и модели). Действует
        только при n_optuna_trials > 0. None → дефолтный search space.
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

        model = AsymmetricLossClassifier(gamma_pos=0, gamma_neg=4, prob_margin=0.05)
        model.fit(X_train, y_train, X_valid, y_valid)

    """

    _loss_spec = _LossSpec(
        loss_cls=_AsymmetricLoss,
        param_bounds={'gamma_pos': (0.0, 2.0), 'gamma_neg': (1.0, 8.0), 'prob_margin': (0.0, 0.2)},
        name='AsymmetricLoss',
    )

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        prob_margin: float = 0.05,
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
            loss_params={'gamma_pos': gamma_pos, 'gamma_neg': gamma_neg, 'prob_margin': prob_margin},
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
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.prob_margin = prob_margin
