"""NNPUClassifier: CatBoost + non-negative PU risk estimator (Kiryo et al., 2017).

В отличие от Элкана-Ното (PULearningClassifier/029, SpyPUClassifier/028,
BaggingPUClassifier/027) — которые лечат PU-структуру данных ПОСЛЕ обычного
обучения (пост-хок коррекция вероятностей / отбор надёжных негативов /
бэггинг с OOB) — nnPU встраивает знание о PU-структуре прямо в градиент через
class_prior pi: модель напрямую оптимизирует несмещённую (в отличие от
наивного P vs U Logloss) оценку истинного риска.

Когда: pi известен заранее и стабилен (например, пересчитан по полным данным
прошлых периодов) — тогда это единственный из PU-методов, не требующий
эвристик (spy_frac, u_sample_size, c-holdout) для получения несмещённой
оценки.

fit/tune/predict реализованы в _CustomLossClassifierBase — этот файл только
объявляет _loss_spec и именованные kwargs; class_prior и beta — фиксированные
внешние знания (не тюнятся Optuna, аналогично n_pos/n_neg у LDAMClassifier),
поэтому _make_loss переопределён, чтобы подмешать их к тюнящемуся gamma.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ml_toolkit.losses import NNPULoss as _NNPULoss
from ml_toolkit.presets.classification.high_pr_auc._custom_loss_base import (
    _CustomLossClassifierBase,
    _LossSpec,
)

if TYPE_CHECKING:
    from catboost import Pool
    import optuna
    from optuna.pruners import BasePruner


class NNPUClassifier(_CustomLossClassifierBase):
    """CatBoost с non-negative PU risk estimator вместо стандартного Logloss.

    Parameters
    ----------
    class_prior:
        pi = P(y=1) — истинная доля позитивов (включая незамеченные в U).
        Обязательный параметр — метод не имеет смысла без известного prior.
    beta:
        Порог срабатывания non-negative коррекции (обычно 0, не тюнится Optuna).
    gamma:
        Множитель "обратного" градиента при срабатывании коррекции.
    base_params:
        Параметры CatBoost (без loss_function — задаётся автоматически).
    n_optuna_trials:
        Число Optuna-триалов для подбора gamma и гиперпараметров CatBoost.
        0 → использовать base_params напрямую.
    param_space:
        Кастомная функция `f(trial) -> dict` — переопределяет search space для
        Optuna (и лосса, и архитектуры CatBoost, в одном пространстве). Любой
        отсутствующий в возвращённом словаре ключ (gamma или iterations/
        max_depth/learning_rate/l2_leaf_reg/subsample/min_data_in_leaf)
        тюнится дефолтным способом — можно переопределить как ни одного
        параметра (default space целиком), так и часть, так и все параметры
        сразу (и лосса, и модели). class_prior/beta в param_space не участвуют
        — фиксированные внешние знания, не тюнятся Optuna вообще. Действует
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

        model = NNPUClassifier(class_prior=0.05)
        model.fit(X_train, y_train, X_valid, y_valid)

    """

    _loss_spec = _LossSpec(
        loss_cls=_NNPULoss,
        param_bounds={'gamma': (0.1, 5.0)},
        name='NNPU',
    )

    def __init__(
        self,
        class_prior: float,
        beta: float = 0.0,
        gamma: float = 1.0,
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
        if not 0.0 < class_prior < 1.0:
            raise ValueError(f'class_prior должен быть в (0, 1), получено {class_prior}')
        super().__init__(
            loss_params={'gamma': gamma},
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
        self.class_prior = class_prior
        self.beta = beta
        self.gamma = gamma

    def _make_loss(self, loss_params: dict[str, float], *, tr_pool: Pool, arch_params: dict) -> _NNPULoss:
        return _NNPULoss(class_prior=self.class_prior, beta=self.beta, gamma=loss_params['gamma'])
