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

from typing import Any

from ml_toolkit.losses import NNPULoss as _NNPULoss
from ml_toolkit.presets.classification.high_pr_auc._custom_loss_base import (
    _CustomLossClassifierBase,
    _LossSpec,
)


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
        optuna_timeout: int | None = None,
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
            optuna_timeout=optuna_timeout,
            random_seed=random_seed,
            cat_features=cat_features,
            selected_features=selected_features,
        )
        self.class_prior = class_prior
        self.beta = beta
        self.gamma = gamma

    def _make_loss(self, loss_params: dict[str, float], *, tr_pool: Any, arch_params: dict) -> _NNPULoss:
        return _NNPULoss(class_prior=self.class_prior, beta=self.beta, gamma=loss_params['gamma'])
