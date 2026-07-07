"""ElkanNotoHoldoutPU: PULearningClassifier + bootstrap-CI для c.

PULearningClassifier (см. pu_learning.py) уже реализует Элкан-Ното с
c-holdout, но выдаёт только точечную оценку c = mean(raw_c[y_c==1]). При
малом числе c-holdout позитивов эта точечная оценка сама по себе шумная — c
может быть смещено, и точечная коррекция proba/c создаёт ложное ощущение
точности. Bootstrap (n_bootstrap ресэмплов c-holdout позитивов с возвратом)
даёт доверительный интервал вместо одного числа: если интервал широк
относительно c_, это прямой сигнал "не доверяйте абсолютным вероятностям
этой модели, доверяйте только ранжированию".

Переопределяется только _estimate_c (см. соответствующий метод в
PULearningClassifier) — остальной fit/predict/Optuna унаследованы как есть.
"""

from __future__ import annotations

import logging

import numpy as np

from ml_toolkit.presets.classification.high_pr_auc.pu_learning import (
    PULearningClassifier,
)

logger = logging.getLogger(__name__)


class ElkanNotoHoldoutPU(PULearningClassifier):
    """PULearningClassifier с bootstrap доверительным интервалом для c.

    Parameters
    ----------
    c_holdout_frac:
        То же самое, что c_estimation_frac у PULearningClassifier — доля
        валидации под оценку c (здесь — под её bootstrap-версию).
    n_bootstrap:
        Число bootstrap-ресэмплов c-holdout позитивов для CI.
    Остальные параметры — см. PULearningClassifier.

    Атрибуты после fit (в дополнение к атрибутам PULearningClassifier)::

        c_ci_        — (c_lower, c_upper), персентильный 95% CI
        c_bootstrap_std_ — std бутстрап-распределения c

    Пример::

        model = ElkanNotoHoldoutPU(c_holdout_frac=0.3, n_bootstrap=100)
        model.fit(X_train, y_train, X_valid, y_valid)
        print(f"c={model.c_:.3f}  95% CI={model.c_ci_}")

    """

    def __init__(
        self,
        c_holdout_frac: float = 0.3,
        n_bootstrap: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(c_estimation_frac=c_holdout_frac, **kwargs)
        if n_bootstrap < 10:
            raise ValueError(f'n_bootstrap должен быть >= 10, получено {n_bootstrap}')
        self.c_holdout_frac = c_holdout_frac
        self.n_bootstrap = n_bootstrap

        self.c_ci_: tuple[float, float] = (0.0, 0.0)
        self.c_bootstrap_std_: float = 0.0

    def _estimate_c(self, raw_c: np.ndarray, y_c: np.ndarray) -> None:
        super()._estimate_c(raw_c, y_c)

        pos_scores = raw_c[y_c == 1]
        if len(pos_scores) < 2:
            logger.warning('[ElkanNotoHoldoutPU] < 2 позитивов в c-holdout — CI не строится')
            self.c_ci_ = (self.c_, self.c_)
            self.c_bootstrap_std_ = 0.0
            return

        rng = np.random.default_rng(self.random_seed)
        boot_c = np.empty(self.n_bootstrap)
        for b in range(self.n_bootstrap):
            sample = rng.choice(pos_scores, size=len(pos_scores), replace=True)
            boot_c[b] = sample.mean()

        self.c_ci_ = (float(np.percentile(boot_c, 2.5)), float(np.percentile(boot_c, 97.5)))
        self.c_bootstrap_std_ = float(boot_c.std())

        rel_width = (self.c_ci_[1] - self.c_ci_[0]) / max(self.c_, 1e-6)
        logger.info(
            '[ElkanNotoHoldoutPU] c=%.4f  95%% CI=[%.4f, %.4f]  std=%.4f (по %d позитивам, %d bootstrap)',
            self.c_, self.c_ci_[0], self.c_ci_[1], self.c_bootstrap_std_,
            len(pos_scores), self.n_bootstrap,
        )
        if rel_width > 0.5:
            logger.warning(
                '[ElkanNotoHoldoutPU] Относительная ширина CI=%.1f%% > 50%% — '
                'оценка c нестабильна, доверяйте ранжированию, а не абсолютным '
                'вероятностям (см. докстринг PULearningClassifier).',
                100.0 * rel_width,
            )
