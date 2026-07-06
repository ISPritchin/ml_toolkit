"""FocalLoss для CatBoost: фокусировка на трудных примерах.

FL = -alpha_t * (1-p_t)^gamma * log(p_t)

gamma >= 1: фокус на трудных примерах; gamma=0 → взвешенный CE.
alpha: вес позитивных примеров (0.25 при сильном дисбалансе).
"""

from __future__ import annotations

import numpy as np


class FocalLoss:
    """CatBoost-совместимая Focal Loss для бинарной классификации.

    Parameters
    ----------
    gamma:
        Фокусирующий параметр (>= 1). Чем выше, тем сильнее подавляются
        «лёгкие» примеры.
    alpha:
        Вес класса 1 (позитивы). 1-alpha — вес класса 0.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        if gamma < 1.0:
            raise ValueError("gamma < 1 инвертирует фокусировку — используйте gamma >= 1")
        self.gamma = gamma
        self.alpha = alpha

    def calc_ders_range(
        self, predictions, targets, weights
    ) -> list[tuple[float, float]]:
        eps = 1e-7
        f = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)

        p = np.clip(1.0 / (1.0 + np.exp(-f)), eps, 1.0 - eps)
        p_t = np.where(y == 1, p, 1.0 - p)
        alpha_t = np.where(y == 1, self.alpha, 1.0 - self.alpha)

        # dFL/dp_t = alpha_t * (1-p_t)^{gamma-1} * [gamma*log(p_t) - (1-p_t)/p_t]
        dfl_dp_t = alpha_t * (1.0 - p_t) ** (self.gamma - 1) * (
            self.gamma * np.log(p_t + eps) - (1.0 - p_t) / (p_t + eps)
        )
        dp_t_dp = np.where(y == 1, 1.0, -1.0)
        dfl_df = dfl_dp_t * dp_t_dp * p * (1.0 - p)

        der1 = -dfl_df
        der2 = -(alpha_t * (1.0 - p_t) ** self.gamma * p * (1.0 - p))

        return list(zip(der1.tolist(), der2.tolist()))
