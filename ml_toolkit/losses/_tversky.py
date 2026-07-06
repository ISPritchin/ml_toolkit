"""TverskyLoss: batch-level дифференцируемый Tversky index.

TI = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
L  = 1 - TI

alpha > beta → штрафуем FP → выше precision.
alpha < beta → штрафуем FN → выше recall.
alpha=beta=0.5 → Dice Loss.

Градиенты вычисляются по всему батчу одновременно (batch-level loss).
"""

from __future__ import annotations

import numpy as np


class TverskyLoss:
    """CatBoost-совместимая batch Tversky Loss.

    Parameters
    ----------
    alpha:
        Вес ложноположительных (FP). Меньше alpha → выше recall.
    beta:
        Вес ложноотрицательных (FN). Больше beta → выше recall.
    smooth:
        Коэффициент сглаживания для численной устойчивости.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def calc_ders_range(
        self, predictions, targets, weights
    ) -> list[tuple[float, float]]:
        eps = 1e-7
        f = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)

        p = np.clip(1.0 / (1.0 + np.exp(-f)), eps, 1.0 - eps)

        tp = np.sum(p * y)
        fp = np.sum(p * (1.0 - y))
        fn = np.sum((1.0 - p) * y)
        D = tp + self.alpha * fp + self.beta * fn + self.smooth
        N = tp + self.smooth

        # ∂L/∂p_i:
        #   y_i=1: -(D - N*(1-beta)) / D²
        #   y_i=0:  N*alpha / D²
        pos = y == 1
        dL_dp = np.empty_like(p)
        dL_dp[pos] = -(D - N * (1.0 - self.beta)) / (D * D)
        dL_dp[~pos] = N * self.alpha / (D * D)

        # chain rule: ∂L/∂f_i = ∂L/∂p_i * p_i*(1-p_i)
        dL_df = dL_dp * p * (1.0 - p)

        der1 = -dL_df
        # Аппроксимация диагонального гессиана через CE-гессиан
        der2 = np.minimum(-(p * (1.0 - p)), -eps)

        return list(zip(der1.tolist(), der2.tolist()))
