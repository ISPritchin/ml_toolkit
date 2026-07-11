"""LabelSmoothingLoss: бинарный CE со сглаженными метками.

y → y*(1-eps) + eps/2. Снижает overconfidence, улучшает калибровку
и часто помогает обобщению при зашумлённой разметке.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class LabelSmoothingLoss:
    """CatBoost-совместимый бинарный CE с label smoothing.

    Parameters
    ----------
    eps:
        Степень сглаживания: 0.1 → целевые метки [0.05, 0.95].
        Рекомендуется 0.05–0.15.

    """

    def __init__(self, eps: float = 0.1) -> None:
        if not 0.0 <= eps < 0.5:
            raise ValueError('eps должен быть в [0, 0.5)')
        self.eps = eps

    def calc_ders_range(
        self,
        predictions: Sequence[float],
        targets: Sequence[float],
        weights: Sequence[float] | None,
    ) -> list[tuple[float, float]]:
        eps_clip = 1e-7
        f = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)

        p = np.clip(1.0 / (1.0 + np.exp(-f)), eps_clip, 1.0 - eps_clip)
        y_s = y * (1.0 - self.eps) + self.eps * 0.5

        # dL/df = p - y_s  →  der1 = y_s - p
        der1 = y_s - p
        der2 = -(p * (1.0 - p))

        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            der1 = der1 * w
            der2 = der2 * w

        return list(zip(der1.tolist(), der2.tolist(), strict=False))
