"""PolyLoss (Poly-1, Leng et al. 2022): CE + eps1*(1-p_t).

Линейное расширение бинарного CE через полиномиальное разложение.
eps1 > 0 усиливает фокус на трудных примерах (аналогично Focal Loss),
eps1 < 0 — акцент на уверенных, eps1=0 — стандартный CE.
"""

from __future__ import annotations

import numpy as np


class PolyLoss:
    """CatBoost-совместимый Poly-1 Loss для бинарной классификации.

    Parameters
    ----------
    eps1:
        Коэффициент линейного члена. Рекомендуется 1.0–3.0 при дисбалансе.

    """

    def __init__(self, eps1: float = 2.0) -> None:
        self.eps1 = eps1

    def calc_ders_range(
        self, predictions, targets, weights
    ) -> list[tuple[float, float]]:
        eps = 1e-7
        f = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)

        p = np.clip(1.0 / (1.0 + np.exp(-f)), eps, 1.0 - eps)
        pos = y == 1

        der1 = np.empty_like(p)
        der2 = np.empty_like(p)

        # y=1: L = -log(p) + eps1*(1-p)
        # dL/df = -(1-p)*(1 + eps1*p)  →  der1 = (1-p)*(1 + eps1*p)
        # d²L/df² = p*(1-p)*(1 + eps1*(2p-1))
        q_pos = 1.0 - p[pos]
        der1[pos] = q_pos * (1.0 + self.eps1 * p[pos])
        der2[pos] = -(p[pos] * q_pos * (1.0 + self.eps1 * (2.0 * p[pos] - 1.0)))

        # y=0: L = -log(1-p) + eps1*p
        # dL/df = p*(1 + eps1*(1-p))  →  der1 = -p*(1 + eps1*(1-p))
        # d²L/df² = p*(1-p)*(1 + eps1*(1-2p))
        q_neg = 1.0 - p[~pos]
        der1[~pos] = -p[~pos] * (1.0 + self.eps1 * q_neg)
        der2[~pos] = -(p[~pos] * q_neg * (1.0 + self.eps1 * (1.0 - 2.0 * p[~pos])))

        # Гессиан должен быть отрицательным (выпуклый лосс)
        clip_eps = 1e-7
        der2 = np.minimum(der2, -clip_eps)

        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            der1 = der1 * w
            der2 = der2 * w

        return list(zip(der1.tolist(), der2.tolist()))
