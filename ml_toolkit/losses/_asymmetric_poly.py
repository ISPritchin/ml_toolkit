"""AsymmetricPolyLoss: ASL (Ridnik et al., 2021) + Poly-1 (Leng et al., 2022) в одном лоссе.

Poly-1 добавляет к любому базовому CE-подобному лоссу линейный по p_t член
eps1*(1-p_t), не трогая базовую функцию — в PolyLoss (см. ml_toolkit/losses)
базой служит обычный CE; здесь та же поправка накладывается на ASL, давая ещё
одну степень свободы поверх уже имеющихся gamma_pos/gamma_neg/prob_margin.
Реализация переиспользует AsymmetricLoss.calc_ders_range как есть (без
повторной деривации ASL) и прибавляет производные линейного члена.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ml_toolkit.losses._asl import AsymmetricLoss


class AsymmetricPolyLoss:
    """CatBoost-совместимый ASL + Poly-1 для бинарной классификации.

    Parameters
    ----------
    gamma_pos, gamma_neg, prob_margin:
        Параметры базового ASL (см. AsymmetricLoss).
    eps1:
        Коэффициент линейного Poly-1 члена. Рекомендуется 1.0-3.0.

    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        prob_margin: float = 0.05,
        eps1: float = 2.0,
    ) -> None:
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.prob_margin = prob_margin
        self.eps1 = eps1
        self._asl = AsymmetricLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, prob_margin=prob_margin)

    def calc_ders_range(
        self,
        predictions: Sequence[float],
        targets: Sequence[float],
        weights: Sequence[float] | None,
    ) -> list[tuple[float, float]]:
        eps = 1e-7
        # Unweighted здесь намеренно (weights=None): внешний sample weight —
        # чистый множитель поверх per-row лосса (см. FocalLoss/AsymmetricLoss),
        # применяется один раз в конце, а не внутри ASL + отдельно к поправке.
        base = self._asl.calc_ders_range(predictions, targets, None)
        der1_base = np.array([d[0] for d in base], dtype=np.float64)
        der2_base = np.array([d[1] for d in base], dtype=np.float64)

        f = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        p = np.clip(1.0 / (1.0 + np.exp(-f)), eps, 1.0 - eps)
        sign = np.where(y == 1, 1.0, -1.0)

        # d/df[eps1*(1-p_t)]: derivative of the Poly-1 linear term, same form as in PolyLoss
        der1_poly = sign * self.eps1 * p * (1.0 - p)
        der2_poly = sign * self.eps1 * (1.0 - 2.0 * p) * p * (1.0 - p)

        der1 = der1_base + der1_poly
        der2 = np.minimum(der2_base + der2_poly, -eps)

        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            der1 = der1 * w
            der2 = der2 * w

        return list(zip(der1.tolist(), der2.tolist(), strict=False))
