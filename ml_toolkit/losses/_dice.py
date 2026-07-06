"""DiceLoss: batch Dice/soft-F1 loss для CatBoost.

Dice index — частный случай Tversky index (см. TverskyLoss) при alpha=beta=0.5
(FP и FN штрафуются одинаково — прямая мягкая аппроксимация F1, а не
произвольный precision/recall трейдофф). Соответствие явно задокументировано
в TverskyLoss; DiceLoss переиспользует её градиенты один-в-один, вместо
повторной деривации того же дифференцирования.
"""

from __future__ import annotations

from ml_toolkit.losses._tversky import TverskyLoss


class DiceLoss(TverskyLoss):
    """CatBoost-совместимая batch Dice Loss (Tversky c alpha=beta=0.5).

    Parameters
    ----------
    smooth:
        Коэффициент сглаживания для численной устойчивости.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__(alpha=0.5, beta=0.5, smooth=smooth)
