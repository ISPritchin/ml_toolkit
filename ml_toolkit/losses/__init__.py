"""ml_toolkit.losses — CatBoost-совместимые функции потерь для задач с дисбалансом.

Все классы реализуют интерфейс calc_ders_range(predictions, targets, weights)
и передаются напрямую в CatBoostClassifier(loss_function=...).

Экспортируемые классы
---------------------
FocalLoss          — фокусировка на трудных примерах (единый gamma).
AsymmetricLoss     — разные γ+/γ- + prob_margin для негативов (ASL).
LabelSmoothingLoss — CE со сглаженными метками (калибровка / зашумлённость).
PolyLoss           — Poly-1: CE + eps1*(1-p_t).
TverskyLoss        — batch Tversky index (управление precision/recall).
LDAMLoss           — Label-Distribution-Aware Margin + Deferred Re-Weighting.

Быстрый старт::

    from ml_toolkit.losses import FocalLoss, TverskyLoss
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        loss_function=FocalLoss(gamma=2.0, alpha=0.25),
        eval_metric='AUC',
        iterations=500,
    )
"""

from ._asl import AsymmetricLoss
from ._focal import FocalLoss
from ._label_smoothing import LabelSmoothingLoss
from ._ldam import LDAMLoss
from ._poly import PolyLoss
from ._tversky import TverskyLoss

__all__ = [
    'FocalLoss',
    'AsymmetricLoss',
    'LabelSmoothingLoss',
    'PolyLoss',
    'TverskyLoss',
    'LDAMLoss',
]
