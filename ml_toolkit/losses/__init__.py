"""ml_toolkit.losses — CatBoost-совместимые функции потерь для задач с дисбалансом.

Все классы реализуют интерфейс calc_ders_range(predictions, targets, weights)
и передаются напрямую в CatBoostClassifier(loss_function=...), кроме
мультиклассовых (calc_ders_multi) — см. ниже.

Экспортируемые классы (бинарная классификация, calc_ders_range)
-----------------------------------------------------------------
FocalLoss              — фокусировка на трудных примерах (единый gamma).
AsymmetricLoss         — разные γ+/γ- + prob_margin для негативов (ASL).
LabelSmoothingLoss     — CE со сглаженными метками (калибровка / зашумлённость).
PolyLoss               — Poly-1: CE + eps1*(1-p_t).
TverskyLoss            — batch Tversky index (управление precision/recall).
DiceLoss               — batch Dice/soft-F1 (Tversky c alpha=beta=0.5).
LDAMLoss               — Label-Distribution-Aware Margin + Deferred Re-Weighting.
GHMLoss                — Gradient Harmonizing Mechanism (подавляет лёгкие И выбросы).
InfluenceBalancedLoss  — по-сэмпловый вес по обратному |grad| + class-balanced.
AsymmetricPolyLoss     — ASL + Poly-1 поправка в одном лоссе.
NNPULoss               — non-negative PU risk estimator (Kiryo et al., 2017).

Экспортируемые классы (мультикласс, calc_ders_multi)
-----------------------------------------------------------------
EqualizationLoss       — Seesaw/EQLv2-style: подавление головных классов на редкие.
BalancedSoftmaxLoss    — logit adjustment на log(class_prior) во время обучения.
LogitNormLoss          — нормализация логитов перед CE (против overconfidence).

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
from ._asymmetric_poly import AsymmetricPolyLoss
from ._balanced_softmax import BalancedSoftmaxLoss
from ._dice import DiceLoss
from ._equalization import EqualizationLoss
from ._focal import FocalLoss
from ._ghm import GHMLoss
from ._ib import InfluenceBalancedLoss
from ._label_smoothing import LabelSmoothingLoss
from ._ldam import LDAMLoss
from ._logitnorm import LogitNormLoss
from ._nnpu import NNPULoss
from ._poly import PolyLoss
from ._tversky import TverskyLoss

__all__ = [
    'FocalLoss',
    'AsymmetricLoss',
    'LabelSmoothingLoss',
    'PolyLoss',
    'TverskyLoss',
    'DiceLoss',
    'LDAMLoss',
    'GHMLoss',
    'InfluenceBalancedLoss',
    'AsymmetricPolyLoss',
    'NNPULoss',
    'EqualizationLoss',
    'BalancedSoftmaxLoss',
    'LogitNormLoss',
]
