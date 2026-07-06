"""Отношения коэффициентов вариации (CV) на фиксированных горизонтах: эволюция нестабильности.

Signal:
    Набор признаков, сравнивающих CV (std/|mean|) на горизонтах 3, 6, 12, 24 месяца.
    Позволяет определить, на каком масштабе концентрирован «хаос»: если CV_3 >> CV_12 —
    нестабильность краткосрочная, на фоне стабильного долгосрочного тренда. vol_accel —
    ускорение роста волатильности (вторая разница std). regime_flag — бинарный детектор
    краткосрочного «шторма» (CV_3 > 2 * CV_12).

Formula:
    CV_w = std_w / (|mean_w| + eps)

    cv_ratio_w3_w6  = CV_3  / (CV_6  + eps)
    cv_ratio_w3_w12 = CV_3  / (CV_12 + eps)
    cv_ratio_w6_w24 = CV_6  / (CV_24 + eps)
    vol_accel       = (std_3 - std_6) - (std_6 - std_12)
    short_excess    = (CV_3 - CV_12) / (CV_12 + eps)
    regime_flag     = 1 if CV_3 > 2 * CV_12 else 0

Outputs:
    {product}__window_volatility_ratios__cv_ratio_w3_w6   — CV_3 / CV_6
    {product}__window_volatility_ratios__cv_ratio_w3_w12  — CV_3 / CV_12
    {product}__window_volatility_ratios__cv_ratio_w6_w24  — CV_6 / CV_24
    {product}__window_volatility_ratios__vol_accel        — ускорение нарастания std
    {product}__window_volatility_ratios__short_excess     — избыточная краткосрочная волатильность
    {product}__window_volatility_ratios__regime_flag      — флаг краткосрочного хаоса

Preset (monthly.yaml):
    window_volatility_ratios: {}

Interpretation:
    cv_ratio_w3_w12 > 2 — последние 3 месяца в 2 раза нестабильнее годового фона.
    regime_flag = 1 — экстремальный краткосрочный режим (CV_3 более чем вдвое выше CV_12).
    vol_accel > 0 — нестабильность нарастает со всё большим темпом (ускорение хаоса).
    cv_ratio_w6_w24 < 1 — полугодие стабильнее двухлетней истории (долгосрочный хаос остался позади).

Example:
    Ряд (6 мес): [20, 20, 20, 10, 40, 10]
    (истории 6 мес, поэтому ws12=ws24=ws6=6)

    CV_3 (посл. 3 [10,40,10]): mean=20, std=14.142 → CV = 0.707
    CV_6 (все 6):              mean=20, std=10.0  → CV = 0.5
    cv_ratio_w3_w6 = 0.707 / 0.5 = 1.414
    short_excess = (CV_3 − CV_12)/CV_12 = (0.707 − 0.5)/0.5 = 0.414
    → window_volatility_ratios__cv_ratio_w3_w6 = 1.414,  short_excess = 0.414
"""

import numba as nb
import numpy as np

from .._windowing import compute_window_mean_and_std, resolve_window_size, safe_ratio

FEATURE = "window_volatility_ratios"


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray):
    n_rows = product_values.shape[0]
    out_cv3_cv6 = np.zeros(n_rows)
    out_cv3_cv12 = np.zeros(n_rows)
    out_cv6_cv24 = np.zeros(n_rows)
    out_vol_accel = np.zeros(n_rows)
    out_short_excess = np.zeros(n_rows)
    out_regime_flag = np.zeros(n_rows)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        ws3 = resolve_window_size(pos, 3)
        ws6 = resolve_window_size(pos, 6)
        ws12 = resolve_window_size(pos, 12)
        ws24 = resolve_window_size(pos, 24)

        mean3, std3 = compute_window_mean_and_std(product_values, row_idx, ws3)
        mean6, std6 = compute_window_mean_and_std(product_values, row_idx, ws6)
        mean12, std12 = compute_window_mean_and_std(product_values, row_idx, ws12)
        mean24, std24 = compute_window_mean_and_std(product_values, row_idx, ws24)

        cv3 = safe_ratio(std3, mean3)
        cv6 = safe_ratio(std6, mean6)
        cv12 = safe_ratio(std12, mean12)
        cv24 = safe_ratio(std24, mean24)

        out_cv3_cv6[row_idx] = safe_ratio(cv3, cv6)
        out_cv3_cv12[row_idx] = safe_ratio(cv3, cv12)
        out_cv6_cv24[row_idx] = safe_ratio(cv6, cv24)
        out_vol_accel[row_idx] = (std3 - std6) - (std6 - std12)
        out_short_excess[row_idx] = safe_ratio(cv3 - cv12, cv12)
        out_regime_flag[row_idx] = 1.0 if cv3 > cv12 * 2.0 else 0.0

    return out_cv3_cv6, out_cv3_cv12, out_cv6_cv24, out_vol_accel, out_short_excess, out_regime_flag


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {} (no params)"""
    cv3_cv6, cv3_cv12, cv6_cv24, va, se, rf = _kernel(values, position)
    return (
        [cv3_cv6, cv3_cv12, cv6_cv24, va, se, rf],
        ["cv_ratio_w3_w6", "cv_ratio_w3_w12", "cv_ratio_w6_w24", "vol_accel", "short_excess", "regime_flag"],
    )
