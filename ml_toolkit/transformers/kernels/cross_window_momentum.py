"""Зум-спектр средних на разных горизонтах: mean_w1/w3/w6/w24 и флаги ускорения.

Signal:
    Иерархия соотношений «текущее к краткосрочному», «краткосрочное к среднесрочному»,
    «среднесрочное к долгосрочному» образует «зум-спектр» ускорения. Все ratios > 1
    одновременно сигнализируют о мощном и устойчивом ускорении на всех горизонтах.

Formula:
    ratio_w1_w3  = v[t] / (|mean_w3| + eps)
    ratio_w3_w6  = mean_w3 / (|mean_w6| + eps)
    ratio_w6_w24 = mean_w6 / (|mean_w24| + eps)
    all_accel    = 1 if ratio_w1_w3 > 1 AND ratio_w3_w6 > 1 AND ratio_w6_w24 > 1
    all_decel    = 1 if ratio_w1_w3 < 1 AND ratio_w3_w6 < 1 AND ratio_w6_w24 < 1
    horizon_spread = ln(|mean_w3| / |mean_w24|), 0 если одно из средних ~ 0

Outputs:
    {product}__cross_window_momentum__ratio_w1_w3    — текущее / среднее 3 мес
    {product}__cross_window_momentum__ratio_w3_w6    — среднее 3 / среднее 6 мес
    {product}__cross_window_momentum__ratio_w6_w24   — среднее 6 / среднее 24 мес
    {product}__cross_window_momentum__all_accel      — флаг полного ускорения
    {product}__cross_window_momentum__all_decel      — флаг полного замедления
    {product}__cross_window_momentum__horizon_spread — лог-разрыв кратко-/долгосрочного

Preset (monthly.yaml):
    cross_window_momentum: {}

Interpretation:
    all_accel = 1 — «бычья» структура на всех горизонтах; сильный кандидат на рост класса.
    all_decel = 1 — системное замедление; потенциальный риск снижения категории.
    horizon_spread > 1 — недавние месяцы значительно выше двухлетнего среднего.
    ratio_w1_w3 < 1, остальные > 1 — краткосрочная просадка на фоне долгосрочного роста.

Example:
    Ряд (6 мес): [10, 20, 30, 40, 50, 60]
    (t=5; ws24=ws6=6, т.к. истории всего 6 мес)

    mean_w3 = (40+50+60)/3 = 50,  mean_w6 = 35,  mean_w24 = 35,  v[t] = 60
    ratio_w1_w3  = 60/50 = 1.2
    ratio_w3_w6  = 50/35 = 1.429
    ratio_w6_w24 = 35/35 = 1.0
    horizon_spread = ln(50/35) = 0.357
    → cross_window_momentum__ratio_w1_w3 = 1.2, ratio_w3_w6 = 1.429, horizon_spread = 0.357

"""

import numba as nb
import numpy as np

from .._windowing import compute_window_mean, resolve_window_size, safe_ratio

FEATURE = 'cross_window_momentum'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray):
    n_rows = product_values.shape[0]
    out_r_w1_w3 = np.zeros(n_rows)
    out_r_w3_w6 = np.zeros(n_rows)
    out_r_w6_w24 = np.zeros(n_rows)
    out_all_accel = np.zeros(n_rows)
    out_all_decel = np.zeros(n_rows)
    out_horizon_spread = np.zeros(n_rows)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        ws3 = resolve_window_size(pos, 3)
        ws6 = resolve_window_size(pos, 6)
        ws24 = resolve_window_size(pos, 24)

        mean3 = compute_window_mean(product_values, row_idx, ws3)
        mean6 = compute_window_mean(product_values, row_idx, ws6)
        mean24 = compute_window_mean(product_values, row_idx, ws24)
        v_now = product_values[row_idx]

        r_w1_w3 = safe_ratio(v_now, mean3)
        r_w3_w6 = safe_ratio(mean3, mean6)
        r_w6_w24 = safe_ratio(mean6, mean24)

        out_r_w1_w3[row_idx] = r_w1_w3
        out_r_w3_w6[row_idx] = r_w3_w6
        out_r_w6_w24[row_idx] = r_w6_w24

        all_accel = 1.0 if r_w1_w3 > 1.0 and r_w3_w6 > 1.0 and r_w6_w24 > 1.0 else 0.0
        all_decel = 1.0 if r_w1_w3 < 1.0 and r_w3_w6 < 1.0 and r_w6_w24 < 1.0 else 0.0
        out_all_accel[row_idx] = all_accel
        out_all_decel[row_idx] = all_decel

        # спред не определён при нулевом mean3/mean24 -> 0 (раньше log(eps) ~ -20.7)
        spread_ratio = safe_ratio(abs(mean3), mean24)
        out_horizon_spread[row_idx] = np.log(spread_ratio) if spread_ratio > 0.0 else 0.0

    return out_r_w1_w3, out_r_w3_w6, out_r_w6_w24, out_all_accel, out_all_decel, out_horizon_spread


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {} (no params)"""
    r13, r36, r624, aa, ad, hs = _kernel(values, position)
    return (
        [r13, r36, r624, aa, ad, hs],
        ['ratio_w1_w3', 'ratio_w3_w6', 'ratio_w6_w24', 'all_accel', 'all_decel', 'horizon_spread'],
    )
