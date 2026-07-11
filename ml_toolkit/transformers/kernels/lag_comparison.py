"""Квартальные (lag-3), lag-9 и годовые (lag-12) сравнения с ускорением YoY.

Signal:
    Сравнивает текущий месяц с тем, что было квартал, три квартала и год назад.
    YoY (lag12_ratio) — ключевой индикатор годовой динамики; lag3_trend — устойчивость
    квартального роста; yoy_accel — ускоряется ли YoY-рост.

Formula:
    lag3_ratio = v[t] / (|v[t-3]| + eps) - 1
    lag9_ratio = v[t] / (|v[t-9]| + eps) - 1
    lag12_ratio = v[t] / (|v[t-12]| + eps) - 1
    lag3_trend  = mean(lag3_ratio[t], lag3_ratio[t-1], lag3_ratio[t-2])
    lag12_consistency = std(lag12_ratio[t], lag12_ratio[t-1], lag12_ratio[t-2])
    yoy_accel   = lag12_ratio[t] - lag12_ratio[t-6]

    Все нули при недостаточной истории (pos < lag).

Outputs:
    {product}__lag_comparison__lag3_ratio         — рост vs 3 мес назад
    {product}__lag_comparison__lag9_ratio         — рост vs 9 мес назад
    {product}__lag_comparison__lag12_ratio        — рост vs 12 мес назад (YoY)
    {product}__lag_comparison__lag3_trend         — средний квартальный рост
    {product}__lag_comparison__lag12_consistency  — σ YoY-роста (стабильность YoY)
    {product}__lag_comparison__yoy_accel          — ускорение YoY за полгода

Preset entry:
    lag_comparison: {}

Interpretation:
    lag12_ratio = +1.0 — значение удвоилось год к году.
    lag12_consistency = 0 — стабильный YoY-рост без колебаний.
    yoy_accel > 0 — YoY-рост ускоряется (всё лучше год к году).
    lag3_ratio > 0, lag12_ratio < 0 — краткосрочный отскок при годовом спаде.

Example:
    Ряд (4 мес): [10, 20, 30, 40]
    (t=3; доступен только лаг 3, истории < 9 мес)

    lag3_ratio = v[t] / v[t-3] − 1 = 40/10 − 1 = 3.0  (рост ×4 за квартал)
    lag9_ratio = lag12_ratio = 0 (недостаточно истории)
    → lag_comparison__lag3_ratio = 3.0

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import EPS, safe_ratio

FEATURE = 'lag_comparison'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray):
    n_rows = product_values.shape[0]
    out_lag3 = np.zeros(n_rows)
    out_lag9 = np.zeros(n_rows)
    out_lag12 = np.zeros(n_rows)
    out_lag3_trend = np.zeros(n_rows)
    out_lag12_consistency = np.zeros(n_rows)
    out_yoy_accel = np.zeros(n_rows)

    # running: последние 3 значения lag3_ratio и lag12_ratio
    lag3_hist = np.zeros(3)
    lag12_hist = np.zeros(3)
    lag3_h_count = 0
    lag12_h_count = 0

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        if pos == 0:
            lag3_hist[:] = 0.0
            lag12_hist[:] = 0.0
            lag3_h_count = 0
            lag12_h_count = 0

        v = product_values[row_idx]

        # при нулевой базе (v[t-k] ~ 0) рост не определён -> 0, а не v/eps
        r3 = 0.0
        r9 = 0.0
        r12 = 0.0
        if pos >= 3:
            v3 = product_values[row_idx - 3]
            if abs(v3) > EPS:
                r3 = safe_ratio(v, v3) - 1.0
        if pos >= 9:
            v9 = product_values[row_idx - 9]
            if abs(v9) > EPS:
                r9 = safe_ratio(v, v9) - 1.0
        if pos >= 12:
            v12 = product_values[row_idx - 12]
            if abs(v12) > EPS:
                r12 = safe_ratio(v, v12) - 1.0

        out_lag3[row_idx] = r3
        out_lag9[row_idx] = r9
        out_lag12[row_idx] = r12

        # lag3_trend: mean последних 3 lag3_ratio
        if lag3_h_count >= 3:
            lag3_hist[0] = lag3_hist[1]
            lag3_hist[1] = lag3_hist[2]
            lag3_hist[2] = r3
        else:
            lag3_hist[lag3_h_count] = r3
            lag3_h_count += 1
        n3 = min(lag3_h_count, 3)
        mean3 = 0.0
        for i in range(n3):
            mean3 += lag3_hist[i]
        out_lag3_trend[row_idx] = mean3 / max(n3, 1)

        # lag12_consistency: std последних 3 lag12_ratio
        if pos >= 12:
            if lag12_h_count >= 3:
                lag12_hist[0] = lag12_hist[1]
                lag12_hist[1] = lag12_hist[2]
                lag12_hist[2] = r12
            else:
                lag12_hist[lag12_h_count] = r12
                lag12_h_count += 1
        n12 = min(lag12_h_count, 3)
        if n12 >= 2:
            m = 0.0
            for i in range(n12):
                m += lag12_hist[i]
            m /= n12
            var = 0.0
            for i in range(n12):
                var += (lag12_hist[i] - m) ** 2
            out_lag12_consistency[row_idx] = (var / n12) ** 0.5
        else:
            out_lag12_consistency[row_idx] = 0.0

        # yoy_accel: lag12_ratio[t] - lag12_ratio[t-6]
        if pos >= 18:
            v12_t6 = product_values[row_idx - 6]
            v12_t6_lag = product_values[row_idx - 18]
            r12_t6 = safe_ratio(v12_t6, v12_t6_lag) - 1.0 if abs(v12_t6_lag) > EPS else 0.0
            out_yoy_accel[row_idx] = r12 - r12_t6
        else:
            out_yoy_accel[row_idx] = 0.0

    return out_lag3, out_lag9, out_lag12, out_lag3_trend, out_lag12_consistency, out_yoy_accel


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {} (no params needed)."""
    r3, r9, r12, t3, c12, accel = _kernel(values, position)
    return (
        [r3, r9, r12, t3, c12, accel],
        ['lag3_ratio', 'lag9_ratio', 'lag12_ratio', 'lag3_trend', 'lag12_consistency', 'yoy_accel'],
    )
