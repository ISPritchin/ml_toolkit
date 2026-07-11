"""Позиция в жизненном цикле клиента: возраст пика, completeness, фаза (разгон/зрелость/спад).

Signal:
    Определяет, в какой фазе жизненного цикла находится клиент: разгон (доходы ещё не
    достигли пика), зрелость (≥80% исторического максимума), или снижение (ниже пика).
    Помогает предсказывать будущую классификацию на основе траектории.

Formula:
    running_max[t] = max(v[0..t])    (первое значение сущности — стартовый пик)
    peak_age_share  = pos_at_peak / (current_pos + 1)
    post_peak_share = 1 - peak_age_share
    completeness    = safe_ratio(v[t], running_max)  (0 при running_max ~ 0)
    ramp_norm       = pos_of_first_half_max / (current_pos + 1)
    is_new_peak     = 1 if v[t] > running_max[t-1] (и на pos=0 — тривиальный пик)
    phase_flag      = 0 (разгон) | 1 (зрелость, v >= 0.8*max) | 2 (снижение)
    post_peak_slope_w = OLS_slope * (-sign(v[t] - running_max))

Outputs:
    {product}__lifecycle_phase__peak_age_share      — доля истории до пика
    {product}__lifecycle_phase__post_peak_share     — доля после пика
    {product}__lifecycle_phase__completeness        — v[t] / all-time max
    {product}__lifecycle_phase__ramp_norm           — нормированное время разгона
    {product}__lifecycle_phase__is_new_peak         — флаг нового исторического пика
    {product}__lifecycle_phase__phase_flag          — фаза: 0/1/2
    {product}__lifecycle_phase__post_peak_slope_w12 — скорость снижения от пика

Preset (monthly.yaml):
    lifecycle_phase:
      windows: [12]

Interpretation:
    phase_flag = 0, completeness = 0.6 — клиент ещё набирает обороты, не достиг пика.
    phase_flag = 1, completeness = 0.95 — стабильная зрелость, высокое плато.
    phase_flag = 2, post_peak_share = 0.6 — на спаде уже 60% всей истории.
    is_new_peak = 1 — положительный сигнал: продолжает устанавливать рекорды.

Example:
    Ряд (6 мес): [10, 20, 30, 40, 50, 60]
    (t=5; running_max обновляется каждый мес.)

    running_max = 60, достигнут на pos=5,  v[t]=60
    completeness = 60/60 = 1.0  (≥ 0.8 → фаза «зрелость»)
    peak_age_share = pos_at_peak/(pos+1) = 5/6 = 0.833
    is_new_peak = 1 (новый рекорд)
    → lifecycle_phase__completeness = 1.0,  phase_flag = 1,  peak_age_share = 0.833

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import EPS, fit_linear_trend_slope, resolve_window_size, safe_ratio

FEATURE = 'lifecycle_phase'


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    windows: np.ndarray,
    maturity_threshold: float,
    ramp_threshold: float,
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]

    out_peak_age_share = np.zeros(n_rows)
    out_post_peak_share = np.zeros(n_rows)
    out_completeness = np.zeros(n_rows)
    out_ramp_norm = np.zeros(n_rows)
    out_is_new_peak = np.zeros(n_rows)
    out_phase = np.zeros(n_rows)
    out_post_peak_slope = np.zeros((n_w, n_rows))

    r_alltime_max = 0.0
    r_alltime_max_pos = 0
    r_half_max_reached_pos = -1

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        v = product_values[row_idx]
        is_new_peak = 0.0
        if pos == 0:
            # первое наблюдение сущности — стартовый пик (корректно и для
            # нулевых/отрицательных рядов: раньше max оставался 0.0)
            r_alltime_max = v
            r_alltime_max_pos = 0
            r_half_max_reached_pos = -1
            is_new_peak = 1.0
        elif v > r_alltime_max:
            r_alltime_max = v
            r_alltime_max_pos = pos
            is_new_peak = 1.0

        if r_half_max_reached_pos < 0 and v >= r_alltime_max * ramp_threshold and r_alltime_max > EPS:
            r_half_max_reached_pos = pos

        peak_age_share = r_alltime_max_pos / (pos + 1)
        post_peak_share = 1.0 - peak_age_share
        completeness = safe_ratio(v, r_alltime_max)
        ramp_norm = (r_half_max_reached_pos / (pos + 1)) if r_half_max_reached_pos >= 0 else 0.0

        if completeness >= maturity_threshold:
            phase = 1.0  # зрелость
        elif pos > r_alltime_max_pos:
            phase = 2.0  # снижение
        else:
            phase = 0.0  # разгон

        out_peak_age_share[row_idx] = peak_age_share
        out_post_peak_share[row_idx] = post_peak_share
        out_completeness[row_idx] = completeness
        out_ramp_norm[row_idx] = ramp_norm
        out_is_new_peak[row_idx] = is_new_peak
        out_phase[row_idx] = phase

        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            s = fit_linear_trend_slope(product_values, row_idx, ws)
            # post_peak_slope: скорость снижения от пика (>0 если падаем)
            sign_val = -1.0 if v < r_alltime_max else 1.0
            out_post_peak_slope[j, row_idx] = s * sign_val

    return (out_peak_age_share, out_post_peak_share, out_completeness,
            out_ramp_norm, out_is_new_peak, out_phase, out_post_peak_slope)


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12], "maturity_threshold": 0.8, "ramp_threshold": 0.5 (опционально)}."""
    windows = np.array(params['windows'], dtype=np.int64)
    maturity_threshold = float(params.get('maturity_threshold', 0.8))
    ramp_threshold = float(params.get('ramp_threshold', 0.5))
    peak_age, post_peak, compl, ramp, is_new, phase, pps = _kernel(
        values, position, windows, maturity_threshold, ramp_threshold
    )
    arrays = [peak_age, post_peak, compl, ramp, is_new, phase]
    suffixes = ['peak_age_share', 'post_peak_share', 'completeness', 'ramp_norm', 'is_new_peak', 'phase_flag']
    for j, w in enumerate(params['windows']):
        arrays.append(pps[j])
        suffixes.append(f'post_peak_slope_w{w}')
    return arrays, suffixes
