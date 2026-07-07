"""Восстановление после просадки: полнота, скорость, факт, длина долины.

Signal:
    Оценивает, насколько клиент восстановился от минимума в окне, как быстро это
    происходит и находится ли он сейчас в стадии активного восстановления. Важен
    для различения «реально растущих» от «всё ещё в долине».

Formula:
    w_min = min(v[t-w+1..t]),  w_max = max(v[t-w+1..t])
    trough_pos — позиция минимума внутри окна
    months_since_trough = (ws - 1) - trough_pos

    completeness_w  = (v[t] - w_min) / (w_max - w_min + eps)  ∈ [0, 1]
    drawdown_dur_w  = count(v[i] < w_max, i in окне)
    post_trough_gain_w = (v[t] - w_min) / (|mean_w| + eps)
    trough_is_recent_w = 1 if months_since_trough <= 3
    speed_w         = (v[t] - w_min) / (months_since_trough + 1)
    is_recovering_now = 1 if v[t] > v[t-1] > v[t-2] AND v[t] < mean_w12

Outputs:
    {product}__recovery_dynamics__completeness_w12    — полнота восстановления [0,1]
    {product}__recovery_dynamics__drawdown_dur_w12    — мес. ниже пика (длина долины)
    {product}__recovery_dynamics__post_trough_gain_w12 — прирост от дна / mean
    {product}__recovery_dynamics__trough_is_recent_w12 — дно было ≤3 мес назад
    {product}__recovery_dynamics__speed_w12           — скорость восстановления
    {product}__recovery_dynamics__is_recovering_now   — флаг активного восстановления

Preset (monthly.yaml):
    recovery_dynamics:
      windows: [12]

Interpretation:
    completeness = 1.0 — полностью восстановился от минимума до максимума окна.
    completeness = 0.3 — восстановился лишь на треть от амплитуды.
    is_recovering_now = 1 — три месяца подряд растём, но ещё ниже среднего.
    speed_w12 = 14.2 ед/мес — высокая скорость восстановления (ряд D, восстановление за 6 мес).

Example:
    Ряд (6 мес): [10, 80, 40, 20, 5, 30],  w=6
    w_min = 5 (offset=4),  w_max = 80,  v[t] = 30

    completeness = (v − w_min)/(w_max − w_min) = (30 − 5)/(80 − 5) = 25/75 = 0.333
    months_since_trough = (6−1) − 4 = 1
    speed = (30 − 5)/(1 + 1) = 12.5
    → recovery_dynamics__completeness_w6 = 0.333,  speed_w6 = 12.5

"""

import numba as nb
import numpy as np

from .._windowing import compute_window_mean, resolve_window_size, safe_ratio

FEATURE = 'recovery_dynamics'


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    windows: np.ndarray,
    trough_recent_months: int,
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_completeness = np.zeros((n_w, n_rows))
    out_drawdown_dur = np.zeros((n_w, n_rows))
    out_is_recovering = np.zeros(n_rows)
    out_post_trough_gain = np.zeros((n_w, n_rows))
    out_trough_is_recent = np.zeros((n_w, n_rows))
    out_recovery_speed = np.zeros((n_w, n_rows))

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        v = product_values[row_idx]
        mean_all = compute_window_mean(product_values, row_idx, min(pos + 1, 12))

        # is_recovering_now: растём 2 периода подряд, но ещё ниже среднего
        if pos >= 2:
            if v > product_values[row_idx - 1] > product_values[row_idx - 2] and v < mean_all:
                out_is_recovering[row_idx] = 1.0

        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            w_max = product_values[row_idx - ws + 1]
            w_min = product_values[row_idx - ws + 1]
            trough_pos_in_window = 0
            peak_val = w_max

            for offset in range(1, ws):
                abs_idx = row_idx - ws + 1 + offset
                vv = product_values[abs_idx]
                if vv > w_max:
                    w_max = vv
                    peak_val = vv
                if vv < w_min:
                    w_min = vv
                    trough_pos_in_window = offset

            months_since_trough = ws - 1 - trough_pos_in_window

            # completeness: (v - min) / (max - min)
            out_completeness[j, row_idx] = safe_ratio(v - w_min, w_max - w_min)

            # drawdown_duration: число месяцев ниже peak_val
            dd_dur = 0
            for offset in range(ws):
                if product_values[row_idx - ws + 1 + offset] < peak_val:
                    dd_dur += 1
            out_drawdown_dur[j, row_idx] = dd_dur

            # post_trough_gain
            mean_w = compute_window_mean(product_values, row_idx, ws)
            out_post_trough_gain[j, row_idx] = safe_ratio(v - w_min, mean_w)

            # trough_is_recent: дно в пределах последних trough_recent_months месяцев
            out_trough_is_recent[j, row_idx] = 1.0 if months_since_trough <= trough_recent_months else 0.0

            # recovery_speed: gain from trough / time since trough
            out_recovery_speed[j, row_idx] = (v - w_min) / (months_since_trough + 1)

    return out_completeness, out_drawdown_dur, out_is_recovering, out_post_trough_gain, out_trough_is_recent, out_recovery_speed


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12], "trough_recent_months": 3 (опционально)}"""
    windows = np.array(params['windows'], dtype=np.int64)
    trough_recent_months = int(params.get('trough_recent_months', 3))
    compl, dd_dur, isr, ptg, tir, rs = _kernel(values, position, windows, trough_recent_months)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(compl[j]);  suffixes.append(f'completeness_w{w}')
        arrays.append(dd_dur[j]); suffixes.append(f'drawdown_dur_w{w}')
        arrays.append(ptg[j]);    suffixes.append(f'post_trough_gain_w{w}')
        arrays.append(tir[j]);    suffixes.append(f'trough_is_recent_w{w}')
        arrays.append(rs[j]);     suffixes.append(f'speed_w{w}')
    arrays.append(isr); suffixes.append('is_recovering_now')
    return arrays, suffixes
