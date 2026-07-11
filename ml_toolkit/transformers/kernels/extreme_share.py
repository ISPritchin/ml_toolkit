"""Доля месяцев с отклонением > 1.5σ от среднего и баланс выше/ниже среднего.

Signal:
    Показывает, насколько часто значения оказываются далеко от средней нормы (extreme_share)
    и нет ли системного смещения — больше времени проводит выше среднего или ниже (balance).
    Используется для отличия «спокойных» рядов от волатильных.

Formula:
    mean_w, std_w — среднее и станд. отклонение окна
    extreme_w = count(|v[i] - mean_w| > 1.5 * std_w) / ws
    balance_w = count(v[i] > mean_w) / ws - 0.5

Outputs:
    {product}__extreme_share__extreme_w6   — доля экстремальных месяцев за 6 мес
    {product}__extreme_share__balance_w6   — баланс выше/ниже среднего за 6 мес
    {product}__extreme_share__extreme_w12  — доля экстремальных месяцев за 12 мес
    {product}__extreme_share__balance_w12  — баланс выше/ниже среднего за 12 мес

Preset entry:
    extreme_share:
      windows: [6, 12]

Interpretation:
    extreme_w12 > 0.3 — более 30% месяцев выходят за 1.5σ: нестабильный ряд.
    balance_w12 > 0.2 — ряд больше времени проводит выше среднего (восходящий тренд).
    balance_w12 = 0 при высоком extreme — осцилляция симметрична (ни рост ни падение).
    extreme близко к 0 при balance ≈ 0 — стабильный ряд около своего среднего.

Example:
    Ряд (6 мес): [10, 10, 10, 10, 10, 40],  w=6
    mean = 90/6 = 15,  std = 11.18,  1.5·std = 16.77

    экстремальные (|v−mean| > 16.77): только 40 (|40−15|=25) → 1 из 6
    выше среднего (>15): только 40 → 1 из 6
    → extreme_share__extreme_w6 = 1/6 = 0.167
    → extreme_share__balance_w6 = 1/6 − 0.5 = −0.333

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import compute_window_mean_and_std, resolve_window_size

FEATURE = 'extreme_share'


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    windows: np.ndarray,
    sigma_threshold: float,
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_extreme = np.zeros((n_w, n_rows))
    out_balance = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            mean, std = compute_window_mean_and_std(product_values, row_idx, ws)
            threshold = sigma_threshold * std
            extreme_count = 0
            above_count = 0
            for offset in range(ws):
                v = product_values[row_idx - ws + 1 + offset]
                if abs(v - mean) > threshold:
                    extreme_count += 1
                if v > mean:
                    above_count += 1
            out_extreme[j, row_idx] = extreme_count / ws
            out_balance[j, row_idx] = above_count / ws - 0.5
    return out_extreme, out_balance


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12], "sigma_threshold": 1.5 (опционально)}."""
    windows = np.array(params['windows'], dtype=np.int64)
    sigma_threshold = float(params.get('sigma_threshold', 1.5))
    out_extreme, out_balance = _kernel(values, position, windows, sigma_threshold)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(out_extreme[j])
        suffixes.append(f'extreme_w{w}')
        arrays.append(out_balance[j])
        suffixes.append(f'balance_w{w}')
    return arrays, suffixes
