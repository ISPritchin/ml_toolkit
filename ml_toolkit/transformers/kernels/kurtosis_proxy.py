"""Избыточный эксцесс (kurtosis) и персентильные соотношения внутри окна.

Signal:
    Описывает форму распределения объёмов: есть ли тяжёлые хвосты (редкие крупные
    месяцы) или плоское, равномерное распределение. Персентильные соотношения дополняют
    kurtosis и полезны при нулях в хвостах (стандартный kurtosis там взрывается).

Formula:
    kurt_w = (1/ws * sum(((v[i] - mean_w) / std_w)^4)) - 3    при std_w > eps
    p10, p25, p75, p90 — персентили из sorted buffer окна
    p75_p25_w   = p75 / (|p25| + eps)
    p90_p10_w   = p90 / (|p10| + eps)
    upper_tail_w = sum(v[i] if v[i] > p75) / (|S_w| + eps)
    lower_tail_w = sum(v[i] if v[i] < p25) / (|S_w| + eps)

Outputs:
    {product}__kurtosis_proxy__kurt_w6        — избыточный эксцесс за 6 мес
    {product}__kurtosis_proxy__p75_p25_w6     — отношение p75/p25 за 6 мес
    {product}__kurtosis_proxy__p90_p10_w6     — отношение p90/p10 за 6 мес
    {product}__kurtosis_proxy__upper_tail_w6  — доля верхнего квартиля в сумме
    {product}__kurtosis_proxy__lower_tail_w6  — доля нижнего квартиля в сумме
    (аналогично для w12)

Preset (monthly.yaml):
    kurtosis_proxy:
      windows: [6, 12]

Interpretation:
    kurt_w12 > 3 — тяжёлые хвосты (экстремальные месяцы, проектный B2B-паттерн).
    kurt_w12 < -1 — плоское, равномерное распределение (потоковый клиент).
    upper_tail > 0.8 — верхние 25% месяцев дают 80% объёма (крайняя концентрация).
    p75_p25 большое при p25 ≈ 0 — разрыв между активными и нулевыми месяцами.

Example:
    Ряд (6 мес): [10, 10, 10, 10, 10, 70],  w=6
    mean = 60/6... = 20,  std = 22.36

    z пяти «десяток» = (10−20)/22.36 = −0.447 → z⁴ = 0.04
    z для 70 = (70−20)/22.36 = 2.236 → z⁴ = 25.0
    sum(z⁴) = 5·0.04 + 25 = 25.2
    kurt = 25.2/6 − 3 = 4.2 − 3 = 1.2
    → kurtosis_proxy__kurt_w6 = 1.2  (тяжёлый правый хвост)
"""

import numba as nb
import numpy as np

from .._windowing import (
    EPS,
    compute_window_mean_and_std,
    compute_window_sorted_buffer,
    compute_window_sum,
    resolve_window_size,
    safe_ratio,
    sorted_quantile,
)

FEATURE = "kurtosis_proxy"


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_kurt = np.zeros((n_w, n_rows))
    out_p75_p25 = np.zeros((n_w, n_rows))
    out_p90_p10 = np.zeros((n_w, n_rows))
    out_upper = np.zeros((n_w, n_rows))
    out_lower = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            mean, std = compute_window_mean_and_std(product_values, row_idx, ws)
            # эксцесс
            if std > EPS:
                fourth = 0.0
                for offset in range(ws):
                    z = (product_values[row_idx - ws + 1 + offset] - mean) / std
                    fourth += z * z * z * z
                out_kurt[j, row_idx] = fourth / ws - 3.0
            # персентили через sorted buffer (единая конвенция sorted[int(q*(ws-1))])
            sorted_buf = compute_window_sorted_buffer(product_values, row_idx, ws)
            p10 = sorted_quantile(sorted_buf, ws, 0.10)
            p25 = sorted_quantile(sorted_buf, ws, 0.25)
            p75 = sorted_quantile(sorted_buf, ws, 0.75)
            p90 = sorted_quantile(sorted_buf, ws, 0.90)
            out_p75_p25[j, row_idx] = safe_ratio(p75, p25)
            out_p90_p10[j, row_idx] = safe_ratio(p90, p10)
            win_sum = compute_window_sum(product_values, row_idx, ws)
            if abs(win_sum) > EPS:
                upper_sum = 0.0; lower_sum = 0.0
                for offset in range(ws):
                    v = product_values[row_idx - ws + 1 + offset]
                    if v > p75: upper_sum += v
                    elif v < p25: lower_sum += v
                out_upper[j, row_idx] = safe_ratio(upper_sum, win_sum)
                out_lower[j, row_idx] = safe_ratio(lower_sum, win_sum)
    return out_kurt, out_p75_p25, out_p90_p10, out_upper, out_lower


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6, 12]}"""
    windows = np.array(params["windows"], dtype=np.int64)
    out_kurt, out_p75p25, out_p90p10, out_upper, out_lower = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params["windows"]):
        arrays.append(out_kurt[j])
        suffixes.append(f"kurt_w{w}")
        arrays.append(out_p75p25[j])
        suffixes.append(f"p75_p25_w{w}")
        arrays.append(out_p90p10[j])
        suffixes.append(f"p90_p10_w{w}")
        arrays.append(out_upper[j])
        suffixes.append(f"upper_tail_w{w}")
        arrays.append(out_lower[j])
        suffixes.append(f"lower_tail_w{w}")
    return arrays, suffixes
