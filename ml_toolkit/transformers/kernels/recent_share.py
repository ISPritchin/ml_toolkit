"""Доля суммы короткого окна в сумме длинного: концентрация активности в последнее время.

Signal:
    Показывает, какую часть долгосрочного объёма составляют последние месяцы. Высокое
    значение — активность сконцентрирована в недавнем прошлом (рост или разгон).
    Низкое — последние месяцы «легче» исторического фона.

Formula:
    S_short = sum(v[t-ws_short+1..t])
    S_long  = sum(v[t-ws_long+1..t])
    recent_share = S_short / (|S_long| + eps)

    При равенстве длин ratio равно долям пропорционально: равномерный ряд даёт short/long.

Outputs:
    {product}__recent_share__r3_w12  — сумма 3 мес / сумма 12 мес
    {product}__recent_share__r6_w24  — сумма 6 мес / сумма 24 мес

Preset (monthly.yaml):
    recent_share:
      pairs:
        - [3, 12]
        - [6, 24]

Interpretation:
    r3_w12 = 0.40 — последний квартал даёт 40% годового объёма (выше нормы 3/12=0.25 = рост).
    r3_w12 = 0.15 — последний квартал слабее нормы (снижение).
    r6_w24 > 0.5 — второе полугодие «тяжелее» первого и всего предыдущего года.
    Равномерный ряд: r3_w12 = 3/12 = 0.25, r6_w24 = 6/24 = 0.25.

Example:
    Ряд (6 мес): [10, 20, 30, 40, 50, 60],  пара (3, 6)

    S_short = 40+50+60 = 150   (последние 3 мес)
    S_long  = 10+...+60 = 210  (все 6 мес)
    recent_share = 150 / 210 = 0.714
    → recent_share__r3_w6 = 0.714  (последний квартал — 71% объёма, рост выше нормы 0.5)

"""

import numba as nb
import numpy as np

from .._windowing import compute_window_sum, resolve_window_size, safe_ratio

FEATURE = 'recent_share'


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    short_windows: np.ndarray,
    long_windows: np.ndarray,
):
    n_rows = product_values.shape[0]
    n_p = short_windows.shape[0]
    out = np.zeros((n_p, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_p):
            ws_short = resolve_window_size(pos, short_windows[j])
            ws_long = resolve_window_size(pos, long_windows[j])
            s_short = compute_window_sum(product_values, row_idx, ws_short)
            s_long = compute_window_sum(product_values, row_idx, ws_long)
            out[j, row_idx] = safe_ratio(s_short, s_long)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"pairs": [[3, 12], [6, 24]]}"""
    pairs = params['pairs']
    short_w = np.array([p[0] for p in pairs], dtype=np.int64)
    long_w = np.array([p[1] for p in pairs], dtype=np.int64)
    out = _kernel(values, position, short_w, long_w)
    suffixes = [f'r{p[0]}_w{p[1]}' for p in pairs]
    return [out[j] for j in range(len(pairs))], suffixes
