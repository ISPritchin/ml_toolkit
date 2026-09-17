"""Доля суммы короткого окна в сумме длинного: концентрация активности в последнее время.

Signal:
    Показывает, какую часть долгосрочного объёма составляют последние месяцы. Высокое
    значение — активность сконцентрирована в недавнем прошлом (рост или разгон).
    Низкое — последние месяцы «легче» исторического фона.

Formula:
    S_short = sum(v[t-ws_short+1..t])
    S_long  = sum(v[t-ws_long+1..t])
    recent_share = S_short / (|S_long| + eps)

    При равномерном (постоянном) ряде ratio равно доле длин окон: short/long
    (см. Interpretation: r3_w12 = 3/12 = 0.25 для равномерного ряда).

Outputs:
    {product}__recent_share__r3_w12  — сумма 3 мес / сумма 12 мес
    {product}__recent_share__r6_w24  — сумма 6 мес / сумма 24 мес

Preset entry:
    recent_share:
      pairs:
        - [3, 12]
        - [6, 24]
      dilations: [1, 2]   # optional, default [1]; applies to both windows of every pair

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

from ml_toolkit.transformers._windowing import (
    FILL_NAN_UNBOUNDED_LOW,
    compute_window_sum,
    resolve_window_size,
    safe_ratio,
)

FEATURE = 'recent_share'
# суммы окон произвольного знака при знакопеременных данных.
FILL_NAN: dict[str | None, float] = {None: FILL_NAN_UNBOUNDED_LOW}


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    short_windows: np.ndarray,
    long_windows: np.ndarray,
    dilations: np.ndarray,
):
    n_rows = product_values.shape[0]
    n_p = short_windows.shape[0]
    n_d = dilations.shape[0]
    out = np.zeros((n_p * n_d, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        k = 0
        for j in range(n_p):
            for d in range(n_d):
                dilation = dilations[d]
                ws_short = resolve_window_size(pos, short_windows[j], dilation)
                ws_long = resolve_window_size(pos, long_windows[j], dilation)
                s_short = compute_window_sum(product_values, row_idx, ws_short, dilation)
                s_long = compute_window_sum(product_values, row_idx, ws_long, dilation)
                out[k, row_idx] = safe_ratio(s_short, s_long)
                k += 1
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"pairs": [[3, 12], [6, 24]], "dilations": [1, 2]}. "dilations" optional, default [1]."""
    pairs = params['pairs']
    dilations_list = params.get('dilations', [1])
    short_w = np.array([p[0] for p in pairs], dtype=np.int64)
    long_w = np.array([p[1] for p in pairs], dtype=np.int64)
    dilations = np.array(dilations_list, dtype=np.int64)
    out = _kernel(values, position, short_w, long_w, dilations)
    suffixes = [f'r{p[0]}_w{p[1]}' if d == 1 else f'r{p[0]}_w{p[1]}_d{d}' for p in pairs for d in dilations_list]
    return [out[k] for k in range(len(suffixes))], suffixes
