"""Тренд волатильности: разница стандартных отклонений на коротком и длинном горизонтах.

Signal:
    Показывает, растёт или падает нестабильность поведения клиента. Положительное значение —
    краткосрочная волатильность выше долгосрочной (нестабильность нарастает). Отрицательное —
    поведение стабилизируется. Полезно в паре с rolling_cv для разделения «масштаб вырос,
    но ровнее» от «масштаб тот же, но всё хаотичнее».

Formula:
    std_short = std(v[t-ws+1..t])      за окно ws (короткое)
    std_long  = std(v[t-wl+1..t])      за окно wl (длинное)
    volatility_trend_wS_wL = std_short - std_long

    Оба std — популяционные (без коррекции Бесселя).

Outputs:
    {product}__volatility_trend__w3_w12  — std_3 - std_12
    {product}__volatility_trend__w6_w12  — std_6 - std_12

Preset (monthly.yaml):
    volatility_trend:
      pairs:
        - [3, 12]
        - [6, 12]

Interpretation:
    > 0 — краткосрочная волатильность выше долгосрочной (нарастающий хаос).
    < 0 — клиент стабилизируется: последние месяцы ровнее длинной истории.
    ≈ 0 — волатильность стационарна (не нарастает и не снижается).
    w3_w12 > 0, w6_w12 < 0 — только последние 3 месяца выбиваются, но полугодие ещё в норме.

Example:
    Ряд (6 мес): [20, 20, 20, 10, 40, 10],  пара (3, 6)

    std_short (посл. 3 [10,40,10]) = 14.142
    std_long  (все 6) = 10.0
    volatility_trend = 14.142 − 10.0 = 4.142
    → volatility_trend__w3_w6 = 4.142  (краткосрочный хаос нарастает)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import compute_window_mean_and_std, resolve_window_size

FEATURE = 'volatility_trend'


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
            _, std_short = compute_window_mean_and_std(product_values, row_idx, ws_short)
            _, std_long = compute_window_mean_and_std(product_values, row_idx, ws_long)
            out[j, row_idx] = std_short - std_long
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"pairs": [[6, 12]]} — ключ обязателен, дефолты задаёт пресет."""
    pairs = params['pairs']
    short_w = np.array([p[0] for p in pairs], dtype=np.int64)
    long_w = np.array([p[1] for p in pairs], dtype=np.int64)
    out = _kernel(values, position, short_w, long_w)
    suffixes = [f'w{p[0]}_w{p[1]}' for p in pairs]
    return [out[j] for j in range(len(pairs))], suffixes
