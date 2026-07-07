"""Взвешенный по давности (временной позиции) импульс приращений на скользящем окне.

Signal:
    Усиливает вес последних приращений относительно ранних: если рост нарастает
    в конце окна — признак выше, чем при равномерном или убывающем темпе. Более
    чувствителен к «свежим» изменениям, чем slope.

Formula:
    Для i in [1..ws-1] (позиции внутри окна):
        d[i] = v[t-ws+1+i] - v[t-ws+i]   (приращение шага)
        weighted_sum = sum(i * d[i], i in [1..ws-1])
    time_weighted_momentum_w = weighted_sum / (|sum_v_w| + eps)

    Весом выступает позиция шага внутри окна (1, 2, ..., ws-1) — последние шаги тяжелее.

Outputs:
    {product}__time_weighted_momentum__w6   — взвеш. импульс за 6 мес
    {product}__time_weighted_momentum__w12  — взвеш. импульс за 12 мес

Preset (monthly.yaml):
    time_weighted_momentum:
      windows: [6, 12]

Interpretation:
    > 0 — нарастающий положительный импульс (свежие приросты важнее ранних).
    < 0 — нарастающий отрицательный импульс (снижение усиливается).
    ≈ 0 — либо стагнация, либо ранние и поздние приросты компенсируют друг друга.
    В паре со slope: оба > 0 и time_weighted_momentum > slope → ускорение роста.

Example:
    Ряд (6 мес): [10, 20, 30, 40, 50, 60],  w=6

    приращения d[i] = +10 для всех шагов i=1..5
    weighted_sum = sum(i·d[i]) = 10·(1+2+3+4+5) = 150
    window_sum = 10+...+60 = 210
    time_weighted_momentum = 150 / 210 = 0.714
    → time_weighted_momentum__w6 = 0.714  (положительный нарастающий импульс)

"""

import numba as nb
import numpy as np

from .._windowing import compute_window_sum, resolve_window_size, safe_ratio

FEATURE = 'time_weighted_momentum'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            if ws >= 2:
                weighted_sum = 0.0
                for offset in range(1, ws):
                    d = (
                        product_values[row_idx - ws + 1 + offset]
                        - product_values[row_idx - ws + offset]
                    )
                    weighted_sum += offset * d
                window_sum = compute_window_sum(product_values, row_idx, ws)
                out[j, row_idx] = safe_ratio(weighted_sum, window_sum)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}"""
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f'w{w}' for w in params['windows']]
