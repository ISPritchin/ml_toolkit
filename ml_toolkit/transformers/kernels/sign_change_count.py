"""Число смен знака приращения (рост ↔ падение) на скользящем окне.

Signal:
    Подсчитывает количество разворотов направления внутри окна. Высокое значение —
    хаотичный осциллирующий ряд. Низкое — монотонный тренд без разворотов. Абсолютная
    версия alternation_rate (там — доля, здесь — количество).

Formula:
    d[i] = v[i] - v[i-1]  для i in [t-w+2..t]
    cur_sign[i] = sign(d[i]) ∈ {-1, 0, +1}  (нули игнорируются)
    sign_change_count_w = count(cur_sign[i] != prev_nonzero_sign[i])

    Требует ws >= 3.

Outputs:
    {product}__sign_change_count__w6   — число смен знака за 6 мес
    {product}__sign_change_count__w12  — число смен знака за 12 мес

Preset (monthly.yaml):
    sign_change_count:
      windows: [6, 12]

Interpretation:
    = 0 — монотонный ряд (чистый рост или чистое падение).
    ≈ ws - 2 — максимальная осцилляция (каждый шаг — смена направления).
    sign_change_count_w12 = 2 — два разворота за год: возможно рост → просадка → рост.
    Полезно как feature interaction с slope: высокий slope + мало смен = чистый тренд.

Example:
    Ряд (6 мес): [10, 30, 20, 40, 30, 50],  w=6

    приращения d: +20, −10, +20, −10, +20
    знаки: +, −, +, −, +  → каждая пара соседних знаков различна
    смен знака = 4
    → sign_change_count__w6 = 4  (пилообразный, осциллирующий ряд)

"""

import numba as nb
import numpy as np

from .._windowing import resolve_window_size

FEATURE = 'sign_change_count'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            if ws >= 3:
                count = 0
                prev_sign = 0
                for offset in range(1, ws):
                    d = (
                        product_values[row_idx - ws + 1 + offset]
                        - product_values[row_idx - ws + offset]
                    )
                    cur_sign = 1 if d > 0.0 else (-1 if d < 0.0 else 0)
                    if prev_sign != 0 and cur_sign != 0 and cur_sign != prev_sign:
                        count += 1
                    if cur_sign != 0:
                        prev_sign = cur_sign
                out[j, row_idx] = count
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}"""
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f'w{w}' for w in params['windows']]
