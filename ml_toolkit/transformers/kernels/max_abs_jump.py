"""Наибольший по модулю скачок между соседними периодами в окне.

Signal:
    Определяет максимальное разовое изменение внутри окна: полезен для выявления
    контрактных переходов, разовых крупных платежей или резких обвалов. Большой
    max_abs_jump при малом alternation_rate — один крупный скачок на фоне стабильного ряда.

Formula:
    max_abs_jump_w = max(|v[i] - v[i-1]|, i in [t-w+2..t])
    Требует ws >= 2.

Outputs:
    {product}__max_abs_jump__w6   — макс. абс. скачок за 6 мес
    {product}__max_abs_jump__w12  — макс. абс. скачок за 12 мес

Preset (monthly.yaml):
    max_abs_jump:
      windows: [6, 12]

Interpretation:
    Высокий max_abs_jump относительно mean — единовременный крупный платёж.
    max_abs_jump_w6 ≈ max_abs_jump_w12 — крупный скачок произошёл именно в последние 6 мес.
    max_abs_jump_w6 << max_abs_jump_w12 — крупный скачок был давно, сейчас стабильно.
    В паре с alternation_rate__max_jump_share дополняет картину структуры TV.

Example:
    Ряд (6 мес): [10, 20, 15, 60, 55, 50],  w=6

    |скачки| соседних: |20−10|=10, |15−20|=5, |60−15|=45, |55−60|=5, |50−55|=5
    максимум = 45
    → max_abs_jump__w6 = 45  (резкий разовый скачок 15→60)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import resolve_window_size

FEATURE = 'max_abs_jump'


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    windows: np.ndarray,
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            if ws < 2:
                continue
            largest = 0.0
            for offset in range(1, ws):
                jump = abs(
                    product_values[row_idx - ws + 1 + offset]
                    - product_values[row_idx - ws + offset]
                )
                largest = max(largest, jump)
            out[j, row_idx] = largest
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6, 12]}.

    """
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    arrays = [out[j] for j in range(len(windows))]
    suffixes = [f'w{w}' for w in params['windows']]
    return arrays, suffixes
