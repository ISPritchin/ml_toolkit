"""Минимум и максимум на скользящем окне.

Signal:
    Абсолютные экстремумы окна — базовые компоненты для других трансформеров
    (max_drawdown, pct_of_max, recovery_dynamics). Самостоятельно полезны для оценки
    диапазона, в котором находился ряд за период.

Formula:
    min_w = min(v[t-w+1..t])
    max_w = max(v[t-w+1..t])

Outputs:
    {product}__rolling_min_max__min_w6   — минимум за 6 мес
    {product}__rolling_min_max__max_w6   — максимум за 6 мес
    {product}__rolling_min_max__min_w12  — минимум за 12 мес
    {product}__rolling_min_max__max_w12  — максимум за 12 мес

Preset entry:
    rolling_min_max:
      windows: [6, 12]
      dilations: [1, 2]   # optional, default [1]; d>1 spaces window taps d rows apart

Interpretation:
    max_w12 / min_w12 — диапазон разброса: чем больше, тем волатильнее ряд.
    min_w12 = 0 — в течение года был хотя бы один нулевой месяц.
    max_w6 = max_w12 — пик был в последние полгода.
    max_w12 = min_w12 — полностью стабильный ряд без каких-либо изменений.

Example:
    Ряд (6 мес): [10, 80, 40, 20, 5, 30],  w=6

    min_w = min(окна) = 5
    max_w = max(окна) = 80
    → rolling_min_max__min_w6 = 5,  max_w6 = 80  (диапазон 5..80)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import (
    FILL_NAN_UNBOUNDED_HIGH,
    FILL_NAN_UNBOUNDED_LOW,
    compute_window_min_and_max,
    resolve_window_size,
)

FEATURE = 'rolling_min_max'
FILL_NAN: dict[str | None, float] = {
    # ряд не обязан быть неотрицательным, поэтому нет доказанно недостижимого
    # числа "рядом с нулём" — берём заведомо большой сентинел, направленный
    # в сторону, противоположную своей роли (min → +, max → -), чтобы
    # заполненное значение не читалось как правдоподобный реальный экстремум.
    'min_w': FILL_NAN_UNBOUNDED_HIGH,
    'max_w': FILL_NAN_UNBOUNDED_LOW,
}


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray, dilations: np.ndarray
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    n_d = dilations.shape[0]
    out_min = np.zeros((n_w * n_d, n_rows))
    out_max = np.zeros((n_w * n_d, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        k = 0
        for j in range(n_w):
            for d in range(n_d):
                ws = resolve_window_size(pos, windows[j], dilations[d])
                lo, hi = compute_window_min_and_max(product_values, row_idx, ws, dilations[d])
                out_min[k, row_idx] = lo
                out_max[k, row_idx] = hi
                k += 1
    return out_min, out_max


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12], "dilations": [1, 2]}. "dilations" optional, default [1]."""
    windows = np.array(params['windows'], dtype=np.int64)
    dilations = np.array(params.get('dilations', [1]), dtype=np.int64)
    out_min, out_max = _kernel(values, position, windows, dilations)
    arrays = []
    suffixes = []
    k = 0
    for w in params['windows']:
        for d in params.get('dilations', [1]):
            w_tag = f'w{w}' if d == 1 else f'w{w}_d{d}'
            arrays.append(out_min[k])
            suffixes.append(f'min_{w_tag}')
            arrays.append(out_max[k])
            suffixes.append(f'max_{w_tag}')
            k += 1
    return arrays, suffixes
