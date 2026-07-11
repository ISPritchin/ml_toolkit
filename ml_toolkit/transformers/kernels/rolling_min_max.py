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

from ml_toolkit.transformers._windowing import compute_window_min_and_max, resolve_window_size

FEATURE = 'rolling_min_max'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_min = np.zeros((n_w, n_rows))
    out_max = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            lo, hi = compute_window_min_and_max(product_values, row_idx, ws)
            out_min[j, row_idx] = lo
            out_max[j, row_idx] = hi
    return out_min, out_max


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    out_min, out_max = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(out_min[j])
        suffixes.append(f'min_w{w}')
        arrays.append(out_max[j])
        suffixes.append(f'max_w{w}')
    return arrays, suffixes
