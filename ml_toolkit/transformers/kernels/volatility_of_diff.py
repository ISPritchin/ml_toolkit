"""Стандартное отклонение помесячных приращений: «дёрганость» изменений ряда.

Signal:
    Измеряет, насколько непостоянны сами изменения значений ряда (не уровень, а дифф уровня).
    В отличие от rolling_std (волатильность уровня), этот признак чувствителен к частым
    чередованиям роста и падения: два ряда с одинаковым std уровня могут иметь разное
    std_diff, если у одного смены плавные, а у другого — резкие.

Formula:
    d[i] = v[i] - v[i-1]   для i in [t-w+2..t]
    mean_d   = mean(d[1..w-1])
    std_diff_w = sqrt(mean((d[i] - mean_d)^2))

    Использует w-1 приращений в пределах окна (популяционное std).

Outputs:
    {product}__volatility_of_diff__w6   — std приращений за 6 мес
    {product}__volatility_of_diff__w12  — std приращений за 12 мес

Preset entry:
    volatility_of_diff:
      windows: [6, 12]

Interpretation:
    = 0 — приращения абсолютно постоянны (рост/спад одним темпом).
    Высокое при низком rolling_std — чередование больших плюсов и минусов, компенсирующих друг друга.
    Высокое при высоком rolling_std — и уровни, и скорость их изменения нестабильны (хаотичный ряд).
    Снижение volatility_of_diff_w6 < w12 — поведение стабилизируется в последние полгода.

Example:
    Ряд (6 мес): [10, 30, 20, 40, 30, 50],  w=6

    приращения d: +20, −10, +20, −10, +20  (n_diffs = 5)
    mean_d = 40/5 = 8
    std_diff = sqrt(mean((d − 8)²)) = sqrt(1080/5) = sqrt(216) = 14.70
    → volatility_of_diff__w6 = 14.70  (резко чередующиеся приращения)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import resolve_window_size

FEATURE = 'volatility_of_diff'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            n_diffs = ws - 1
            if n_diffs >= 2:
                # сумма приращений телескопируется: v[t] - v[t-ws+1]; второй
                # проход считает дисперсию без промежуточного буфера
                d_mean = (
                    product_values[row_idx] - product_values[row_idx - ws + 1]
                ) / n_diffs
                d_var = 0.0
                for offset in range(1, ws):
                    abs_idx = row_idx - ws + 1 + offset
                    d = product_values[abs_idx] - product_values[abs_idx - 1]
                    d_var += (d - d_mean) ** 2
                out[j, row_idx] = (d_var / n_diffs) ** 0.5
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6, 12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f'w{w}' for w in params['windows']]
