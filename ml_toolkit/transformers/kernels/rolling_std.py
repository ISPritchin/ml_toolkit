"""Стандартное отклонение (смещённое) на скользящем окне.

Signal:
    Абсолютная мера разброса в единицах исходной колонки. В отличие от CV не нормирован
    на уровень, поэтому ряды большего масштаба автоматически имеют большее std. Полезен как
    компонент других признаков (CV = std/mean, zscore = diff/std).

Formula:
    mean_w = mean(v[t-w+1..t])
    std_w  = sqrt(mean((v[i] - mean_w)², i in [t-w+1..t]))    (смещённый, делитель ws)

Outputs:
    {product}__rolling_std__w6   — σ за 6 мес
    {product}__rolling_std__w12  — σ за 12 мес
    {product}__rolling_std__w24  — σ за 24 мес

Preset entry:
    rolling_std:
      windows: [6, 12, 24]
      dilations: [1, 2]   # optional, default [1]; d>1 spaces window taps d rows apart

Interpretation:
    std = 0 — абсолютно стабильный ряд.
    std_w6 > std_w12 — волатильность нарастает в краткосроке.
    std_w12 ≈ 38.2 для осциллирующего ряда V при mean = 50.
    В паре с mean через CV позволяет сравнивать ряды разного масштаба.

Example:
    Ряд (6 мес): [10, 10, 10, 10, 10, 40],  w=6

    mean = 90/6 = 15
    отклонения²: пять·(10−15)²=25 и (40−15)²=625 → sum = 5·25 + 625 = 750
    std = sqrt(750/6) = sqrt(125) = 11.18
    → rolling_std__w6 = 11.18

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import compute_window_mean_and_std, resolve_window_size

FEATURE = 'rolling_std'
FILL_NAN: dict[str | None, float] = {None: -1.0}  # std >= 0 всегда, независимо от масштаба колонки


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray, dilations: np.ndarray
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    n_d = dilations.shape[0]
    out = np.zeros((n_w * n_d, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        k = 0
        for j in range(n_w):
            for d in range(n_d):
                ws = resolve_window_size(pos, windows[j], dilations[d])
                _, std = compute_window_mean_and_std(product_values, row_idx, ws, dilations[d])
                out[k, row_idx] = std
                k += 1
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6, 12, 24], "dilations": [1, 2]}. "dilations" optional, default [1]."""
    windows = np.array(params['windows'], dtype=np.int64)
    dilations = np.array(params.get('dilations', [1]), dtype=np.int64)
    out = _kernel(values, position, windows, dilations)
    suffixes = [f'w{w}' if d == 1 else f'w{w}_d{d}' for w in params['windows'] for d in params.get('dilations', [1])]
    return [out[k] for k in range(len(suffixes))], suffixes
