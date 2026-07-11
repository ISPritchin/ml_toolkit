"""Отношение суммы второй половины окна к первой: тренд «половина-к-половине».

Signal:
    Грубый, но устойчивый к выбросам индикатор направления: больше ли объём в последние
    полокна, чем в первые. Значение > 1 — восходящий тренд внутри окна, < 1 — нисходящий.

Formula:
    Вычисляется только при ws >= w (полное окно доступно).
    half = ws // 2
    first_half_sum  = sum(v[t-2*half+1..t-half])
    second_half_sum = sum(v[t-half+1..t])
    half_ratio_w = second_half_sum / (|first_half_sum| + eps)

Outputs:
    {product}__half_ratio__w6   — соотношение вторая/первая половина 6 мес окна
    {product}__half_ratio__w12  — соотношение вторая/первая половина 12 мес окна

Preset entry:
    half_ratio:
      windows: [6, 12]

Interpretation:
    > 1 — вторая половина периода «тяжелее»: рост, разгон или восстановление.
    < 1 — первая половина была лучше: снижение или затухание.
    ≈ 1 — стабильный уровень без изменений.
    half_ratio_w6 > 1 при half_ratio_w12 < 1 — краткосрочный отскок на фоне годового спада.

Example:
    Ряд (6 мес): [10, 10, 10, 20, 20, 20],  w=6
    half = 3

    first_half_sum  = 10+10+10 = 30
    second_half_sum = 20+20+20 = 60
    half_ratio = 60 / 30 = 2.0
    → half_ratio__w6 = 2.0  (вторая половина вдвое «тяжелее»)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import compute_window_sum, resolve_window_size, safe_ratio

FEATURE = 'half_ratio'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            if ws >= windows[j]:  # полное окно доступно
                half = ws // 2
                first_half_sum = compute_window_sum(product_values, row_idx - half, half)
                second_half_sum = compute_window_sum(product_values, row_idx, half)
                out[j, row_idx] = safe_ratio(second_half_sum, first_half_sum)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f'w{w}' for w in params['windows']]
