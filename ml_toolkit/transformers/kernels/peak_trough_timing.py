"""Месяцев с момента пика и впадины внутри скользящего окна.

Signal:
    Датирует, когда внутри окна произошёл максимум и минимум. Позволяет определить,
    находится ли ряд на «выходе из дна» (trough недавно) или «уходящем от пика»
    (peak давно). Нулевое значение означает, что пик/дно — прямо сейчас.

Formula:
    running_peak = max(v[start..offset])
    running_trough = min(v[start..offset])
    peak_offset   — позиция внутри окна, где был max
    trough_offset — позиция внутри окна, где был min
    months_since_peak_w   = (ws - 1) - peak_offset
    months_since_trough_w = (ws - 1) - trough_offset

Outputs:
    {product}__peak_trough_timing__peak_w6    — месяцев с пика (окно 6)
    {product}__peak_trough_timing__trough_w6  — месяцев с дна (окно 6)
    {product}__peak_trough_timing__peak_w12   — месяцев с пика (окно 12)
    {product}__peak_trough_timing__trough_w12 — месяцев с дна (окно 12)

Preset entry:
    peak_trough_timing:
      windows: [6, 12]
      dilations: [1, 2]   # optional, default [1]; d>1 spaces window taps d rows apart

Interpretation:
    months_since_trough_w12 = 5 при max_drawdown_w12 > 0.9 — ряд восстанавливается 5 мес после глубокого дна.
    months_since_peak_w12 = 0 — пик в текущем месяце (новый максимум).
    months_since_peak_w12 = 11 — пик в начале окна, значит последние 11 мес падение.
    peak ≈ trough ≈ 0 — ряд монотонен в этом окне (один из них совпадает с текущим).

Example:
    Ряд (6 мес): [10, 80, 40, 20, 5, 30],  w=6
    (позиции внутри окна offset=0..5)

    максимум 80 на offset=1,  минимум 5 на offset=4
    months_since_peak   = (6−1) − 1 = 4
    months_since_trough = (6−1) − 4 = 1
    → peak_trough_timing__peak_w6 = 4,  trough_w6 = 1  (дно недавно, восстановление)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import resolve_window_size

FEATURE = 'peak_trough_timing'
FILL_NAN: dict[str | None, float] = {}  # трансформер NaN не производит по конструкции (позиционный индекс, без деления)


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray, dilations: np.ndarray
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    n_d = dilations.shape[0]
    out_peak = np.zeros((n_w * n_d, n_rows))
    out_trough = np.zeros((n_w * n_d, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        k = 0
        for j in range(n_w):
            for d in range(n_d):
                dilation = dilations[d]
                ws = resolve_window_size(pos, windows[j], dilation)
                base = row_idx - (ws - 1) * dilation
                running_peak = product_values[base]
                running_trough = running_peak
                peak_offset = 0
                trough_offset = 0
                for offset in range(ws):
                    v = product_values[base + offset * dilation]
                    if v > running_peak:
                        running_peak = v
                        peak_offset = offset
                    if v < running_trough:
                        running_trough = v
                        trough_offset = offset
                out_peak[k, row_idx] = (ws - 1) - peak_offset
                out_trough[k, row_idx] = (ws - 1) - trough_offset
                k += 1
    return out_peak, out_trough


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12], "dilations": [1, 2]}. "dilations" optional, default [1]."""
    windows = np.array(params['windows'], dtype=np.int64)
    dilations = np.array(params.get('dilations', [1]), dtype=np.int64)
    out_peak, out_trough = _kernel(values, position, windows, dilations)
    arrays = []
    suffixes = []
    k = 0
    for w in params['windows']:
        for d in params.get('dilations', [1]):
            w_tag = f'w{w}' if d == 1 else f'w{w}_d{d}'
            arrays.append(out_peak[k])
            suffixes.append(f'peak_{w_tag}')
            arrays.append(out_trough[k])
            suffixes.append(f'trough_{w_tag}')
            k += 1
    return arrays, suffixes
