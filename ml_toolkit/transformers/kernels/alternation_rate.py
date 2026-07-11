"""Доля смен направления приращений, доля максимального скачка и средний прыжок.

Signal:
    Характеризует «нервозность» ряда: как часто движение вверх/вниз меняется на
    противоположное (alternation_rate), доминирует ли один скачок над всей вариацией
    (max_jump_share) и каков средний размер приращения (mean_abs_jump).

Formula:
    TV_w = sum(|v[i] - v[i-1]|, i in [t-w+2..t])
    alternation_rate_w = count(sign(d[i]) != sign(d[i-1])) / (n_pairs - 1)
        где d[i] = v[i] - v[i-1], знак игнорирует нули
    max_jump_share_w = max(|d[i]|) / (TV_w + eps)
    mean_abs_jump_w  = TV_w / (n_pairs)

Outputs:
    {product}__alternation_rate__alt_rate_w6    — доля смен направления за 6 мес
    {product}__alternation_rate__max_jump_share_w6 — доля крупнейшего скачка в TV
    {product}__alternation_rate__mean_abs_jump_w6  — средний |прирост| за 6 мес
    (аналогично для w12)

Preset (monthly.yaml):
    alternation_rate:
      windows: [6, 12]

Interpretation:
    alt_rate = 1.0 — каждый месяц смена направления (пилообразный ряд).
    alt_rate = 0.0 — монотонный тренд без разворотов.
    max_jump_share > 0.5 — один скачок доминирует; может быть контрактным разовым платежом.

Example:
    Ряд (6 мес): [10, 30, 20, 40, 30, 50]
    (t=5, w=6; приращения d[i] для i=1..5)

    d = +20, −10, +20, −10, +20   (n_pairs = 5)
    TV = 20+10+20+10+20 = 80,  max_jump = 20
    знаки: +,−,+,−,+ → смен направления 4 из 4 пар
    → alternation_rate__alt_rate_w6   = 4/4 = 1.0
    → alternation_rate__max_jump_share_w6 = 20/80 = 0.25
    → alternation_rate__mean_abs_jump_w6  = 80/5 = 16.0

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import resolve_window_size, safe_ratio

FEATURE = 'alternation_rate'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_alt = np.zeros((n_w, n_rows))
    out_max_share = np.zeros((n_w, n_rows))
    out_mean_jump = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            n_pairs = ws - 1
            if n_pairs < 1:
                continue
            tv = 0.0
            max_jump = 0.0
            alternations = 0
            prev_sign = 0
            for offset in range(1, ws):
                abs_idx = row_idx - ws + 1 + offset
                d_abs = abs(product_values[abs_idx] - product_values[abs_idx - 1])
                tv += d_abs
                max_jump = max(max_jump, d_abs)
                raw_d = product_values[abs_idx] - product_values[abs_idx - 1]
                cur_sign = 1 if raw_d > 0.0 else (-1 if raw_d < 0.0 else 0)
                if offset >= 2 and prev_sign != 0 and cur_sign != 0 and cur_sign != prev_sign:
                    alternations += 1
                if cur_sign != 0:
                    prev_sign = cur_sign
            out_max_share[j, row_idx] = safe_ratio(max_jump, tv)
            out_mean_jump[j, row_idx] = tv / n_pairs
            if n_pairs >= 2:
                out_alt[j, row_idx] = alternations / (n_pairs - 1)
    return out_alt, out_max_share, out_mean_jump


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    out_alt, out_max, out_mean = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(out_alt[j])
        suffixes.append(f'alt_rate_w{w}')
        arrays.append(out_max[j])
        suffixes.append(f'max_jump_share_w{w}')
        arrays.append(out_mean[j])
        suffixes.append(f'mean_abs_jump_w{w}')
    return arrays, suffixes
