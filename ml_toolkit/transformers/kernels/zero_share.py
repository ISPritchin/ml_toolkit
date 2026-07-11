"""Доля нулевых (неактивных) месяцев в скользящем окне: интенсивность использования.

Signal:
    Базовый показатель регулярности: какая доля месяцев в окне была «тихой» (нулевой оборот).
    Дополняет activity_rate, который считает expanding-долю: zero_share считает только окно
    фиксированного размера, что позволяет сравнивать клиентов на разных этапах жизненного цикла
    с одинаковым горизонтом. Три окна (3, 6, 12) улавливают краткосрочные и долгосрочные режимы.

Formula:
    zero_share_w = count(v[i] == 0, i in [t-w+1..t]) / w

    Включает текущий месяц. При w < w_preset используется фактически доступная глубина.

Outputs:
    {product}__zero_share__w3   — доля нулей за 3 мес
    {product}__zero_share__w6   — доля нулей за 6 мес
    {product}__zero_share__w12  — доля нулей за 12 мес

Preset (monthly.yaml):
    zero_share:
      windows: [3, 6, 12]

Interpretation:
    = 0.0 — все месяцы в окне активны (нет ни одного нулевого).
    = 1.0 — клиент полностью неактивен в окне (все месяцы нулевые).
    zero_share_w3 > zero_share_w12 — активность снижается именно в последнее время.
    zero_share_w3 = 0 при zero_share_w12 = 0.5 — восстановился после длинного перерыва.

Example:
    Ряд (6 мес): [10, 0, 0, 10, 0, 10],  w=6

    нулевых месяцев: idx1, idx2, idx4 → 3 из 6
    zero_share = 3 / 6
    → zero_share__w6 = 0.5  (половина месяцев неактивна)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import resolve_window_size

FEATURE = 'zero_share'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            zero_count = 0
            for offset in range(ws):
                if product_values[row_idx - ws + 1 + offset] == 0.0:
                    zero_count += 1
            out[j, row_idx] = zero_count / ws
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f'w{w}' for w in params['windows']]
