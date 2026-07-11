"""Число активных (ненулевых) месяцев на скользящем окне.

Signal:
    Показывает регулярность присутствия клиента/холдинга в периоде. Высокое значение
    означает стабильный поток поступлений, низкое — прерывистую или сезонную активность.

Formula:
    active_months_w = count(v[i] != 0, i in [t-w+1..t])
    effective_w = min(position+1, w)

Outputs:
    {product}__active_months__w6   — число активных месяцев за 6 мес
    {product}__active_months__w12  — число активных месяцев за 12 мес
    {product}__active_months__w24  — число активных месяцев за 24 мес

Preset (monthly.yaml):
    active_months:
      windows: [6, 12, 24]

Interpretation:
    active_months_w12 = 12 — клиент активен каждый месяц (потоковый, B2C-like).
    active_months_w12 = 3–5 — проектный B2B-клиент с редкими крупными поступлениями.
    active_months_w12 = 0–1 — практически неактивный или новый/чёрствый клиент.
    Сопоставление w6 и w12 позволяет обнаружить затухание активности.

Example:
    Ряд (6 мес): [10, 0, 5, 0, 8, 3]
    (t=5, w=6; окно охватывает весь ряд)

    ненулевые: 10, 5, 8, 3 → 4 шт.
    нулевые:   0, 0       → 2 шт.
    → active_months__w6 = 4

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import resolve_window_size

FEATURE = 'active_months'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            count = 0
            for offset in range(ws):
                if product_values[row_idx - ws + 1 + offset] != 0.0:
                    count += 1
            out[j, row_idx] = count
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12, 24]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f'w{w}' for w in params['windows']]
