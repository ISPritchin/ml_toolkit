"""Корреляция Пирсона значений со временем внутри окна: нормированная сила тренда.

Signal:
    Показывает, насколько линеен тренд внутри окна. Близко к +1 — монотонный рост,
    близко к -1 — монотонное падение, близко к 0 — нет линейного тренда (шум или
    U-образная кривая). Дополняет slope: одинаковый slope может иметь разный r.

Formula:
    r_w = Pearson({0, 1, ..., ws-1}, {v[t-ws+1], ..., v[t]})
        = (ws*sum(t*v) - sum(t)*sum(v)) /
          sqrt((ws*sum(t²)-sum(t)²) * (ws*sum(v²)-sum(v)²))

    Требует ws >= 3, иначе 0.

Outputs:
    {product}__corr_with_time__w6   — корреляция со временем за 6 мес
    {product}__corr_with_time__w12  — корреляция со временем за 12 мес

Preset entry:
    corr_with_time:
      windows: [6, 12]

Interpretation:
    |r| > 0.9 — очень чистый линейный тренд; slope значим и надёжен.
    |r| < 0.3 — хаотический ряд или нелинейная кривая (проверь nonlinearity).
    r_w6 >> r_w12 — последние полгода значительно линейнее, чем вся история.
    Близкие r_w6 и r_w12 при высоких значениях = стабильный долгосрочный тренд.

Example:
    Ряд (6 мес): [10, 30, 20, 40, 30, 50]
    (t=5, w=6; время t=0..5)

    sum(t)=15, sum(v)=180, sum(t·v)=560, sum(t²)=55, sum(v²)=6400
    cov = 6·560 − 15·180 = 3360 − 2700 = 660
    var = sqrt((6·55−15²)(6·6400−180²)) = sqrt(105·6000) = 793.7
    → corr_with_time__w6 = 660/793.7 = 0.832  (растущий, но зубчатый тренд)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import EPS, resolve_window_size

FEATURE = 'corr_with_time'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            if ws >= 3:
                st = 0.0
                sv = 0.0
                stv = 0.0
                st2 = 0.0
                sv2 = 0.0
                for offset in range(ws):
                    t = float(offset)
                    v = product_values[row_idx - ws + 1 + offset]
                    st += t
                    sv += v
                    stv += t * v
                    st2 += t * t
                    sv2 += v * v
                cov = ws * stv - st * sv
                var = ((ws * st2 - st * st) * (ws * sv2 - sv * sv)) ** 0.5
                out[j, row_idx] = cov / var if var > EPS else 0.0
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6, 12, 24]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f'w{w}' for w in params['windows']]
