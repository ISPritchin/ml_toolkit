"""OLS-наклон (линейный тренд) на скользящем окне.

Signal:
    Направление и скорость тренда: на сколько единиц в месяц в среднем меняется доход
    за последние w месяцев. Положительный — рост, отрицательный — падение. Масштаб
    зависит от единиц продуктовой колонки (не нормирован).

Formula:
    Для окна n = effective_w (позиции 0..n-1, значения v[t-n+1..t]):
        slope_w = (n*sum(i*v[i]) - sum(i)*sum(v[i])) /
                  (n*sum(i²) - (sum(i))²)

    При n < 2 возвращает 0.

Outputs:
    {product}__slope__w6   — OLS-наклон за 6 мес
    {product}__slope__w12  — OLS-наклон за 12 мес
    {product}__slope__w24  — OLS-наклон за 24 мес

Preset (monthly.yaml):
    slope:
      windows: [6, 12, 24]

Interpretation:
    = +5.0 — рост +5 единиц в месяц (пример ряда G, slope_w6).
    > 0 — восходящий тренд в окне.
    < 0 — нисходящий тренд.
    slope_w6 > slope_w12 > 0 — ускорение роста: последние полгода наклон крутче.
    В паре с corr_with_time: slope + |r| ≈ 1 = надёжный линейный тренд.

Example:
    Ряд (5 мес): [10, 20, 30, 40, 50]
    (t=4, w=5; i=0..4 — позиции внутри окна)

    sum(i)   = 0+1+2+3+4 = 10
    sum(v)   = 150,  sum(i·v) = 0+20+60+120+200 = 400,  sum(i²) = 30
    slope    = (5·400 − 10·150) / (5·30 − 10²) = (2000−1500)/(150−100) = 500/50 = 10.0
    → slope__w5 = 10.0  (ровный рост +10 ед/мес)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import fit_linear_trend_slope, resolve_window_size

FEATURE = 'slope'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            out[j, row_idx] = fit_linear_trend_slope(product_values, row_idx, ws)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6, 12, 24]}.

    Returns:
        (arrays, suffixes) — по одному массиву и суффиксу на каждое окно.

    """
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    arrays = [out[j] for j in range(len(windows))]
    suffixes = [f'w{w}' for w in params['windows']]
    return arrays, suffixes
