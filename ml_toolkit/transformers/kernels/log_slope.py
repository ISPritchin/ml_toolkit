"""OLS-наклон log1p(|v|) на скользящем окне: среднемесячный темп роста в log-шкале.

Signal:
    Наклон в log-шкале интерпретируется как приблизительный среднемесячный процент
    роста (log-slope ≈ ln(1+r) ≈ r при малых r). Устойчив к экспоненциальному
    распределению значений — не взрывается при крупных разовых выбросах.

Formula:
    log_v[i] = log1p(|v[t-w+1+i]|)   для i in [0..w-1]
    log_slope_w = OLS_slope({(i, log_v[i])} for i in [0..ws-1])
                = (ws*sum(i*log_v[i]) - sum(i)*sum(log_v[i])) /
                  (ws*sum(i²) - sum(i)²)

Outputs:
    {product}__log_slope__w6   — log-наклон за 6 мес
    {product}__log_slope__w12  — log-наклон за 12 мес
    {product}__log_slope__w24  — log-наклон за 24 мес

Preset entry:
    log_slope:
      windows: [6, 12, 24]

Interpretation:
    ≈ +0.175/мес — рост ≈ 20% в месяц (экспоненциальный разгон, ln(1.2)).
    ≈ 0 — стагнация в log-шкале.
    < 0 — систематическое снижение.
    log_slope_w6 >> log_slope_w24 — ускорение в последние полгода.

Example:
    Ряд (4 мес): [10, 20, 40, 80],  w=4  (удвоение каждый месяц)

    log1p(|v|) = ln11, ln21, ln41, ln81 ≈ 2.398, 3.045, 3.714, 4.394
    OLS-наклон по точкам (i=0..3, log_v) = 0.666/мес
    → log_slope__w4 = 0.666  (≈ устойчивый экспоненциальный рост)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import fit_linear_trend_slope, resolve_window_size

FEATURE = 'log_slope'


@nb.njit(cache=True)
def _kernel(log_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = log_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            out[j, row_idx] = fit_linear_trend_slope(log_values, row_idx, ws)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6, 12, 24]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    # log1p считается один раз на колонку (векторно), а не в каждом окне заново
    log_values = np.log1p(np.abs(values))
    out = _kernel(log_values, position, windows)
    return [out[j] for j in range(len(windows))], [f'w{w}' for w in params['windows']]
