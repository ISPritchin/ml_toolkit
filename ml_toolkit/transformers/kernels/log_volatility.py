"""Реализованная волатильность в лог-шкале: σ(log_diff) на скользящем окне.

Signal:
    Измеряет «разброс» месячных темпов роста в log-шкале. Высокое значение — нестабильный
    мультипликативный рост (нельзя предсказать следующий месяц). Низкое — стабильный
    ежемесячный темп. Более устойчив к выбросам, чем rolling_std.

Formula:
    log_diff[i] = log1p(|v[i]|) - log1p(|v[i-1]|)   для i in [t-w+2..t]
    ld_mean_w   = mean(log_diff)
    log_vol_w   = sqrt(mean((log_diff[i] - ld_mean_w)²))

    Требует n_diffs >= 1 (ws >= 2).

Outputs:
    {product}__log_volatility__w6   — лог-волатильность за 6 мес
    {product}__log_volatility__w12  — лог-волатильность за 12 мес

Preset (monthly.yaml):
    log_volatility:
      windows: [6, 12]

Interpretation:
    ≈ 0 — темп роста стабилен месяц к месяцу в log-шкале.
    > 0.3 — высокая нестабильность темпов: типично для B2B-проектных клиентов.
    В паре с geometric_return: высокий возврат + высокий log_vol = рискованный рост.
    log_vol_w6 > log_vol_w12 — нестабильность нарастает в последнее время.

Example:
    Ряд (4 мес): [10, 30, 15, 45],  w=4

    log_diff = log1p(30)−log1p(10), log1p(15)−log1p(30), log1p(45)−log1p(15)
             ≈ +1.036, −0.661, +1.056   (n_diffs = 3)
    ld_mean = 0.477
    log_vol = sqrt(mean((ld − 0.477)²)) = 0.805
    → log_volatility__w4 = 0.805  (резкие колебания темпа)
"""

import numba as nb
import numpy as np

from .._windowing import resolve_window_size

FEATURE = "log_volatility"


@nb.njit(cache=True)
def _log_vol(log_values: np.ndarray, row_idx: int, ws: int) -> float:
    n_diffs = ws - 1
    if n_diffs < 1:
        return 0.0
    # сумма лог-разностей телескопируется: lv[t] - lv[t-ws+1]
    ld_mean = (log_values[row_idx] - log_values[row_idx - ws + 1]) / n_diffs
    ld_sq = 0.0
    for offset in range(1, ws):
        abs_idx = row_idx - ws + 1 + offset
        ld = log_values[abs_idx] - log_values[abs_idx - 1]
        ld_sq += (ld - ld_mean) ** 2
    return (ld_sq / n_diffs) ** 0.5


@nb.njit(cache=True)
def _kernel(log_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = log_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            out[j, row_idx] = _log_vol(log_values, row_idx, ws)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6, 12]}"""
    windows = np.array(params["windows"], dtype=np.int64)
    # log1p считается один раз на колонку (векторно)
    log_values = np.log1p(np.abs(values))
    out = _kernel(log_values, position, windows)
    return [out[j] for j in range(len(windows))], [f"w{w}" for w in params["windows"]]
