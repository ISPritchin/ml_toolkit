"""Auto-detected period: доминирующий период ряда через пик автокорреляции, без предположения лага.

Signal:
    seasonal_autocorr проверяет ЗАРАНЕЕ выбранные лаги (6 и 12) — если реальный период другой
    (например, 3 или 9 месяцев), сигнал пропущен. Этот кернел ищет период сам: перебирает лаги
    в диапазоне [min_lag..max_lag], берёт лаг с максимальной windowed-корреляцией — «бедный, но
    честный» аналог спектрального пика без FFT (numba не умеет np.fft в nopython-режиме — поиск
    пика по автокорреляции работает как замена для обнаружения доминирующего периода).

Formula:
    Для lag in [min_lag..max_lag], где ws >= lag + 2:
        r_lag = windowed_lag_pearson(v, row_idx, ws, lag)
    dominant_period_w   = lag*, при котором r_lag максимален (при равенстве — наименьший
        лаг: гармоники периода не должны маскировать сам период)
    dominant_strength_w = r_lag* (сила корреляции на найденном периоде)

    Если ни один лаг не валиден — period=0, strength=0 (недостаточно истории).

Outputs:
    {product}__auto_period__period_w24   — обнаруженный доминирующий период (в шагах)
    {product}__auto_period__strength_w24 — сила автокорреляции на этом периоде

Preset entry:
    auto_period:
      windows: [24]
      min_lag: 2
      max_lag: 12

Interpretation:
    period=6, strength=0.85 — устойчивый полугодовой цикл, сильнее, чем на любом другом
        лаге в проверенном диапазоне.
    strength низкий (< 0.3) при любом period — периодичности как таковой нет, найденный
        лаг — просто наименее плохой из проверенных, не воспринимайте period как факт.
    period около min_lag или max_lag — вероятно, реальный период за пределами диапазона:
        расширьте min_lag/max_lag и проверьте снова.

Example:
    Ряд (12 мес): [10,30,10,30,10,30,10,30,10,30,10,30],  w=12,  min_lag=2, max_lag=4
    (чёткий бимесячный ритм — период 2)

    r_lag2 = 1.0   (пары (10,10),(30,30),... идеально совпадают — та же фаза)
    r_lag3 = -1.0  (полупериод — точная противофаза, не считается «периодом»)
    r_lag4 = 1.0   (гармоника периода 2, совпадает по силе с lag2, но lag2 найден раньше)
    → auto_period__period_w12 = 2.0,  strength_w12 = 1.0

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import resolve_window_size, windowed_lag_pearson

FEATURE = 'auto_period'


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    windows: np.ndarray,
    min_lag: int,
    max_lag: int,
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_period = np.zeros((n_w, n_rows))
    out_strength = np.zeros((n_w, n_rows))

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            best_lag = 0
            best_r = -2.0  # ниже любого валидного r (r ∈ [-1, 1])
            for lag in range(min_lag, max_lag + 1):
                if ws < lag + 2:
                    break
                r = windowed_lag_pearson(product_values, row_idx, ws, lag)
                if r > best_r:
                    best_r = r
                    best_lag = lag
            if best_lag > 0:
                out_period[j, row_idx] = float(best_lag)
                out_strength[j, row_idx] = best_r

    return out_period, out_strength


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [24], "max_lag": 12, "min_lag": 2 (опционально)}."""
    windows = np.array(params['windows'], dtype=np.int64)
    max_lag = int(params['max_lag'])
    min_lag = int(params.get('min_lag', 2))
    period, strength = _kernel(values, position, windows, min_lag, max_lag)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(period[j])
        suffixes.append(f'period_w{w}')
        arrays.append(strength[j])
        suffixes.append(f'strength_w{w}')
    return arrays, suffixes
