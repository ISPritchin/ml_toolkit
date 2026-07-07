"""Флаг разворота тренда и количественное изменение OLS-наклона между двумя периодами.

Signal:
    trend_flip__flag срабатывает, когда знак OLS-наклона поменялся по сравнению с lag месяцев
    назад — сигнал разворота. slope_change_lagL_wW — числовая дельта наклона: насколько быстро
    тренд ускорился или замедлился/сменился. Позволяет отловить «V-образные» восстановления и
    «перевёрнутые V» при заблаговременном снижении.

Formula:
    slope_now  = OLS_slope(v[t-w+1..t])
    slope_ago  = OLS_slope(v[(t-lag)-w+1..(t-lag)])

    flag = 1 if sign(slope_now) != sign(slope_ago) and (|slope_now| > eps or |slope_ago| > eps)
         = 0 otherwise
    slope_change_lagL_wW = slope_now - slope_ago

    flag вычисляется только по первой паре в lag_window_pairs.

Outputs:
    {product}__trend_flip__flag                    — бинарный флаг смены знака тренда
    {product}__trend_flip__slope_change_lag6_w6    — дельта наклона lag=6, w=6
    {product}__trend_flip__slope_change_lag12_w12  — дельта наклона lag=12, w=12

Preset (monthly.yaml):
    trend_flip:
      lag_window_pairs:
        - [6, 6]
        - [12, 12]

Interpretation:
    flag = 1 — тренд поменял знак относительно полугода/года назад (разворот рынка).
    slope_change_lag6_w6 > 0 — краткосрочный тренд ускоряется.
    slope_change_lag12_w12 < 0 при flag = 0 — долгосрочное замедление без смены знака.
    Комбинация flag=1 + slope_change > 0 — восстановление после спада.

Example:
    Ряд (12 мес): [10,20,30,40,50,60, 55,45,35,25,15,5],  пара (lag=6, w=6)
    (t=11; рост в первой половине, падение во второй)

    slope_now (посл. 6 [55..5]) = −10
    slope_ago (6 мес назад, [10..60]) = +10
    знаки противоположны → flag = 1
    slope_change = slope_now − slope_ago = −10 − 10 = −20
    → trend_flip__flag = 1,  slope_change_lag6_w6 = −20  (разворот вниз)

"""

import numba as nb
import numpy as np

from .._windowing import EPS, fit_linear_trend_slope, resolve_window_size

FEATURE = 'trend_flip'


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    lags: np.ndarray,
    windows: np.ndarray,
):
    """lags[j] и windows[j] определяют пару: наклон сейчас vs lag[j] назад за windows[j]."""
    n_rows = product_values.shape[0]
    n_p = lags.shape[0]
    flip_flag = np.zeros(n_rows)
    slope_change = np.zeros((n_p, n_rows))

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_p):
            lag = lags[j]
            w = windows[j]
            if pos >= lag:
                ws_now = resolve_window_size(pos, w)
                slope_now = fit_linear_trend_slope(product_values, row_idx, ws_now)
                pos_ago = position_within_entity[row_idx - lag]
                ws_ago = resolve_window_size(pos_ago, w)
                slope_ago = fit_linear_trend_slope(product_values, row_idx - lag, ws_ago)
                slope_change[j, row_idx] = slope_now - slope_ago
                # flip flag: только для первой пары; требуем оба наклона явно ненулевые
                if j == 0:
                    if (slope_now > EPS and slope_ago < -EPS) or (slope_now < -EPS and slope_ago > EPS):
                        flip_flag[row_idx] = 1.0
    return flip_flag, slope_change


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"lag_window_pairs": [[6, 6], [12, 12]]} — ключ обязателен."""
    pairs = params['lag_window_pairs']
    lags = np.array([p[0] for p in pairs], dtype=np.int64)
    windows = np.array([p[1] for p in pairs], dtype=np.int64)
    flip, slope_ch = _kernel(values, position, lags, windows)
    arrays = [flip]
    suffixes = ['flag']
    for j, p in enumerate(pairs):
        arrays.append(slope_ch[j])
        suffixes.append(f'slope_change_lag{p[0]}_w{p[1]}')
    return arrays, suffixes
