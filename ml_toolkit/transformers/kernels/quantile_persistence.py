"""Устойчивость квантильного положения: время в зонах, trend рангов, стабильность.

Signal:
    Показывает, стабильно ли ряд удерживает «хорошие» или «плохие» позиции
    в своей истории. Высокий above_med при высоком q_stability — устойчиво выше медианы.
    rank_trend — улучшается ли позиция ряда в последнее время.

Formula:
    Из sorted buffer окна определяются median (честная: среднее двух центральных
    при чётном ws) и p25/p75 (единая конвенция: sorted[int(q * (ws - 1))]).
    above_med_w   = count(v[i] > median) / ws
    top_q_w       = count(v[i] >= p75) / ws
    bot_q_w       = count(v[i] <= p25) / ws
    rank[i]       = count(v[j] <= v[i]) / ws (перцентильный ранг)
    rank_trend_w  = OLS slope(rank[i] последних half_w) — улучшается ли ранг
                    (считается только при ws >= 2)
    q_stability_w = 1 - CV(rank[i]) ∈ (-inf, 1]
    above_ewma_w  = count(v[i] > EWMA_alpha03_now) / ws

Outputs:
    {product}__quantile_persistence__above_med_w12   — доля мес. выше медианы
    {product}__quantile_persistence__top_q_w12       — доля мес. в верхнем квартиле
    {product}__quantile_persistence__bot_q_w12       — доля мес. в нижнем квартиле
    {product}__quantile_persistence__rank_trend_w12  — тренд рангов (улучшение/ухудшение)
    {product}__quantile_persistence__q_stability_w12 — 1 - CV рангов
    {product}__quantile_persistence__above_ewma_w12  — доля мес. выше EWMA

Preset entry:
    quantile_persistence:
      windows: [12]

Interpretation:
    above_med = 0.5, q_stability = 0 — ряд линейно растёт (симметрично пересекает медиану).
    top_q = 0.9 при q_stability > 0.8 — стабильно в верхнем квартиле последние 12 мес.
    rank_trend > 0 — ранги последовательно растут (ряд улучшает позицию).
    bot_q > 0.5 — более половины месяцев в нижнем квартиле: хронически низкий уровень.

Example:
    Ряд (6 мес): [10, 20, 30, 40, 50, 60],  w=6
    median = (30 + 40) / 2 = 35

    выше медианы (> 35): 40, 50, 60 → above_med = 3/6 = 0.5
    ранги растут монотонно → rank_trend > 0 (позиция улучшается)
    → quantile_persistence__above_med_w6 = 0.5,  rank_trend_w6 > 0

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import (
    EPS,
    fill_window_sorted,
    resolve_window_size,
    sorted_median,
    sorted_quantile,
)

FEATURE = 'quantile_persistence'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_above_med = np.zeros((n_w, n_rows))
    out_top_q = np.zeros((n_w, n_rows))
    out_bot_q = np.zeros((n_w, n_rows))
    out_rank_trend = np.zeros((n_w, n_rows))
    out_q_stability = np.zeros((n_w, n_rows))
    out_above_ewma = np.zeros((n_w, n_rows))

    max_w = 1
    for j in range(n_w):
        max_w = max(max_w, windows[j])
    sorted_buf = np.empty(max_w)
    ranks = np.empty(max_w)

    # running EWMA alpha=0.3 state
    r_ewma = 0.0
    alpha = 0.3

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        if pos == 0:
            r_ewma = product_values[row_idx]
        else:
            r_ewma = alpha * product_values[row_idx] + (1.0 - alpha) * r_ewma
        ewma_now = r_ewma

        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            fill_window_sorted(sorted_buf, product_values, row_idx, ws)
            median = sorted_median(sorted_buf, ws)
            p25 = sorted_quantile(sorted_buf, ws, 0.25)
            p75 = sorted_quantile(sorted_buf, ws, 0.75)

            above_med = 0
            top_q = 0
            bot_q = 0
            above_ewma_cnt = 0
            for offset in range(ws):
                abs_idx = row_idx - ws + 1 + offset
                v = product_values[abs_idx]
                if v > median:
                    above_med += 1
                if v >= p75:
                    top_q += 1
                if v <= p25:
                    bot_q += 1
                if v > ewma_now:
                    above_ewma_cnt += 1
                # rank within window: доля точек <= v (бинарный поиск по sorted_buf)
                ranks[offset] = np.searchsorted(sorted_buf[:ws], v, side='right') / ws

            out_above_med[j, row_idx] = above_med / ws
            out_top_q[j, row_idx] = top_q / ws
            out_bot_q[j, row_idx] = bot_q / ws
            out_above_ewma[j, row_idx] = above_ewma_cnt / ws

            # rank trend: OLS slope of ranks over recent half window.
            # При ws < 2 не считается: half > ws означал бы чтение за границей буфера.
            if ws >= 2:
                half = ws // 2
                half = max(half, 2)
                start = ws - half
                mean_r = 0.0
                for i in range(half):
                    mean_r += ranks[start + i]
                mean_r /= half
                sx = 0.0
                sxy = 0.0
                for i in range(half):
                    dx = i - (half - 1) / 2.0
                    sx += dx * dx
                    sxy += dx * (ranks[start + i] - mean_r)
                out_rank_trend[j, row_idx] = sxy / (sx + EPS)

            # quartile stability: 1 - CV of ranks
            mean_all_r = 0.0
            for i in range(ws):
                mean_all_r += ranks[i]
            mean_all_r /= ws
            var_r = 0.0
            for i in range(ws):
                var_r += (ranks[i] - mean_all_r) ** 2
            std_r = (var_r / ws) ** 0.5
            out_q_stability[j, row_idx] = 1.0 - std_r / (mean_all_r + EPS)

    return out_above_med, out_top_q, out_bot_q, out_rank_trend, out_q_stability, out_above_ewma


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    am, tq, bq, rt, qs, ae = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(am[j])
        suffixes.append(f'above_med_w{w}')
        arrays.append(tq[j])
        suffixes.append(f'top_q_w{w}')
        arrays.append(bq[j])
        suffixes.append(f'bot_q_w{w}')
        arrays.append(rt[j])
        suffixes.append(f'rank_trend_w{w}')
        arrays.append(qs[j])
        suffixes.append(f'q_stability_w{w}')
        arrays.append(ae[j])
        suffixes.append(f'above_ewma_w{w}')
    return arrays, suffixes
