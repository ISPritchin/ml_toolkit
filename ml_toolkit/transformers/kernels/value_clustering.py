"""Концентрация значений ряда: top-N доли, Herfindahl и плотность.

Signal:
    Описывает, насколько объём сосредоточен в нескольких «пиковых» месяцах. Высокая
    top1_share — один месяц доминирует (сезонный выброс или единоразовый крупный всплеск).
    Herfindahl > 1/w — концентрация выше равномерного распределения. density — насколько
    активные месяцы «работают» на уровне максимума, а не тихо присутствуют.

Formula:
    total_w  = sum(v[t-w+1..t])
    top1_share_w  = safe_ratio(max_v, total_w)
    top3_share_w  = safe_ratio(sum(top-3 values), total_w)
    bot3_share_w  = safe_ratio(sum(bot-3 values), total_w)
    concentration_w = safe_ratio(top3_sum, bot3_sum)  (0 при bot3_sum ~ 0 —
        «полюсное» отношение не определено, когда нижние месяцы нулевые)
    density_w     = safe_ratio(total_w, n_active * max_v)
    herfindahl_w  = sum((v_i / total_w)^2, i in window)

Outputs:
    {product}__value_clustering__top1_share_w12    — доля топ-1 месяца в объёме
    {product}__value_clustering__top3_share_w12    — доля топ-3 месяцев в объёме
    {product}__value_clustering__bot3_share_w12    — доля нижних-3 месяцев в объёме
    {product}__value_clustering__concentration_w12 — top3 / bot3 (отношение полюсов)
    {product}__value_clustering__density_w12       — плотность: total / (n_active * max)
    {product}__value_clustering__herfindahl_w12    — индекс Херфиндаля по месяцам

Preset entry:
    value_clustering:
      windows: [12]

Interpretation:
    top1_share = 0.5 — половина годового объёма пришлась в один месяц (сезонный пик).
    herfindahl = 1/12 ≈ 0.083 — идеально равномерное распределение по месяцам.
    herfindahl > 0.25 — сильная концентрация (условно: «доминирует 1 из 4»).
    density ≈ 1.0 — все активные месяцы работают на уровне максимума (нет «тихих» пиков).

Example:
    Ряд (6 мес): [10, 10, 10, 10, 10, 50],  w=6
    total = 100

    top1_share = max/total = 50/100 = 0.5
    herfindahl = Σ(v_i/total)² = 5·(0.1)² + (0.5)² = 0.05 + 0.25 = 0.3
    → value_clustering__top1_share_w6 = 0.5,  herfindahl_w6 = 0.3  (один мес. доминирует)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import (
    EPS,
    compute_window_sum,
    fill_window_sorted,
    resolve_window_size,
    safe_ratio,
)

FEATURE = 'value_clustering'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_top1 = np.zeros((n_w, n_rows))
    out_top3 = np.zeros((n_w, n_rows))
    out_bot3 = np.zeros((n_w, n_rows))
    out_conc = np.zeros((n_w, n_rows))
    out_density = np.zeros((n_w, n_rows))
    out_herf = np.zeros((n_w, n_rows))

    max_w = 1
    for j in range(n_w):
        max_w = max(max_w, windows[j])
    sorted_buf = np.empty(max_w)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            total = compute_window_sum(product_values, row_idx, ws)
            if abs(total) < EPS:
                continue
            fill_window_sorted(sorted_buf, product_values, row_idx, ws)
            # top-1: largest element (last in sorted asc)
            top1 = sorted_buf[ws - 1]
            top3_sum = 0.0
            bot3_sum = 0.0
            n_top = min(3, ws)
            for i in range(n_top):
                top3_sum += sorted_buf[ws - 1 - i]
                bot3_sum += sorted_buf[i]

            out_top1[j, row_idx] = safe_ratio(top1, total)
            out_top3[j, row_idx] = safe_ratio(top3_sum, total)
            out_bot3[j, row_idx] = safe_ratio(bot3_sum, total)
            # bot3_sum = 0 (нулевые месяцы) -> отношение не определено -> 0
            out_conc[j, row_idx] = safe_ratio(top3_sum, bot3_sum)

            # active months + max for density
            v_max = 0.0
            active = 0
            for offset in range(ws):
                vv = product_values[row_idx - ws + 1 + offset]
                v_max = max(v_max, vv)
                if vv != 0.0:
                    active += 1
            out_density[j, row_idx] = safe_ratio(total, active * v_max)

            # herfindahl: Σ(share²)
            herf = 0.0
            for offset in range(ws):
                s = product_values[row_idx - ws + 1 + offset] / total
                herf += s * s
            out_herf[j, row_idx] = herf

    return out_top1, out_top3, out_bot3, out_conc, out_density, out_herf


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    t1, t3, b3, conc, dens, herf = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(t1[j])
        suffixes.append(f'top1_share_w{w}')
        arrays.append(t3[j])
        suffixes.append(f'top3_share_w{w}')
        arrays.append(b3[j])
        suffixes.append(f'bot3_share_w{w}')
        arrays.append(conc[j])
        suffixes.append(f'concentration_w{w}')
        arrays.append(dens[j])
        suffixes.append(f'density_w{w}')
        arrays.append(herf[j])
        suffixes.append(f'herfindahl_w{w}')
    return arrays, suffixes
