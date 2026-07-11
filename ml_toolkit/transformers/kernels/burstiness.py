"""Паттерны проектного дохода: всплески, burst-периоды, соотношение peak/mean.

Signal:
    Разграничивает потоковых клиентов (равномерный доход) от проектных (редкие
    крупные всплески с долгими паузами). Высокий peak_mean при низком burst_count
    указывает на единственный крупный контракт.

Formula:
    peak_mean_w    = max(v[i], i in окне) / (|mean_w| + eps)
    peak_med_w     = max(v[i]) / (|median_w| + eps)
    burst_count_w  = число переходов 0 → ненулевое в окне
    burst_dur_w    = mean длин непрерывных ненулевых серий
    burst_cv_w     = std(длины серий) / (mean_длины + eps)
    gap_mean_w     = zero_count / max(burst_count, 1)
    calm_share_w   = zero_count / ws

Outputs:
    {product}__burstiness__peak_mean_w12    — пик/среднее за 12 мес
    {product}__burstiness__peak_med_w12     — пик/медиана за 12 мес
    {product}__burstiness__gap_mean_w12     — среднее число нулей между вспышками
    {product}__burstiness__burst_count_w12  — число вспышек активности
    {product}__burstiness__burst_dur_w12    — средняя длина вспышки
    {product}__burstiness__burst_cv_w12     — CV длин вспышек
    {product}__burstiness__calm_share_w12   — доля нулевых месяцев

Preset (monthly.yaml):
    burstiness:
      windows: [12]

Interpretation:
    peak_mean_w12 > 4, calm_share > 0.5 — типичный B2B-проектный клиент.
    peak_mean_w12 ≈ 1.1, calm_share ≈ 0 — потоковый клиент без всплесков.
    burst_cv_w12 = 0 — строго равномерные по длине вспышки (ритмичный клиент).
    gap_mean_w12 = 2, burst_count = 4 — поступления раз в квартал.

Example:
    Ряд (6 мес): [0, 40, 30, 0, 20, 0],  w=6
    mean = 90/6 = 15,  median = 20 (6-й элемент сорт. [0,0,0,20,30,40]),  max = 40

    вспышки (серии ненулевых): [40,30] и [20] → burst_count = 2
    длины вспышек: 2, 1 → burst_dur = (2+1)/2 = 1.5
    нулей: 3 → gap_mean = 3/2 = 1.5,  calm_share = 3/6 = 0.5
    → burstiness__peak_mean_w6  = 40/15 = 2.667
    → burstiness__peak_med_w6   = 40/20 = 2.0
    → burstiness__burst_count_w6 = 2,  burst_dur_w6 = 1.5

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import (
    EPS,
    compute_window_mean,
    fill_window_sorted,
    resolve_window_size,
    safe_ratio,
    sorted_median,
)

FEATURE = 'burstiness'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_peak_mean = np.zeros((n_w, n_rows))
    out_peak_med = np.zeros((n_w, n_rows))
    out_gap_mean = np.zeros((n_w, n_rows))
    out_burst_count = np.zeros((n_w, n_rows))
    out_burst_dur = np.zeros((n_w, n_rows))
    out_burst_cv = np.zeros((n_w, n_rows))
    out_calm_share = np.zeros((n_w, n_rows))

    max_w = 1
    for j in range(n_w):
        max_w = max(max_w, windows[j])
    sorted_buf = np.empty(max_w)
    burst_durs = np.zeros(max_w)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            mean = compute_window_mean(product_values, row_idx, ws)
            v_max = product_values[row_idx - ws + 1]
            for offset in range(1, ws):
                vv = product_values[row_idx - ws + 1 + offset]
                v_max = max(v_max, vv)

            fill_window_sorted(sorted_buf, product_values, row_idx, ws)
            median = sorted_median(sorted_buf, ws)

            out_peak_mean[j, row_idx] = safe_ratio(v_max, mean)
            out_peak_med[j, row_idx] = safe_ratio(v_max, median)

            # burst analysis: count bursts (transitions 0 → nonzero), their durations, gaps
            burst_count = 0
            total_burst_dur = 0
            zero_count = 0
            in_burst = False
            cur_dur = 0
            n_bursts_tracked = 0

            for offset in range(ws):
                abs_idx = row_idx - ws + 1 + offset
                vv = product_values[abs_idx]
                active = vv != 0.0
                if active:
                    if not in_burst:
                        burst_count += 1
                        in_burst = True
                    cur_dur += 1
                else:
                    if in_burst:
                        if n_bursts_tracked < ws:
                            burst_durs[n_bursts_tracked] = cur_dur
                            n_bursts_tracked += 1
                        total_burst_dur += cur_dur
                        cur_dur = 0
                        in_burst = False
                    zero_count += 1
            if in_burst:
                if n_bursts_tracked < ws:
                    burst_durs[n_bursts_tracked] = cur_dur
                    n_bursts_tracked += 1
                total_burst_dur += cur_dur

            out_burst_count[j, row_idx] = burst_count
            if burst_count > 0:
                mean_dur = total_burst_dur / burst_count
                out_burst_dur[j, row_idx] = mean_dur
                if n_bursts_tracked >= 2:
                    var_dur = 0.0
                    for bi in range(n_bursts_tracked):
                        var_dur += (burst_durs[bi] - mean_dur) ** 2
                    out_burst_cv[j, row_idx] = (var_dur / n_bursts_tracked) ** 0.5 / (mean_dur + EPS)

            out_gap_mean[j, row_idx] = zero_count / max(burst_count, 1)
            out_calm_share[j, row_idx] = zero_count / ws

    return out_peak_mean, out_peak_med, out_gap_mean, out_burst_count, out_burst_dur, out_burst_cv, out_calm_share


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    pm, pmed, gm, bc, bd, bcv, cs = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(pm[j])
        suffixes.append(f'peak_mean_w{w}')
        arrays.append(pmed[j])
        suffixes.append(f'peak_med_w{w}')
        arrays.append(gm[j])
        suffixes.append(f'gap_mean_w{w}')
        arrays.append(bc[j])
        suffixes.append(f'burst_count_w{w}')
        arrays.append(bd[j])
        suffixes.append(f'burst_dur_w{w}')
        arrays.append(bcv[j])
        suffixes.append(f'burst_cv_w{w}')
        arrays.append(cs[j])
        suffixes.append(f'calm_share_w{w}')
    return arrays, suffixes
