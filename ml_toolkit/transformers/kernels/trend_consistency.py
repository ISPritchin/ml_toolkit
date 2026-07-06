"""Чистота тренда: согласованность направлений, noise-to-signal, R² и длина чистой серии.

Signal:
    Оценивает, насколько «чист» тренд: если slope положительный, то доля месячных приростов
    с тем же знаком — dir_consistency. noise_signal измеряет остаточный шум (RMSE residuals)
    относительно амплитуды тренда. R² — доля дисперсии, объяснённой линейным трендом.
    clean_streak — длинейшая непрерывная серия шагов, согласованных с трендом.

Formula:
    slope     = OLS_slope(v[t-w+1..t])
    slope_sign = sign(slope)

    dir_consistency_w  = count(sign(diff_i) == slope_sign) / (w-1)
    noise_signal_w     = RMSE_residuals / (|slope| * w + eps)
    r_squared_w        = 1 - SS_res / (SS_tot + eps)
    clean_streak_w     = max_run(sign(diff_i) == slope_sign)
    sub_sign_consist_w = share of 3-month sub-windows with slope matching slope_sign

Outputs:
    {product}__trend_consistency__dir_consistency_w6    — согласованность направлений, 6 мес
    {product}__trend_consistency__noise_signal_w6       — noise-to-signal, 6 мес
    {product}__trend_consistency__clean_streak_w6       — длина чистой серии, 6 мес
    {product}__trend_consistency__sub_sign_consist_w6   — доля под-окон с верным знаком, 6 мес
    {product}__trend_consistency__r_squared_w6          — R² линейного тренда, 6 мес
    {product}__trend_consistency__dir_consistency_w12   — то же за 12 мес
    {product}__trend_consistency__noise_signal_w12      — noise-to-signal за 12 мес
    {product}__trend_consistency__clean_streak_w12      — длина чистой серии, 12 мес
    {product}__trend_consistency__sub_sign_consist_w12  — доля под-окон с верным знаком, 12 мес
    {product}__trend_consistency__r_squared_w12         — R² линейного тренда, 12 мес

Preset (monthly.yaml):
    trend_consistency:
      windows: [6, 12]

Interpretation:
    dir_consistency = 1.0 — каждый шаг совпадает с общим трендом (идеальная монотонность).
    noise_signal = 0 — ряд лежит точно на линии тренда (нет шума).
    R² = 0.95 — 95% дисперсии объяснено линейным трендом; клиент предсказуемо растёт/падает.
    clean_streak_w12 = 10 при dir_consistency_w12 = 0.9 — один откат за весь год.

Example:
    Ряд (6 мес): [10, 20, 30, 40, 50, 60],  w=6  (ровный рост)

    slope > 0, все 5 приращений = +10 (совпадают со знаком тренда)
    dir_consistency = 5/5 = 1.0
    точки лежат точно на линии → ss_res = 0 → r_squared = 1.0
    → trend_consistency__dir_consistency_w6 = 1.0,  r_squared_w6 = 1.0,  clean_streak_w6 = 5
"""

import numba as nb
import numpy as np

from .._windowing import EPS, fit_linear_trend_slope, resolve_window_size, safe_ratio

FEATURE = "trend_consistency"


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_dir_consistency = np.zeros((n_w, n_rows))
    out_noise_signal = np.zeros((n_w, n_rows))
    out_clean_streak = np.zeros((n_w, n_rows))
    out_sub_sign_consist = np.zeros((n_w, n_rows))
    out_r_squared = np.zeros((n_w, n_rows))

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            start = row_idx - ws + 1
            slope = fit_linear_trend_slope(product_values, row_idx, ws)
            slope_sign = 1 if slope > 0 else (-1 if slope < 0 else 0)

            # direction_consistency: доля diff с тем же знаком, что и slope
            consistent = 0
            n_diffs = ws - 1
            if n_diffs > 0:
                for i in range(1, ws):
                    d = product_values[start + i] - product_values[start + i - 1]
                    d_sign = 1 if d > 0.0 else (-1 if d < 0.0 else 0)
                    if d_sign == slope_sign and slope_sign != 0:
                        consistent += 1
                out_dir_consistency[j, row_idx] = consistent / n_diffs

            # residuals: noise_to_signal and R²
            mean = 0.0
            for i in range(ws):
                mean += product_values[start + i]
            mean /= ws
            # intercept = mean - slope * (ws-1)/2
            intercept = mean - slope * (ws - 1) / 2.0
            ss_res = 0.0; ss_tot = 0.0
            for i in range(ws):
                pred = intercept + slope * i
                res = product_values[start + i] - pred
                ss_res += res * res
                ss_tot += (product_values[start + i] - mean) ** 2
            r2 = 1.0 - ss_res / (ss_tot + EPS)
            out_r_squared[j, row_idx] = r2
            # осциллирующий ряд с нулевым наклоном давал rmse/eps ~ 1e10
            out_noise_signal[j, row_idx] = safe_ratio((ss_res / ws) ** 0.5, abs(slope) * ws)

            # clean_trend_streak: longest run of diffs consistent with slope
            best_run = 0; cur_run = 0
            for i in range(1, ws):
                d = product_values[start + i] - product_values[start + i - 1]
                d_sign = 1 if d > 0.0 else (-1 if d < 0.0 else 0)
                if d_sign == slope_sign and slope_sign != 0:
                    cur_run += 1
                    if cur_run > best_run:
                        best_run = cur_run
                else:
                    cur_run = 0
            out_clean_streak[j, row_idx] = best_run

            # sub_slope_sign_consistency: доля sub-slopes с тем же знаком
            sub_len = 3
            n_subs = ws // sub_len
            if n_subs >= 2:
                same_sign = 0
                for s in range(n_subs):
                    sub_end = (s + 1) * sub_len
                    if sub_end > ws:
                        break
                    # под-окно заканчивается на абсолютном индексе start + sub_end - 1
                    sub_slope = fit_linear_trend_slope(product_values, start + sub_end - 1, sub_len)
                    ss = 1 if sub_slope > 0 else (-1 if sub_slope < 0 else 0)
                    if ss == slope_sign and slope_sign != 0:
                        same_sign += 1
                out_sub_sign_consist[j, row_idx] = same_sign / n_subs

    return out_dir_consistency, out_noise_signal, out_clean_streak, out_sub_sign_consist, out_r_squared


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6, 12]}"""
    windows = np.array(params["windows"], dtype=np.int64)
    dc, ns, cs, ssc, r2 = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params["windows"]):
        arrays.append(dc[j]);  suffixes.append(f"dir_consistency_w{w}")
        arrays.append(ns[j]);  suffixes.append(f"noise_signal_w{w}")
        arrays.append(cs[j]);  suffixes.append(f"clean_streak_w{w}")
        arrays.append(ssc[j]); suffixes.append(f"sub_sign_consist_w{w}")
        arrays.append(r2[j]);  suffixes.append(f"r_squared_w{w}")
    return arrays, suffixes
