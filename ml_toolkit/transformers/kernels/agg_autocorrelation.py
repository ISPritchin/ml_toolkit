"""Agg autocorrelation: сводка формы ACF по диапазону лагов (mean/|mean|/std), не по одному лагу.

Signal:
    autocorr.py даёт корреляцию на КОНКРЕТНЫХ лагах (1, 2, 3, 6, 12) — если «интересный»
    лаг не входит в этот список, сигнал пропущен. agg_autocorrelation считает windowed
    Pearson для ВСЕХ лагов 1..max_lag внутри окна и сводит их в одну сводку: средний
    уровень памяти (mean_acf), средний МОДУЛЬ памяти (abs_mean_acf — не даёт лагам с
    противоположным знаком взаимно погаситься) и разброс по лагам (std_acf — плавно ли
    угасает автокорреляция или скачет от лага к лагу).

Formula:
    Для каждого lag in [1..max_lag], где ws >= lag + 2:
        r_lag = windowed_lag_pearson(v, row_idx, ws, lag)
    (лаги с ws < lag+2 пропускаются, а не считаются нулём — не искажают среднее)

    mean_acf_w     = mean(r_lag по валидным лагам)
    abs_mean_acf_w = mean(|r_lag|)
    std_acf_w      = std(r_lag)   (population, 0 если < 2 валидных лагов)

    Если ни один лаг не валиден (ws < 3) — все три 0.

Outputs:
    {product}__agg_autocorrelation__mean_w24     — средняя автокорреляция по лагам 1..max_lag
    {product}__agg_autocorrelation__abs_mean_w24 — средний |r| по лагам
    {product}__agg_autocorrelation__std_w24      — σ r по лагам

Preset entry:
    agg_autocorrelation:
      windows: [24]
      max_lag: 6

Interpretation:
    abs_mean_acf высокий при mean_acf ≈ 0 — есть заметная память на каких-то лагах, но
        знаки чередуются (осцилляция) и взаимно гасятся в обычном среднем — не путать
        с «памяти нет вообще».
    std_acf ≈ 0 — автокорреляция устойчиво одного уровня на всех лагах (гладкое,
        предсказуемое затухание/сохранение памяти).
    std_acf высокий — ACF скачет от лага к лагу — шумный, ненадёжный сигнал памяти;
        отдельным autocorr__lagK в этом случае лучше не доверять без этой сводки.

Example:
    Ряд (6 мес): [10, 20, 30, 40, 50, 60],  w=6,  max_lag=2  (строго линейный рост)

    r_lag1 = windowed_lag_pearson(lag=1) = 1.0  (идеальная линейная связь соседних точек)
    r_lag2 = windowed_lag_pearson(lag=2) = 1.0  (линейность сохраняется и на лаге 2)
    → agg_autocorrelation__mean_w6 = 1.0,  abs_mean_w6 = 1.0,  std_w6 = 0.0
      (идеально гладкая, устойчивая память на обоих лагах)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import resolve_window_size, windowed_lag_pearson

FEATURE = 'agg_autocorrelation'


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    windows: np.ndarray,
    max_lag: int,
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_mean = np.zeros((n_w, n_rows))
    out_abs_mean = np.zeros((n_w, n_rows))
    out_std = np.zeros((n_w, n_rows))

    r_buf = np.empty(max_lag)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            n_valid = 0
            for lag in range(1, max_lag + 1):
                if ws >= lag + 2:
                    r_buf[n_valid] = windowed_lag_pearson(product_values, row_idx, ws, lag)
                    n_valid += 1
            if n_valid == 0:
                continue
            s = 0.0
            s_abs = 0.0
            for i in range(n_valid):
                s += r_buf[i]
                s_abs += abs(r_buf[i])
            mean_r = s / n_valid
            out_mean[j, row_idx] = mean_r
            out_abs_mean[j, row_idx] = s_abs / n_valid
            if n_valid >= 2:
                var_r = 0.0
                for i in range(n_valid):
                    var_r += (r_buf[i] - mean_r) ** 2
                out_std[j, row_idx] = (var_r / n_valid) ** 0.5

    return out_mean, out_abs_mean, out_std


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [24], "max_lag": 6} — оба ключа обязательны."""
    windows = np.array(params['windows'], dtype=np.int64)
    max_lag = int(params['max_lag'])
    out_mean, out_abs_mean, out_std = _kernel(values, position, windows, max_lag)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(out_mean[j])
        suffixes.append(f'mean_w{w}')
        arrays.append(out_abs_mean[j])
        suffixes.append(f'abs_mean_w{w}')
        arrays.append(out_std[j])
        suffixes.append(f'std_w{w}')
    return arrays, suffixes
