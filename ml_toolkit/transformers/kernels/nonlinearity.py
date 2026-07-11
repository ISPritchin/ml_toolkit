"""Выпуклость/вогнутость тренда: квадратичная компонента и ускорение кривизны.

Signal:
    Определяет форму тренда: U-образный (вогнутый вниз, quad_proxy < 0),
    перевёрнутый U (выпуклый, quad_proxy > 0), или равномерное ускорение.
    Полезен для обнаружения «разгона после плато» и «схода с пика».

Formula:
    Метод третей: third = ws // 3
        Q1 = mean(v[0..third-1])
        Q2 = mean(v[third..2*third-1])
        Q3 = mean(v[2*third..3*third-1])
        quad_proxy_w = (Q1 - 2*Q2 + Q3) / (|mean_w| + eps)
        convexity_sign_w = sign(quad_proxy_w)

    Средняя вторая разность:
        accel[i] = v[i] - 2*v[i-1] + v[i-2]
        mean_accel_w = mean(accel[i], i in [t-w+3..t])
        accel_std_w  = std(accel)
        frac_concave_w = count(accel[i] < 0) / n_accels

Outputs:
    {product}__nonlinearity__quad_proxy_w6     — квадратичный коэф., окно 6
    {product}__nonlinearity__convexity_sign_w6 — знак кривизны
    {product}__nonlinearity__mean_accel_w6     — средняя вторая разность
    {product}__nonlinearity__accel_std_w6      — σ вторых разностей
    {product}__nonlinearity__frac_concave_w6   — доля шагов с торможением
    (аналогично для w12)

Preset entry:
    nonlinearity:
      windows: [6, 12]

Interpretation:
    quad_proxy < 0 — дугообразный: пик посередине периода (рост → спад).
    quad_proxy > 0 — U-образный: середина ниже концов (просадка → восстановление).
    frac_concave > 0.7 — большинство шагов тормозит (затухающий рост).
    accel_std высокий — кривизна нестабильна, нет устойчивой квадратичной формы.

Example:
    Ряд (6 мес): [10, 20, 30, 20, 10, 5],  w=6  (дугообразный)
    mean = 95/6 = 15.833,  third = 2

    Q1 = (10+20)/2 = 15,  Q2 = (30+20)/2 = 25,  Q3 = (10+5)/2 = 7.5
    quad_proxy = (15 − 2·25 + 7.5) / 15.833 = −27.5 / 15.833 = −1.737
    → nonlinearity__quad_proxy_w6 = −1.737,  convexity_sign_w6 = −1  (пик посередине)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import compute_window_mean, resolve_window_size, safe_ratio

FEATURE = 'nonlinearity'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_quad_proxy = np.zeros((n_w, n_rows))
    out_convexity_sign = np.zeros((n_w, n_rows))
    out_mean_accel = np.zeros((n_w, n_rows))
    out_accel_std = np.zeros((n_w, n_rows))
    out_frac_concave = np.zeros((n_w, n_rows))

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            mean = compute_window_mean(product_values, row_idx, ws)

            # метод третей
            third = ws // 3
            if third >= 1:
                q1 = 0.0
                q2 = 0.0
                q3 = 0.0
                for i in range(third):
                    q1 += product_values[row_idx - ws + 1 + i]
                    q2 += product_values[row_idx - ws + 1 + third + i]
                    q3 += product_values[row_idx - ws + 1 + 2 * third + i]
                q1 /= third
                q2 /= third
                q3 /= third
                quad = safe_ratio(q1 - 2.0 * q2 + q3, mean)
                out_quad_proxy[j, row_idx] = quad
                out_convexity_sign[j, row_idx] = 1.0 if quad > 0.0 else (-1.0 if quad < 0.0 else 0.0)

            # mean and std of second differences (два прохода без буфера)
            n_accels = ws - 2
            if n_accels >= 1:
                accel_sum = 0.0
                for i in range(n_accels):
                    abs_idx = row_idx - ws + 1 + i + 2
                    accel_sum += (product_values[abs_idx]
                                  - 2.0 * product_values[abs_idx - 1]
                                  + product_values[abs_idx - 2])
                mean_accel = accel_sum / n_accels
                out_mean_accel[j, row_idx] = mean_accel
                var_a = 0.0
                concave_count = 0
                for i in range(n_accels):
                    abs_idx = row_idx - ws + 1 + i + 2
                    a = (product_values[abs_idx]
                         - 2.0 * product_values[abs_idx - 1]
                         + product_values[abs_idx - 2])
                    var_a += (a - mean_accel) ** 2
                    if a < 0.0:
                        concave_count += 1
                out_accel_std[j, row_idx] = (var_a / n_accels) ** 0.5
                out_frac_concave[j, row_idx] = concave_count / n_accels

    return out_quad_proxy, out_convexity_sign, out_mean_accel, out_accel_std, out_frac_concave


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6, 12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    qp, cs, ma, as_, fc = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(qp[j])
        suffixes.append(f'quad_proxy_w{w}')
        arrays.append(cs[j])
        suffixes.append(f'convexity_sign_w{w}')
        arrays.append(ma[j])
        suffixes.append(f'mean_accel_w{w}')
        arrays.append(as_[j])
        suffixes.append(f'accel_std_w{w}')
        arrays.append(fc[j])
        suffixes.append(f'frac_concave_w{w}')
    return arrays, suffixes
