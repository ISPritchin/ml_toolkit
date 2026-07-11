"""Change quantiles: волатильность приращений внутри коридора значений [ql, qh] своего окна.

Signal:
    Обычные волатильностные признаки (rolling_std, rolling_cv, mean_deviation_shape) меряют
    разброс безусловно, по всему окну. change_quantiles — условно: сначала определяет
    коридор значений [ql, qh] (квантили распределения САМОГО окна), затем считает
    волатильность приращений ТОЛЬКО между точками, обе из которых попали в этот коридор.
    Это отделяет «нормальный шум в типичном диапазоне» от волатильности, вызванной редкими
    экстремумами — два ряда с одинаковым rolling_std могут иметь совершенно разный
    change_quantiles, если у одного шум сосредоточен в хвостах, а у другого — в теле
    распределения.

Formula:
    lo_w, hi_w = квантили ql/qh отсортированного окна (sorted_quantile, та же конвенция,
        что и везде: sorted[int(q*(ws-1))])
    Для каждой пары соседних точек (v[i-1], v[i]) внутри окна, где ОБЕ точки в [lo_w, hi_w]:
        d = |v[i] - v[i-1]|
    change_mean_w = mean(d по квалифицирующим парам)   (0, если ни одной пары)
    change_std_w  = std(d по квалифицирующим парам)    (population, 0 если < 2 пар)

Outputs:
    {product}__change_quantiles__mean_w6   — средняя |приращение| в коридоре, окно 6
    {product}__change_quantiles__std_w6    — σ |приращений| в коридоре, окно 6
    {product}__change_quantiles__mean_w12  — то же за 12 мес
    {product}__change_quantiles__std_w12   — то же за 12 мес

Preset entry:
    change_quantiles:
      windows: [6, 12]
      ql: 0.2
      qh: 0.8

Interpretation:
    change_mean высокий при обычном rolling_std низком — шум сосредоточен именно в
        типичном диапазоне значений, а не только в редких выбросах (выбросы уже
        исключены коридором [ql, qh]).
    change_std ≈ 0 при change_mean > 0 — приращения в коридоре стабильны по размеру
        (равномерный «дребезг»), не путать с отсутствием шума вообще.
    Сильное расхождение с rolling_cv — сигнал, что волатильность ряда сконцентрирована
        в хвостах распределения (см. extreme_share), а не в его основной массе.

Example:
    Ряд (6 мес): [5, 20, 22, 21, 23, 5],  w=6,  ql=0.2, qh=0.8

    отсортировано: [5, 5, 20, 21, 22, 23]
    lo = sorted[int(0.2·5)] = sorted[1] = 5
    hi = sorted[int(0.8·5)] = sorted[4] = 22

    пары (обе точки в [5, 22]):
      (5,20)→d=15,  (20,22)→d=2,  (22,21)→d=1   — квалифицируют (n=3)
      (21,23) — 23 вне коридора, (23,5) — 23 вне коридора — не квалифицируют
    mean = (15+2+1)/3 = 6.0
    var  = (15²+2²+1²)/3 − 6.0² = 76.667 − 36 = 40.667  →  std = 6.377
    → change_quantiles__mean_w6 = 6.0,  std_w6 = 6.377

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import fill_window_sorted, resolve_window_size, sorted_quantile

FEATURE = 'change_quantiles'


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    windows: np.ndarray,
    ql: float,
    qh: float,
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_mean = np.zeros((n_w, n_rows))
    out_std = np.zeros((n_w, n_rows))

    max_w = 1
    for j in range(n_w):
        max_w = max(max_w, windows[j])
    sorted_buf = np.empty(max_w)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            if ws < 2:
                continue
            fill_window_sorted(sorted_buf, product_values, row_idx, ws)
            lo = sorted_quantile(sorted_buf, ws, ql)
            hi = sorted_quantile(sorted_buf, ws, qh)

            start = row_idx - ws + 1
            count = 0
            d_sum = 0.0
            d_sq_sum = 0.0
            for offset in range(1, ws):
                prev = product_values[start + offset - 1]
                cur = product_values[start + offset]
                if prev >= lo and prev <= hi and cur >= lo and cur <= hi:
                    d = abs(cur - prev)
                    count += 1
                    d_sum += d
                    d_sq_sum += d * d
            if count > 0:
                mean_d = d_sum / count
                out_mean[j, row_idx] = mean_d
                if count >= 2:
                    var_d = d_sq_sum / count - mean_d * mean_d
                    out_std[j, row_idx] = (max(var_d, 0.0)) ** 0.5
    return out_mean, out_std


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6, 12], "ql": 0.2, "qh": 0.8 (ql/qh опциональны)}."""
    windows = np.array(params['windows'], dtype=np.int64)
    ql = float(params.get('ql', 0.2))
    qh = float(params.get('qh', 0.8))
    out_mean, out_std = _kernel(values, position, windows, ql, qh)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(out_mean[j])
        suffixes.append(f'mean_w{w}')
        arrays.append(out_std[j])
        suffixes.append(f'std_w{w}')
    return arrays, suffixes
