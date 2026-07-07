"""Асимметрия отклонений: semi-deviation вверх/вниз, mean-reversion crosses.

Signal:
    Разграничивает направление «сюрпризов» клиента: больше ли случается крупных подъёмов
    (σ_up > σ_down, dev_asym > 0) или глубоких провалов (σ_down > σ_up). Число пересечений
    среднего (cross_count) показывает, как часто значение «болтается» вокруг нормы.

Formula:
    mean_w, std_w — среднее и станд. отклонение окна
    up_semi_w    = sqrt(mean((v[i]-mean)² for v[i] >= mean))
    down_semi_w  = sqrt(mean((mean-v[i])² for v[i] < mean))
    semi_ratio_w = up_semi_w / (down_semi_w + eps)
    max_up_z_w   = max((v[i]-mean)/std) для v[i] >= mean
    max_down_z_w = max((mean-v[i])/std) для v[i] < mean
    dev_asym_w   = sum(max(0,v[i]-mean)) / (sum(|v[i]-mean|) + eps) - 0.5
    cross_count_w = число смен стороны (выше → ниже среднего или наоборот)

Outputs:
    {product}__mean_deviation_shape__up_semi_w12    — σ положительных отклонений
    {product}__mean_deviation_shape__down_semi_w12  — σ отрицательных отклонений
    {product}__mean_deviation_shape__semi_ratio_w12 — up_semi / down_semi
    {product}__mean_deviation_shape__max_up_z_w12   — лучший месяц (в σ)
    {product}__mean_deviation_shape__max_down_z_w12 — худший месяц (в σ)
    {product}__mean_deviation_shape__dev_asym_w12   — взвешенный дисбаланс
    {product}__mean_deviation_shape__cross_count_w12 — число пересечений среднего
    (аналогично для w6)

Preset (monthly.yaml):
    mean_deviation_shape:
      windows: [6, 12]

Interpretation:
    semi_ratio > 1 — «приятные сюрпризы» сильнее «неприятных» (правосторонняя асимметрия).
    dev_asym > 0.1 — клиент чаще бывает выше среднего значительно, чем ниже.
    cross_count_w12 > 8 — ряд интенсивно пересекает среднее (осциллирующий паттерн).
    semi_ratio ≈ 1 при высоком std — симметричная осцилляция (ни роста ни падения).

Example:
    Ряд (6 мес): [10, 10, 10, 10, 10, 40],  w=6
    mean = 90/6 = 15

    отклонения вверх (≥ mean): только 40 → dev=+25 → up_semi = sqrt(25²) = 25
    отклонения вниз (< mean): пять «десяток» → dev=−5 → down_semi = sqrt(5²) = 5
    semi_ratio = 25 / 5 = 5.0
    → mean_deviation_shape__up_semi_w6 = 25,  down_semi_w6 = 5,  semi_ratio_w6 = 5.0

"""

import numba as nb
import numpy as np

from .._windowing import (
    EPS,
    compute_window_mean_and_std,
    resolve_window_size,
    safe_ratio,
)

FEATURE = 'mean_deviation_shape'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_up_semi = np.zeros((n_w, n_rows))
    out_down_semi = np.zeros((n_w, n_rows))
    out_semi_ratio = np.zeros((n_w, n_rows))
    out_max_up_z = np.zeros((n_w, n_rows))
    out_max_down_z = np.zeros((n_w, n_rows))
    out_dev_asym = np.zeros((n_w, n_rows))
    out_cross_count = np.zeros((n_w, n_rows))

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            mean, std = compute_window_mean_and_std(product_values, row_idx, ws)

            up_sq = 0.0; down_sq = 0.0
            n_up = 0; n_down = 0
            max_up_z = 0.0; max_down_z = 0.0
            sum_abs_dev = 0.0; sum_pos_dev = 0.0
            cross_count = 0
            prev_side = 0  # +1 above mean, -1 below

            for offset in range(ws):
                abs_idx = row_idx - ws + 1 + offset
                v = product_values[abs_idx]
                dev = v - mean
                abs_dev = abs(dev)
                sum_abs_dev += abs_dev
                z = safe_ratio(abs_dev, std)
                if dev >= 0.0:
                    up_sq += dev * dev
                    n_up += 1
                    sum_pos_dev += abs_dev
                    max_up_z = max(max_up_z, z)
                    cur_side = 1
                else:
                    down_sq += dev * dev
                    n_down += 1
                    max_down_z = max(max_down_z, z)
                    cur_side = -1
                if offset >= 1 and cur_side != prev_side:
                    cross_count += 1
                prev_side = cur_side

            out_up_semi[j, row_idx] = (up_sq / max(n_up, 1)) ** 0.5
            out_down_semi[j, row_idx] = (down_sq / max(n_down, 1)) ** 0.5
            out_semi_ratio[j, row_idx] = safe_ratio(out_up_semi[j, row_idx], out_down_semi[j, row_idx])
            out_max_up_z[j, row_idx] = max_up_z
            out_max_down_z[j, row_idx] = max_down_z
            # у константного ряда отклонений нет — дисбаланс 0, а не -0.5
            if sum_abs_dev > EPS:
                out_dev_asym[j, row_idx] = safe_ratio(sum_pos_dev, sum_abs_dev) - 0.5
            out_cross_count[j, row_idx] = cross_count

    return out_up_semi, out_down_semi, out_semi_ratio, out_max_up_z, out_max_down_z, out_dev_asym, out_cross_count


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}"""
    windows = np.array(params['windows'], dtype=np.int64)
    us, ds, sr, muz, mdz, da, cc = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(us[j]);  suffixes.append(f'up_semi_w{w}')
        arrays.append(ds[j]);  suffixes.append(f'down_semi_w{w}')
        arrays.append(sr[j]);  suffixes.append(f'semi_ratio_w{w}')
        arrays.append(muz[j]); suffixes.append(f'max_up_z_w{w}')
        arrays.append(mdz[j]); suffixes.append(f'max_down_z_w{w}')
        arrays.append(da[j]);  suffixes.append(f'dev_asym_w{w}')
        arrays.append(cc[j]);  suffixes.append(f'cross_count_w{w}')
    return arrays, suffixes
