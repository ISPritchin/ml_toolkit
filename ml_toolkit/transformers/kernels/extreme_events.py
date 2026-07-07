"""Редкие экстремальные события: всплески, обвалы, их количество и давность.

Signal:
    Детектирует аномальные месяцы: всплески (z > 2σ) и обвалы (падение > 50% MoM).
    Высокий spike_count при низком crash_count — клиент с нечастыми, но крупными
    поступлениями. Наличие is_spike_now = 1 означает аномально сильный текущий месяц.

Formula:
    mean_w, std_w — среднее и станд. отклонение окна
    z[i] = (v[i] - mean_w) / (std_w + eps)
    spike: z[i] > 2.0
    crash: (v[i-1] - v[i]) / (|v[i-1]| + eps) > 0.5

    spike_count_w  = count(z[i] > 2)
    max_spike_z_w  = max(z[i])
    crash_count_w  = count(relative_drop[i] > 0.5)
    max_drop_w     = max(relative_drop[i])
    recency_w      = ws - 1 - last_extreme_offset  (шагов с последнего экстремума)
    balance_w      = spike_count - crash_count
    is_spike_now   = 1 if z[t] > 2 (по первому окну)

Outputs:
    {product}__extreme_events__spike_count_w12   — число всплесков за 12 мес
    {product}__extreme_events__max_spike_z_w12   — макс z-score всплеска
    {product}__extreme_events__crash_count_w12   — число обвалов за 12 мес
    {product}__extreme_events__max_drop_w12      — макс относительный обвал
    {product}__extreme_events__recency_w12       — месяцев с последнего экстремума
    {product}__extreme_events__balance_w12       — разница всплески - обвалы
    {product}__extreme_events__is_spike_now      — флаг текущего всплеска

Preset (monthly.yaml):
    extreme_events:
      windows: [12]

Interpretation:
    spike_count = 1, max_spike_z = 3.5 — один мощный выброс за год, вероятно проектный.
    crash_count > 2, balance < 0 — волатильный клиент со склонностью к обвалам.
    recency = 0 — только что был экстремум (будь то всплеск или падение).
    is_spike_now = 1 — хороший момент для оценки контрактной активности.

Example:
    Ряд (7 мес): [10, 10, 10, 10, 10, 10, 100],  w=7
    mean = 160/7 = 22.857,  std = 31.493

    z[t] = (100 − 22.857) / 31.493 = 2.449 > 2 → всплеск
    обвалов нет (значения росли или равны)
    → extreme_events__spike_count_w7 = 1,  max_spike_z_w7 = 2.449
    → extreme_events__is_spike_now = 1,  recency_w7 = 0

"""

import numba as nb
import numpy as np

from .._windowing import compute_window_mean_and_std, resolve_window_size, safe_ratio

FEATURE = 'extreme_events'


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    windows: np.ndarray,
    spike_z: float,
    crash_drop: float,
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_spike_count = np.zeros((n_w, n_rows))
    out_max_spike_z = np.zeros((n_w, n_rows))
    out_crash_count = np.zeros((n_w, n_rows))
    out_max_drop = np.zeros((n_w, n_rows))
    out_extreme_recency = np.zeros((n_w, n_rows))
    out_is_spike_now = np.zeros(n_rows)
    out_balance = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            mean, std = compute_window_mean_and_std(product_values, row_idx, ws)
            spike_count = 0
            crash_count = 0
            max_spike_z = 0.0
            max_drop = 0.0
            last_extreme_ago = ws  # worse case: never
            z_now = 0.0
            for offset in range(ws):
                abs_idx = row_idx - ws + 1 + offset
                v = product_values[abs_idx]
                z = safe_ratio(v - mean, std)
                if offset == ws - 1:
                    z_now = z
                is_extreme = False
                if z > spike_z:
                    spike_count += 1
                    max_spike_z = max(max_spike_z, z)
                    is_extreme = True
                if offset >= 1:
                    prev = product_values[abs_idx - 1]
                    drop = safe_ratio(prev - v, prev)
                    if drop > crash_drop:
                        crash_count += 1
                        max_drop = max(max_drop, drop)
                        is_extreme = True
                if is_extreme:
                    last_extreme_ago = ws - 1 - offset
            out_spike_count[j, row_idx] = spike_count
            out_max_spike_z[j, row_idx] = max_spike_z
            out_crash_count[j, row_idx] = crash_count
            out_max_drop[j, row_idx] = max_drop
            out_extreme_recency[j, row_idx] = last_extreme_ago
            out_balance[j, row_idx] = spike_count - crash_count
            if j == 0:
                # is_spike_now: z текущего месяца уже вычислен в цикле окна
                out_is_spike_now[row_idx] = 1.0 if z_now > spike_z else 0.0
    return out_spike_count, out_max_spike_z, out_crash_count, out_max_drop, out_extreme_recency, out_is_spike_now, out_balance


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12], "spike_z": 2.0, "crash_drop": 0.5 (опционально)}"""
    windows = np.array(params['windows'], dtype=np.int64)
    spike_z = float(params.get('spike_z', 2.0))
    crash_drop = float(params.get('crash_drop', 0.5))
    sc, mz, cc, md, er, isn, bal = _kernel(values, position, windows, spike_z, crash_drop)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(sc[j]);  suffixes.append(f'spike_count_w{w}')
        arrays.append(mz[j]);  suffixes.append(f'max_spike_z_w{w}')
        arrays.append(cc[j]);  suffixes.append(f'crash_count_w{w}')
        arrays.append(md[j]);  suffixes.append(f'max_drop_w{w}')
        arrays.append(er[j]);  suffixes.append(f'recency_w{w}')
        arrays.append(bal[j]); suffixes.append(f'balance_w{w}')
    arrays.append(isn); suffixes.append('is_spike_now')
    return arrays, suffixes
