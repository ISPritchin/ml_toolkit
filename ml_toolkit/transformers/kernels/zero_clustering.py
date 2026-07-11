"""Структура нулей внутри окна: серии, давность, пространственное распределение.

Signal:
    Анализирует не просто долю нулей, а их «архитектуру»: слиплись ли они в длинные серии
    (max_zero_run) или рассеяны по всему окну (run_count высокий при малом max_run). recent_vs_long
    определяет, нарастает ли доля нулей в последнее время. last_zero_rec — сколько месяцев назад
    был последний нуль: 0 = сейчас, ws = нулей не было. zero_after_active — флаг немедленного
    выхода из активного состояния (рисковый паттерн).

Formula:
    max_zero_run_w    = longest consecutive run of zeros in [t-w+1..t]
    zero_run_count_w  = number of distinct zero runs in window
    recent_vs_long_w  = (zero_share in [t-2..t]) / (zero_share_w + eps)
    last_zero_rec_w   = w - 1 - last_zero_offset (0 если текущий нуль, w если нулей нет)
    front_back_w      = zero_share(first half) / (zero_share(second half) + eps)
    zero_after_active = 1 if v[t]=0 and v[t-1]!=0 else 0

Outputs:
    {product}__zero_clustering__max_zero_run_w12    — длиннейшая нулевая серия (мес)
    {product}__zero_clustering__zero_run_count_w12  — число отдельных нулевых серий
    {product}__zero_clustering__recent_vs_long_w12  — нарастание нулей в последние 3 мес
    {product}__zero_clustering__last_zero_rec_w12   — давность последнего нуля (мес назад)
    {product}__zero_clustering__front_back_w12      — нули в начале vs. конце окна
    {product}__zero_clustering__zero_after_active   — флаг немедленного выхода из активности

Preset (monthly.yaml):
    zero_clustering:
      windows: [12]

Interpretation:
    max_zero_run = 6 — клиент полгода подряд не совершал транзакций (длительная «спячка»).
    zero_run_count = 4 при max_zero_run = 2 — рваная активность: частые короткие перерывы.
    recent_vs_long > 2 — доля нулей в последние 3 месяца вдвое выше среднегодовой (уходит).
    zero_after_active = 1 — прошлый месяц был активным, а этот — нулевой (немедленный уход).

Example:
    Ряд (6 мес): [10, 0, 0, 10, 0, 10],  w=6

    нулевые серии: idx1-2 (длина 2) и idx4 (длина 1) → run_count = 2, max_zero_run = 2
    последний нуль на offset=4 → last_zero_rec = (6−1) − 4 = 1
    → zero_clustering__max_zero_run_w6 = 2,  zero_run_count_w6 = 2,  last_zero_rec_w6 = 1

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import resolve_window_size, safe_ratio

FEATURE = 'zero_clustering'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_max_run = np.zeros((n_w, n_rows))
    out_run_count = np.zeros((n_w, n_rows))
    out_recent_vs_long = np.zeros((n_w, n_rows))
    out_last_zero_rec = np.zeros((n_w, n_rows))
    out_front_back = np.zeros((n_w, n_rows))
    out_zero_after_active = np.zeros(n_rows)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        v_now = product_values[row_idx]

        # zero_after_active: текущий 0, предыдущий активный
        if pos >= 1 and v_now == 0.0 and product_values[row_idx - 1] != 0.0:
            out_zero_after_active[row_idx] = 1.0

        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            max_run = 0
            run_count = 0
            cur_run = 0
            in_zero = False
            last_zero_ago = ws  # если нулей не было
            zero_front = 0
            zero_back = 0
            half = ws // 2

            for offset in range(ws):
                abs_idx = row_idx - ws + 1 + offset
                vv = product_values[abs_idx]
                if vv == 0.0:
                    if not in_zero:
                        run_count += 1
                        in_zero = True
                    cur_run += 1
                    max_run = max(max_run, cur_run)
                    last_zero_ago = ws - 1 - offset
                    if offset < half:
                        zero_front += 1
                    else:
                        zero_back += 1
                else:
                    cur_run = 0
                    in_zero = False

            out_max_run[j, row_idx] = max_run
            out_run_count[j, row_idx] = run_count
            out_last_zero_rec[j, row_idx] = last_zero_ago

            # zero_share_recent3_vs_w12
            ws3 = min(3, ws)
            z3 = 0
            for offset in range(ws3):
                if product_values[row_idx - ws3 + 1 + offset] == 0.0:
                    z3 += 1
            z_all = 0
            for offset in range(ws):
                if product_values[row_idx - ws + 1 + offset] == 0.0:
                    z_all += 1
            z_share_all = z_all / ws
            out_recent_vs_long[j, row_idx] = safe_ratio(z3 / ws3, z_share_all)

            # front vs back zero share
            out_front_back[j, row_idx] = safe_ratio(zero_front / max(half, 1), zero_back / max(ws - half, 1))

    return out_max_run, out_run_count, out_recent_vs_long, out_last_zero_rec, out_front_back, out_zero_after_active


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    mrun, rcnt, rvl, lzr, fb, zaa = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(mrun[j])
        suffixes.append(f'max_zero_run_w{w}')
        arrays.append(rcnt[j])
        suffixes.append(f'zero_run_count_w{w}')
        arrays.append(rvl[j])
        suffixes.append(f'recent_vs_long_w{w}')
        arrays.append(lzr[j])
        suffixes.append(f'last_zero_rec_w{w}')
        arrays.append(fb[j])
        suffixes.append(f'front_back_w{w}')
    arrays.append(zaa)
    suffixes.append('zero_after_active')
    return arrays, suffixes
