"""Текущая серия месяцев подряд выше скользящего среднего (running state).

Signal:
    Подсчитывает, как долго клиент непрерывно превышает собственное среднее.
    Высокое значение указывает на устойчивое превышение нормы — потенциальный рост
    или период высокой активности. Сбрасывается при первом ненарушении.

Formula:
    mean_ref_w = mean(v[t-w+1..t])   (скользящее среднее эталонного окна)
    cur_run: running state, сбрасывается при v[t] <= mean_ref_w
    run_above_mean_w = cur_run (инкрементируется при v[t] > mean_ref_w)

Outputs:
    {product}__run_above_mean__w12  — текущая серия мес. выше среднего окна 12

Preset (monthly.yaml):
    run_above_mean:
      window: 12

Interpretation:
    = 0 — текущий месяц не выше среднего (или только что сбросился).
    = 3 — три месяца подряд выше 12-месячного среднего.
    = 12 — весь год выше среднего: аномальная активность или растущий тренд.
    Для линейного ряда G: последние 6 месяцев выше среднего 37.5, run = 6.

Example:
    Ряд (6 мес): [10, 20, 30, 40, 50, 60],  ref_window=6

    на каждом шаге t≥1 значение выше своего скользящего среднего:
      t=1: v=20 > mean(10,20)=15 → run=1
      t=2: v=30 > 20 → run=2  ... t=5: v=60 > 35 → run=5
    → run_above_mean__w6 = 5  (5 месяцев подряд выше среднего)
"""

import numba as nb
import numpy as np

from .._windowing import compute_window_mean, resolve_window_size

FEATURE = "run_above_mean"


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, ref_window: int):
    n_rows = product_values.shape[0]
    out = np.zeros(n_rows)
    cur_run = 0
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        if pos == 0:
            cur_run = 0
        ws = resolve_window_size(pos, ref_window)
        mean = compute_window_mean(product_values, row_idx, ws)
        if product_values[row_idx] > mean:
            cur_run += 1
        else:
            cur_run = 0
        out[row_idx] = cur_run
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"window": 12} — ключ обязателен, дефолты задаёт пресет."""
    w = params["window"]
    return [_kernel(values, position, w)], [f"w{w}"]
