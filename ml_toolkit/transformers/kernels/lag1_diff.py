"""Разности с лагом 1: абсолютная, логарифмическая, процентная.

Signal:
    Простейшие MoM-индикаторы изменения: насколько вырос/упал доход за последний месяц
    в абсолютных единицах, в log-шкале и в процентах. Логарифмическая разность устойчива
    к экспоненциальному распределению оборотов.

Formula:
    diff      = v[t] - v[t-1]
    log_diff  = log1p(|v[t]|) - log1p(|v[t-1]|)
    pct_change = (v[t] - v[t-1]) / (|v[t-1]| + eps)

    Все три равны 0 при position == 0.

Outputs:
    {product}__lag1_diff__diff       — абсолютная разность MoM
    {product}__lag1_diff__log_diff   — лог-разность MoM
    {product}__lag1_diff__pct_change — процентное изменение MoM

Preset (monthly.yaml):
    lag1_diff: {}

Interpretation:
    diff > 0, pct_change > 0 — рост относительно прошлого месяца.
    log_diff ≈ pct_change при малых изменениях; расходятся при больших.
    pct_change = +0.083 = +8.3% при v=65, v_prev=60 (линейный рост).
    diff = 0 при pct_change > 0 невозможно; зато log_diff может расходиться с diff
    при переходе 0→ненулевое (log_diff ненулевой, diff = v[t]).

Example:
    Ряд (3 мес): [50, 60, 65]
    (t=2; сравнение с предыдущим мес. v[t-1]=60)

    diff       = 65 − 60 = 5
    log_diff   = log1p(65) − log1p(60) = 4.190 − 4.111 = 0.079
    pct_change = (65 − 60)/60 = 0.0833
    → lag1_diff__diff = 5.0,  log_diff = 0.079,  pct_change = 0.083

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import safe_ratio

FEATURE = 'lag1_diff'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, log_values: np.ndarray, position_within_entity: np.ndarray):
    n_rows = product_values.shape[0]
    diff = np.zeros(n_rows)
    log_diff = np.zeros(n_rows)
    pct_change = np.zeros(n_rows)
    for row_idx in range(n_rows):
        if position_within_entity[row_idx] >= 1:
            v = product_values[row_idx]
            v_prev = product_values[row_idx - 1]
            diff[row_idx] = v - v_prev
            log_diff[row_idx] = log_values[row_idx] - log_values[row_idx - 1]
            pct_change[row_idx] = safe_ratio(v - v_prev, v_prev)
    return diff, log_diff, pct_change


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {} — параметры не используются."""
    log_values = np.log1p(np.abs(values))
    d, ld, pc = _kernel(values, log_values, position)
    return [d, ld, pc], ['diff', 'log_diff', 'pct_change']
