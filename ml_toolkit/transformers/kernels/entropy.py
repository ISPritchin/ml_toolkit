"""Нормированная энтропия Шеннона распределения объёмов внутри окна.

Signal:
    Измеряет равномерность распределения объёмов по месяцам. Высокая энтропия
    (близко к 1) — доходы равномерны; низкая — сконцентрированы в нескольких
    месяцах (типично для проектных B2B-клиентов).

Formula:
    S_w = sum(v[i], i in окне, v[i] > 0)
    H_w = -sum(v[i]/S_w * ln(v[i]/S_w)) / ln(ws)    для v[i] > 0
    H_w = 0 если S_w <= eps или ws == 1

    Нормировка на ln(ws) дает значение в [0, 1].

Outputs:
    {product}__entropy__w6   — нормированная энтропия за 6 мес
    {product}__entropy__w12  — нормированная энтропия за 12 мес

Preset (monthly.yaml):
    entropy:
      windows: [6, 12]

Interpretation:
    ≈ 1.0 — доходы равномерно распределены по всем месяцам окна.
    ≈ 0.5 — умеренная концентрация (половина объёма в 2-3 месяцах).
    ≈ 0.0 — весь доход сосредоточен в одном месяце (разовый крупный контракт).
    Сопоставь с gini: оба сигнализируют о концентрации, но энтропия больше чувствительна
    к нулевым месяцам, а gini — к разбросу между ненулевыми.

Example:
    Ряд (4 мес): [10, 20, 30, 40],  w=4
    S = 100

    p = [0.1, 0.2, 0.3, 0.4]
    H = −(0.1·ln0.1 + 0.2·ln0.2 + 0.3·ln0.3 + 0.4·ln0.4)
      = −(−0.230 − 0.322 − 0.361 − 0.367) = 1.280
    H_norm = 1.280 / ln(4) = 1.280 / 1.386 = 0.924
    → entropy__w4 = 0.924

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import EPS, compute_window_sum, resolve_window_size

FEATURE = 'entropy'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            win_sum = compute_window_sum(product_values, row_idx, ws)
            if win_sum > EPS and ws > 1:
                h = 0.0
                for offset in range(ws):
                    v = product_values[row_idx - ws + 1 + offset]
                    if v > 0.0:
                        share = v / win_sum
                        h -= share * np.log(share)
                out[j, row_idx] = h / np.log(ws)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12, 24]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f'w{w}' for w in params['windows']]
