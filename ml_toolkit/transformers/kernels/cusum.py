"""CUSUM: накопленные положительные и отрицательные отклонения от среднего окна.

Signal:
    Отображает накопленный «излишек» (cusum_pos) и «дефицит» (cusum_neg) относительно
    среднего уровня окна. Большой cusum_pos при малом |cusum_neg| — доходы регулярно
    превышают норму; обратное — систематический недобор.

Formula:
    mean_w = mean(v[i], i in [t-w+1..t])
    cusum_pos_w = sum(max(0, v[i] - mean_w), i in окне)
    cusum_neg_w = sum(min(0, v[i] - mean_w), i in окне)

    Имеют размерность исходной колонки (не нормированы).

Outputs:
    {product}__cusum__pos_w6   — накоп. положительные отклонения за 6 мес
    {product}__cusum__neg_w6   — накоп. отрицательные отклонения за 6 мес
    {product}__cusum__pos_w12  — накоп. положительные отклонения за 12 мес
    {product}__cusum__neg_w12  — накоп. отрицательные отклонения за 12 мес

Preset (monthly.yaml):
    cusum:
      windows: [6, 12]

Interpretation:
    cusum_pos_w12 >> |cusum_neg_w12| — распределение смещено вправо; редкие большие
    месяцы перевешивают частые умеренные (B2B-проектный профиль).
    cusum_pos ≈ |cusum_neg| — симметричные отклонения (осциллирующий паттерн).
    Нормируй cusum на mean_w чтобы получить безразмерный сигнал для сравнения клиентов.

Example:
    Ряд (4 мес): [10, 40, 20, 30],  w=4
    mean = 100/4 = 25

    отклонения: −15, +15, −5, +5
    cusum_pos = 15 + 5 = 20
    cusum_neg = −15 + (−5) = −20
    → cusum__pos_w4 = 20,  cusum__neg_w4 = −20

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import compute_window_sum, resolve_window_size

FEATURE = 'cusum'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_pos = np.zeros((n_w, n_rows))
    out_neg = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            win_sum = compute_window_sum(product_values, row_idx, ws)
            mean = win_sum / ws
            pos_sum = 0.0
            neg_sum = 0.0
            for offset in range(ws):
                dev = product_values[row_idx - ws + 1 + offset] - mean
                if dev > 0.0:
                    pos_sum += dev
                else:
                    neg_sum += dev
            out_pos[j, row_idx] = pos_sum
            out_neg[j, row_idx] = neg_sum
    return out_pos, out_neg


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    out_pos, out_neg = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(out_pos[j])
        suffixes.append(f'pos_w{w}')
        arrays.append(out_neg[j])
        suffixes.append(f'neg_w{w}')
    return arrays, suffixes
