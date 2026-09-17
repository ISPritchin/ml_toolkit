"""CUSUM: накопленные положительные и отрицательные отклонения от среднего окна.

Signal:
    Отображает накопленный «излишек» (cusum_pos) и «дефицит» (cusum_neg) относительно
    среднего уровня окна. Большой cusum_pos при малом |cusum_neg| — значения ряда регулярно
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

Preset entry:
    cusum:
      windows: [6, 12]
      dilations: [1, 2]   # optional, default [1]; d>1 spaces window taps d rows apart

Interpretation:
    cusum_pos_w12 >> |cusum_neg_w12| — распределение смещено вправо; редкие большие
    всплески перевешивают частые умеренные значения.
    cusum_pos ≈ |cusum_neg| — симметричные отклонения (осциллирующий паттерн).
    Нормируй cusum на mean_w чтобы получить безразмерный сигнал для сравнения разных рядов.

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

from ml_toolkit.transformers._windowing import FILL_NAN_UNBOUNDED_LOW, compute_window_sum, resolve_window_size

FEATURE = 'cusum'
FILL_NAN: dict[str | None, float] = {
    'pos': -1.0,  # cusum_pos >= 0 всегда (сумма положительных отклонений) — -1 недостижим
    'neg': FILL_NAN_UNBOUNDED_LOW,  # cusum_neg <= 0, сырой масштаб без универсальной нижней границы
}


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray, dilations: np.ndarray
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    n_d = dilations.shape[0]
    out_pos = np.zeros((n_w * n_d, n_rows))
    out_neg = np.zeros((n_w * n_d, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        k = 0
        for j in range(n_w):
            for d in range(n_d):
                dilation = dilations[d]
                ws = resolve_window_size(pos, windows[j], dilation)
                win_sum = compute_window_sum(product_values, row_idx, ws, dilation)
                mean = win_sum / ws
                base = row_idx - (ws - 1) * dilation
                pos_sum = 0.0
                neg_sum = 0.0
                for offset in range(ws):
                    dev = product_values[base + offset * dilation] - mean
                    if dev > 0.0:
                        pos_sum += dev
                    else:
                        neg_sum += dev
                out_pos[k, row_idx] = pos_sum
                out_neg[k, row_idx] = neg_sum
                k += 1
    return out_pos, out_neg


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12], "dilations": [1, 2]}. "dilations" optional, default [1]."""
    windows = np.array(params['windows'], dtype=np.int64)
    dilations = np.array(params.get('dilations', [1]), dtype=np.int64)
    out_pos, out_neg = _kernel(values, position, windows, dilations)
    arrays = []
    suffixes = []
    k = 0
    for w in params['windows']:
        for d in params.get('dilations', [1]):
            w_tag = f'w{w}' if d == 1 else f'w{w}_d{d}'
            arrays.append(out_pos[k])
            suffixes.append(f'pos_{w_tag}')
            arrays.append(out_neg[k])
            suffixes.append(f'neg_{w_tag}')
            k += 1
    return arrays, suffixes
