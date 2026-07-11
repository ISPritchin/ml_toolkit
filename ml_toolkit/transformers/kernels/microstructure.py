"""Предсказуемость отдельного месяца: SNR, surprise, conditional mean.

Signal:
    Оценивает, насколько «типичен» текущий месяц для ряда. Высокая предсказуемость
    при активном месяце = стабильный ряд. Большой surprise = нетипичный месяц, требует
    проверки (всплеск или провал). Conditional mean — ожидаемое значение при активном месяце.

Formula:
    snr_w        = std_short_w / (std_long_w + eps)   где short_w = max(w//4, 1)
    surprise_w   = |v[t] - mean_w| / (std_w + eps)
    surprise_dir = +1 if v[t] >= mean_w else -1
    predictability_w = 1 / (1 + std_w / (|mean_w| + eps))  = 1 / (1 + CV_w)
    active_rate  = count(v[i] != 0) / ws
    cond_mean_w  = mean_w / (active_rate + eps)
    vs_cond_mean_w = v[t] / (|cond_mean_w| + eps)

Outputs:
    {product}__microstructure__snr_w12          — краткосрочный/долгосрочный std
    {product}__microstructure__surprise_w12     — |v-mean|/std текущего месяца
    {product}__microstructure__predictability_w12 — 1/(1+CV) ∈ (0,1]
    {product}__microstructure__cond_mean_w12    — ожид. значение при активном мес.
    {product}__microstructure__vs_cond_mean_w12 — текущее / conditional mean
    {product}__microstructure__surprise_dir     — знак отклонения от среднего

Preset entry:
    microstructure:
      windows: [12]

Interpretation:
    predictability ≈ 0.99 — почти идеально предсказуемый ряд (плоский, CV≈0).
    predictability ≈ 0.40 — высокая нестабильность (пульсирующий паттерн).
    surprise > 2 при surprise_dir = -1 — текущий месяц значительно ниже нормы.
    vs_cond_mean ≈ 1 при активном месяце — типичный активный месяц, нет сюрпризов.

Example:
    Ряд (6 мес): [20, 20, 20, 20, 20, 30],  w=6
    mean = 130/6 = 21.667,  std = 3.727

    surprise = |v[t] − mean| / std = |30 − 21.667| / 3.727 = 2.236
    predictability = 1 / (1 + std/mean) = 1 / (1 + 3.727/21.667) = 0.853
    → microstructure__surprise_w6 = 2.236,  predictability_w6 = 0.853

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import EPS, compute_window_mean_and_std, resolve_window_size, safe_ratio

FEATURE = 'microstructure'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_snr = np.zeros((n_w, n_rows))
    out_surprise = np.zeros((n_w, n_rows))
    out_surprise_dir = np.zeros(n_rows)
    out_predictability = np.zeros((n_w, n_rows))
    out_cond_mean = np.zeros((n_w, n_rows))
    out_vs_cond = np.zeros((n_w, n_rows))

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        v = product_values[row_idx]

        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            ws_short = resolve_window_size(pos, max(windows[j] // 4, 1))

            mean, std = compute_window_mean_and_std(product_values, row_idx, ws)
            _, std_short = compute_window_mean_and_std(product_values, row_idx, ws_short)

            out_snr[j, row_idx] = safe_ratio(std_short, std)
            out_surprise[j, row_idx] = safe_ratio(abs(v - mean), std)
            # predictability = 1/(1+CV) ограничена (0,1] — деление с eps здесь
            # намеренно: CV -> inf корректно даёт predictability -> 0
            out_predictability[j, row_idx] = 1.0 / (1.0 + std / (abs(mean) + EPS))

            # conditional mean given active: mean/active_rate = window_sum/active_count
            active_count = 0
            for offset in range(ws):
                if product_values[row_idx - ws + 1 + offset] != 0.0:
                    active_count += 1
            cond_mean = mean * ws / active_count if active_count > 0 else 0.0
            out_cond_mean[j, row_idx] = cond_mean
            out_vs_cond[j, row_idx] = safe_ratio(v, cond_mean)

            if j == 0:
                out_surprise_dir[row_idx] = 1.0 if v >= mean else -1.0

    return out_snr, out_surprise, out_surprise_dir, out_predictability, out_cond_mean, out_vs_cond


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    snr, surp, sdir, pred, cm, vsc = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(snr[j])
        suffixes.append(f'snr_w{w}')
        arrays.append(surp[j])
        suffixes.append(f'surprise_w{w}')
        arrays.append(pred[j])
        suffixes.append(f'predictability_w{w}')
        arrays.append(cm[j])
        suffixes.append(f'cond_mean_w{w}')
        arrays.append(vsc[j])
        suffixes.append(f'vs_cond_mean_w{w}')
    arrays.append(sdir)
    suffixes.append('surprise_dir')
    return arrays, suffixes
