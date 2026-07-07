"""Структурный сдвиг уровня: оптимальный разрыв внутри окна, флаг смены режима.

Signal:
    Обнаруживает «смену режима»: резкий переход с одного уровня дохода на другой.
    Алгоритм перебирает все точки разбиения окна и ищет оптимальный разрыв.
    Высокий magnitude при flag = 1 — явная структурная смена (контракт/расторжение).

Formula:
    Для каждого k in [1..ws-1]:
        mean_left  = mean(v[t-ws+1..t-ws+k])
        mean_right = mean(v[t-ws+k+1..t])
        magnitude(k) = |mean_left - mean_right| / (std_w + eps)
    level_shift_magnitude_w = max(magnitude(k))
    split_pos_w             = k* (оптимальная точка разрыва)
    flag_w                  = 1 if magnitude > 2.0
    late_vs_early_w         = (mean_late_half - mean_early_half) / (std_w + eps)
    asymmetry_w             = mean_last3 / (|mean_first3| + eps)
    current_regime_len      — running: мес. без сдвига флага (сбрасывается при flag = 1)

Outputs:
    {product}__regime_change__magnitude_w12      — сила оптимального разрыва в σ
    {product}__regime_change__split_pos_w12      — позиция разрыва внутри окна
    {product}__regime_change__flag_w12           — флаг смены режима (> 2σ)
    {product}__regime_change__late_vs_early_w12  — поздняя vs ранняя половина (в σ)
    {product}__regime_change__asymmetry_w12      — среднее последних 3 / первых 3
    {product}__regime_change__current_regime_len — длина текущего стабильного периода

Preset (monthly.yaml):
    regime_change:
      windows: [12]

Interpretation:
    magnitude = 2.33 при флаге = 1 — явная смена режима (пример R: 0→100).
    magnitude < 1 — единый уровень в окне, нет смены.
    late_vs_early > 2 — вторая половина года значительно выше первой.
    current_regime_len = 0 — прямо сейчас обнаруженная смена режима.

Example:
    Ряд (6 мес): [0, 0, 100, 100, 100, 100],  w=6
    mean = 400/6 = 66.667,  std = 47.14

    оптимальный разрыв при k=2: mean_left = 0,  mean_right = 100
    magnitude = |0 − 100| / 47.14 = 2.121 > 2 → flag = 1
    → regime_change__magnitude_w6 = 2.121,  split_pos_w6 = 2,  flag_w6 = 1

"""

import numba as nb
import numpy as np

from .._windowing import compute_window_mean_and_std, resolve_window_size, safe_ratio

FEATURE = 'regime_change'


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    windows: np.ndarray,
    shift_threshold: float,
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_magnitude = np.zeros((n_w, n_rows))
    out_split_pos = np.zeros((n_w, n_rows))
    out_flag = np.zeros((n_w, n_rows))
    out_late_vs_early = np.zeros((n_w, n_rows))
    out_regime_asym = np.zeros((n_w, n_rows))

    # running state: длина текущего режима (сбрасывается при флаге сдвига)
    r_regime_len = 0
    out_regime_len = np.zeros(n_rows)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        if pos == 0:
            r_regime_len = 0

        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            mean_all, std_all = compute_window_mean_and_std(product_values, row_idx, ws)
            total = mean_all * ws

            # накопительная левая сумма вместо пересуммирования обеих половин:
            # O(ws) на окно вместо O(ws^2)
            best_diff = 0.0
            best_k = 0
            sum_l = 0.0
            for k in range(1, ws):
                sum_l += product_values[row_idx - ws + k]
                mean_l = sum_l / k
                mean_r = (total - sum_l) / (ws - k)
                d = safe_ratio(abs(mean_l - mean_r), std_all)
                if d > best_diff:
                    best_diff = d
                    best_k = k

            out_magnitude[j, row_idx] = best_diff
            out_split_pos[j, row_idx] = best_k
            is_shift = 1.0 if best_diff > shift_threshold else 0.0
            out_flag[j, row_idx] = is_shift

            # late half vs early half
            half = ws // 2
            sum_late = 0.0
            sum_early = 0.0
            for i in range(half):
                sum_early += product_values[row_idx - ws + 1 + i]
            for i in range(half, ws):
                sum_late += product_values[row_idx - ws + 1 + i]
            mean_early = sum_early / max(half, 1)
            mean_late = sum_late / max(ws - half, 1)
            out_late_vs_early[j, row_idx] = safe_ratio(mean_late - mean_early, std_all)

            # regime asymmetry: mean_last3 / mean_first3
            n3 = min(3, ws)
            s_first = 0.0; s_last = 0.0
            for i in range(n3):
                s_first += product_values[row_idx - ws + 1 + i]
                s_last += product_values[row_idx - n3 + 1 + i]
            out_regime_asym[j, row_idx] = safe_ratio(s_last / n3, s_first / n3)

            if j == 0:
                if is_shift:
                    r_regime_len = 0
                else:
                    r_regime_len += 1

        out_regime_len[row_idx] = r_regime_len

    return out_magnitude, out_split_pos, out_flag, out_late_vs_early, out_regime_asym, out_regime_len


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12], "shift_threshold": 2.0 (опционально, в сигмах)}"""
    windows = np.array(params['windows'], dtype=np.int64)
    shift_threshold = float(params.get('shift_threshold', 2.0))
    mag, spos, flag, lve, asym, rlen = _kernel(values, position, windows, shift_threshold)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(mag[j]);   suffixes.append(f'magnitude_w{w}')
        arrays.append(spos[j]);  suffixes.append(f'split_pos_w{w}')
        arrays.append(flag[j]);  suffixes.append(f'flag_w{w}')
        arrays.append(lve[j]);   suffixes.append(f'late_vs_early_w{w}')
        arrays.append(asym[j]);  suffixes.append(f'asymmetry_w{w}')
    arrays.append(rlen); suffixes.append('current_regime_len')
    return arrays, suffixes
