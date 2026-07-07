"""Органичность роста: равномерность положительных приростов, Gini роста.

Signal:
    Разграничивает «органичный» рост (много равных небольших приростов) от «проектного»
    (один крупный скачок = вся прибавка). Высокий organic_share при высоком consist_score
    — здоровый, стабильный рост. Низкий best_share означает прирост распределён равномерно.

Formula:
    delta_pos[i] = max(0, v[i] - v[i-1])   для i in [t-w+2..t]
    best_share_w    = max(delta_pos) / (sum(delta_pos) + eps)
    organic_w       = 1 - best_share_w
    consist_score_w = count(delta_pos > 0) / max(active_months, 1)
    pos_count_w     = count(delta_pos > 0)
    growth_gini_w   = Gini(delta_pos)   (только по положительным)
    neg_sum_share_w = sum(max(0, -delta[i])) / (|mean_w| * ws + eps)

Outputs:
    {product}__growth_quality__best_share_w12    — доля роста от лучшего скачка
    {product}__growth_quality__consist_score_w12 — доля активных мес. с ростом
    {product}__growth_quality__pos_count_w12     — число месяцев с ростом
    {product}__growth_quality__growth_gini_w12   — Gini положительных приростов
    {product}__growth_quality__organic_w12       — 1 - best_share (органичность)
    {product}__growth_quality__neg_sum_share_w12 — доля суммарных потерь к обороту

Preset (monthly.yaml):
    growth_quality:
      windows: [12]

Interpretation:
    organic_w12 = 0.91 — рост равномерно распределён по месяцам (органичный).
    organic_w12 ≈ 0 — весь рост от одного контракта/скачка (проектный).
    consist_score ≈ 1 + growth_gini ≈ 0 — идеальный равномерный рост.
    neg_sum_share > 0.3 — значительные откаты, чистый прирост невелик.

Example:
    Ряд (6 мес): [10, 20, 30, 40, 50, 60],  w=6
    положительные приросты: +10, +10, +10, +10, +10 (5 шт., сумма 50)

    best_share = max(delta)/sum(delta) = 10/50 = 0.2
    organic = 1 − 0.2 = 0.8
    pos_count = 5,  consist_score = 5/5 = 1.0,  growth_gini = 0 (все приросты равны)
    → growth_quality__organic_w6 = 0.8,  best_share_w6 = 0.2,  pos_count_w6 = 5

"""

import numba as nb
import numpy as np

from .._windowing import EPS, compute_window_mean, resolve_window_size, safe_ratio

FEATURE = 'growth_quality'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_best_share = np.zeros((n_w, n_rows))
    out_consist_score = np.zeros((n_w, n_rows))
    out_pos_count = np.zeros((n_w, n_rows))
    out_growth_gini = np.zeros((n_w, n_rows))
    out_organic = np.zeros((n_w, n_rows))
    out_neg_sum_share = np.zeros((n_w, n_rows))

    max_w = 1
    for j in range(n_w):
        max_w = max(max_w, windows[j])
    pos_diffs = np.zeros(max_w)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            n_diffs = ws - 1
            if n_diffs < 1:
                continue

            n_pos = 0
            sum_pos = 0.0
            sum_neg_abs = 0.0
            max_pos = 0.0
            active_count = 0

            for offset in range(1, ws):
                abs_idx = row_idx - ws + 1 + offset
                d = product_values[abs_idx] - product_values[abs_idx - 1]
                if d > 0.0:
                    pos_diffs[n_pos] = d
                    n_pos += 1
                    sum_pos += d
                    max_pos = max(max_pos, d)
                elif d < 0.0:
                    sum_neg_abs += abs(d)
                if product_values[abs_idx] != 0.0:
                    active_count += 1

            out_pos_count[j, row_idx] = n_pos
            out_best_share[j, row_idx] = max_pos / (sum_pos + EPS)
            out_organic[j, row_idx] = 1.0 - max_pos / (sum_pos + EPS)
            out_consist_score[j, row_idx] = n_pos / max(active_count, 1)

            # growth gini on positive diffs
            if n_pos >= 2:
                # сортировка вставками (вместо пузырька): O(k) на почти
                # отсортированных данных, k <= ws-1
                for a in range(1, n_pos):
                    key = pos_diffs[a]
                    b = a - 1
                    while b >= 0 and pos_diffs[b] > key:
                        pos_diffs[b + 1] = pos_diffs[b]
                        b -= 1
                    pos_diffs[b + 1] = key
                gini_num = 0.0
                for i in range(n_pos):
                    gini_num += (2 * (i + 1) - n_pos - 1) * pos_diffs[i]
                out_growth_gini[j, row_idx] = gini_num / (n_pos * sum_pos + EPS)

            mean = compute_window_mean(product_values, row_idx, ws)
            out_neg_sum_share[j, row_idx] = safe_ratio(sum_neg_abs, abs(mean) * ws)

    return out_best_share, out_consist_score, out_pos_count, out_growth_gini, out_organic, out_neg_sum_share


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}"""
    windows = np.array(params['windows'], dtype=np.int64)
    bs, cs, pc, gg, org, nss = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(bs[j]);  suffixes.append(f'best_share_w{w}')
        arrays.append(cs[j]);  suffixes.append(f'consist_score_w{w}')
        arrays.append(pc[j]);  suffixes.append(f'pos_count_w{w}')
        arrays.append(gg[j]);  suffixes.append(f'growth_gini_w{w}')
        arrays.append(org[j]); suffixes.append(f'organic_w{w}')
        arrays.append(nss[j]); suffixes.append(f'neg_sum_share_w{w}')
    return arrays, suffixes
