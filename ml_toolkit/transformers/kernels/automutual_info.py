"""Automutual information: нелинейное обобщение автокорреляции через взаимную информацию.

Signal:
    autocorr/agg_autocorrelation меряют линейную связь через Пирсона — если зависимость v[t]
    от v[t-lag] нелинейна (например, U-образная или пороговая), Пирсон может показать r≈0
    при реально сильной зависимости. Automutual information квантует окно в 3 терцильных
    состояния (та же квантовка, что в transition_matrix) и считает взаимную информацию между
    состоянием lag назад и текущим — ловит ЛЮБУЮ статистическую зависимость, не только линейную.

Formula:
    t1, t2 = терцильные пороги ВСЕГО окна (sorted_quantile, q=1/3 и 2/3)
    state[i] = 0/1/2 по терцилям
    n_pairs = ws - lag
    count[a][b] = число пар (state[i-lag]=a, state[i]=b), i in [lag..ws-1]
    p_ab = count[a][b]/n_pairs;  p_a, p_b — маргинальные частоты по count
    MI = Σ(p_ab · ln(p_ab / (p_a·p_b)), p_ab > 0)     (взаимная информация, в нат)
    automutual_info_lagL_wW = MI / ln(3)   (нормировка на ln(K), K=3 состояния — тот же
        приём приближённой нормировки, что и в entropy.py/permutation_entropy, не точная
        верхняя граница MI, а практический масштаб для сравнения между рядами)

    Требует ws > lag и ws >= 4, иначе 0.

Outputs:
    {product}__automutual_info__lag1_w12  — взаимная информация, lag=1, окно 12
    {product}__automutual_info__lag1_w24  — то же, окно 24

Preset entry:
    automutual_info:
      lag_window_pairs:
        - [1, 12]
        - [1, 24]

Interpretation:
    ≈ 0 — состояние lag месяцев назад не даёт информации о текущем состоянии (ни линейной,
        ни нелинейной зависимости).
    заметно > 0 при autocorr__lag1 ≈ 0 — зависимость есть, но нелинейная (Пирсон её не
        видит, а MI видит) — сверьте с c3 для направления асимметрии.
    automutual_info по силе близко к autocorr — зависимость в основном линейная, MI не
        добавляет новой информации сверх того, что уже видно в autocorr.

Example:
    Ряд (6 мес): [10, 10, 50, 50, 90, 90],  lag=1, w=6

    состояния (терцили, как в transition_matrix): [0,0,1,1,2,2]
    пары (state[i-1],state[i]) i=1..5: (0,0),(0,1),(1,1),(1,2),(2,2)   (n_pairs=5, все разные)
    p_a: {0:2/5, 1:2/5, 2:1/5};  p_b: {0:1/5, 1:2/5, 2:2/5}
    MI = Σ p_ab·ln(p_ab/(p_a·p_b)) по 5 парам = 0.500 нат
    → automutual_info__lag1_w6 = 0.500 / ln(3) = 0.455

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import fill_window_sorted, resolve_window_size, sorted_quantile

FEATURE = 'automutual_info'

_LN3 = np.log(3.0)


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    lags: np.ndarray,
    windows: np.ndarray,
):
    n_rows = product_values.shape[0]
    n_p = lags.shape[0]
    out = np.zeros((n_p, n_rows))

    max_w = 1
    for j in range(n_p):
        max_w = max(max_w, windows[j])
    sorted_buf = np.empty(max_w)
    states = np.empty(max_w, dtype=np.int64)
    count = np.empty((3, 3))
    p_a = np.empty(3)
    p_b = np.empty(3)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_p):
            lag = lags[j]
            ws = resolve_window_size(pos, windows[j])
            if ws < 4 or ws <= lag:
                continue
            fill_window_sorted(sorted_buf, product_values, row_idx, ws)
            t1 = sorted_quantile(sorted_buf, ws, 1.0 / 3.0)
            t2 = sorted_quantile(sorted_buf, ws, 2.0 / 3.0)

            start = row_idx - ws + 1
            for offset in range(ws):
                v = product_values[start + offset]
                if v <= t1:
                    states[offset] = 0
                elif v <= t2:
                    states[offset] = 1
                else:
                    states[offset] = 2

            count[:, :] = 0.0
            n_pairs = ws - lag
            for i in range(lag, ws):
                count[states[i - lag], states[i]] += 1.0

            p_a[:] = 0.0
            p_b[:] = 0.0
            for a in range(3):
                for b in range(3):
                    p_a[a] += count[a, b]
                    p_b[b] += count[a, b]
            p_a /= n_pairs
            p_b /= n_pairs

            mi = 0.0
            for a in range(3):
                for b in range(3):
                    if count[a, b] > 0.0:
                        p_ab = count[a, b] / n_pairs
                        denom = p_a[a] * p_b[b]
                        if denom > 0.0:
                            mi += p_ab * np.log(p_ab / denom)
            out[j, row_idx] = mi / _LN3
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"lag_window_pairs": [[1, 12], [1, 24]]} — ключ обязателен."""
    pairs = params['lag_window_pairs']
    lags = np.array([p[0] for p in pairs], dtype=np.int64)
    windows = np.array([p[1] for p in pairs], dtype=np.int64)
    out = _kernel(values, position, lags, windows)
    arrays = [out[j] for j in range(len(pairs))]
    suffixes = [f'lag{p[0]}_w{p[1]}' for p in pairs]
    return arrays, suffixes
