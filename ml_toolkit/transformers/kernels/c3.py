"""Time-reversal asymmetry (c3): детектор нелинейной, необратимой во времени динамики.

Signal:
    Обычная автокорреляция (autocorr, corr_with_time) — линейная мера памяти, симметричная
    относительно направления времени: она не отличит ряд от его инверсии во времени.
    c3 — момент третьего порядка E[v[t]·v[t+lag]·v[t+2·lag]], нормированный на std_w³.
    Для линейного стационарного (гауссовского) процесса это среднее стремится к нулю
    независимо от лага; ненулевое значение указывает на нелинейную структуру динамики —
    например, резкие скачки с медленным затуханием (или наоборот), которые autocorr
    и corr_with_time видят как «просто шум», а c3 — как устойчивый паттерн асимметрии.

Formula:
    n = ws - 2·lag  (число валидных троек в окне)
    c3_raw_lagL_wW = mean(v[i]·v[i+L]·v[i+2L], i in [0..n-1] внутри окна)
    c3_lagL_wW     = safe_ratio(c3_raw, std_w³)  (0 при почти постоянном ряде — std_w ~ 0)

    Требует ws > 2·lag, иначе 0 (недостаточно истории для этой пары).

Outputs:
    {product}__c3__lag1_w6   — нормированный c3, lag=1, окно 6
    {product}__c3__lag1_w12  — нормированный c3, lag=1, окно 12

Preset entry:
    c3:
      lag_window_pairs:
        - [1, 6]
        - [1, 12]

Interpretation:
    ≈ 0 — динамика ряда согласуется с линейным, симметричным во времени процессом
        (то, что видят autocorr/corr_with_time, — это всё, что там есть).
    сильно ≠ 0 (в любую сторону) — обнаружена нелинейная, time-irreversible структура:
        типична для рядов с резкими скачками и медленным откатом (или наоборот),
        которую линейная автокорреляция пропускает.
    Знак малоинформативен сам по себе (зависит от направления асимметрии) — используйте
        |c3| как силу нелинейности, а не как индикатор направления тренда.

Example:
    Ряд (5 мес): [1, 2, 4, 2, 1],  lag=1, w=5

    тройки (v[i],v[i+1],v[i+2]): (1,2,4)→8, (2,4,2)→16, (4,2,1)→8   (n=3)
    c3_raw = (8+16+8)/3 = 10.667
    mean_w = 2.0,  std_w = sqrt(mean((v-2)²)) = sqrt(6/5) = 1.095
    c3 = 10.667 / 1.095³ = 10.667 / 1.315 = 8.11  (заметная нелинейность)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import compute_window_mean_and_std, resolve_window_size, safe_ratio

FEATURE = 'c3'


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
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_p):
            lag = lags[j]
            ws = resolve_window_size(pos, windows[j])
            n_triples = ws - 2 * lag
            if n_triples < 1:
                continue
            start = row_idx - ws + 1
            raw_sum = 0.0
            for i in range(n_triples):
                raw_sum += (
                    product_values[start + i]
                    * product_values[start + i + lag]
                    * product_values[start + i + 2 * lag]
                )
            c3_raw = raw_sum / n_triples
            _, std_w = compute_window_mean_and_std(product_values, row_idx, ws)
            out[j, row_idx] = safe_ratio(c3_raw, std_w ** 3)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"lag_window_pairs": [[1, 6], [1, 12]]} — ключ обязателен."""
    pairs = params['lag_window_pairs']
    lags = np.array([p[0] for p in pairs], dtype=np.int64)
    windows = np.array([p[1] for p in pairs], dtype=np.int64)
    out = _kernel(values, position, lags, windows)
    arrays = [out[j] for j in range(len(pairs))]
    suffixes = [f'lag{p[0]}_w{p[1]}' for p in pairs]
    return arrays, suffixes
