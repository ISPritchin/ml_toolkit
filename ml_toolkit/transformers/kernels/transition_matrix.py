"""Transition matrix: марковская структура режимов — квантование в терцили + матрица переходов.

Signal:
    regime_change ищет ОДИН оптимальный разрыв уровня в окне; здесь другой взгляд — окно
    квантуется в 3 состояния (терцили собственного распределения: низкое/среднее/высокое),
    и строится матрица частот переходов между состояниями по соседним точкам. stickiness —
    вероятность, что ряд остаётся в том же состоянии на следующем шаге («липкость» режима).
    trans_entropy — насколько предсказуемы сами переходы между состояниями (энтропия
    совместного распределения пар состояний), а не насколько предсказуем следующий уровень.

Formula:
    t1, t2 = терцильные пороги окна (sorted_quantile, q=1/3 и 2/3)
    state[i] = 0 если v[i] <= t1, 1 если t1 < v[i] <= t2, 2 если v[i] > t2
    n_pairs = ws - 1
    count[a][b] = число переходов state[i-1]=a -> state[i]=b
    stickiness_w    = (count[0][0]+count[1][1]+count[2][2]) / n_pairs
    trans_entropy_w = -Σ(p_ab·ln(p_ab), p_ab=count[a][b]/n_pairs, p_ab>0) / ln(9)   ∈ [0, 1]

    Требует ws >= 4, иначе 0.

Outputs:
    {product}__transition_matrix__stickiness_w12    — доля переходов «в то же состояние»
    {product}__transition_matrix__trans_entropy_w12 — энтропия матрицы переходов

Preset entry:
    transition_matrix:
      windows: [12, 24]

Interpretation:
    stickiness ≈ 1.0 — ряд почти всегда остаётся в своей терцили (устойчивый режим,
        редкие переключения между низким/средним/высоким уровнем).
    stickiness ≈ 1/3 — переключения происходят не реже случайных (типично для быстро
        осциллирующего ряда без выраженных режимов).
    trans_entropy низкий при stickiness невысоком — переходы между состояниями хоть
        и частые, но идут по немногим устойчивым «маршрутам» (например, всегда
        низкое→высокое→низкое, без промежуточного состояния).
    trans_entropy высокий — структура переходов близка к случайной по всем 9 парам.

Example:
    Ряд (6 мес): [10, 10, 50, 50, 90, 90],  w=6

    отсортировано: [10,10,50,50,90,90] → t1=sorted[int(1/3·5)]=sorted[1]=10,
                                          t2=sorted[int(2/3·5)]=sorted[3]=50
    состояния (v<=10→0, 10<v<=50→1, v>50→2): [0,0,1,1,2,2]
    переходы (5 пар): 0→0, 0→1, 1→1, 1→2, 2→2
    count[0][0]=1, count[0][1]=1, count[1][1]=1, count[1][2]=1, count[2][2]=1
    stickiness = (1+1+1)/5 = 0.6
    p_ab по 5 ненулевым ячейкам из 9 = 0.2 каждая → h = -5·(0.2·ln0.2) = 1.609
    trans_entropy = 1.609 / ln(9) = 1.609 / 2.197 = 0.733
    → transition_matrix__stickiness_w6 = 0.6,  trans_entropy_w6 = 0.733

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import fill_window_sorted, resolve_window_size, sorted_quantile

FEATURE = 'transition_matrix'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_stick = np.zeros((n_w, n_rows))
    out_entropy = np.zeros((n_w, n_rows))

    max_w = 1
    for j in range(n_w):
        max_w = max(max_w, windows[j])
    sorted_buf = np.empty(max_w)
    count = np.empty((3, 3))
    ln9 = np.log(9.0)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            if ws < 4:
                continue
            fill_window_sorted(sorted_buf, product_values, row_idx, ws)
            t1 = sorted_quantile(sorted_buf, ws, 1.0 / 3.0)
            t2 = sorted_quantile(sorted_buf, ws, 2.0 / 3.0)

            start = row_idx - ws + 1
            count[:, :] = 0.0
            prev_state = 0
            for offset in range(ws):
                v = product_values[start + offset]
                if v <= t1:
                    state = 0
                elif v <= t2:
                    state = 1
                else:
                    state = 2
                if offset >= 1:
                    count[prev_state, state] += 1.0
                prev_state = state

            n_pairs = ws - 1
            out_stick[j, row_idx] = (count[0, 0] + count[1, 1] + count[2, 2]) / n_pairs

            h = 0.0
            for a in range(3):
                for b in range(3):
                    if count[a, b] > 0.0:
                        p = count[a, b] / n_pairs
                        h -= p * np.log(p)
            out_entropy[j, row_idx] = h / ln9

    return out_stick, out_entropy


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12, 24]} — ключ обязателен."""
    windows = np.array(params['windows'], dtype=np.int64)
    stick, entropy = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(stick[j])
        suffixes.append(f'stickiness_w{w}')
        arrays.append(entropy[j])
        suffixes.append(f'trans_entropy_w{w}')
    return arrays, suffixes
