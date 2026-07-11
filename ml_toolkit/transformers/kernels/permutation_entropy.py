"""Permutation entropy: энтропия ordinal-паттернов (порядка соседних троек), а не уровня.

Signal:
    entropy.py (Шеннон) меряет равномерность распределения ОБЪЁМОВ по месяцам — ряд может
    быть крайне неравномерным по уровню, но при этом полностью предсказуемым по ДИНАМИКЕ
    (строго монотонный рост даёт entropy≈1, но абсолютно регулярен по порядку). Permutation
    entropy — про порядок: разбивает окно на перекрывающиеся тройки соседних точек, у каждой
    определяет один из 6 возможных ordinal-паттернов (кто больше кого), и считает энтропию
    распределения этих паттернов. Строго монотонный участок — всегда один и тот же паттерн
    (энтропия 0), белый шум — все 6 паттернов равновероятны (энтропия 1).

Formula:
    Для каждой тройки (v[i], v[i+1], v[i+2]) в окне определяется тип ordinal-паттерна
    (одна из 3! = 6 перестановок; тай-брейк равных значений — по позиции, чтобы повторы
    и нули классифицировались детерминированно, а не ломали подсчёт).
    n = ws - 2  (число троек)
    p_k = count(тип == k) / n,  k = 0..5
    permutation_entropy_w = -Σ(p_k · ln(p_k), p_k > 0) / ln(6)   ∈ [0, 1]

    embedding dimension m=3 и delay tau=1 зафиксированы внутри (не параметризуются):
    при типичных для этой библиотеки окнах 6/12/24 m=4 (4!=24 паттерна) не даёт
    статистически надёжной оценки — m=3 (6 паттернов) практический потолок.
    Требует ws >= 3, иначе 0.

Outputs:
    {product}__permutation_entropy__w6   — нормированная ordinal-энтропия за 6 мес
    {product}__permutation_entropy__w12  — то же за 12 мес
    {product}__permutation_entropy__w24  — то же за 24 мес

Preset entry:
    permutation_entropy:
      windows: [6, 12, 24]

Interpretation:
    ≈ 1.0 — все 6 ordinal-паттернов равновероятны: порядок взлётов/падений максимально
        непредсказуем (по динамике, не по уровню — сравните с entropy.py).
    ≈ 0.0 — почти все тройки одного паттерна (например, строго монотонный рост/падение
        внутри всего окна) — крайне регулярная, предсказуемая динамика.
    Высокий entropy.py при низком permutation_entropy — ряд неравномерен по объёму
        (редкие крупные месяцы), но их последовательность строго упорядочена (например,
        монотонно растущие всплески) — разные грани одного ряда, не противоречие.

Example:
    Ряд (6 мес): [10, 20, 30, 40, 50, 60],  w=6  (строго монотонный рост)

    тройки: (10,20,30), (20,30,40), (30,40,50), (40,50,60) — все возрастающие, один
    и тот же ordinal-паттерн (тип 0)   (n=4)
    p_0 = 4/4 = 1.0 → h = -1.0·ln(1.0) = 0
    → permutation_entropy__w6 = 0/ln(6) = 0.0  (максимально регулярная динамика)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import resolve_window_size

FEATURE = 'permutation_entropy'


@nb.njit(cache=True)
def _perm_type3(x: float, y: float, z: float) -> int:
    """Тип ordinal-паттерна тройки (x,y,z) — одна из 6 перестановок; тай-брейк равных
    значений по позиции (x перед y перед z), через total order на <=, а не строгое <.
    """
    lt_xy = x <= y
    lt_yz = y <= z
    lt_xz = x <= z
    if lt_xy and lt_yz:
        return 0  # x<=y<=z
    elif lt_xy and lt_xz:
        return 1  # x<=z<y
    elif lt_xy:
        return 2  # z<x<=y
    elif lt_yz and lt_xz:
        return 3  # y<x<=z
    elif lt_yz:
        return 4  # y<=z<x
    else:
        return 5  # z<y<x


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    ln6 = np.log(6.0)
    counts = np.empty(6)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            n_triples = ws - 2
            if n_triples < 1:
                continue
            counts[:] = 0.0
            start = row_idx - ws + 1
            for i in range(n_triples):
                t = _perm_type3(
                    product_values[start + i],
                    product_values[start + i + 1],
                    product_values[start + i + 2],
                )
                counts[t] += 1.0
            h = 0.0
            for k in range(6):
                if counts[k] > 0.0:
                    p = counts[k] / n_triples
                    h -= p * np.log(p)
            out[j, row_idx] = h / ln6
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6, 12, 24]} — ключ обязателен."""
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f'w{w}' for w in params['windows']]
