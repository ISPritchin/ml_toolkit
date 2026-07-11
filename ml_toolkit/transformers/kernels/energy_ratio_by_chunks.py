"""Energy ratio by chunks: доля энергии (Σv²) ряда, приходящаяся на треть окна.

Signal:
    Делит окно на три равные части (метод третей, как в nonlinearity.py) и считает, какая
    доля суммарной энергии (Σv²) приходится на каждую треть. В отличие от value_clustering
    (доли по РАНГУ значений, независимо от позиции) и cross_window_momentum (отношения
    СРЕДНИХ, знакочувствительные) — здесь позиция во времени первична, а квадрат снимает
    знак и сильно взвешивает именно крупные выбросы: один большой всплеск в последней трети
    сдвигает last-долю к 1, даже если он единственный за всё окно.

Formula:
    third = ws // 3   (остаток 0-2 строк в конце окна не учитывается — та же
        конвенция, что и в nonlinearity.py)
    total_energy_w = Σ(v[i]², i in [0..ws-1])           (по ВСЕМУ окну, не только 3·third)
    e1 = Σ(v[i]², i in [0..third-1])          — первая треть
    e2 = Σ(v[i]², i in [third..2·third-1])    — средняя треть
    e3 = Σ(v[i]², i in [2·third..3·third-1])  — последняя треть
    {first,mid,last}_share_w = safe_ratio(e{1,2,3}, total_energy_w)

    Требует ws >= 3 (third >= 1), иначе 0.

Outputs:
    {product}__energy_ratio_by_chunks__first_w6/12 — доля энергии в первой трети окна
    {product}__energy_ratio_by_chunks__mid_w6/12    — доля энергии в средней трети
    {product}__energy_ratio_by_chunks__last_w6/12   — доля энергии в последней трети

Preset entry:
    energy_ratio_by_chunks:
      windows: [6, 12]

Interpretation:
    last_share >> first_share — энергия (крупные значения) сконцентрирована в конце
        окна: недавний всплеск доминирует над всей историей окна.
    first_share >> last_share — было бурно в начале окна, сейчас затихло.
    Доли не обязаны давать в сумме 1 (остаток окна, не кратный 3, не учитывается) —
        не использовать как строгое распределение вероятностей.
    Высокий last_share при низком window_volatility_ratios__vol_accel — энергия
        сконцентрирована на нескольких крупных уровнях, а не на растущей волатильности
        приращений; это разные сигналы, сверяйте оба.

Example:
    Ряд (6 мес): [5, 5, 10, 10, 30, 30],  w=6,  third=2

    e1 = 5²+5² = 50,  e2 = 10²+10² = 200,  e3 = 30²+30² = 1800
    total = 50+200+1800 = 2050
    → energy_ratio_by_chunks__first_w6 = 50/2050 = 0.024
    → energy_ratio_by_chunks__mid_w6   = 200/2050 = 0.098
    → energy_ratio_by_chunks__last_w6  = 1800/2050 = 0.878  (энергия в конце окна)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import resolve_window_size, safe_ratio

FEATURE = 'energy_ratio_by_chunks'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_first = np.zeros((n_w, n_rows))
    out_mid = np.zeros((n_w, n_rows))
    out_last = np.zeros((n_w, n_rows))

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            third = ws // 3
            if third < 1:
                continue
            start = row_idx - ws + 1
            total_energy = 0.0
            for offset in range(ws):
                v = product_values[start + offset]
                total_energy += v * v
            e1 = 0.0
            e2 = 0.0
            e3 = 0.0
            for i in range(third):
                v1 = product_values[start + i]
                v2 = product_values[start + third + i]
                v3 = product_values[start + 2 * third + i]
                e1 += v1 * v1
                e2 += v2 * v2
                e3 += v3 * v3
            out_first[j, row_idx] = safe_ratio(e1, total_energy)
            out_mid[j, row_idx] = safe_ratio(e2, total_energy)
            out_last[j, row_idx] = safe_ratio(e3, total_energy)

    return out_first, out_mid, out_last


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6, 12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    first, mid, last = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(first[j])
        suffixes.append(f'first_w{w}')
        arrays.append(mid[j])
        suffixes.append(f'mid_w{w}')
        arrays.append(last[j])
        suffixes.append(f'last_w{w}')
    return arrays, suffixes
