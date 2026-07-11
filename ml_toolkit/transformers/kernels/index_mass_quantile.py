"""Index mass quantile: доля окна (по времени), за которую накапливается X% суммарного объёма.

Signal:
    lifecycle_phase__peak_age_share показывает время ЕДИНСТВЕННОГО пика — чувствительно
    к одному выбросу. index_mass_quantile — устойчивая альтернатива: считает, к какой доле
    длины окна накопилось 25%/50%/75% суммарного объёма (по кумулятивной сумме, а не по
    единичному максимуму). Раннее достижение половины массы (низкий q50) — активность
    сфронтирована в начало окна; позднее (высокий q50) — сзади нагружена.

Formula:
    total_w = Σ(v[i], i in [0..ws-1])   (values ожидаются неотрицательными — иначе
        кумулятивная сумма немонотонна и порог не определён однозначно)
    running[i] = Σ(v[0..i])
    idx_q = первый i, для которого running[i] >= q · total_w
    qXX_w = idx_q / (ws - 1)   ∈ [0, 1]   (0 = масса набирается мгновенно в начале
        окна, 1 = только к самой последней точке)

    При total_w <= eps (весь ряд в окне нулевой) или ws < 2 — 0 (недостаточно сигнала).
    Квантили массы зафиксированы (25/50/75%), как квартили в kurtosis_proxy —
    не параметризуются.

Outputs:
    {product}__index_mass_quantile__q25_w12 — доля окна до накопления 25% объёма
    {product}__index_mass_quantile__q50_w12 — доля окна до накопления 50% объёма (медиана массы)
    {product}__index_mass_quantile__q75_w12 — доля окна до накопления 75% объёма

Preset entry:
    index_mass_quantile:
      windows: [12, 24]

Interpretation:
    q50 ≈ 0.5 — масса распределена равномерно по окну (как у линейно растущего
        или равномерного ряда).
    q50 < 0.3 — большая часть объёма пришлась на первую треть окна: активность
        затухает к концу.
    q50 > 0.7 — большая часть объёма — в последней трети: разгон к концу окна.
    q25 и q75 близки друг к другу — почти вся масса сосредоточена в узком
        промежутке времени внутри окна (концентрированный всплеск), а не размазана.

Example:
    Ряд (6 мес): [0, 0, 10, 10, 10, 10],  w=6

    total = 40,  running = [0, 0, 10, 20, 30, 40]
    q25: порог 10  → первый i с running>=10 — i=2 (running=10)  → 2/5 = 0.4
    q50: порог 20  → i=3 (running=20)                            → 3/5 = 0.6
    q75: порог 30  → i=4 (running=30)                            → 4/5 = 0.8
    → index_mass_quantile__q25_w6=0.4, q50_w6=0.6, q75_w6=0.8  (масса нагружена во вторую половину)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import EPS, resolve_window_size

FEATURE = 'index_mass_quantile'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_q25 = np.zeros((n_w, n_rows))
    out_q50 = np.zeros((n_w, n_rows))
    out_q75 = np.zeros((n_w, n_rows))

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            if ws < 2:
                continue
            start = row_idx - ws + 1
            total = 0.0
            for offset in range(ws):
                total += product_values[start + offset]
            if total <= EPS:
                continue

            thr25 = 0.25 * total
            thr50 = 0.5 * total
            thr75 = 0.75 * total
            running = 0.0
            found25 = False
            found50 = False
            found75 = False
            for offset in range(ws):
                running += product_values[start + offset]
                if not found25 and running >= thr25:
                    out_q25[j, row_idx] = offset / (ws - 1)
                    found25 = True
                if not found50 and running >= thr50:
                    out_q50[j, row_idx] = offset / (ws - 1)
                    found50 = True
                if not found75 and running >= thr75:
                    out_q75[j, row_idx] = offset / (ws - 1)
                    found75 = True
    return out_q25, out_q50, out_q75


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12, 24]} — ключ обязателен."""
    windows = np.array(params['windows'], dtype=np.int64)
    q25, q50, q75 = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(q25[j])
        suffixes.append(f'q25_w{w}')
        arrays.append(q50[j])
        suffixes.append(f'q50_w{w}')
        arrays.append(q75[j])
        suffixes.append(f'q75_w{w}')
    return arrays, suffixes
