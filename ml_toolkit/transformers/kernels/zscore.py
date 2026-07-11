"""Z-score текущего значения относительно скользящего окна (нормированное отклонение).

Signal:
    Показывает, насколько текущее значение выбивается из «нормы» своего же окна. Значение > 2 —
    экстремальный пик относительно последних w месяцев; < -2 — аномально низкое значение.
    В отличие от rank_in_window (порядковая статистика), zscore учитывает расстояние в единицах
    std, что делает его чувствительным к «выбросам» разного масштаба.

Formula:
    mean_w = mean(v[t-w+1..t])
    std_w  = std(v[t-w+1..t])
    zscore_w = (v[t] - mean_w) / (std_w + eps)

    std_w — популяционное (без коррекции Бесселя).
    Текущий месяц входит в окно, поэтому zscore привязан к самой свежей «норме».

Outputs:
    {product}__zscore__w6   — z-score относительно 6 мес
    {product}__zscore__w12  — z-score относительно 12 мес
    {product}__zscore__w24  — z-score относительно 24 мес

Preset entry:
    zscore:
      windows: [6, 12, 24]

Interpretation:
    zscore_w12 = 0 — значение точно на уровне годового среднего.
    zscore_w12 > 2 — аномально высокий месяц: возможно, разовый крупный выброс или сезонный пик.
    zscore_w12 < -1.5 при положительном slope — временный откат на растущем тренде.
    zscore_w6 > zscore_w24 — выброс свежее и резче, чем в двухлетней перспективе.

Example:
    Ряд (6 мес): [10, 10, 10, 10, 10, 40],  w=6,  v[t]=40

    mean = 90/6 = 15,  std = 11.18
    zscore = (v[t] − mean) / std = (40 − 15) / 11.18 = 2.236
    → zscore__w6 = 2.236  (аномально высокий текущий месяц, > 2σ)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import compute_window_mean_and_std, resolve_window_size, safe_ratio

FEATURE = 'zscore'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        v = product_values[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            mean, std = compute_window_mean_and_std(product_values, row_idx, ws)
            out[j, row_idx] = safe_ratio(v - mean, std)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6, 12, 24]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f'w{w}' for w in params['windows']]
