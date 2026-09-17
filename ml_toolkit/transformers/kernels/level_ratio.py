"""Отношение средних на коротком и длинном окнах: смещение недавнего уровня.

Signal:
    Показывает, насколько недавний уровень отличается от исторического. Значение > 1
    — последние месяцы в среднем выше долгосрочной нормы (рост); < 1 — ниже нормы
    (снижение или коррекция). Устойчивее к шуму, чем slope, — опирается на средние.

Formula:
    mean_short = mean(v[t-ws_short+1..t])
    mean_long  = mean(v[t-ws_long+1..t])
    level_ratio = mean_short / (|mean_long| + eps)

Outputs:
    {product}__level_ratio__w3_w12  — среднее 3 мес / среднее 12 мес
    {product}__level_ratio__w6_w24  — среднее 6 мес / среднее 24 мес

Preset entry:
    level_ratio:
      pairs:
        - [3, 12]
        - [6, 24]
      dilations: [1, 2]   # optional, default [1]; applies to both windows of every pair

Interpretation:
    w3_w12 = 1.4 — последний квартал на 40% выше годовой нормы.
    w6_w24 = 0.7 — последние полгода на 30% ниже двухлетней нормы.
    Близко к 1 — уровень стабилен, нет явного тренда.
    w3_w12 > 1, w6_w24 < 1 — краткосрочный отскок при долгосрочном снижении.

Example:
    Ряд (6 мес): [10, 20, 30, 40, 50, 60],  пара (3, 6)
    (t=5; короткое окно 3 мес, длинное 6 мес)

    mean_short = (40+50+60)/3 = 50
    mean_long  = 35  (среднее всех 6)
    level_ratio = 50 / 35 = 1.429
    → level_ratio__w3_w6 = 1.429  (последний квартал на ~43% выше нормы)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import (
    FILL_NAN_UNBOUNDED_LOW,
    compute_window_mean,
    resolve_window_size,
    safe_ratio,
)

FEATURE = 'level_ratio'
FILL_NAN: dict[str | None, float] = {None: FILL_NAN_UNBOUNDED_LOW}


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    short_windows: np.ndarray,
    long_windows: np.ndarray,
    dilations: np.ndarray,
):
    n_rows = product_values.shape[0]
    n_p = short_windows.shape[0]
    n_d = dilations.shape[0]
    out = np.zeros((n_p * n_d, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        k = 0
        for j in range(n_p):
            for d in range(n_d):
                dilation = dilations[d]
                ws_short = resolve_window_size(pos, short_windows[j], dilation)
                ws_long = resolve_window_size(pos, long_windows[j], dilation)
                mean_short = compute_window_mean(product_values, row_idx, ws_short, dilation)
                mean_long = compute_window_mean(product_values, row_idx, ws_long, dilation)
                out[k, row_idx] = safe_ratio(mean_short, mean_long)
                k += 1
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"pairs": [[6, 12]], "dilations": [1, 2]}. "dilations" optional, default [1]."""
    pairs = params['pairs']
    dilations_list = params.get('dilations', [1])
    short_w = np.array([p[0] for p in pairs], dtype=np.int64)
    long_w = np.array([p[1] for p in pairs], dtype=np.int64)
    dilations = np.array(dilations_list, dtype=np.int64)
    out = _kernel(values, position, short_w, long_w, dilations)
    suffixes = [f'w{p[0]}_w{p[1]}' if d == 1 else f'w{p[0]}_w{p[1]}_d{d}' for p in pairs for d in dilations_list]
    return [out[k] for k in range(len(suffixes))], suffixes
