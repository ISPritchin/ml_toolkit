"""Автокорреляция: expanding (lag 1/2/3) и windowed (lag 1/2 в окне 12) + partial.

Signal:
    Измеряет «память» ряда: насколько текущий месяц похож на предыдущий (lag1),
    позапрошлый (lag2) и т.д. Высокий положительный lag1 — инерционный клиент;
    сильный отрицательный lag1 — осциллирующий (чётные/нечётные месяцы разные).

Formula:
    Expanding Pearson (накопленный с начала истории):
        r_k = (n*sum(x*y) - sum(x)*sum(y)) /
              sqrt((n*sum(x²)-sum(x)²)*(n*sum(y²)-sum(y)²))
        где пара (x, y) = (v[t-k], v[t])

    Windowed Pearson lag k в окне 12:
        вычисляется по парам (v[i], v[i+k]) для i in [t-ws+1..t-k]

    Partial autocorrelation lag 2 (Yule-Walker):
        PACF2 = (r2 - r1²) / (1 - r1² + eps)

Outputs:
    {product}__autocorr__lag1        — expanding Pearson, лаг 1
    {product}__autocorr__lag2        — expanding Pearson, лаг 2
    {product}__autocorr__lag3        — expanding Pearson, лаг 3
    {product}__autocorr__lag1_w12    — windowed Pearson lag 1, окно 12
    {product}__autocorr__lag2_w12    — windowed Pearson lag 2, окно 12
    {product}__autocorr__partial_lag2 — частичная автокорреляция лага 2

Preset (monthly.yaml):
    autocorr: {}

Interpretation:
    lag1 ≈ +1 — сильная инерция (рост следует за ростом, плато остаётся плато).
    lag1 ≈ -1 — жёсткая осцилляция между высокими и низкими значениями.
    partial_lag2 ≈ 0 при высоком lag2 — лаг-2 корреляция полностью объяснена лагом-1.
    lag3 высокий — квартальная периодичность (подтверждается seasonal_autocorr).

Example:
    Ряд (5 мес): [10, 20, 15, 25, 20]
    (t=4; expanding Pearson лага 1 по парам (v[t-1], v[t]))

    пары (x,y): (10,20),(20,15),(15,25),(25,20),  n=4
    sum(x)=70, sum(y)=80, sum(x·y)=1375, sum(x²)=1350, sum(y²)=1650
    num = 4·1375 − 70·80 = 5500 − 5600 = −100
    den = sqrt((4·1350−70²)(4·1650−80²)) = sqrt(800·200) = 316.23
    → autocorr__lag1 = −100/316.23 = −0.316  (лёгкая осцилляция)
"""

import numba as nb
import numpy as np

from .._windowing import pearson_from_sums, safe_ratio, windowed_lag_pearson

FEATURE = "autocorr"


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray):
    n_rows = product_values.shape[0]
    lag1 = np.zeros(n_rows)
    lag2 = np.zeros(n_rows)
    lag3 = np.zeros(n_rows)
    lag1_w12 = np.zeros(n_rows)
    lag2_w12 = np.zeros(n_rows)
    partial_lag2 = np.zeros(n_rows)

    n1 = sx1 = sy1 = sxy1 = sx21 = sy21 = 0.0
    n2 = sx2 = sy2 = sxy2 = sx22 = sy22 = 0.0
    n3 = sx3 = sy3 = sxy3 = sx23 = sy23 = 0.0

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        if pos == 0:
            n1 = sx1 = sy1 = sxy1 = sx21 = sy21 = 0.0
            n2 = sx2 = sy2 = sxy2 = sx22 = sy22 = 0.0
            n3 = sx3 = sy3 = sxy3 = sx23 = sy23 = 0.0

        v = product_values[row_idx]
        if pos >= 1:
            x1 = product_values[row_idx - 1]
            n1 += 1.0; sx1 += x1; sy1 += v; sxy1 += x1*v; sx21 += x1*x1; sy21 += v*v
        if pos >= 2:
            x2 = product_values[row_idx - 2]
            n2 += 1.0; sx2 += x2; sy2 += v; sxy2 += x2*v; sx22 += x2*x2; sy22 += v*v
        if pos >= 3:
            x3 = product_values[row_idx - 3]
            n3 += 1.0; sx3 += x3; sy3 += v; sxy3 += x3*v; sx23 += x3*x3; sy23 += v*v

        r1 = pearson_from_sums(n1, sx1, sy1, sxy1, sx21, sy21)
        r2 = pearson_from_sums(n2, sx2, sy2, sxy2, sx22, sy22)
        r3 = pearson_from_sums(n3, sx3, sy3, sxy3, sx23, sy23)

        lag1[row_idx] = r1
        lag2[row_idx] = r2
        lag3[row_idx] = r3

        ws12 = min(pos + 1, 12)
        lag1_w12[row_idx] = windowed_lag_pearson(product_values, row_idx, ws12, 1)
        lag2_w12[row_idx] = windowed_lag_pearson(product_values, row_idx, ws12, 2)

        partial_lag2[row_idx] = safe_ratio(r2 - r1 * r1, 1.0 - r1 * r1)

    return lag1, lag2, lag3, lag1_w12, lag2_w12, partial_lag2


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {} — параметры не используются."""
    r = _kernel(values, position)
    arrays = list(r)
    suffixes = ["lag1", "lag2", "lag3", "lag1_w12", "lag2_w12", "partial_lag2"]
    return arrays, suffixes
