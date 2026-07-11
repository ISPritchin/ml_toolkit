"""Периодичность и сезонность: expanding Pearson лаги 6/12 + windowed + квартальные прокси.

Signal:
    Обнаруживает повторяющиеся паттерны с полугодовым (lag6) или годовым (lag12)
    периодом. Высокая корреляция указывает на сезонного клиента. quarter_cv_proxy
    меряет неравномерность по кварталам (традиционная метрика сезонности).

Formula:
    Expanding Pearson lag k (аналог autocorr):
        при каждом новом шаге аккумулируем пары (v[t-k], v[t])
        ac_lag6  = Pearson над парами (v[t-6], v[t]) с момента pos >= 6
        ac_lag12 = Pearson над парами (v[t-12], v[t]) с момента pos >= 12

    Windowed Pearson lag k в окне 24:
        ac_lag6_w24  = Pearson(v[i], v[i+6]) для i in [t-24+1..t-6]
        ac_lag12_w24 = Pearson(v[i], v[i+12]) для i in [t-24+1..t-12]

    Квартальные прокси (только при ws12 >= 12):
        Q1..Q4 = средние 4 троек из окна 12
        quarter_cv_proxy_w12 = std(Q1..Q4) / |mean(Q1..Q4)|
        seasonal_amplitude_w12 = (max(Q) - min(Q)) / |mean_w12|
        even_odd_w12 = mean(мес. с чётной позицией) / |mean(мес. с нечётной)|
        Чётность берётся от позиции месяца ВНУТРИ СУЩНОСТИ (pos % 2), а не от
        начала окна — иначе при сдвиге окна на месяц чётности инвертируются и
        признак осциллирует на стабильном бимесячном паттерне.

Outputs:
    {product}__seasonal_autocorr__lag6              — expanding Pearson, лаг 6
    {product}__seasonal_autocorr__lag6_w24          — windowed Pearson lag 6 в окне 24
    {product}__seasonal_autocorr__lag12             — expanding Pearson, лаг 12
    {product}__seasonal_autocorr__lag12_w24         — windowed Pearson lag 12 в окне 24
    {product}__seasonal_autocorr__quarter_cv_w12    — CV квартальных средних
    {product}__seasonal_autocorr__even_odd_w12      — бимесячный паттерн
    {product}__seasonal_autocorr__amplitude_w12     — амплитуда квартальной сезонности

Preset (monthly.yaml):
    seasonal_autocorr: {}

Interpretation:
    lag12 близко к +1 — сильная годовая сезонность (те же месяцы ежегодно активны).
    lag6 близко к +1 — полугодовая периодичность (типично для госзаказа/бюджетных клиентов).
    quarter_cv > 0.5 — ярко выраженная квартальная неравномерность.
    even_odd = 3.67 — чётные месяцы в 3.67× больше нечётных (бимесячный ритм).

Example:
    Ряд (6 мес): [10, 30, 10, 30, 10, 30]  (чёткий бимесячный ритм)

    чётные позиции (0,2,4): 10,10,10 → mean = 10
    нечётные позиции (1,3,5): 30,30,30 → mean = 30
    even_odd = 10 / 30 = 0.333
    → seasonal_autocorr__even_odd_w12 = 0.333  (чётные мес. втрое слабее нечётных)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import EPS, pearson_from_sums, safe_ratio, windowed_lag_pearson

FEATURE = 'seasonal_autocorr'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray):
    n_rows = product_values.shape[0]
    ac_lag6 = np.zeros(n_rows)
    ac_lag6_w24 = np.zeros(n_rows)
    ac_lag12 = np.zeros(n_rows)
    ac_lag12_w24 = np.zeros(n_rows)
    quarter_cv = np.zeros(n_rows)
    even_odd = np.zeros(n_rows)
    seasonal_amp = np.zeros(n_rows)

    n6 = sx6 = sy6 = sxy6 = sx26 = sy26 = 0.0
    n12 = sx12 = sy12 = sxy12 = sx212 = sy212 = 0.0
    q_means = np.empty(4)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        if pos == 0:
            n6 = sx6 = sy6 = sxy6 = sx26 = sy26 = 0.0
            n12 = sx12 = sy12 = sxy12 = sx212 = sy212 = 0.0

        v = product_values[row_idx]
        if pos >= 6:
            x6 = product_values[row_idx - 6]
            n6 += 1.0
            sx6 += x6
            sy6 += v
            sxy6 += x6*v
            sx26 += x6*x6
            sy26 += v*v
        if pos >= 12:
            x12 = product_values[row_idx - 12]
            n12 += 1.0
            sx12 += x12
            sy12 += v
            sxy12 += x12*v
            sx212 += x12*x12
            sy212 += v*v

        ac_lag6[row_idx] = pearson_from_sums(n6, sx6, sy6, sxy6, sx26, sy26)
        ac_lag12[row_idx] = pearson_from_sums(n12, sx12, sy12, sxy12, sx212, sy212)

        ws24 = min(pos + 1, 24)
        ac_lag6_w24[row_idx] = windowed_lag_pearson(product_values, row_idx, ws24, 6)
        ac_lag12_w24[row_idx] = windowed_lag_pearson(product_values, row_idx, ws24, 12)

        # квартальная неравномерность: 4 тройки из окна 12
        ws12 = min(pos + 1, 12)
        if ws12 >= 12:
            for q in range(4):
                qsum = 0.0
                for i in range(3):
                    qsum += product_values[row_idx - 12 + 1 + q*3 + i]
                q_means[q] = qsum / 3.0
            qmean = (q_means[0] + q_means[1] + q_means[2] + q_means[3]) / 4.0
            if abs(qmean) > EPS:
                qstd_sq = 0.0
                for q in range(4):
                    qstd_sq += (q_means[q] - qmean) ** 2
                quarter_cv[row_idx] = (qstd_sq / 4.0) ** 0.5 / abs(qmean)
                qmax = q_means[0]
                qmin = q_means[0]
                for q in range(1, 4):
                    qmax = max(qmax, q_means[q])
                    qmin = min(qmin, q_means[q])
                seasonal_amp[row_idx] = (qmax - qmin) / abs(qmean)

        if ws12 >= 2:
            even_sum = 0.0
            odd_sum = 0.0
            even_cnt = 0
            odd_cnt = 0
            for offset in range(ws12):
                val = product_values[row_idx - ws12 + 1 + offset]
                # чётность позиции месяца внутри сущности — стабильна при сдвиге окна
                month_pos = pos - (ws12 - 1) + offset
                if month_pos % 2 == 0:
                    even_sum += val
                    even_cnt += 1
                else:
                    odd_sum += val
                    odd_cnt += 1
            even_mean = even_sum / even_cnt if even_cnt > 0 else 0.0
            odd_mean = odd_sum / odd_cnt if odd_cnt > 0 else 0.0
            even_odd[row_idx] = safe_ratio(even_mean, odd_mean)

    return ac_lag6, ac_lag6_w24, ac_lag12, ac_lag12_w24, quarter_cv, even_odd, seasonal_amp


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {} — параметры не используются."""
    r = _kernel(values, position)
    suffixes = ['lag6', 'lag6_w24', 'lag12', 'lag12_w24', 'quarter_cv_w12', 'even_odd_w12', 'amplitude_w12']
    return list(r), suffixes
