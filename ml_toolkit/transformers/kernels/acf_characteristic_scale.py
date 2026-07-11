"""Характерный масштаб памяти ряда: первое пересечение 1/e и первый локальный минимум ACF.

Signal:
    autocorr.py даёт значения на фиксированных лагах (1,2,3,6,12); agg_autocorrelation — сводку
    по диапазону. Этот кернел ищет САМИ лаги, на которых память ряда характерно меняется — без
    предположений о периоде. f1ecac — на каком лаге автокорреляция впервые опускается ниже 1/e
    (~0.368) — стандартная «декорреляционная длина» из анализа временных рядов. first_min_ac —
    первый локальный минимум ACF — классический выбор задержки для embedding в анализе
    нелинейной динамики (после первого минимума добавление лага перестаёт давать новую
    информацию о структуре ряда).

Formula:
    r_0 = 1.0 (тривиально, корреляция ряда с самим собой), r_lag = windowed_lag_pearson(...)
    для lag = 1..max_lag, где ws >= lag + 2 (иначе лаг не измеряется)

    f1ecac_w       = первый lag, для которого r_lag <= 1/e ≈ 0.368
                     (если такого лага нет в пределах max_lag — max_lag: censored, память
                     не истощилась в пределах измеренного диапазона, а не «нет памяти»)
    first_min_ac_w = первый lag (1 <= lag < max_lag), где r_{lag-1} > r_lag < r_{lag+1}
                     (то же censored-соглашение при отсутствии минимума в диапазоне)

    Требует хотя бы lag=1 валиден (ws >= 3), иначе оба 0 — единственный случай, где 0
    означает именно «недостаточно истории», а не censored-верхнюю границу.

Outputs:
    {product}__acf_characteristic_scale__f1ecac_w24      — лаг первого пересечения 1/e
    {product}__acf_characteristic_scale__first_min_ac_w24 — лаг первого локального минимума ACF

Preset entry:
    acf_characteristic_scale:
      windows: [24]
      max_lag: 12

Interpretation:
    f1ecac = 1 — память угасает почти мгновенно (следующий месяц уже почти независим).
    f1ecac = max_lag (censored) — память устойчиво держится дольше всего измеренного
        диапазона лагов — увеличьте max_lag, если важно различать точнее.
    first_min_ac маленький — короткий «естественный» горизонт памяти; за его пределами
        линейная корреляция уже не добавляет новой информации о структуре ряда.
    first_min_ac = max_lag (censored) — ACF монотонно убывает во всём измеренном диапазоне,
        разворот (если есть) — за пределами max_lag.

Example:
    Ряд (8 мес): [10, 9, 7, 4, 3, 4, 7, 9],  w=8,  max_lag=4  (V-образное падение и разворот)

    r_1=0.643, r_2=-0.283, r_3=-0.921, r_4=-0.984 (r_0=1.0 по определению)
    f1ecac: первый лаг с r_lag <= 0.368 — это lag=2 (r_2=-0.283)
    first_min_ac: r монотонно убывает на всём диапазоне 1..4, разворота внутри не найдено
        → first_min_ac = max_lag = 4 (censored)
    → acf_characteristic_scale__f1ecac_w8 = 2.0,  first_min_ac_w8 = 4.0

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import resolve_window_size, windowed_lag_pearson

FEATURE = 'acf_characteristic_scale'

_INV_E = 1.0 / np.e


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    windows: np.ndarray,
    max_lag: int,
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_f1ecac = np.zeros((n_w, n_rows))
    out_first_min = np.zeros((n_w, n_rows))

    r_buf = np.empty(max_lag + 1)  # r_buf[0] = r_0 = 1.0, r_buf[k] = r_lag(k)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            if ws < 3:
                continue
            r_buf[0] = 1.0
            n_valid = 0
            for lag in range(1, max_lag + 1):
                if ws < lag + 2:
                    break
                r_buf[lag] = windowed_lag_pearson(product_values, row_idx, ws, lag)
                n_valid = lag
            if n_valid == 0:
                continue

            f1ecac = float(n_valid)  # censored по умолчанию
            for lag in range(1, n_valid + 1):
                if r_buf[lag] <= _INV_E:
                    f1ecac = float(lag)
                    break
            out_f1ecac[j, row_idx] = f1ecac

            first_min = float(n_valid)  # censored по умолчанию
            for lag in range(1, n_valid):
                if r_buf[lag - 1] > r_buf[lag] and r_buf[lag] < r_buf[lag + 1]:
                    first_min = float(lag)
                    break
            out_first_min[j, row_idx] = first_min

    return out_f1ecac, out_first_min


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [24], "max_lag": 12} — оба ключа обязательны."""
    windows = np.array(params['windows'], dtype=np.int64)
    max_lag = int(params['max_lag'])
    f1ecac, first_min = _kernel(values, position, windows, max_lag)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(f1ecac[j])
        suffixes.append(f'f1ecac_w{w}')
        arrays.append(first_min[j])
        suffixes.append(f'first_min_ac_w{w}')
    return arrays, suffixes
