"""Позиция строки в наблюдаемой истории сущности: флаг первых строк и нормированная позиция.

Signal:
    Показывает, сколько строк уже накоплено для сущности В НАБЛЮДАЕМОМ ДАТАСЕТЕ — и
    только это. Признак строится ИСКЛЮЧИТЕЛЬНО по position_within_entity и НЕ смотрит
    на product_values: строка со значением 0 учитывается наравне со строкой с любым
    ненулевым значением.

    Где это ломает интуицию: если датасет — прямоугольная календарная сетка (у каждой
    сущности одинаковый набор дат, а периоды до реального появления клиента зафилены
    нулями), то position_within_entity считает позицию В СЕТКЕ, а не «сколько клиент
    реально существует». Два клиента с одинаковым числом строк получат ПОБАЙТОВО
    ОДИНАКОВЫЕ значения этого признака на одинаковых датах, даже если один был активен
    с первой строки, а второй появился только в середине окна — «новый» здесь означает
    «одна из первых строк присутствия сущности в датафрейме», а не «недавно начал быть
    активным». Если нужен признак от первой ненулевой точки — это `tenure`
    (tenure_months / first_active_flag), не этот модуль.

Formula:
    new_client_flag         = 1 if position < 3 else 0
    months_since_start_norm = position / (position + 12)

    position — 0-индексированная позиция строки сущности в отсортированных по ts
    данных (`_windowing.compute_position_within_entity`); не зависит от values.

    Нормировка через (pos + 12) — насыщающаяся гипербола (не S-образный сигмоид):
    pos=0 → 0, pos=12 → 0.5 (точка полунасыщения = norm_months), pos=60 → 0.83,
    pos→∞ → 1 (асимптота, не достигается). Ранние строки дают больший прирост
    признака, чем поздние — диминишинг-ретёрнс, а не линейный счётчик.

Outputs:
    {product}__client_age__new_client_flag        — 1 на первых 3 строках сущности в датасете
    {product}__client_age__months_since_start_norm — нормированная позиция строки в [0, 1)

Preset (monthly.yaml):
    client_age: {}

Interpretation:
    new_client_flag = 1 — это одна из первых 3 строк сущности в наблюдаемых данных;
        НЕ обязательно означает «клиент недавно стал активным» — если ряд зафилен
        нулями до реальной активации, флаг всё равно встанет на этих нулевых строках.
    months_since_start_norm — насколько далеко эта строка от начала наблюдаемого окна
        сущности, а не от начала реальной активности клиента.
    Если у всех сущностей одинаковое число строк и для обучения берётся фиксированная
    относительная позиция (например, «последняя строка» каждой сущности) — признак
    вырождается в константу по всей выборке (ловится FeatureScreener на стадии
    low_variance / quasi_constant).
    Для «сколько времени клиент реально активен с первой ненулевой транзакции» —
    используйте `tenure` (tenure_months / first_active_flag), а не этот модуль.

Example:
    Клиент A (активен с первой строки): values = [35, 20, 12, ...], pos = 0..N-1
    Клиент B (те же даты, но появился в середине; до этого — нули):
        values = [0, 0, ..., 0, 34, 25, ...], pos = 0..N-1

    На одинаковой позиции pos=15 (та же календарная дата) оба клиента получат
    ОДИНАКОВЫЙ months_since_start_norm = 15/(15+12) = 0.556 — несмотря на то что A
    активен уже 16 месяцев, а B — только 4 (см. tenure_months для корректного различия).

    Однострочный пример:
    Ряд (5 мес): [10, 20, 30, 40, 50], t=4, pos=4
    new_client_flag = 1 if pos<3 else 0 → pos=4 ≥ 3 → 0
    months_since_start_norm = pos/(pos+12) = 4/16 = 0.25
    → client_age__new_client_flag = 0.0
    → client_age__months_since_start_norm = 0.25

"""

import numba as nb
import numpy as np

FEATURE = 'client_age'


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    new_client_months: int,
    norm_months: float,
):
    n_rows = product_values.shape[0]
    new_client_flag = np.zeros(n_rows)
    months_since_start_norm = np.zeros(n_rows)
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        new_client_flag[row_idx] = 1.0 if pos < new_client_months else 0.0
        months_since_start_norm[row_idx] = pos / (pos + norm_months)
    return new_client_flag, months_since_start_norm


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"new_client_months": 3, "norm_months": 12 (опционально)}"""
    new_client_months = int(params.get('new_client_months', 3))
    norm_months = float(params.get('norm_months', 12.0))
    ncf, msn = _kernel(values, position, new_client_months, norm_months)
    return [ncf, msn], ['new_client_flag', 'months_since_start_norm']
