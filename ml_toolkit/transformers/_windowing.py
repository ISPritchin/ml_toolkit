"""Shared numba primitives для последовательного прохода по (entity, ts_key).

Контракты слоя кернелов:
- product_values ожидаются неотрицательными (денежные объёмы). Кернелы не
  падают на отрицательных значениях, но лог-признаки (log1p(|v|)) и ratio-
  признаки теряют знак/интерпретацию.
- 0 в выходе — конвенция «недостаточно истории» (pos < требуемого лага/окна);
  он неотличим от легитимного нулевого сигнала. Учитывать при интерпретации.
- Деления нормируются через safe_ratio: при |знаменателе| <= EPS результат 0,
  иначе num / |den| с клампом в [-RATIO_CAP, RATIO_CAP] — чтобы нулевые
  месяцы не порождали выбросы ~1e10, ломающие float32-каст и корреляции.
"""

import numba as nb
import numpy as np

EPS = 1e-9
RATIO_CAP = 1e6


@nb.njit(cache=True)
def compute_position_within_entity(entity_codes: np.ndarray) -> np.ndarray:
    n = entity_codes.shape[0]
    out = np.empty(n, dtype=np.int64)
    if n == 0:
        return out
    cur = entity_codes[0]
    pos = 0
    out[0] = 0
    for i in range(1, n):
        if entity_codes[i] != cur:
            cur = entity_codes[i]
            pos = 0
        else:
            pos += 1
        out[i] = pos
    return out


@nb.njit(cache=True)
def resolve_window_size(position_in_entity: int, requested: int) -> int:
    available = position_in_entity + 1
    return min(requested, available)


@nb.njit(cache=True)
def safe_ratio(num: float, den: float) -> float:
    """Num / |den| с защитой: 0.0 при |den| <= EPS, кламп в [-RATIO_CAP, RATIO_CAP].

    Знак результата определяется числителем (как в прежней конвенции
    x / (|y| + eps)). Нулевой знаменатель трактуется как «отношение не
    определено» -> 0, а не взрыв num/eps.
    """
    a = abs(den)
    if a <= EPS:
        return 0.0
    r = num / a
    if r > RATIO_CAP:
        return RATIO_CAP
    if r < -RATIO_CAP:
        return -RATIO_CAP
    return r


@nb.njit(cache=True)
def fit_linear_trend_slope(product_values: np.ndarray, row_idx: int, window_size: int) -> float:
    """OLS-наклон по равноотстоящим точкам окна [row_idx-window_size+1, row_idx].

    sum(i) и sum(i^2) для i in [0..n-1] считаются замкнутыми формулами —
    n(n-1)/2 и (n-1)n(2n-1)/6 (точны в float64 для месячных окон).
    """
    if window_size < 2:
        return 0.0
    n = float(window_size)
    sx = n * (n - 1.0) / 2.0
    sxx = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0
    sy = 0.0
    sxy = 0.0
    for offset in range(window_size):
        y = product_values[row_idx - window_size + 1 + offset]
        sy += y
        sxy += offset * y
    denom = n * sxx - sx * sx
    if denom == 0.0:
        return 0.0
    return (n * sxy - sx * sy) / denom


@nb.njit(cache=True)
def compute_window_mean_and_std(product_values: np.ndarray, row_idx: int, window_size: int):
    s = 0.0
    for offset in range(window_size):
        s += product_values[row_idx - window_size + 1 + offset]
    mean = s / window_size
    sq = 0.0
    for offset in range(window_size):
        d = product_values[row_idx - window_size + 1 + offset] - mean
        sq += d * d
    return mean, (sq / window_size) ** 0.5


@nb.njit(cache=True)
def compute_window_sum(product_values: np.ndarray, row_idx: int, window_size: int) -> float:
    s = 0.0
    for offset in range(window_size):
        s += product_values[row_idx - window_size + 1 + offset]
    return s


@nb.njit(cache=True)
def compute_window_mean(product_values: np.ndarray, row_idx: int, window_size: int) -> float:
    """Среднее окна без вычисления std (один проход вместо двух)."""
    return compute_window_sum(product_values, row_idx, window_size) / window_size


@nb.njit(cache=True)
def compute_window_min_and_max(product_values: np.ndarray, row_idx: int, window_size: int):
    lo = product_values[row_idx - window_size + 1]
    hi = lo
    for offset in range(1, window_size):
        v = product_values[row_idx - window_size + 1 + offset]
        lo = min(lo, v)
        hi = max(hi, v)
    return lo, hi


@nb.njit(cache=True)
def compute_window_sorted_buffer(product_values: np.ndarray, row_idx: int, window_size: int) -> np.ndarray:
    buf = np.empty(window_size)
    for offset in range(window_size):
        buf[offset] = product_values[row_idx - window_size + 1 + offset]
    buf.sort()
    return buf


@nb.njit(cache=True)
def fill_window_sorted(buf: np.ndarray, product_values: np.ndarray, row_idx: int, window_size: int) -> None:
    """Как compute_window_sorted_buffer, но в предвыделенный buf (без аллокации).

    Использует buf[:window_size]; buf должен иметь длину >= window_size.
    """
    for offset in range(window_size):
        buf[offset] = product_values[row_idx - window_size + 1 + offset]
    sub = buf[:window_size]
    sub.sort()


@nb.njit(cache=True)
def sorted_median(sorted_buf: np.ndarray, window_size: int) -> float:
    """Честная медиана отсортированного окна (среднее двух центральных при чётном ws)."""
    mid = window_size // 2
    if window_size % 2 == 1:
        return sorted_buf[mid]
    return 0.5 * (sorted_buf[mid - 1] + sorted_buf[mid])


@nb.njit(cache=True)
def sorted_quantile(sorted_buf: np.ndarray, window_size: int, q: float) -> float:
    """Квантиль по отсортированному окну: элемент с индексом int(q * (ws - 1)).

    Единая конвенция для всех кернелов (nearest-rank, 'lower'): индексы
    комплементарных квантилей симметричны (p25 <-> p75, p10 <-> p90).
    """
    idx = int(q * (window_size - 1))
    idx = max(idx, 0)
    idx = min(idx, window_size - 1)
    return sorted_buf[idx]


@nb.njit(cache=True)
def pearson_from_sums(n: float, sx: float, sy: float, sxy: float, sx2: float, sy2: float) -> float:
    """Pearson r из накопленных сумм; 0 при n < 2 или вырожденной дисперсии."""
    if n < 2.0:
        return 0.0
    num = n * sxy - sx * sy
    denom = ((n * sx2 - sx * sx) * (n * sy2 - sy * sy)) ** 0.5
    return num / denom if denom > EPS else 0.0


@nb.njit(cache=True)
def windowed_lag_pearson(product_values: np.ndarray, row_idx: int, ws: int, lag: int) -> float:
    """Pearson по парам (v[i], v[i+lag]) внутри окна [row_idx-ws+1, row_idx]."""
    n_pairs = ws - lag
    if n_pairs < 2:
        return 0.0
    sx = sy = sxy = sx2 = sy2 = 0.0
    start = row_idx - ws + 1
    for i in range(n_pairs):
        x = product_values[start + i]
        y = product_values[start + i + lag]
        sx += x
        sy += y
        sxy += x * y
        sx2 += x * x
        sy2 += y * y
    return pearson_from_sums(float(n_pairs), sx, sy, sxy, sx2, sy2)
