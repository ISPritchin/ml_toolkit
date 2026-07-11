"""DFA (detrended fluctuation analysis): экспонента фрактального масштабирования, устойчивая к тренду.

Signal:
    Обычная автокорреляция/rolling_std путаются на нестационарных рядах — линейный тренд внутри
    окна сам по себе создаёт видимость «памяти». DFA сначала убирает локальный линейный тренд
    внутри каждого блока окна, а потом смотрит, как растёт остаточная флуктуация с увеличением
    размера блока — это устойчиво к тренду по конструкции. Экспонента alpha показывает, насколько
    ряд самоподобен (fractal): ~0.5 — белый шум без памяти, ~1.0 — устойчивая долгосрочная память
    (persistent), >1 — нестационарный/трендовый, <0.5 — antipersistent (частые развороты).

Formula:
    profile[k] = Σ(v[i] - mean_w, i in [0..k])           (интегрированный, центрированный ряд)
    Для размера блока n (small = ws//4, large = ws//2), неперекрывающихся блоков (n_boxes = ws//n):
        в каждом блоке — OLS-детрендинг profile по локальному индексу [0..n-1],
        F(n)² = mean(residual², по всем n_boxes·n точкам)
    alpha_w = (ln(F(n_large)) - ln(F(n_small))) / (ln(n_large) - ln(n_small))

    Требует ws >= 8 (n_small = ws//4 >= 2 и минимум 2 блока на каждом масштабе), иначе 0.
    При вырожденной F(n) (постоянный профиль на одном из масштабов) — 0 (не определено).
    Только 2 масштаба (small/large) — компромисс из-за коротких окон этой библиотеки (6-24 точки);
    классический DFA использует 10+ масштабов на рядах из сотен-тысяч точек.

Outputs:
    {product}__dfa__alpha_w12  — экспонента фрактального масштабирования, окно 12
    {product}__dfa__alpha_w24  — то же, окно 24

Preset entry:
    dfa:
      windows: [12, 24]

Interpretation:
    alpha ≈ 0.5 — ряд ведёт себя как белый шум относительно тренда: приращения независимы.
    alpha ≈ 1.0 — сильная долгосрочная память (persistent): рост порождает продолжение роста.
    alpha > 1.2 — не просто память, а нестационарность/явный тренд, не убранный локальным
        детрендингом полностью (двух масштабов не хватает, чтобы это разделить надёжно).
    alpha < 0.4 — antipersistent: ряд чаще разворачивается, чем продолжает движение
        (mean-reverting сильнее, чем случайное блуждание).

Example:
    Ряд (12 мес): [10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 17],  w=12
    (линейный тренд +0.5/мес с колебанием ±1 вокруг него; n_small=3, n_large=6)

    После построения profile и OLS-детрендинга по 4 блокам (small) и 2 блокам (large):
    → dfa__alpha_w12 ≈ 0.949  (близко к persistent — устойчивый тренд с малым шумом)

    Для сравнения на этом же ряде (не входит в Outputs, только для интуиции):
    чисто линейный тренд без шума даёт alpha ≈ 2.40 (нестационарность доминирует),
    чередующийся ряд без тренда — alpha ≈ 0.02 (антиперсистентный, у детрендинга
    почти не остаётся зависимости от масштаба блока), константа — 0 (F(n)=0 всюду).

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import EPS, compute_window_mean, resolve_window_size

FEATURE = 'dfa'


@nb.njit(cache=True)
def _box_residual_sumsq(profile: np.ndarray, start: int, n: int) -> float:
    """Сумма квадратов остатков OLS-детрендинга profile[start:start+n] по локальному индексу."""
    nf = float(n)
    sx = nf * (nf - 1.0) / 2.0
    sxx = (nf - 1.0) * nf * (2.0 * nf - 1.0) / 6.0
    sy = 0.0
    sxy = 0.0
    for i in range(n):
        y = profile[start + i]
        sy += y
        sxy += i * y
    denom = nf * sxx - sx * sx
    slope = (nf * sxy - sx * sy) / denom if denom != 0.0 else 0.0
    intercept = (sy - slope * sx) / nf
    ss = 0.0
    for i in range(n):
        pred = intercept + slope * i
        resid = profile[start + i] - pred
        ss += resid * resid
    return ss


@nb.njit(cache=True)
def _fluctuation(profile: np.ndarray, ws: int, box_size: int) -> float:
    n_boxes = ws // box_size
    if n_boxes < 1:
        return 0.0
    total_ss = 0.0
    for b in range(n_boxes):
        total_ss += _box_residual_sumsq(profile, b * box_size, box_size)
    return (total_ss / (n_boxes * box_size)) ** 0.5


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))

    max_w = 1
    for j in range(n_w):
        max_w = max(max_w, windows[j])
    profile = np.empty(max_w)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            n_small = ws // 4
            n_large = ws // 2
            if n_small < 2 or n_large <= n_small or ws // n_small < 2 or ws // n_large < 2:
                continue
            mean_w = compute_window_mean(product_values, row_idx, ws)
            start = row_idx - ws + 1
            running = 0.0
            for k in range(ws):
                running += product_values[start + k] - mean_w
                profile[k] = running
            f_small = _fluctuation(profile, ws, n_small)
            f_large = _fluctuation(profile, ws, n_large)
            if f_small <= EPS or f_large <= EPS:
                continue
            out[j, row_idx] = (
                (np.log(f_large) - np.log(f_small)) / (np.log(float(n_large)) - np.log(float(n_small)))
            )
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12, 24]} — ключ обязателен."""
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f'alpha_w{w}' for w in params['windows']]
