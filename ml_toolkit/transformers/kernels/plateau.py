"""Стагнация: доля плоских месяцев, текущая и наибольшая плоская серия.

Signal:
    Выявляет ряды в «плато»: где изменения слишком малы, чтобы считаться ростом
    или падением. «Плоский» шаг: |v[t]-v[t-1]| < 5% от локального среднего.
    Отличается от low-CV: плато возможно при монотонном тренде (там CV → 0, но нет плато).

Formula:
    flat_threshold = 0.05
    is_flat(i) = |v[i] - v[i-1]| < flat_threshold * ((|v[i]| + |v[i-1]|) / 2 + eps)
    flat_share_w   = count(is_flat) / (ws - 1)
    longest_flat_w = max длина непрерывной плоской серии
    near_mean_w    = count(|v[i] - mean_w| < 0.1 * |mean_w|) / ws
    current_flat_streak — running: текущая непрерывная плоская серия (сбрасывается при выходе)
    plateau_exit_recency — running: месяцев с последнего выхода из плато
        (0 = в плато или вышел в текущем месяце, -1 = плато ещё не завершалось)

Outputs:
    {product}__plateau__flat_share_w6           — доля плоских шагов за 6 мес
    {product}__plateau__longest_flat_w6         — наибольшая плоская серия за 6 мес
    {product}__plateau__near_mean_w6            — доля мес. вблизи среднего за 6 мес
    {product}__plateau__flat_share_w12          — доля плоских шагов за 12 мес
    {product}__plateau__longest_flat_w12        — наибольшая плоская серия за 12 мес
    {product}__plateau__near_mean_w12           — доля мес. вблизи среднего за 12 мес
    {product}__plateau__current_flat_streak     — текущая серия плато (running)
    {product}__plateau__plateau_exit_recency    — месяцев с выхода из плато (running)

Preset entry:
    plateau:
      windows: [6, 12]

Interpretation:
    flat_share_w12 = 1.0 — полное плато весь год (ряд F: все шаги < 5% среднего).
    flat_share_w12 = 0 — ни одного плоского шага: динамичный ряд.
    plateau_exit_recency = 2 — 2 месяца назад вышел из плато (возможен импульс роста).
    near_mean_w12 = 1 + flat_share_w12 = 1 — абсолютно стабильный равномерный ряд.

Example:
    Ряд (6 мес): [100, 101, 100, 101, 100, 101],  w=6
    (порог «плоско»: |diff| < 5% от среднего соседей)

    каждый шаг |diff| = 1 < 0.05·~100 = 5 → все 5 шагов плоские
    flat_share = 5/5 = 1.0,  longest_flat = 5
    → plateau__flat_share_w6 = 1.0,  longest_flat_w6 = 5  (полное плато)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import EPS, compute_window_mean, resolve_window_size

FEATURE = 'plateau'

_FLAT_THRESHOLD = 0.05  # |diff| < 5% от mean считается плоским


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    windows: np.ndarray,
    flat_threshold: float,
    near_mean_threshold: float,
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_flat_share = np.zeros((n_w, n_rows))
    out_longest_flat = np.zeros((n_w, n_rows))
    out_near_mean = np.zeros((n_w, n_rows))
    out_exit_recency = np.full(n_rows, -1.0)  # running state: -1 если плато не было

    cur_flat_streak = np.zeros(n_rows)  # running per-row (built via position)

    # running state scalars (reset at position == 0)
    r_cur_flat = 0
    r_last_exit_ago = -1  # -1 = плато не завершалось

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        if pos == 0:
            r_cur_flat = 0
            r_last_exit_ago = -1

        # давность прошлого выхода стареет на месяц ДО обработки текущего шага,
        # чтобы в сам месяц выхода из плато значение было 0, а не 1
        if r_last_exit_ago >= 0:
            r_last_exit_ago += 1

        v = product_values[row_idx]
        if pos >= 1:
            prev = product_values[row_idx - 1]
            mean_approx = (abs(v) + abs(prev)) / 2.0
            is_flat = abs(v - prev) < flat_threshold * (mean_approx + EPS)
            if is_flat:
                r_cur_flat += 1
            else:
                if r_cur_flat > 0:
                    r_last_exit_ago = 0
                r_cur_flat = 0
        else:
            is_flat = False

        # если сейчас плато — выход ещё не случился
        exit_recency = 0.0 if r_cur_flat > 0 else float(r_last_exit_ago)
        out_exit_recency[row_idx] = exit_recency
        cur_flat_streak[row_idx] = r_cur_flat

        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            mean = compute_window_mean(product_values, row_idx, ws)
            flat_count = 0
            near_count = 0
            longest = 0
            run = 0
            for offset in range(1, ws):
                abs_idx = row_idx - ws + 1 + offset
                vv = product_values[abs_idx]
                pp = product_values[abs_idx - 1]
                mm = (abs(vv) + abs(pp)) / 2.0
                f = abs(vv - pp) < flat_threshold * (mm + EPS)
                if f:
                    flat_count += 1
                    run += 1
                    longest = max(longest, run)
                else:
                    run = 0
            # near_mean не нуждается в предыдущей точке (в отличие от flat/longest выше) —
            # отдельный проход по ПОЛНОМУ окну range(0, ws), иначе offset=0 никогда не
            # засчитывается и near_mean_w физически не может достичь 1.0 (см. supported_transformers.md).
            for offset in range(ws):
                abs_idx = row_idx - ws + 1 + offset
                vv = product_values[abs_idx]
                if abs(vv - mean) < near_mean_threshold * (abs(mean) + EPS):
                    near_count += 1
            out_flat_share[j, row_idx] = flat_count / max(ws - 1, 1)
            out_longest_flat[j, row_idx] = longest
            out_near_mean[j, row_idx] = near_count / ws

    return out_flat_share, out_longest_flat, out_near_mean, cur_flat_streak, out_exit_recency


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6, 12], "flat_threshold": 0.05, "near_mean_threshold": 0.10 (опционально)}."""
    windows = np.array(params['windows'], dtype=np.int64)
    flat_threshold = float(params.get('flat_threshold', _FLAT_THRESHOLD))
    near_mean_threshold = float(params.get('near_mean_threshold', 0.10))
    flat_share, longest_flat, near_mean, cur_streak, exit_rec = _kernel(
        values, position, windows, flat_threshold, near_mean_threshold
    )
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(flat_share[j])
        suffixes.append(f'flat_share_w{w}')
        arrays.append(longest_flat[j])
        suffixes.append(f'longest_flat_w{w}')
        arrays.append(near_mean[j])
        suffixes.append(f'near_mean_w{w}')
    arrays.append(cur_streak)
    suffixes.append('current_flat_streak')
    arrays.append(exit_rec)
    suffixes.append('plateau_exit_recency')
    return arrays, suffixes
