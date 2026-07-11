"""Регулярность активности: промежутки между активными сериями, CV ритма.

Signal:
    Измеряет предсказуемость ритма ряда: как равномерны интервалы между «включениями»
    и насколько стабильны сами всплески по длине. Низкий CV промежутков при нескольких
    всплесках — ряд с квартальным/полугодовым ритмом (ценно для прогноза).

Formula:
    gap[k] — длина k-й нулевой серии между активными сериями
    gap_mean_w   = mean(gap[k]) для k in окне
    gap_std_w    = std(gap[k])
    gap_cv_w     = gap_std / (gap_mean + eps)
    is_monthly_w = 1 if active_count >= ws - 1  (почти каждый месяц активен)
    cadence_shift_w = 1 if gap_mean(последние ws/2) > 1.5 * gap_mean_w
    active_len_cv_w = CV длин активных серий

Outputs:
    {product}__flow_regularity__gap_mean_w12       — средний интервал между вспышками
    {product}__flow_regularity__gap_std_w12        — σ промежутков
    {product}__flow_regularity__gap_cv_w12         — CV промежутков (ритмичность)
    {product}__flow_regularity__is_monthly_w12     — флаг практически ежемесячной активности
    {product}__flow_regularity__cadence_shift_w12  — флаг замедления ритма
    {product}__flow_regularity__active_len_cv_w12  — CV длин активных серий

Preset entry:
    flow_regularity:
      windows: [12]

Interpretation:
    gap_cv = 0 — идеально ритмичный ряд (активность через строго равные интервалы).
    gap_cv > 1 — хаотичный, непредсказуемый ритм.
    cadence_shift = 1 — в последние месяцы ряд «замолчал» дольше обычного, чем раньше.
    is_monthly = 1 при gap_mean = 0 — непрерывный поток без перебоев.

Example:
    Ряд (6 мес): [5, 0, 0, 8, 0, 0],  w=6
    активные серии: [5] (idx0) и [8] (idx3)

    промежуток между ними: нули idx1,idx2 → gap = 2 (одна серия)
    gap_mean = 2,  gap_std = 0 (единственный промежуток)
    → flow_regularity__gap_mean_w6 = 2.0
    → flow_regularity__gap_cv_w6   = 0.0  (ритм идеально равномерен)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import EPS, resolve_window_size

FEATURE = 'flow_regularity'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_gap_mean = np.zeros((n_w, n_rows))
    out_gap_std = np.zeros((n_w, n_rows))
    out_gap_cv = np.zeros((n_w, n_rows))
    out_is_monthly = np.zeros((n_w, n_rows))
    out_cadence_shift = np.zeros((n_w, n_rows))
    out_active_len_cv = np.zeros((n_w, n_rows))

    max_w = 1
    for j in range(n_w):
        max_w = max(max_w, windows[j])
    gaps = np.zeros(max_w)
    burst_lens = np.zeros(max_w)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])

            # collect burst gaps and burst lengths
            n_gaps = 0
            n_bursts = 0
            in_burst = False
            cur_gap = 0
            cur_burst = 0
            active_count = 0

            for offset in range(ws):
                abs_idx = row_idx - ws + 1 + offset
                active = product_values[abs_idx] != 0.0
                if active:
                    active_count += 1
                    if not in_burst:
                        if n_gaps < ws and cur_gap > 0:
                            gaps[n_gaps] = cur_gap
                            n_gaps += 1
                        in_burst = True
                        cur_gap = 0
                    cur_burst += 1
                else:
                    if in_burst:
                        if n_bursts < ws:
                            burst_lens[n_bursts] = cur_burst
                            n_bursts += 1
                        cur_burst = 0
                        in_burst = False
                    cur_gap += 1
            if in_burst and n_bursts < ws:
                burst_lens[n_bursts] = cur_burst
                n_bursts += 1

            # gap stats
            if n_gaps >= 1:
                m_gap = 0.0
                for i in range(n_gaps):
                    m_gap += gaps[i]
                m_gap /= n_gaps
                v_gap = 0.0
                for i in range(n_gaps):
                    v_gap += (gaps[i] - m_gap) ** 2
                std_gap = (v_gap / n_gaps) ** 0.5
                out_gap_mean[j, row_idx] = m_gap
                out_gap_std[j, row_idx] = std_gap
                out_gap_cv[j, row_idx] = std_gap / (m_gap + EPS)

            out_is_monthly[j, row_idx] = 1.0 if active_count >= ws - 1 else 0.0

            # cadence_shift: средний промежуток в последней половине окна vs по всему окну
            if ws >= 6:
                ws_half_r = resolve_window_size(pos, ws // 2)
                n_gaps_r = 0
                m_gap_r = 0.0
                in_b = False
                c_gap = 0
                for offset in range(ws_half_r):
                    abs_idx = row_idx - ws_half_r + 1 + offset
                    act = product_values[abs_idx] != 0.0
                    if act:
                        if not in_b:
                            if c_gap > 0:
                                m_gap_r += c_gap
                                n_gaps_r += 1
                            c_gap = 0
                            in_b = True
                    else:
                        in_b = False
                        c_gap += 1
                if n_gaps_r > 0:
                    m_gap_r /= n_gaps_r
                    out_cadence_shift[j, row_idx] = 1.0 if m_gap_r > out_gap_mean[j, row_idx] * 1.5 else 0.0

            # burst length CV
            if n_bursts >= 2:
                m_b = 0.0
                for i in range(n_bursts):
                    m_b += burst_lens[i]
                m_b /= n_bursts
                v_b = 0.0
                for i in range(n_bursts):
                    v_b += (burst_lens[i] - m_b) ** 2
                out_active_len_cv[j, row_idx] = (v_b / n_bursts) ** 0.5 / (m_b + EPS)

    return out_gap_mean, out_gap_std, out_gap_cv, out_is_monthly, out_cadence_shift, out_active_len_cv


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    gm, gs, gcv, im, cs, alcv = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(gm[j])
        suffixes.append(f'gap_mean_w{w}')
        arrays.append(gs[j])
        suffixes.append(f'gap_std_w{w}')
        arrays.append(gcv[j])
        suffixes.append(f'gap_cv_w{w}')
        arrays.append(im[j])
        suffixes.append(f'is_monthly_w{w}')
        arrays.append(cs[j])
        suffixes.append(f'cadence_shift_w{w}')
        arrays.append(alcv[j])
        suffixes.append(f'active_len_cv_w{w}')
    return arrays, suffixes
