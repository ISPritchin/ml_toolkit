"""AsymmetricLoss (ASL, Ridnik et al. 2021) для CatBoost.

Расширение Focal Loss с разными гаммами для pos/neg и margin для негативов:
  gamma_pos (γ+): фокус на трудных позитивах (обычно 0–1).
  gamma_neg (γ-): подавление уверенных негативов (обычно 2–6).
  prob_margin (m): негативы с p < m полностью исключаются из градиента.
"""

from __future__ import annotations

import numpy as np


class AsymmetricLoss:
    """CatBoost-совместимый ASL для бинарной классификации.

    Parameters
    ----------
    gamma_pos:
        Фокусирующий параметр для позитивов. 0 = стандартный CE для pos.
    gamma_neg:
        Фокусирующий параметр для негативов. Обычно 2–6.
    prob_margin:
        Порог обрезки: негативы с p < prob_margin исключаются из градиента.
        Полезно при зашумлённых метках.

    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        prob_margin: float = 0.05,
    ) -> None:
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.prob_margin = prob_margin

    def calc_ders_range(
        self, predictions, targets, weights
    ) -> list[tuple[float, float]]:
        eps = 1e-7
        f = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)

        p = np.clip(1.0 / (1.0 + np.exp(-f)), eps, 1.0 - eps)

        der1 = np.empty_like(p)
        der2 = np.empty_like(p)

        # ── Позитивы (y == 1) ─────────────────────────────────────────────────
        # L+ = -(1-p)^γ+ * log(p + ε)
        # der1 = (1-p)^(γ++1) - γ+ * p * (1-p)^γ+ * log(p+ε)
        pos = y == 1
        if pos.any():
            p_p = p[pos]
            q_p = 1.0 - p_p
            gp = self.gamma_pos
            focal_w = q_p ** gp
            der1[pos] = q_p ** (gp + 1) - gp * p_p * focal_w * np.log(p_p + eps)
            der2[pos] = -(focal_w * p_p * q_p)

        # ── Негативы (y == 0) ─────────────────────────────────────────────────
        # p_s = max(p - m, 0); L- = -(p_s)^γ- * log(1-p_s+ε) if p_s > 0
        neg = ~pos
        if neg.any():
            p_n = p[neg]
            p_s = np.maximum(p_n - self.prob_margin, 0.0)
            active = p_s > 0.0
            d1 = np.zeros_like(p_n)
            d2 = np.zeros_like(p_n)
            if active.any():
                ps_a = p_s[active]
                pn_a = p_n[active]
                qn_a = 1.0 - pn_a
                gn = self.gamma_neg
                log1m = np.log(1.0 - ps_a + eps)
                focal_w_neg = ps_a ** max(gn - 1, 0)
                # dLdf здесь уже равно der1 = -dL/df (см. регрессию в тестах) —
                # предыдущая версия ошибочно негировала его повторно, разворачивая
                # знак градиента для негативов с p > prob_margin.
                der1_neg = (
                    -gn * focal_w_neg * log1m
                    + ps_a ** gn / (1.0 - ps_a + eps)
                ) * pn_a * qn_a
                d1[active] = der1_neg
                d2[active] = -(ps_a ** gn * pn_a * qn_a)
            der1[neg] = d1
            der2[neg] = d2

        der2 = np.minimum(der2, -eps)

        # sample weight из Pool(weight=...) — CatBoost не применяет его сам к
        # результату кастомного лосса, это обязанность самого лосса.
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            der1 = der1 * w
            der2 = der2 * w

        return list(zip(der1.tolist(), der2.tolist()))

    def is_max_optimal(self) -> bool:
        return False

    def get_final_error(self, error: float, weight: float) -> float:
        return error / max(weight, 1.0)
