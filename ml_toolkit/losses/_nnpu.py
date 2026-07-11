"""NNPULoss: non-negative PU risk estimator (Kiryo et al., 2017) для CatBoost.

Стандартный (небезопасный, uPU) PU-риск:
  R_pu(g) = pi*R_p+(g) + [R_u-(g) - pi*R_p-(g)]
где pi = class_prior = P(y=1) (истинная доля позитивов, включая незамеченные
в U), R_p+ = E_p[l(g,+1)], R_p- = E_p[l(g,-1)], R_u- = E_u[l(g,-1)],
l(g,+1) = -log(p), l(g,-1) = -log(1-p) (логистический суррогат).

Проблема uPU: гибкие модели (в т.ч. бустинг) переобучаются так, что
эмпирическая оценка [R_u- - pi*R_p-] уходит в отрицательную область — чего
истинный риск не может (это оценка pi_n*R_n-, неотрицательной величины по
построению). nnPU (Kiryo et al.) заменяет это на max(0, ...) на уровне
градиента:

  R_u- - pi*R_p- >= -beta:  L = pi*R_p+ + (R_u- - pi*R_p-)         (обычный шаг)
  R_u- - pi*R_p- <  -beta:  L = pi*R_p+ - gamma*(R_u- - pi*R_p-)   (шаг НАЗАД:
                             градиент на этом слагаемом обращается, чтобы
                             оттолкнуть модель от переобучения, а не продолжать
                             тянуть уже отрицательную оценку риска ещё ниже)

Батч-уровневый лосс (как TverskyLoss): P/U здесь — это метки y=1/y=0 самого
train (0 = unlabeled, трактуется как "U", а не как достоверный негативный
класс). der2 в ветке "шаг назад" для точек U математически положителен (риск
специально обращается) — что ломает предположение CatBoost о вогнутости для
Newton-шага; здесь, как и в остальных лоссах этого пакета (см. TverskyLoss/
GHMLoss), der2 клампится к отрицательному, направление же (der1) остаётся
корректным и несёт всю логику nnPU-поправки.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class NNPULoss:
    """CatBoost-совместимый non-negative PU risk estimator (Kiryo et al., 2017).

    Parameters
    ----------
    class_prior:
        pi = P(y=1) — истинная (не наблюдаемая по одним лишь меткам) доля
        позитивов. Должен быть известен заранее (например, из полного
        пересчёта прошлых периодов).
    beta:
        Порог срабатывания non-negative коррекции (обычно 0).
    gamma:
        Множитель "обратного" градиента при срабатывании коррекции (обычно 1).

    """

    def __init__(self, class_prior: float, beta: float = 0.0, gamma: float = 1.0) -> None:
        if not 0.0 < class_prior < 1.0:
            raise ValueError(f'class_prior должен быть в (0, 1), получено {class_prior}')
        self.class_prior = class_prior
        self.beta = beta
        self.gamma = gamma

    def calc_ders_range(
        self,
        predictions: Sequence[float],
        targets: Sequence[float],
        weights: Sequence[float] | None,
    ) -> list[tuple[float, float]]:
        eps = 1e-7
        f = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        w = np.ones_like(f) if weights is None else np.asarray(weights, dtype=np.float64)
        pi = self.class_prior

        p = np.clip(1.0 / (1.0 + np.exp(-f)), eps, 1.0 - eps)
        pos = y == 1
        # R_p+/R_p-/R_u- становятся ВЗВЕШЕННЫМИ средними (сумма весов вместо
        # count в знаменателе) — та же генерализация, что у Tversky/GHM. w_p/w_u
        # ниже — сумма весов подмножества (P/U), не количество строк.
        w_p = max(float(w[pos].sum()), eps)
        w_u = max(float(w[~pos].sum()), eps)

        r_p_minus = float(np.sum(w[pos] * -np.log(1.0 - p[pos] + eps)) / w_p) if pos.any() else 0.0
        r_u_minus = float(np.sum(w[~pos] * -np.log(1.0 - p[~pos] + eps)) / w_u) if (~pos).any() else 0.0
        neg_risk = r_u_minus - pi * r_p_minus

        der1 = np.empty_like(p)
        der2 = np.empty_like(p)

        if neg_risk >= -self.beta:
            der1[pos] = pi * w[pos] / w_p
            der2[pos] = -eps
            der1[~pos] = -(w[~pos] * p[~pos] / w_u)
            der2[~pos] = -(w[~pos] * p[~pos] * (1.0 - p[~pos]) / w_u)
        else:
            gamma = self.gamma
            der1[pos] = -(pi * w[pos] / w_p) * ((1.0 + gamma) * p[pos] - 1.0)
            der2[pos] = -(pi * w[pos] / w_p) * (1.0 + gamma) * p[pos] * (1.0 - p[pos])
            der1[~pos] = gamma * w[~pos] * p[~pos] / w_u
            der2[~pos] = gamma * w[~pos] * p[~pos] * (1.0 - p[~pos]) / w_u

        der2 = np.minimum(der2, -eps)
        return list(zip(der1.tolist(), der2.tolist(), strict=False))
