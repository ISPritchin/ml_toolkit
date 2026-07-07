"""LDAMLoss: Label-Distribution-Aware Margin + Deferred Re-Weighting (Cao et al., 2019).

Два механизма, оба нацелены на миноритарный класс:

1. Margin: минority-класс получает больший обязательный отступ от границы
   решения. Δ_j = C / n_j^{1/4}, нормировано так, что max_j(Δ_j) = max_margin.
   Перед сигмоидой логит сдвигается на Δ_y в сторону "усложнения" правильного
   ответа:
     y=1: f_adj = f - Δ_pos   (нужно больше уверенности, чтобы быть "верно")
     y=0: f_adj = f + Δ_neg
   f_adj = f + const(y), поэтому градиент/гессиан обычного sigmoid CE в точке
   f_adj — точные (без аппроксимации), в отличие от Focal/Tversky.

2. Deferred Re-Weighting (DRW): первые reweight_epoch_frac * n_total_iterations
   итераций margin-лосс обучается с равными весами классов; затем включаются
   веса по effective number of samples (Cui et al., 2019):
     w_j = (1-beta)/(1-beta^n_j), нормированные к среднему 1.0.
   Ранняя фаза без реweight даёт модели сначала выучить общую форму границы,
   поздняя — точнее откалибровать её под дисбаланс (эмпирически лучше, чем
   reweight с первой итерации — оригинальная мотивация LDAM-DRW).

Стейтфулность: экземпляр считает собственные вызовы calc_ders_range как
"итерации" бустинга (CatBoost вызывает calc_ders_range один раз на дерево) —
поэтому n_total_iterations должен совпадать с iterations финальной модели.
Если early stopping остановит обучение раньше расчётной точки переключения,
DRW не успеет включиться за этот запуск — ожидаемое поведение, не ошибка.
"""

from __future__ import annotations

import numpy as np


class LDAMLoss:
    """CatBoost-совместимый LDAM + DRW для бинарной классификации.

    Parameters
    ----------
    n_pos, n_neg:
        Число позитивных/негативных примеров в train (для расчёта margin и DRW-весов).
    max_margin:
        Максимальный margin C среди классов (рекомендуется 0.1–1.0).
    reweight_epoch_frac:
        Доля n_total_iterations, после которой включается DRW (рекомендуется 0.5–0.95).
    n_total_iterations:
        Ожидаемое число итераций финальной модели — точка переключения DRW
        считается как reweight_epoch_frac * n_total_iterations вызовов
        calc_ders_range.
    beta:
        Коэффициент effective number of samples для DRW-весов (не тюнится Optuna
        в LDAMClassifier — фиксированный гиперпараметр).

    """

    def __init__(
        self,
        n_pos: int,
        n_neg: int,
        max_margin: float = 0.5,
        reweight_epoch_frac: float = 0.8,
        n_total_iterations: int = 1,
        beta: float = 0.9999,
    ) -> None:
        if n_pos <= 0 or n_neg <= 0:
            raise ValueError(f'n_pos и n_neg должны быть положительными, получено {n_pos}, {n_neg}')
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.max_margin = max_margin
        self.reweight_epoch_frac = reweight_epoch_frac
        self.n_total_iterations = max(1, n_total_iterations)
        self.beta = beta
        self._iteration = 0

        raw_pos = 1.0 / (n_pos ** 0.25)
        raw_neg = 1.0 / (n_neg ** 0.25)
        scale = max_margin / max(raw_pos, raw_neg)
        self.delta_pos = raw_pos * scale
        self.delta_neg = raw_neg * scale

        eff_pos = (1.0 - beta) / (1.0 - beta ** n_pos)
        eff_neg = (1.0 - beta) / (1.0 - beta ** n_neg)
        mean_w = (eff_pos + eff_neg) / 2.0
        self.w_pos = eff_pos / mean_w
        self.w_neg = eff_neg / mean_w

    def calc_ders_range(
        self, predictions, targets, weights
    ) -> list[tuple[float, float]]:
        eps = 1e-7
        f = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        pos = y == 1

        margin_shift = np.where(pos, -self.delta_pos, self.delta_neg)
        p_adj = np.clip(1.0 / (1.0 + np.exp(-(f + margin_shift))), eps, 1.0 - eps)

        self._iteration += 1
        if self._iteration > self.reweight_epoch_frac * self.n_total_iterations:
            class_w = np.where(pos, self.w_pos, self.w_neg)
        else:
            class_w = np.ones_like(y)

        der1 = -(class_w * (p_adj - y))
        der2 = -(class_w * p_adj * (1.0 - p_adj))

        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            der1 = der1 * w
            der2 = der2 * w

        return list(zip(der1.tolist(), der2.tolist()))
