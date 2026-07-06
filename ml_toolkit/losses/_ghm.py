"""GHMLoss: Gradient Harmonizing Mechanism (Li et al., 2019) для CatBoost.

Идея: и «лёгкие» негативы (p близко к 0, y=0), и настоящие выбросы (шумные
метки, p далеко от y при уже уверенной модели) дают экстремальные значения
градиента |p-y|, только на разных концах — лёгкие примеры сбиваются в одну
плотную область у |p-y|~0, выбросы — в другую у |p-y|~1. Focal Loss (см.
FocalLoss) подавляет только первую область; GHM подавляет ОБЕ, взвешивая
каждый пример обратно пропорционально плотности градиента в его окрестности
(gradient density, GD): регионы с большим количеством примеров получают
меньший вес каждый, редкие регионы (умеренная трудность — основной
обучающий сигнал) — больший.

GD оценивается гистограммой из `bins` равных интервалов по |p-y| in [0,1).
Т.к. calc_ders_range вызывается по одному разу на дерево (а не на мини-батч,
как в оригинальной статье), сырые per-iteration counts по бинам шумные;
`momentum` — EMA той же формы, что в оригинальной статье (там — по
мини-батчам эпохи, здесь — по итерациям бустинга), сглаживает GD между
деревьями.
"""

from __future__ import annotations

import numpy as np


class GHMLoss:
    """CatBoost-совместимый GHM-C Loss для бинарной классификации.

    Parameters
    ----------
    bins:
        Число интервалов гистограммы плотности градиента по |p-y| in [0,1).
    momentum:
        Коэффициент EMA для накопленных по бинам счётчиков между итерациями
        бустинга (0 → использовать только текущую итерацию, без сглаживания).
    """

    def __init__(self, bins: int = 30, momentum: float = 0.75) -> None:
        if bins < 1:
            raise ValueError(f"bins должен быть >= 1, получено {bins}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum должен быть в [0, 1), получено {momentum}")
        self.bins = bins
        self.momentum = momentum
        self._acc_counts: np.ndarray | None = None

    def calc_ders_range(
        self, predictions, targets, weights
    ) -> list[tuple[float, float]]:
        eps = 1e-7
        f = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        w = np.ones_like(f) if weights is None else np.asarray(weights, dtype=np.float64)
        total_w = w.sum()

        p = np.clip(1.0 / (1.0 + np.exp(-f)), eps, 1.0 - eps)
        g = np.abs(p - y)

        bin_idx = np.minimum((g * self.bins).astype(np.int64), self.bins - 1)
        # Взвешенные счётчики бина — sample weight'ы участвуют в оценке
        # плотности градиента наравне с "количеством" (более важная строка
        # ощутимее "занимает" свой бин), не только домножают итоговый градиент.
        counts = np.bincount(bin_idx, weights=w, minlength=self.bins).astype(np.float64)

        if self._acc_counts is None:
            self._acc_counts = counts
        else:
            self._acc_counts = self.momentum * self._acc_counts + (1.0 - self.momentum) * counts

        bin_width = 1.0 / self.bins
        gradient_density = np.maximum(self._acc_counts[bin_idx] / bin_width, eps)
        beta = total_w / gradient_density
        # Взвешенное среднее (не простое) — нормировка должна соответствовать
        # тем же весам, что и сами счётчики бинов.
        beta = beta / (np.sum(w * beta) / total_w)

        # dCE/df = p - y  →  der1 = beta*(y-p); der2 = -beta*p*(1-p) (стандартный CE,
        # взвешенный по GHM), затем домножается на sample weight w — GHM-вес
        # (важность по редкости градиента) и внешний sample weight ортогональны.
        der1 = w * beta * (y - p)
        der2 = -(w * beta * p * (1.0 - p))

        return list(zip(der1.tolist(), der2.tolist()))
