"""BalancedSoftmaxLoss (Ren et al., 2020) для CatBoost.

Training-time аналог пост-хок logit adjustment (см. LogitAdjustmentClassifier/005): вместо сдвига
логитов ПОСЛЕ обучения обычной модели, сдвиг на log(class_prior) встроен в
сам softmax CE во время обучения.

z'_j = z_j + tau*log(n_j/N), где n_j — частота класса j в train, N — общий
размер train. Сдвиг — аддитивная константа на класс (не зависит от логитов),
поэтому градиент обычного softmax CE не меняет форму:

  p' = softmax(z');  L = -log(p'_y);  der1_k = onehot(k=y) - p'_k
  der2[k][j] = -p'_k*(1{k=j} - p'_j)   (полный, точный softmax-CE Hessian)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class BalancedSoftmaxLoss:
    """CatBoost-совместимый Balanced Softmax Loss для мультиклассовой классификации.

    Parameters
    ----------
    class_counts:
        Число примеров каждого класса в train (длина n_classes).
    tau:
        Сила сдвига логитов на log(class_prior). tau=1.0 — полная поправка
        (как в оригинальной статье), tau=0 — обычный softmax CE.

    """

    def __init__(self, class_counts: Sequence[float], tau: float = 1.0) -> None:
        counts = np.asarray(class_counts, dtype=np.float64)
        if counts.ndim != 1 or counts.shape[0] < 2:
            raise ValueError('class_counts должен быть 1D массивом длины >= 2 (n_classes)')
        if np.any(counts <= 0):
            raise ValueError('все class_counts должны быть положительными')
        self.class_counts = counts
        self.n_classes = counts.shape[0]
        self.tau = tau
        self._log_prior = tau * np.log(counts / counts.sum())

    def calc_ders_multi(
        self, approx: Sequence[float], target: float, weight: float
    ) -> tuple[list[float], list[list[float]]]:
        z = np.asarray(approx, dtype=np.float64)
        y = int(target)
        n = self.n_classes

        z_adj = z + self._log_prior
        z_adj_shift = z_adj - z_adj.max()
        exp_adj = np.exp(z_adj_shift)
        p_adj = exp_adj / exp_adj.sum()

        onehot = np.zeros(n)
        onehot[y] = 1.0
        der1 = weight * (onehot - p_adj)
        der2 = -weight * (np.diag(p_adj) - np.outer(p_adj, p_adj))

        return der1.tolist(), der2.tolist()
