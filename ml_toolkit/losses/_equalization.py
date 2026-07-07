"""EqualizationLoss: Seesaw Loss (Wang et al., 2021) + EQLv2-style (Tan et al., 2021)
момент-сглаживание в одном мультиклассовом лоссе для CatBoost.

Мультиклассовый softmax CE подавляется головными классами: градиент от
частого негативного класса j доминирует над редким истинным классом y просто
потому, что j встречается намного чаще. Seesaw Loss решает это, домножая
softmax-вклад каждого негативного логита z_j (j != y) на множитель
S_yj = M_yj * C_yj перед вычислением CE:

  M_yj (mitigation, статический) = (n_y/n_j)^seesaw_p, если n_j > n_y, иначе 1
      — сколько раз j встречается чаще y, тем сильнее давится его вклад.
  C_yj (compensation, динамический) = (q_j/q_y)^seesaw_q, если q_j > q_y, иначе 1
      — если модель ошибочно уверена в j сильнее, чем в истинном y, вклад j
      дополнительно давится.

S_yj трактуется как константа (stop-gradient, как в оригинальной реализации
Seesaw) — модифицированные логиты z'_j = z_j + log(S_yj) для j != y, z'_y = z_y,
и от z' берётся обычный softmax CE; поскольку сдвиг для каждого k — аддитивная
константа (не зависящая от z_k при остановленном градиенте), градиент softmax
CE по z_k не меняет форму: der1_k = onehot(k=y) - p'_k, где p' = softmax(z').

Отличие от ванильного Seesaw: q_j/q_y в C_yj берутся не из текущего сэмпла (в
CatBoost calc_ders_multi вызывается по одному разу на объект на дерево — сырая
softmax-вероятность одного объекта слишком шумная для устойчивой поправки), а
из EMA (момент `lambda_`, как в EQLv2 и в GHMLoss) по средней предсказанной
вероятности класса за всё обучение — то же сглаживание, что и в EQLv2's
running pos/neg gradient statistics.
"""

from __future__ import annotations

import numpy as np


class EqualizationLoss:
    """CatBoost-совместимый Seesaw/EQLv2 Loss для мультиклассовой классификации.

    Parameters
    ----------
    class_counts:
        Число примеров каждого класса в train (длина n_classes), для
        статического mitigation-множителя.
    lambda_:
        EMA-момент для сглаживания средней предсказанной вероятности класса
        (используется в динамическом compensation-множителе).
    seesaw_p:
        Степень mitigation-множителя (сильнее подавляет частые негативы).
    seesaw_q:
        Степень compensation-множителя (сильнее подавляет уверенные ошибки).

    """

    def __init__(
        self,
        class_counts,
        lambda_: float = 0.9,
        seesaw_p: float = 0.8,
        seesaw_q: float = 2.0,
    ) -> None:
        counts = np.asarray(class_counts, dtype=np.float64)
        if counts.ndim != 1 or counts.shape[0] < 2:
            raise ValueError('class_counts должен быть 1D массивом длины >= 2 (n_classes)')
        if np.any(counts <= 0):
            raise ValueError('все class_counts должны быть положительными')
        if not 0.0 <= lambda_ < 1.0:
            raise ValueError(f'lambda_ должен быть в [0, 1), получено {lambda_}')
        self.class_counts = counts
        self.n_classes = counts.shape[0]
        self.lambda_ = lambda_
        self.seesaw_p = seesaw_p
        self.seesaw_q = seesaw_q
        self._avg_p: np.ndarray | None = None

    def calc_ders_multi(self, approx, target, weight):
        eps = 1e-7
        z = np.asarray(approx, dtype=np.float64)
        y = int(target)
        n = self.n_classes

        z_shift = z - z.max()
        exp_z = np.exp(z_shift)
        p = exp_z / exp_z.sum()

        if self._avg_p is None:
            self._avg_p = np.full(n, 1.0 / n)
        avg_p = self._avg_p

        ratio_n = self.class_counts[y] / self.class_counts
        mitigation = np.where(self.class_counts > self.class_counts[y], ratio_n ** self.seesaw_p, 1.0)

        ratio_p = avg_p / max(avg_p[y], eps)
        compensation = np.where(avg_p > avg_p[y], ratio_p ** self.seesaw_q, 1.0)

        s = mitigation * compensation
        s[y] = 1.0
        z_adj = z + np.log(s + eps)
        z_adj_shift = z_adj - z_adj.max()
        exp_adj = np.exp(z_adj_shift)
        p_adj = exp_adj / exp_adj.sum()

        self._avg_p = self.lambda_ * avg_p + (1.0 - self.lambda_) * p

        onehot = np.zeros(n)
        onehot[y] = 1.0
        der1 = weight * (onehot - p_adj)
        der2 = -weight * (np.diag(p_adj) - np.outer(p_adj, p_adj))

        return der1.tolist(), der2.tolist()
