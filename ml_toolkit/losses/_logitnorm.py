"""LogitNormLoss (Wei et al., 2022) для CatBoost — нормализация логитов перед CE.

Переобученные модели дают экстремально большую L2-норму вектора логитов
(||z||), из-за чего softmax становится overconfident независимо от того,
насколько сигнал действительно разделим — особенно заметно на голове
длиннохвостого мультикласса, где частые классы быстро выходят на большую
||z||. LogitNorm делит логиты на их собственную L2-норму (умноженную на
temperature) перед softmax CE:

  z'_j = z_j / (temperature * ||z||_2)

В отличие от BalancedSoftmax/EqualizationLoss, сдвиг здесь не аддитивная
константа на класс, а полноценная перенормировка, зависящая от ВСЕХ
компонент z — градиент правильно учитывает эту зависимость (см. вывод
в docstring calc_ders_multi), но чтобы не тащить через Гессиан вторую
производную самой нормы (шумная и почти всегда доминируется первым слагаемым),
der2 использует ту же аппроксимацию диагональным CE-Гессианом, что и
TverskyLoss/DiceLoss в этом пакете (см. их докстринги).
"""

from __future__ import annotations

import numpy as np


class LogitNormLoss:
    """CatBoost-совместимый LogitNorm Loss для мультиклассовой классификации.

    Parameters
    ----------
    temperature:
        Температура нормализации. Меньше temperature → сильнее нормализация
        → менее уверенные вероятности. Рекомендуется 0.01-0.1.
    """

    def __init__(self, temperature: float = 0.04) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature должна быть положительной, получено {temperature}")
        self.temperature = temperature

    def calc_ders_multi(self, approx, target, weight):
        eps = 1e-7
        z = np.asarray(approx, dtype=np.float64)
        y = int(target)
        n = z.shape[0]
        t = self.temperature

        norm = max(np.sqrt(np.sum(z * z)), eps)
        z_norm = z / (t * norm)
        z_shift = z_norm - z_norm.max()
        exp_z = np.exp(z_shift)
        p = exp_z / exp_z.sum()

        onehot = np.zeros(n)
        onehot[y] = 1.0

        # dL/dz_k = (p_k - onehot_k)/(t*norm) - z_k/(t*norm^3) * sum_m (p_m-onehot_m)*z_m
        residual = p - onehot
        a = float(np.dot(residual, z))
        der1 = weight * -(residual / (t * norm) - z * a / (t * norm ** 3))

        # Аппроксимация диагональю Гессиана через масштабированный softmax-CE
        # Гессиан (см. Tversky/Dice) — вторая производная нормы отброшена.
        # CatBoost всегда ожидает полную n x n матрицу для calc_ders_multi,
        # даже если внедиагональные элементы нулевые.
        der2_diag = -weight * (p * (1.0 - p)) / (t * norm) ** 2
        der2 = np.diag(der2_diag)

        return der1.tolist(), der2.tolist()
