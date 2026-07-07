"""InfluenceBalancedLoss (IB Loss, Park et al., 2021) для CatBoost.

Оригинальная IB Loss взвешивает пример обратно пропорционально его
"influence" = ||x||_1 * |grad| — сумма модуля признаков, умноженная на норму
градиента, что требует доступа к вектору признаков примера. CatBoost
calc_ders_range получает только predictions/targets/weights (без X) — влияние
признаков недоступно, поэтому используется адаптация: influence оценивается
только через |grad| = |p-y| (доступную часть оригинальной формулы). Примеры,
на которых модель уже уверенно права (|p-y| маленький), получают вес, близкий
к базовому class-balanced весу; примеры с большим |p-y| (доминирующие в
агрегированном градиенте, по мотивации статьи — обычно мажоритарный класс) —
подавляются множителем 1/(1+alpha*|p-y|).

Второй множитель — per-class effective-number-of-samples вес (Cui et al.,
2019, тот же w=(1-b)/(1-b^n), что и в ClassBalancedWeightClassifier/006 и в
LDAMLoss) — покрывает статический дисбаланс, а по-сэмпловый influence-множитель
покрывает дисбаланс "внутри класса" (006 не различает лёгкие дубликаты и
редкие паттерны внутри одного класса, IB — различает).

der1 обязан дифференцировать ПОЛНОСТЬЮ через ib_w(p) (а не трактовать вес как
константу/stop-gradient) — тот же принцип, что и в FocalLoss/AsymmetricLoss,
где фокусирующий множитель дифференцируется целиком, а не заморожен. Ранняя
версия этого файла ошибочно использовала der1 = ib_w*(y-p) без члена
d(ib_w)/df — численная проверка против независимо реконструированного лосса
показала расхождение порядка 1e-3, устойчивое при уменьшении шага (не
артефакт конечных разностей). der2 — по общему для пакета соглашению
(TverskyLoss/DiceLoss/LogitNormLoss) приближённая диагональ через
CE-Гессиан — точная вторая производная композиции ib_w(p)*CE(p,y) не
вычисляется.
"""

from __future__ import annotations

import numpy as np


class InfluenceBalancedLoss:
    """CatBoost-совместимый IB Loss (per-sample) + class-balanced веса.

    Parameters
    ----------
    n_pos, n_neg:
        Число позитивных/негативных примеров в train (для class-balanced части).
    alpha:
        Сила подавления примеров с большим influence (|p-y|). Рекомендуется
        порядка 100-1000 — при |p-y|~1 вес почти полностью подавляется.
    beta:
        Коэффициент effective number of samples (Cui et al., 2019) для
        per-class части веса.

    """

    def __init__(
        self,
        n_pos: int,
        n_neg: int,
        alpha: float = 1000.0,
        beta: float = 0.9999,
    ) -> None:
        if n_pos <= 0 or n_neg <= 0:
            raise ValueError(f'n_pos и n_neg должны быть положительными, получено {n_pos}, {n_neg}')
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.alpha = alpha
        self.beta = beta

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

        p = np.clip(1.0 / (1.0 + np.exp(-f)), eps, 1.0 - eps)
        class_w = np.where(pos, self.w_pos, self.w_neg)

        influence = np.abs(p - y)
        denom = 1.0 + self.alpha * influence
        ib_w = class_w / denom
        ce = -(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))

        # d(ib_w)/dp = -class_w*alpha*sign(p-y)/denom^2; sign(p-y) = -1 при y=1
        # (p<1 всегда), +1 при y=0 (p>0 всегда) — совпадает с d|p-y|/dp.
        sign_py = np.where(pos, -1.0, 1.0)
        dw_dp = -class_w * self.alpha * sign_py / (denom * denom)
        dw_df = dw_dp * p * (1.0 - p)

        der1 = -dw_df * ce + ib_w * (y - p)
        der2 = -(ib_w * p * (1.0 - p))

        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            der1 = der1 * w
            der2 = der2 * w

        return list(zip(der1.tolist(), der2.tolist()))
