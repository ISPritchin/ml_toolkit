"""Кастомные CatBoost-совместимые Python-лоссы для пресетов регрессии.

Приватный модуль (в отличие от ml_toolkit.losses, который документирован в
CLAUDE.md как «лоссы для дисбаланса классов» и целиком про calc_ders_range для
вероятностей). Здесь — тот же интерфейс calc_ders_range(predictions, targets,
weights), но derivatives считаются относительно непрерывного прогноза
(CatBoostRegressor), не вероятности.

Соглашение о знаках то же, что в ml_toolkit.losses: der1 = -dL/df (антиградиент,
модель делает шаг в сторону уменьшения L), der2 = -d²L/df² (или отрицательная
стабилизирующая константа там, где истинная вторая производная равна нулю
почти всюду — как у собственного MAE/Quantile лосса CatBoost).
"""

from __future__ import annotations

import numpy as np


class LogCoshLoss:
    """L(f, y) = log(cosh(f - y)) — гладкий робастный аналог Huber без параметра delta.

    Ведёт себя как 0.5*r^2 при малых |r| и как |r| - log(2) при больших —
    переход гладкий и не требует подбора порога, в отличие от Huber.
    """

    def calc_ders_range(
        self, predictions: list[float], targets: list[float], weights: list[float] | None,
    ) -> list[tuple[float, float]]:
        f = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        r = f - y
        t = np.tanh(r)

        der1 = -t
        der2 = -(1.0 - t * t)

        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            der1 = der1 * w
            der2 = der2 * w
        return list(zip(der1.tolist(), der2.tolist(), strict=False))


class AsymmetricMSELoss:
    """Asymmetric MSE: разная цена пере- и недо-прогноза (newsvendor-подобные задачи).

    r = f - y > 0 (over-forecast, предсказали больше факта) штрафуется
    `over_cost`, r < 0 (under-forecast) — `under_cost`.
    """

    def __init__(self, over_cost: float = 1.0, under_cost: float = 1.0) -> None:
        if over_cost <= 0 or under_cost <= 0:
            raise ValueError('over_cost и under_cost должны быть положительными')
        self.over_cost = over_cost
        self.under_cost = under_cost

    def calc_ders_range(
        self, predictions: list[float], targets: list[float], weights: list[float] | None,
    ) -> list[tuple[float, float]]:
        f = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        r = f - y
        cost = np.where(r > 0, self.over_cost, self.under_cost)

        der1 = -(2.0 * cost * r)
        der2 = -(2.0 * cost)

        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            der1 = der1 * w
            der2 = der2 * w
        return list(zip(der1.tolist(), der2.tolist(), strict=False))


class QuantileHuberLoss:
    """Quantile Huber (сглаженный pinball, как в QR-DQN): L = |q - 1{r<0}| * Huber_kappa(r).

    r = y - f. При kappa → 0 стремится к обычному pinball loss (излом в r=0);
    kappa сглаживает излом квадратичным участком |r| <= kappa, убирая шумный
    градиент чистого pinball рядом с нулём.
    """

    def __init__(self, quantile: float = 0.5, kappa: float = 1.0) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError(f'quantile должен быть в (0, 1), получено {quantile}')
        if kappa <= 0:
            raise ValueError('kappa должен быть положительным')
        self.quantile = quantile
        self.kappa = kappa

    def calc_ders_range(
        self, predictions: list[float], targets: list[float], weights: list[float] | None,
    ) -> list[tuple[float, float]]:
        f = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        r = y - f
        kappa = self.kappa
        abs_r = np.abs(r)
        in_quad = abs_r <= kappa

        dhuber_dr = np.where(in_quad, r, kappa * np.sign(r))
        tail_weight = np.where(r >= 0, self.quantile, 1.0 - self.quantile)

        # dL/df = tail_weight * dhuber_dr * dr/df, dr/df = -1
        der1 = tail_weight * dhuber_dr
        # Стабилизатор Ньютона: истинная d²huber/dr² = 1 в квадратичной зоне, 0 в линейной.
        der2 = -tail_weight * np.where(in_quad, 1.0, 1e-2)

        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            der1 = der1 * w
            der2 = der2 * w
        return list(zip(der1.tolist(), der2.tolist(), strict=False))


class RelativeErrorLoss:
    """Лосс на относительной ошибке: MAPE / SMAPE / WAPE surrogate.

    Per-row поверхность:
      mape:  |f-y| / max(|y|, floor)
      smape: 2|f-y| / (|y| + |f| + floor)
      wape:  |f-y| / global_denom   (global_denom = max(mean(|y_train|), floor),
             считается один раз по обучающей выборке — единственный вариант,
             честно приближающий sum|e|/sum|y| per-row градиентом)

    denom_floor защищает от деления на ~0 при y около нуля (клиенты без
    оборота и т.п.) — без floor такие строки давали бы выбросы градиента.
    """

    def __init__(self, metric: str = 'wape', denom_floor: float = 1.0) -> None:
        if metric not in ('mape', 'smape', 'wape'):
            raise ValueError(f"metric должен быть 'mape'/'smape'/'wape', получено {metric!r}")
        if denom_floor <= 0:
            raise ValueError('denom_floor должен быть положительным')
        self.metric = metric
        self.denom_floor = denom_floor
        self.global_denom: float | None = None  # для wape — проставляется извне перед fit

    def calc_ders_range(
        self, predictions: list[float], targets: list[float], weights: list[float] | None,
    ) -> list[tuple[float, float]]:
        f = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        r = f - y
        sign_r = np.sign(r)

        if self.metric == 'mape':
            d = np.maximum(np.abs(y), self.denom_floor)
            der1 = -sign_r / d
            der2 = -1.0 / d
        elif self.metric == 'wape':
            if self.global_denom is None:
                raise RuntimeError('RelativeErrorLoss(metric="wape"): global_denom не проставлен')
            d = self.global_denom
            der1 = -sign_r / d
            der2 = -np.full_like(f, 1.0 / d)
        else:  # smape
            d = np.abs(y) + np.abs(f) + self.denom_floor
            der1 = -2.0 * sign_r / d + 2.0 * np.abs(r) * np.sign(f) / (d * d)
            der2 = -2.0 / (d * d)

        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            der1 = der1 * w
            der2 = der2 * w
        return list(zip(der1.tolist(), der2.tolist(), strict=False))
