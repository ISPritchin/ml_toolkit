"""GreedyForwardEnsembleSelection: Caruana ensemble selection (Caruana et al., 2004).

В отличие от всех остальных пресетов этого пакета, здесь НЕТ обучения с нуля:
model_library — уже обученные (на своих собственных, возможно разных, train)
модели/пресеты. Задача — не обучить что-то новое, а выбрать оптимальную
КОМБИНАЦИЮ из уже существующего "зоопарка" построенных ранее моделей.

Алгоритм (жадный отбор с возвратом):
  Пока не достигнут max_members: из библиотеки берётся модель, чьё ДОБАВЛЕНИЕ
  (усреднение с уже отобранными) даёт наибольший прирост val-метрики; эта
  модель может быть выбрана повторно (with replacement) — тем самым неявно
  получая больший вес в итоговом среднем. Останавливаемся раньше max_members,
  если очередное добавление не улучшает метрику.

Bootstrap-регуляризация (n_bags, Caruana et al. §2.4): жадный отбор целиком
повторяется n_bags раз на bootstrap-ресэмплах val, чтобы не переобучиться под
конкретный val-сплит (одиночный жадный прогон на маленьком val нестабилен —
легко "нащупать" случайную комбинацию, выигрышную именно на этих объектах).
Итоговый вес модели = доля её появлений среди всех выборов всех бэгов.

Не покрыто (упрощение относительно оригинала): Caruana инициализирует
стартовый ансамбль N лучшими одиночными моделями библиотеки; здесь отбор
стартует с пустого ансамбля — на library из 10+ моделей разница на практике
малозаметна, а код проще.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.presets.classification._base import BasePreset

logger = logging.getLogger(__name__)


def _get_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    p = np.asarray(model.predict_proba(X))
    return p[:, 1] if p.ndim == 2 else p


class GreedyForwardEnsembleSelection(BasePreset):
    """Жадный (bagged) отбор комбинации из библиотеки уже обученных моделей.

    Parameters
    ----------
    model_library:
        Список уже обученных объектов с методом predict_proba(X). Каждый
        элемент — независимо обученная модель/пресет (X_train/y_train этого
        пресета не используются — члены библиотеки обучены заранее, каждый на
        своих данных).
    max_members:
        Максимальное число слотов в ансамбле за один жадный прогон (с учётом
        повторов).
    n_bags:
        Число bootstrap-повторов жадного отбора на val (регуляризация).
    random_seed:
        Зерно bootstrap-ресэмплинга val.

    Атрибуты после fit::

        weights_      — вес каждой модели библиотеки (доля выборов, сумма = 1)
        pick_counts_  — сырые счётчики выборов по бэгам

    Пример::

        model = GreedyForwardEnsembleSelection(model_library=[m1, m2, m3, ...])
        model.fit(X_train, y_train, X_valid, y_valid)
        print(dict(zip(range(len(model.weights_)), model.weights_)))

    """

    def __init__(
        self,
        model_library: list[Any],
        max_members: int = 10,
        n_bags: int = 20,
        random_seed: int = 42,
    ) -> None:
        super().__init__(params=None, n_optuna_trials=0)
        if len(model_library) < 2:
            raise ValueError(f'model_library должна содержать >= 2 моделей, получено {len(model_library)}')
        if max_members < 1:
            raise ValueError(f'max_members должен быть >= 1, получено {max_members}')
        if n_bags < 1:
            raise ValueError(f'n_bags должен быть >= 1, получено {n_bags}')
        self.model_library = model_library
        self.max_members = max_members
        self.n_bags = n_bags
        self.random_seed = random_seed

        self.weights_: np.ndarray = np.array([])
        self.pick_counts_: np.ndarray = np.array([])

    def _greedy_bag(self, proba_bag: np.ndarray, y_bag: np.ndarray) -> list[int]:
        n_models = proba_bag.shape[1]
        selected: list[int] = []
        running_sum = np.zeros(len(y_bag))
        best_score = -np.inf

        for _ in range(self.max_members):
            k = len(selected)
            scores = np.array([
                average_precision_score(y_bag, (running_sum + proba_bag[:, j]) / (k + 1))
                for j in range(n_models)
            ])
            j_best = int(np.argmax(scores))
            if selected and scores[j_best] <= best_score:
                break
            best_score = scores[j_best]
            selected.append(j_best)
            running_sum += proba_bag[:, j_best]
        return selected

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> GreedyForwardEnsembleSelection:
        _, _, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        y_va = y_valid.values
        n_models = len(self.model_library)
        self.selected_features_ = list(X_valid.columns)

        va_matrix = np.stack([_get_proba(m, X_valid) for m in self.model_library], axis=1)
        single_scores = [float(average_precision_score(y_va, va_matrix[:, j])) for j in range(n_models)]
        logger.info('[GreedyEnsembleSelection] library PR-AUCs: %s', [f'{s:.4f}' for s in single_scores])

        rng = np.random.default_rng(self.random_seed)
        pick_counts = np.zeros(n_models, dtype=np.float64)

        for b in range(self.n_bags):
            bag_idx = rng.integers(0, len(y_va), size=len(y_va))
            selected = self._greedy_bag(va_matrix[bag_idx], y_va[bag_idx])
            for j in selected:
                pick_counts[j] += 1

        if pick_counts.sum() == 0:
            logger.warning('[GreedyEnsembleSelection] Ни одна модель не выбрана ни в одном бэге — '
                           'используем равные веса как fallback')
            pick_counts = np.ones(n_models)

        self.pick_counts_ = pick_counts
        self.weights_ = pick_counts / pick_counts.sum()

        self.valid_pred_ = va_matrix @ self.weights_
        val_pr_auc = float(average_precision_score(y_va, self.valid_pred_))
        self.train_pred_ = None
        self.best_params_ = {'max_members': self.max_members, 'n_bags': self.n_bags,
                             'weights': self.weights_.tolist()}
        self._model = True

        logger.info('[GreedyEnsembleSelection] weights=%s  ensemble val PR-AUC=%.4f (best single=%.4f)',
                    np.round(self.weights_, 3).tolist(), val_pr_auc, max(single_scores))
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.stack([_get_proba(m, X) for m in self.model_library], axis=1)
        return preds @ self.weights_
