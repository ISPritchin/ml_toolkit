"""Общая логика undersampling мажоритарного класса для Optuna-тюнинга классификаторов.

Используется CatBoostClassifier, LightGBMClassifier, XGBoostClassifier — механизм
одинаковый независимо от библиотеки: за каждый trial урезается мажоритарный класс
(бинарный случай — majority_fraction, мультикласс — balance_fraction), а финальная
модель обучается на том же сэмпле, что и лучший trial (тот же fraction и тот же
seed 42 + trial.number), а не на всех данных — иначе гиперпараметры оцениваются на
одном объёме данных, а обучается финальная модель на другом.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class UndersampleSampler:
    """Строит индексы trial-сэмпла с урезанным мажоритарным классом.

    Signal: одна доля (0..1) определяет, насколько сильно урезаются мажоритарные
    классы относительно миноритарного.
    Formula: бинарный — n_keep = n_majority * majority_fraction, миноритарный класс
    берётся полностью; мультикласс — n_keep_k = n_minority + balance_fraction *
    (n_k - n_minority) для каждого класса k.
    Outputs: sample_idx(fraction, trial_number) -> отсортированный np.ndarray позиций.
    Interpretation: fraction=1.0 — исходное распределение без урезания;
    для бинарного fraction=0.05..1.0, для мультикласса 0.0 (идеальный баланс)..1.0.
    """

    def __init__(self, y: np.ndarray, is_binary: bool, log_prefix: str = '') -> None:
        self.y = y
        self.is_binary = is_binary
        self.classes, self.class_counts = np.unique(y, return_counts=True)
        self.n_minority = int(self.class_counts.min())
        self.class_idx_map = {cls: np.where(y == cls)[0] for cls in self.classes}
        self.full_idx = np.arange(len(y))

        for cls, cnt in zip(self.classes, self.class_counts):
            logger.info('%s Class %s: %d (%.1f%%)', log_prefix, cls, cnt, cnt / len(y) * 100)

        if is_binary:
            maj_pos = int(np.argmax(self.class_counts))
            self.majority_cls = self.classes[maj_pos]
            self.minority_cls = self.classes[1 - maj_pos]
            self.majority_idx = self.class_idx_map[self.majority_cls]
            self.minority_idx = self.class_idx_map[self.minority_cls]
            self.n_majority = int(self.class_counts[maj_pos])
            logger.info('%s Majority class: %s (%d), minority: %s (%d)',
                        log_prefix, self.majority_cls, self.n_majority, self.minority_cls, self.n_minority)
            self.fraction_key = 'majority_fraction'
        else:
            self.fraction_key = 'balance_fraction'

    def suggest_fraction(self, trial) -> float:
        if self.is_binary:
            # Меньше → быстрее trial, сильнее балансировка.
            return trial.suggest_float('majority_fraction', 0.05, 1.0)
        # balance_fraction=0 → все классы до размера минорного (идеальный баланс).
        # balance_fraction=1 → исходное распределение (без подвыборки).
        return trial.suggest_float('balance_fraction', 0.0, 1.0)

    def sample_idx(self, fraction_value: float, trial_number: int) -> np.ndarray:
        rng = np.random.default_rng(42 + trial_number)
        if self.is_binary:
            n_keep = max(1, int(self.n_majority * fraction_value))
            sampled_maj = rng.choice(self.majority_idx, size=n_keep, replace=False)
            return np.sort(np.concatenate([self.minority_idx, sampled_maj]))
        parts = []
        for cls, cnt in zip(self.classes, self.class_counts):
            n_keep = max(1, int(self.n_minority + fraction_value * (cnt - self.n_minority)))
            cidx = self.class_idx_map[cls]
            if n_keep < cnt:
                cidx = rng.choice(cidx, size=n_keep, replace=False)
            parts.append(cidx)
        return np.sort(np.concatenate(parts))
