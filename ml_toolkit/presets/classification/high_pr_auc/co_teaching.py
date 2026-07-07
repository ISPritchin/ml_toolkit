"""CoTeachingClassifier: Co-teaching (Han et al., 2018) для шумных меток.

Идея: две модели с разными случайными зёрнами обучаются ПАРАЛЛЕЛЬНО; на
каждом раунде обе оценивают per-example loss на полном train, отбирают
R(t)-долю примеров с наименьшим собственным loss ("small-loss trick" —
предположение, что корректно размеченные примеры в среднем легче), но
KEY IDEA: модель A переобучается на small-loss выборке МОДЕЛИ B, и наоборот
(cross-update). Без этого перекрёстного обмена одна модель, отбирающая свои
же "лёгкие" примеры, быстро скатывается в confirmation bias — то же самое
поведение, что и у self-training (см. SelfTrainingBooster), только без
партнёра, который замечает разные ошибки.

R(t) убывает линейно от 1.0 (раунд 0, полный train) до 1-forget_rate (раунд
n_rounds) — ранние раунды почти не фильтруют (модели ещё не научились
отличать шум), поздние — фильтруют максимально жёстко.

Отличие от одиночной чистки (ConfidentLearningCleaner/025): Co-teaching не
делает один статический прогон чистки, а итеративно уточняет и модель, и
набор "доверенных" примеров совместно; полезно при большей (>5%) и не
обязательно случайной доле шума, где единая OOF-оценка (025) менее стабильна.

Финальное предсказание — среднее вероятностей двух моделей (стандартная
практика co-teaching: ансамбль надёжнее любой из двух по отдельности).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.presets.classification._base import BasePreset

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    'iterations': 500,
    'max_depth': 5,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'min_data_in_leaf': 10,
    'loss_function': 'Logloss',
    'eval_metric': 'PRAUC',
    'verbose': 0,
}

_EPS = 1e-7


def _bce(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


class CoTeachingClassifier(BasePreset):
    """Две CatBoost-модели, взаимно обучающиеся на small-loss примерах партнёра.

    Parameters
    ----------
    forget_rate:
        Итоговая (на последнем раунде) доля train, отбрасываемая как
        вероятный шум (рекомендуется 0.1-0.3 при доле шума >5%).
    n_rounds:
        Число раундов co-teaching (рекомендуется 3-8).
    base_params:
        Параметры CatBoost для обеих моделей. None → дефолтные. Игнорируется,
        если n_optuna_trials > 0.
    n_optuna_trials:
        Если > 0, общая архитектура (одна на модели A и B, на всех раундах)
        подбирается через Optuna: каждый trial обучает пару моделей на полном
        train (как раунд 0) и оценивается по val PR-AUC ансамбля.
    optuna_timeout:
        Ограничение по времени (сек) на весь Optuna-поиск. None — без ограничения.
    random_seed:
        Модель A получает random_seed, модель B — random_seed + 1. Также сид
        Optuna sampler'а.

    Атрибуты после fit::

        round_scores_a_, round_scores_b_ — val PR-AUC каждой модели по раундам
        ensemble_scores_                 — val PR-AUC среднего (A+B)/2 по раундам
        keep_fraction_history_           — R(t) по раундам

    Пример::

        model = CoTeachingClassifier(forget_rate=0.2, n_rounds=5)
        model.fit(X_train, y_train, X_valid, y_valid)
    """

    def __init__(
        self,
        forget_rate: float = 0.2,
        n_rounds: int = 5,
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        optuna_timeout: int | None = None,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(params=base_params, n_optuna_trials=n_optuna_trials)
        if not 0.0 <= forget_rate < 1.0:
            raise ValueError(f'forget_rate должен быть в [0, 1), получено {forget_rate}')
        if n_rounds < 1:
            raise ValueError(f'n_rounds должен быть >= 1, получено {n_rounds}')
        self.forget_rate = forget_rate
        self.n_rounds = n_rounds
        self.base_params = base_params
        self.optuna_timeout = optuna_timeout
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.round_scores_a_: list[float] = []
        self.round_scores_b_: list[float] = []
        self.ensemble_scores_: list[float] = []
        self.keep_fraction_history_: list[float] = []

    def _fit_one(
        self, X: pd.DataFrame, y: np.ndarray, seed: int, params: dict[str, Any] | None = None,
    ) -> Any:
        from catboost import CatBoostClassifier, Pool

        p = {**(params or self.base_params or _DEFAULT_PARAMS), 'random_seed': seed}
        m = CatBoostClassifier(**p)
        m.fit(Pool(X, y, cat_features=self.cat_features_), verbose=False)
        return m

    def _predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        return model.predict_proba(Pool(X, cat_features=self.cat_features_))[:, 1]

    def _tune(self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray) -> dict[str, Any]:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                'iterations': trial.suggest_int('iterations', 200, 800, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
                'loss_function': 'Logloss',
                'eval_metric': 'PRAUC',
                'verbose': 0,
            }
            m_a = self._fit_one(X_tr, y_tr, self.random_seed, params)
            m_b = self._fit_one(X_tr, y_tr, self.random_seed + 1, params)
            blend = 0.5 * (self._predict(m_a, X_va) + self._predict(m_b, X_va))
            return float(average_precision_score(y_va, blend))

        logger.info('[CoTeaching] Optuna: %d trials (общая архитектура для A/B на всех раундах)',
                    self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed))
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        return {**study.best_params, 'loss_function': 'Logloss', 'eval_metric': 'PRAUC', 'verbose': 0}

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> 'CoTeachingClassifier':
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        X_tr = X_train[feats].reset_index(drop=True)
        y_tr = y_train.values
        X_va = X_valid[feats]
        y_va = y_valid.values
        n = len(y_tr)

        logger.info('[CoTeaching] n_rounds=%d  forget_rate=%.2f  n_train=%d',
                    self.n_rounds, self.forget_rate, n)

        tuned_params = self._tune(X_tr, y_tr, X_va, y_va) if self.n_optuna_trials > 0 else None

        model_a = self._fit_one(X_tr, y_tr, self.random_seed, tuned_params)
        model_b = self._fit_one(X_tr, y_tr, self.random_seed + 1, tuned_params)

        self.round_scores_a_ = [float(average_precision_score(y_va, self._predict(model_a, X_va)))]
        self.round_scores_b_ = [float(average_precision_score(y_va, self._predict(model_b, X_va)))]
        self.ensemble_scores_ = [float(average_precision_score(
            y_va, 0.5 * (self._predict(model_a, X_va) + self._predict(model_b, X_va))
        ))]
        self.keep_fraction_history_ = [1.0]
        logger.info('[CoTeaching] Раунд 0 (init)  A=%.4f  B=%.4f  ensemble=%.4f',
                    self.round_scores_a_[-1], self.round_scores_b_[-1], self.ensemble_scores_[-1])

        class_idx = {cls: np.where(y_tr == cls)[0] for cls in np.unique(y_tr)}

        def _small_loss_stratified(loss: np.ndarray, keep_frac: float) -> np.ndarray:
            # Отбор наименьшего loss ВНУТРИ каждого класса отдельно, а не глобально:
            # позитивы (миноритарный класс) систематически имеют более высокий BCE,
            # чем негативы, особенно на ранних раундах — глобальный top-k по loss
            # при малой доле позитивов и агрессивном forget_rate может целиком
            # вымыть позитивы из отобранной "чистой" выборки, схлопывая её до
            # одного класса (CatBoost не может обучаться на константной метке).
            picks = []
            for idx in class_idx.values():
                n_keep_cls = max(1, int(round(keep_frac * len(idx))))
                order = idx[np.argsort(loss[idx])[:n_keep_cls]]
                picks.append(order)
            return np.concatenate(picks)

        for t in range(1, self.n_rounds + 1):
            ramp = t / self.n_rounds
            keep_frac = 1.0 - self.forget_rate * ramp
            self.keep_fraction_history_.append(keep_frac)

            loss_a = _bce(y_tr, self._predict(model_a, X_tr))
            loss_b = _bce(y_tr, self._predict(model_b, X_tr))
            small_a = _small_loss_stratified(loss_a, keep_frac)  # A считает эти примеры "чистыми"
            small_b = _small_loss_stratified(loss_b, keep_frac)  # B считает эти примеры "чистыми"

            # Перекрёстное обновление: A учится на выборе B, B учится на выборе A.
            model_a = self._fit_one(X_tr.iloc[small_b], y_tr[small_b], self.random_seed + 2 * t, tuned_params)
            model_b = self._fit_one(X_tr.iloc[small_a], y_tr[small_a], self.random_seed + 2 * t + 1, tuned_params)

            proba_a = self._predict(model_a, X_va)
            proba_b = self._predict(model_b, X_va)
            self.round_scores_a_.append(float(average_precision_score(y_va, proba_a)))
            self.round_scores_b_.append(float(average_precision_score(y_va, proba_b)))
            self.ensemble_scores_.append(
                float(average_precision_score(y_va, 0.5 * (proba_a + proba_b)))
            )
            logger.info(
                '[CoTeaching] Раунд %d  keep=%.2f (small_a=%d, small_b=%d)  A=%.4f  B=%.4f  ensemble=%.4f',
                t, keep_frac, len(small_a), len(small_b), self.round_scores_a_[-1], self.round_scores_b_[-1],
                self.ensemble_scores_[-1],
            )

        self._model = (model_a, model_b)
        proba_a_va = self._predict(model_a, X_va)
        proba_b_va = self._predict(model_b, X_va)
        self.valid_pred_ = 0.5 * (proba_a_va + proba_b_va)
        self.train_pred_ = 0.5 * (self._predict(model_a, X_tr) + self._predict(model_b, X_tr))
        self.best_params_ = {
            'forget_rate': self.forget_rate,
            'n_rounds': self.n_rounds,
            'final_keep_fraction': self.keep_fraction_history_[-1],
            'base_params': tuned_params or (self.base_params or _DEFAULT_PARAMS),
        }
        logger.info('[CoTeaching] Итог: ensemble val PR-AUC=%.4f (init=%.4f, delta=%.4f)',
                    self.ensemble_scores_[-1], self.ensemble_scores_[0],
                    self.ensemble_scores_[-1] - self.ensemble_scores_[0])
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        model_a, model_b = self._model
        X_feats = X[self.selected_features_]
        return 0.5 * (self._predict(model_a, X_feats) + self._predict(model_b, X_feats))
