"""JackknifePlusRegressor: jackknife+/CV+ предиктивные интервалы через K-fold остатки.

Split-conformal (см. ConformalRegressionWrapper) тратит часть данных на отдельную
calibration-выборку — цена, которую не всегда можно себе позволить на маленьких
датасетах. Jackknife+/CV+ (Barber, Candès, Ramdas, Tibshirani, 2021,
"Predictive inference with the jackknife+") получает интервалы с покрытием без
отдельного calibration split: K моделей обучаются на K фолдах (каждая — на всех
данных, кроме своего фолда), их out-of-fold остатки и играют роль калибровочных
scores.

Для нового x интервал строится из ВСЕХ n обучающих остатков сразу (не только
своего фолда): для строки i (её fold-модель f_{-k(i)}) кандидат нижней границы —
f_{-k(i)}(x) - |resid_i|, верхней — f_{-k(i)}(x) + |resid_i|; итоговые границы —
alpha- и (1-alpha)-квантили этих n кандидатов. Точечный прогноз predict() —
среднее по K fold-моделей (стандартный CV+ point estimate, ансамблирование —
побочный бонус этой схемы, не только интервалы).

X_valid/y_valid необязательны (в отличие от большинства пресетов пакета) — если
переданы, добавляются в пул перед K-fold разбиением (данных на raw-разбиение
без потерь на отдельный holdout и так достаточно — в этом весь смысл
jackknife+ по сравнению со split-conformal).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_toolkit.models._base import XInput, YInput, _to_pandas
from ml_toolkit.presets.regression._base import BasePreset

_DEFAULT_PARAMS: dict[str, Any] = {
    'iterations': 500, 'depth': 5, 'learning_rate': 0.05, 'verbose': 0,
}


class JackknifePlusRegressor(BasePreset):
    """Jackknife+/CV+ интервалы через K-fold CatBoost-модели.

    Parameters
    ----------
    n_folds:
        Число фолдов K (>= 2).
    base_params:
        Параметры CatBoost для всех K fold-моделей. Если None — дефолтные.
    random_seed:
        Зерно KFold-разбиения и CatBoost.

    Атрибуты после fit::

        models_          — список из K обученных CatBoostRegressor
        fold_id_         — {0..K-1} на каждую обучающую строку
        abs_residuals_   — |остаток| out-of-fold на каждую обучающую строку

    Пример::

        model = JackknifePlusRegressor(n_folds=10)
        model.fit(X_train, y_train)                 # X_valid необязателен
        pred = model.predict(X_test)
        lower, upper = model.predict_interval(X_test, alpha=0.1)

    """

    def __init__(
        self,
        n_folds: int = 10,
        base_params: dict[str, Any] | None = None,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(params=base_params, n_optuna_trials=0)
        if n_folds < 2:
            raise ValueError(f'n_folds должен быть >= 2, получено {n_folds}')
        self.n_folds = n_folds
        self.base_params = base_params
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.models_: list = []
        self.fold_id_: np.ndarray | None = None
        self.abs_residuals_: np.ndarray | None = None

    def _fold_preds(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        pool = Pool(X[self.selected_features_], cat_features=self.cat_features_)
        return np.column_stack([m.predict(pool) for m in self.models_])

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: XInput,
        y_train: YInput,
        X_valid: XInput | None = None,
        y_valid: YInput | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> JackknifePlusRegressor:
        from catboost import CatBoostRegressor, Pool
        from sklearn.model_selection import KFold

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        if X_valid is not None:
            X_full = pd.concat([X_train, X_valid], axis=0, ignore_index=True)
            y_full = pd.concat([y_train, y_valid], axis=0, ignore_index=True)
        else:
            X_full, y_full = X_train, y_train

        feats = self._resolve_features(X_full, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_arr = y_full.values
        n = len(y_arr)
        if self.n_folds > n:
            raise ValueError(f'n_folds={self.n_folds} не может превышать число строк ({n})')

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        fold_id = np.empty(n, dtype=int)
        oof_pred = np.empty(n, dtype=float)
        params = {**(self.base_params or _DEFAULT_PARAMS), 'random_seed': self.random_seed, 'verbose': 0}
        self.models_ = []

        for k, (tr_idx, ho_idx) in enumerate(kf.split(X_full)):
            model = CatBoostRegressor(**params)
            model.fit(Pool(X_full.iloc[tr_idx][feats], y_arr[tr_idx], cat_features=self.cat_features_))
            oof_pred[ho_idx] = model.predict(
                Pool(X_full.iloc[ho_idx][feats], cat_features=self.cat_features_)
            )
            fold_id[ho_idx] = k
            self.models_.append(model)

        self.fold_id_ = fold_id
        self.abs_residuals_ = np.abs(y_arr - oof_pred)
        self._model = self.models_
        self.best_params_ = params

        self.train_pred_ = self._fold_preds(X_full).mean(axis=1)
        self.valid_pred_ = self._fold_preds(X_valid).mean(axis=1) if X_valid is not None else None
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict_interval(self, X: XInput, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Jackknife+ интервал уровня (1 - alpha) (Barber et al., 2021 — см. докстринг модуля)."""
        self._check_fitted()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f'alpha должен быть в (0, 1), получено {alpha}')

        Xp = _to_pandas(X)
        per_row_fold_pred = self._fold_preds(Xp)[:, self.fold_id_]  # (m, n_train)
        lower_candidates = per_row_fold_pred - self.abs_residuals_[None, :]
        upper_candidates = per_row_fold_pred + self.abs_residuals_[None, :]
        lower = np.quantile(lower_candidates, alpha, axis=1)
        upper = np.quantile(upper_candidates, 1.0 - alpha, axis=1)
        return lower, upper

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        return self._fold_preds(X).mean(axis=1)
