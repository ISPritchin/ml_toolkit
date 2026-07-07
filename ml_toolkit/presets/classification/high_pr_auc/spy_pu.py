"""SpyPUClassifier: S-EM ("spy") техника для PU-данных (Liu et al., 2003).

Идея: чтобы найти надёжные негативы (RN) внутри U (unlabeled/"негативы",
среди которых на деле могут быть незамеченные позитивы), часть настоящих
позитивов (spy_frac) временно маскируется под U — это "шпионы": известно, что
они позитивы, но модель обучается так, будто это не так. После обучения
P\\spies против U+spies, шпионы дают эталонное распределение score'ов
настоящих позитивов ВНУТРИ U-подобного контекста.

Порог reliable-negative выбирается так, что spy_threshold_pct% шпионов
(заведомо позитивных!) окажутся НИЖЕ порога — то есть мы сознательно
допускаем до spy_threshold_pct% "потерянных" позитивов среди RN, взамен
получая контролируемую, а не произвольную, оценку порога.

RN = {u из U : score(u) < порог}. Финальная модель обучается уже как обычная
supervised P vs RN — без каких-либо весов/коррекций (в отличие от
Элкана-Ното/029, который использует ВСЕ U и корректирует post-hoc через c).

Когда: нужно явное множество надёжных негативов — например, для последующей
ручной проверки/разметки, а не только вероятность на выходе.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.presets.classification._base import BasePreset

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    'iterations': 400,
    'max_depth': 5,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'min_data_in_leaf': 10,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': 0,
}


class SpyPUClassifier(BasePreset):
    """S-EM spy-техника: находит reliable negatives внутри U, затем P vs RN.

    Parameters
    ----------
    spy_frac:
        Доля позитивов, временно маскируемых под U как "шпионы".
    spy_threshold_pct:
        Процент шпионов, которым разрешено оказаться ниже порога RN
        (контролируемая цена ошибки; рекомендуется 5-15).
    base_params:
        Параметры CatBoost (обе стадии). None → дефолтные. Игнорируется, если
        n_optuna_trials > 0.
    n_optuna_trials:
        Если > 0, общая архитектура (обе стадии) подбирается через Optuna: каждый
        trial целиком прогоняет stage1 (поиск RN) + stage2 (P vs RN) с
        кандидатными параметрами и оценивается по val PR-AUC финальной модели.
    param_space:
        Кастомная функция `f(trial) -> dict` — search space для Optuna вместо
        дефолтного. Может как включать только часть тюнящихся параметров
        (недостающие из loss_function/eval_metric/verbose подставляются
        дефолтами), так и переопределять любой из них, включая loss_function/
        eval_metric — то, что вернула param_space, имеет приоритет над
        дефолтами. Действует только при n_optuna_trials > 0. None → дефолтный space.
    optuna_timeout:
        Ограничение по времени (сек) на весь Optuna-поиск. None — без ограничения.
    optuna_verbose:
        Если True — не глушит логи Optuna. Если False (по умолчанию) —
        форсирует WARNING на время поиска.
    random_seed:
        Зерно выбора шпионов, обеих моделей и Optuna sampler'а.

    Атрибуты после fit::

        threshold_          — порог score для reliable negative
        n_reliable_negative_ — размер найденного RN
        n_spies_             — число шпионов, использованных на первой стадии
        stage1_pr_auc_       — PR-AUC вспомогательной модели P\\spies vs U+spies

    Пример::

        model = SpyPUClassifier(spy_frac=0.1, spy_threshold_pct=5)
        model.fit(X_train, y_train, X_valid, y_valid)
        print(f"Reliable negatives: {model.n_reliable_negative_}")

    """

    def __init__(
        self,
        spy_frac: float = 0.1,
        spy_threshold_pct: float = 5.0,
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        param_space: Callable[[Any], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(params=base_params, n_optuna_trials=n_optuna_trials)
        if not 0.0 < spy_frac < 0.5:
            raise ValueError(f'spy_frac должен быть в (0, 0.5), получено {spy_frac}')
        if not 0.0 < spy_threshold_pct < 100.0:
            raise ValueError(f'spy_threshold_pct должен быть в (0, 100), получено {spy_threshold_pct}')
        self.spy_frac = spy_frac
        self.spy_threshold_pct = spy_threshold_pct
        self.base_params = base_params
        self.param_space = param_space
        self.optuna_timeout = optuna_timeout
        self.optuna_verbose = optuna_verbose
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.threshold_: float = 0.0
        self.n_reliable_negative_: int = 0
        self.n_spies_: int = 0
        self.stage1_pr_auc_: float = 0.0

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

    def _tune(
        self,
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
        pos_idx: np.ndarray,
        u_idx: np.ndarray,
        spy_idx: np.ndarray,
        X_va: pd.DataFrame,
        y_va: np.ndarray,
    ) -> dict[str, Any]:
        import optuna

        if not self.optuna_verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        stage1_pos_idx = np.setdiff1d(pos_idx, spy_idx, assume_unique=True)
        stage1_neg_idx = np.concatenate([u_idx, spy_idx])
        stage1_idx = np.concatenate([stage1_pos_idx, stage1_neg_idx])
        y_stage1 = np.concatenate([np.ones(len(stage1_pos_idx)), np.zeros(len(stage1_neg_idx))])

        def _default_space(trial: optuna.Trial) -> dict[str, Any]:
            return {
                'iterations': trial.suggest_int('iterations', 100, 600, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
            }

        def objective(trial: optuna.Trial) -> float:
            tunable = self.param_space(trial) if self.param_space is not None else _default_space(trial)
            params = {'loss_function': 'Logloss', 'eval_metric': 'AUC', 'verbose': 0, **tunable}
            trial.set_user_attr('cb_params', params)
            m1 = self._fit_one(X_tr.iloc[stage1_idx], y_stage1, self.random_seed, params)
            spy_scores = self._predict(m1, X_tr.iloc[spy_idx])
            u_scores = self._predict(m1, X_tr.iloc[u_idx])
            threshold = float(np.percentile(spy_scores, self.spy_threshold_pct))
            rn_idx = u_idx[u_scores < threshold]
            if len(rn_idx) == 0:
                rn_idx = u_idx
            stage2_idx = np.concatenate([pos_idx, rn_idx])
            m2 = self._fit_one(X_tr.iloc[stage2_idx], y_tr[stage2_idx], self.random_seed + 1, params)
            p = self._predict(m2, X_va)
            return float(average_precision_score(y_va, p))

        logger.info('[SpyPU] Optuna: %d trials (обе стадии на trial)', self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed))
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        return dict(study.best_trial.user_attrs['cb_params'])

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> SpyPUClassifier:
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        X_tr = X_train[feats].reset_index(drop=True)
        y_tr = y_train.values.copy()
        pos_idx = np.where(y_tr == 1)[0]
        u_idx = np.where(y_tr == 0)[0]

        rng = np.random.default_rng(self.random_seed)
        n_spies = max(1, int(round(self.spy_frac * len(pos_idx))))
        spy_idx = rng.choice(pos_idx, size=n_spies, replace=False)
        self.n_spies_ = n_spies

        tuned_params = (
            self._tune(X_tr, y_tr, pos_idx, u_idx, spy_idx, X_valid[feats], y_valid.values)
            if self.n_optuna_trials > 0 else None
        )

        # Стадия 1: P\spies (label=1) vs U+spies (label=0).
        stage1_pos_idx = np.setdiff1d(pos_idx, spy_idx, assume_unique=True)
        stage1_neg_idx = np.concatenate([u_idx, spy_idx])
        stage1_idx = np.concatenate([stage1_pos_idx, stage1_neg_idx])
        y_stage1 = np.concatenate([
            np.ones(len(stage1_pos_idx)), np.zeros(len(stage1_neg_idx)),
        ])
        model1 = self._fit_one(X_tr.iloc[stage1_idx], y_stage1, self.random_seed, tuned_params)

        spy_scores = self._predict(model1, X_tr.iloc[spy_idx])
        u_scores = self._predict(model1, X_tr.iloc[u_idx])
        self.threshold_ = float(np.percentile(spy_scores, self.spy_threshold_pct))

        stage1_full_scores = self._predict(model1, X_tr.iloc[stage1_idx])
        self.stage1_pr_auc_ = float(average_precision_score(y_stage1, stage1_full_scores))

        rn_mask = u_scores < self.threshold_
        rn_idx = u_idx[rn_mask]
        self.n_reliable_negative_ = len(rn_idx)
        logger.info(
            '[SpyPU] n_spies=%d  порог=%.4f (перцентиль %.1f от score шпионов)  '
            'reliable negatives=%d/%d (%.1f%%)  stage1 PR-AUC=%.4f',
            n_spies, self.threshold_, self.spy_threshold_pct, self.n_reliable_negative_, len(u_idx),
            100.0 * self.n_reliable_negative_ / max(len(u_idx), 1), self.stage1_pr_auc_,
        )
        if self.n_reliable_negative_ == 0:
            logger.warning('[SpyPU] Reliable negatives не найдены — используем весь U как fallback')
            rn_idx = u_idx

        # Стадия 2: обычный supervised P vs RN.
        stage2_idx = np.concatenate([pos_idx, rn_idx])
        y_stage2 = y_tr[stage2_idx]
        self._model = self._fit_one(X_tr.iloc[stage2_idx], y_stage2, self.random_seed + 1, tuned_params)
        self.best_params_ = {
            'spy_frac': self.spy_frac,
            'spy_threshold_pct': self.spy_threshold_pct,
            'n_reliable_negative': self.n_reliable_negative_,
            'base_params': tuned_params or (self.base_params or _DEFAULT_PARAMS),
        }

        X_va = X_valid[feats]
        self.valid_pred_ = self._predict(self._model, X_va)
        self.train_pred_ = self._predict(self._model, X_tr)
        val_pr_auc = float(average_precision_score(y_valid.values, self.valid_pred_))
        logger.info('[SpyPU] val PR-AUC (P vs RN model)=%.4f', val_pr_auc)
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict(self._model, X[self.selected_features_])
