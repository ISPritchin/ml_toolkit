"""ConfidentLearningCleaner: облегчённая нативная реализация Confident Learning

(Northcutt et al., 2021, тот же алгоритм, что стоит за библиотекой cleanlab —
переиспользуем идею, а не саму библиотеку: cleanlab не входит в зависимости
проекта, а нужный нам объём — только бинарный prune_by_noise_rate).

Алгоритм:
  1. Честные OOF-вероятности p(x) на train через StratifiedKFold (n_folds).
  2. Self-confidence порог t_j для каждого класса j — средняя предсказанная
     вероятность класса j СРЕДИ примеров, помеченных j:
       t_1 = mean(p[y=1]),  t_0 = mean(1-p[y=0])
     (у "типичного" примера класса j модель предсказывает j увереннее, чем в
     среднем по всей выборке — порог калибруется по самим данным, а не
     фиксируется произвольно на 0.5).
  3. Confident joint C[i][j] — число примеров с меткой i, для которых модель
     "уверенно" считает их классом j (p_j(x) >= t_j); при попадании в оба
     диапазона сразу (t_1 <= p <= 1-t_0) выбирается класс с большим запасом
     уверенности (margin = p_j(x) - t_j).
  4. Калибровка: C[i][:] масштабируется так, чтобы сумма строки i совпадала с
     реальным числом примеров метки i (Northcutt et al., п. calibration) —
     иначе out-of-fold шум завышает/занижает объём подозрительных меток.
  5. prune_by_noise_rate: для каждой внедиагональной ячейки (i, j) удаляется
     round(C_calibrated[i][j]) примеров с меткой i — те, у кого margin в
     пользу j максимален.
  6. Переобучение на очищенном train.

Ограничение (не покрыто): полная 2D-калибровка cleanlab (по строкам И
столбцам одновременно) — здесь только построчная, для двух классов расхождение
обычно небольшое, но при экстремальном дисбалансе может быть заметнее.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from ml_toolkit.presets.classification._base import BasePreset
from ml_toolkit.presets.classification._optuna_utils import CatBoostPruningCallback, make_pruner

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    'iterations': 600,
    'max_depth': 5,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'min_data_in_leaf': 10,
    'early_stopping_rounds': 80,
    'loss_function': 'Logloss',
    'eval_metric': 'PRAUC',
    'random_seed': 42,
    'verbose': 0,
}


class ConfidentLearningCleaner(BasePreset):
    """CatBoost + нативная confident-learning чистка шумных меток (бинарный случай).

    Parameters
    ----------
    n_folds:
        Число фолдов для честных OOF-вероятностей (на которых строится
        confident joint).
    filter:
        Метод чистки. Реализован только 'prune_by_noise_rate' (см. докстринг
        модуля) — другое значение вызывает ValueError.
    base_params:
        Параметры CatBoost для OOF- и финальной модели. None → дефолтные.
        Игнорируется, если n_optuna_trials > 0.
    n_optuna_trials:
        Если > 0, общая архитектура (одна для OOF-моделей и финальной модели)
        подбирается через Optuna по val PR-AUC на исходном (ещё не очищенном)
        train/val, до запуска OOF и чистки.
    param_space:
        Кастомная функция `f(trial) -> dict` — search space для Optuna вместо
        дефолтного. Может как включать только часть тюнящихся параметров
        (недостающие из loss_function/eval_metric/early_stopping_rounds/
        random_seed/verbose подставляются дефолтами), так и переопределять
        любой из них, включая loss_function/eval_metric — то, что вернула
        param_space, имеет приоритет над дефолтами. Действует только при
        n_optuna_trials > 0. None → дефолтный search space.
    optuna_timeout:
        Ограничение по времени (сек) на весь Optuna-поиск. None — без ограничения.
    optuna_verbose:
        Если True — не глушит логи Optuna. Если False (по умолчанию) —
        форсирует WARNING на время поиска.
    random_seed:
        Зерно StratifiedKFold, CatBoost и Optuna sampler'а.

    Атрибуты после fit::

        class_thresholds_   — (t_0, t_1) self-confidence пороги
        confident_joint_    — 2x2 np.ndarray (после калибровки)
        removed_indices_    — позиции (в исходном X_train) удалённых примеров
        oof_pr_auc_         — PR-AUC OOF-вероятностей до чистки

    Пример::

        model = ConfidentLearningCleaner(n_folds=5)
        model.fit(X_train, y_train, X_valid, y_valid)
        print(f"Удалено {len(model.removed_indices_)} подозрительных меток")
    """

    def __init__(
        self,
        n_folds: int = 5,
        filter: str = 'prune_by_noise_rate',
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
        if filter != 'prune_by_noise_rate':
            raise ValueError(
                f"Реализован только filter='prune_by_noise_rate', получено {filter!r}"
            )
        if n_folds < 2:
            raise ValueError(f'n_folds должен быть >= 2, получено {n_folds}')
        self.n_folds = n_folds
        self.filter = filter
        self.base_params = base_params
        self.param_space = param_space
        self.optuna_timeout = optuna_timeout
        self.optuna_verbose = optuna_verbose
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.class_thresholds_: tuple[float, float] = (0.5, 0.5)
        self.confident_joint_: np.ndarray = np.zeros((2, 2))
        self.removed_indices_: np.ndarray = np.array([], dtype=int)
        self.oof_pr_auc_: float = 0.0

    # ── OOF-вероятности ──────────────────────────────────────────────────────

    def _tune(self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray) -> dict[str, Any]:
        import optuna
        from catboost import CatBoostClassifier, Pool

        if not self.optuna_verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        tr_pool = Pool(X_tr, y_tr, cat_features=self.cat_features_)
        va_pool = Pool(X_va, y_va, cat_features=self.cat_features_)

        def _default_space(trial: optuna.Trial) -> dict[str, Any]:
            return {
                'iterations': trial.suggest_int('iterations', 300, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
            }

        def objective(trial: optuna.Trial) -> float:
            tunable = self.param_space(trial) if self.param_space is not None else _default_space(trial)
            params = {
                'loss_function': 'Logloss',
                'eval_metric': 'PRAUC',
                'early_stopping_rounds': 80,
                'random_seed': self.random_seed,
                'verbose': 0,
                **tunable,
            }
            trial.set_user_attr('cb_params', params)
            pruning_cb = CatBoostPruningCallback(trial, params['eval_metric'])
            m = CatBoostClassifier(**params)
            m.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=[pruning_cb])
            pruning_cb.check_pruned()
            p = m.predict_proba(va_pool)[:, 1]
            return float(average_precision_score(y_va, p))

        logger.info('[ConfidentLearning] Optuna: %d trials (архитектура для OOF и финальной модели)',
                    self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed),
                                    pruner=make_pruner())
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        return dict(study.best_trial.user_attrs['cb_params'])

    def _fit_oof(self, X: pd.DataFrame, y: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
        from catboost import CatBoostClassifier, Pool

        params = {**(params or self.base_params or _DEFAULT_PARAMS), 'random_seed': self.random_seed}
        min_class = int(min(np.bincount(y.astype(int))))
        n_splits = min(self.n_folds, min_class)
        if n_splits < self.n_folds:
            logger.warning('[ConfidentLearning] n_folds снижен %d → %d (мало примеров класса)',
                           self.n_folds, n_splits)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)

        oof = np.zeros(len(y))
        for tr_idx, te_idx in skf.split(X, y):
            m = CatBoostClassifier(**params)
            m.fit(Pool(X.iloc[tr_idx], y[tr_idx], cat_features=self.cat_features_), verbose=False)
            oof[te_idx] = m.predict_proba(Pool(X.iloc[te_idx], cat_features=self.cat_features_))[:, 1]
        return oof

    # ── Confident joint ──────────────────────────────────────────────────────

    def _confident_joint(self, oof: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        pos, neg = y == 1, y == 0
        t1 = float(oof[pos].mean()) if pos.any() else 1.0
        t0 = float((1.0 - oof[neg]).mean()) if neg.any() else 1.0

        # margin_j(x) = p_j(x) - t_j; классифицируем x как "уверенно j" по большему margin
        # среди классов, для которых margin >= 0 (т.е. порог пройден).
        margin1 = oof - t1
        margin0 = (1.0 - oof) - t0
        conf1 = margin1 >= 0
        conf0 = margin0 >= 0
        pred_j = np.full(len(y), -1, dtype=int)  # -1 = не уверены ни в одном классе
        both = conf1 & conf0
        pred_j[conf1 & ~both] = 1
        pred_j[conf0 & ~both] = 0
        pred_j[both] = np.where(margin1[both] >= margin0[both], 1, 0)

        joint = np.zeros((2, 2))
        for i in (0, 1):
            for j in (0, 1):
                joint[i, j] = int(((y == i) & (pred_j == j)).sum())

        # Построчная калибровка к реальным частотам меток.
        n_i = np.array([neg.sum(), pos.sum()], dtype=np.float64)
        row_sums = joint.sum(axis=1)
        for i in (0, 1):
            if row_sums[i] > 0:
                joint[i] *= n_i[i] / row_sums[i]

        return joint, (t0, t1)

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> 'ConfidentLearningCleaner':
        from catboost import CatBoostClassifier, Pool

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        X_tr_feats = X_train[feats].reset_index(drop=True)
        y_tr = y_train.values

        tuned_params = (
            self._tune(X_tr_feats, y_tr, X_valid[feats], y_valid.values)
            if self.n_optuna_trials > 0 else None
        )

        oof = self._fit_oof(X_tr_feats, y_tr, tuned_params)
        self.oof_pr_auc_ = float(average_precision_score(y_tr, oof))
        joint, thresholds = self._confident_joint(oof, y_tr)
        self.confident_joint_ = joint
        self.class_thresholds_ = thresholds

        # margin в пользу альтернативного класса — ранжирует кандидатов внутри каждой
        # off-diagonal ячейки (i, j): чем больше margin, тем увереннее модель считает
        # пример скорее j, чем i.
        margin_alt = np.where(y_tr == 0, oof - thresholds[1], (1.0 - oof) - thresholds[0])

        removed = []
        for i, j in [(0, 1), (1, 0)]:
            n_prune = int(round(joint[i, j]))
            if n_prune <= 0:
                continue
            cand_idx = np.where(y_tr == i)[0]
            cand_margin = margin_alt[cand_idx]
            order = np.argsort(cand_margin)[::-1]
            n_prune = min(n_prune, len(cand_idx))
            removed.extend(cand_idx[order[:n_prune]].tolist())

        self.removed_indices_ = np.array(sorted(removed), dtype=int)
        keep_mask = np.ones(len(y_tr), dtype=bool)
        keep_mask[self.removed_indices_] = False

        logger.info(
            '[ConfidentLearning] thresholds t0=%.4f t1=%.4f  confident_joint=%s  '
            'удалено %d/%d (%.2f%%)  OOF PR-AUC=%.4f',
            thresholds[0], thresholds[1], joint.round(1).tolist(),
            len(self.removed_indices_), len(y_tr),
            100.0 * len(self.removed_indices_) / max(len(y_tr), 1), self.oof_pr_auc_,
        )

        params = {**(tuned_params or self.base_params or _DEFAULT_PARAMS), 'random_seed': self.random_seed}
        tr_pool = Pool(X_tr_feats[keep_mask], y_tr[keep_mask], cat_features=self.cat_features_)
        va_pool = Pool(X_valid[feats], y_valid.values, cat_features=self.cat_features_)

        self._model = CatBoostClassifier(**params)
        self._model.fit(tr_pool, eval_set=va_pool, verbose=False)
        self.best_params_ = params

        self.valid_pred_ = self._model.predict_proba(va_pool)[:, 1]
        self.train_pred_ = self._model.predict_proba(
            Pool(X_tr_feats, y_tr, cat_features=self.cat_features_)
        )[:, 1]

        val_pr_auc = float(average_precision_score(y_valid.values, self.valid_pred_))
        logger.info('[ConfidentLearning] val PR-AUC после чистки=%.4f', val_pr_auc)
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool

        pool = Pool(X[self.selected_features_], cat_features=self.cat_features_)
        return self._model.predict_proba(pool)[:, 1]
