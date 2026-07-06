"""TwoStageCascade: двухэтапный каскад CatBoost.

Stage 1 (высокий recall): обучается с высоким scale_pos_weight.
  На val ищется наименьший порог, при котором recall ≥ recall_target.

Stage 2 (высокая precision): обучается только на тех train-примерах,
  которые прошли порог Stage 1 (=позитивы + трудные негативы).
  Stage 2 учится отличать настоящие позитивы от ложных срабатываний Stage 1.
  Train-кандидаты отбираются по OUT-OF-FOLD предсказаниям Stage 1 (K-fold),
  а не по in-sample скорам: полная модель Stage 1 переобучена на своём train,
  и её in-sample скоры не соответствуют инференс-распределению кандидатов.

Итоговый score на инференсе:
  - прошедшие Stage 1: mapped to [threshold1, 1.0] через Stage 2 score
  - не прошедшие Stage 1: Stage 1 score остаётся в [0, threshold1)
  Это обеспечивает непрерывный глобальный ранкинг без «дыры» в рейтинге.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, recall_score

from ml_toolkit.presets.classification._base import BasePreset
from ml_toolkit.presets.classification._optuna_utils import CatBoostPruningCallback, make_pruner

logger = logging.getLogger(__name__)


class TwoStageCascade(BasePreset):
    """Двухэтапный каскад CatBoost: recall → precision.

    Parameters
    ----------
    recall_target:
        Минимальный recall Stage 1 на val. Порог подбирается автоматически.
    stage1_params:
        Параметры CatBoost для Stage 1 (если None — используются дефолтные).
    stage2_params:
        Параметры CatBoost для Stage 2 (если None — используются дефолтные).
    stage1_n_trials:
        Optuna-trials для Stage 1 (0 — использовать stage1_params напрямую).
    stage2_n_trials:
        Optuna-trials для Stage 2 (0 — использовать stage2_params напрямую).
    random_seed:
        Зерно CatBoost, Optuna sampler'а и OOF-разбиения (StratifiedKFold).

    Атрибуты после fit::

        model1_       — Stage 1 CatBoost
        model2_       — Stage 2 CatBoost
        threshold1_   — порог Stage 1 (float)
        stage1_recall_ — фактический recall Stage 1 на val
        stage2_coverage_ — доля val-примеров, дошедших до Stage 2

    Пример::

        model = TwoStageCascade(recall_target=0.90)
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=[...])
        proba = model.predict_proba(X_test)
    """

    def __init__(
        self,
        recall_target: float = 0.90,
        stage1_params: dict[str, Any] | None = None,
        stage2_params: dict[str, Any] | None = None,
        stage1_n_trials: int = 0,
        stage2_n_trials: int = 0,
        optuna_timeout: int | None = None,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ):
        super().__init__(params=None, n_optuna_trials=0)
        self.recall_target = recall_target
        self.stage1_params = stage1_params
        self.stage2_params = stage2_params
        self.stage1_n_trials = stage1_n_trials
        self.stage2_n_trials = stage2_n_trials
        self.optuna_timeout = optuna_timeout
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.model1_: Any = None
        self.model2_: Any = None
        self.threshold1_: float = 0.5
        self.stage1_recall_: float = 0.0
        self.stage2_coverage_: float = 0.0

    # ── Stage 1 ─────────────────────────────────────────────────────────────

    def _default_stage1_params(self) -> dict:
        return {
            'iterations': 700,
            'max_depth': 5,
            'learning_rate': 0.05,
            'scale_pos_weight': 10.0,   # агрессивно в пользу recall
            'l2_leaf_reg': 3.0,
            'subsample': 0.8,
            'loss_function': 'Logloss',
            'eval_metric': 'Recall',
            'early_stopping_rounds': 100,
            'random_seed': self.random_seed,
            'verbose': 0,
        }

    def _default_stage2_params(self) -> dict:
        return {
            'iterations': 700,
            'max_depth': 5,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3.0,
            'subsample': 0.8,
            'loss_function': 'Logloss',
            'eval_metric': 'PRAUC',
            'early_stopping_rounds': 100,
            'random_seed': self.random_seed,
            'verbose': 0,
        }

    def _train_with_optuna(self, CB, tr_pool, va_pool, n_trials: int, is_stage2: bool):
        import optuna
        from catboost import CatBoostClassifier

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        metric_fn = average_precision_score

        def objective(trial: optuna.Trial) -> float:
            params = {
                'iterations': trial.suggest_int('iterations', 300, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
                'scale_pos_weight': (
                    1.0 if is_stage2 else
                    trial.suggest_float('scale_pos_weight', 1.0, 30.0, log=True)
                ),
                'loss_function': 'Logloss',
                'eval_metric': 'PRAUC',
                'early_stopping_rounds': 100,
                'random_seed': self.random_seed,
                'verbose': 0,
            }
            pruning_cb = CatBoostPruningCallback(trial, 'PRAUC')
            m = CatBoostClassifier(**params)
            m.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=[pruning_cb])
            pruning_cb.check_pruned()
            p = m.predict_proba(va_pool)[:, 1]
            return float(metric_fn(va_pool.get_label(), p))

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed),
                                    pruner=make_pruner())
        study.optimize(objective, n_trials=n_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        best = {**study.best_params, 'loss_function': 'Logloss', 'eval_metric': 'PRAUC',
                'early_stopping_rounds': 100, 'random_seed': self.random_seed, 'verbose': 0}
        m = CatBoostClassifier(**best)
        m.fit(tr_pool, eval_set=va_pool, verbose=False)
        return m, best

    # ── Поиск порога ────────────────────────────────────────────────────────

    def _find_threshold(self, y_val: np.ndarray, stage1_proba: np.ndarray) -> float:
        p_curve, r_curve, t_curve = precision_recall_curve(y_val, stage1_proba)
        # r_curve убывает слева направо (r_curve[0]=1.0); последняя точка
        # (precision=1, recall=0) — dummy без threshold, r_curve[:-1] соответствует t_curve
        mask = r_curve[:-1] >= self.recall_target
        if mask.any():
            # Берём наибольший threshold при котором recall ещё ≥ target
            idx = int(np.where(mask)[0][-1])
            return float(t_curve[idx])
        logger.warning('[Cascade] Не удалось достичь recall=%.2f даже при threshold=0 — ставим 0',
                       self.recall_target)
        return 0.0

    # ── OOF-скоры Stage 1 ───────────────────────────────────────────────────

    def _stage1_oof_scores(
        self,
        X_feats: pd.DataFrame,
        y: np.ndarray,
        params: dict,
        va_pool: Any,
    ) -> np.ndarray:
        """Out-of-fold предсказания Stage 1 на train (для отбора кандидатов Stage 2).

        In-sample скоры полной model1_ переобучены и завышены на train —
        отбор «трудных негативов» по ним не соответствует тому, что Stage 2
        увидит на инференсе. OOF-скоры воспроизводят инференс-распределение.
        """
        from catboost import CatBoostClassifier, Pool
        from sklearn.model_selection import StratifiedKFold

        min_class = int(min(np.bincount(y.astype(int))))
        if min_class < 2:
            logger.warning(
                '[Cascade] OOF невозможен (класс с %d примерами) — '
                'используем in-sample скоры Stage 1', min_class,
            )
            return self.model1_.predict_proba(
                Pool(X_feats, cat_features=self.cat_features_)
            )[:, 1]

        n_splits = min(5, min_class)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        oof = np.zeros(len(y))
        for tr_idx, te_idx in skf.split(np.zeros(len(y)), y):
            m = CatBoostClassifier(**params)
            m.fit(
                Pool(X_feats.iloc[tr_idx], y[tr_idx], cat_features=self.cat_features_),
                eval_set=va_pool, verbose=False,
            )
            oof[te_idx] = m.predict_proba(
                Pool(X_feats.iloc[te_idx], cat_features=self.cat_features_)
            )[:, 1]
        return oof

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> 'TwoStageCascade':
        from catboost import CatBoostClassifier, Pool

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values

        # ── Stage 1 ─────────────────────────────────────────────────────────
        logger.info('[Cascade] Stage 1: target recall=%.2f', self.recall_target)
        s1_params = {**(self.stage1_params or self._default_stage1_params()), 'random_seed': self.random_seed}

        tr1 = Pool(X_train[feats], y_tr, cat_features=self.cat_features_)
        va1 = Pool(X_valid[feats], y_va, cat_features=self.cat_features_)

        if self.stage1_n_trials > 0:
            self.model1_, s1_used_params = self._train_with_optuna(
                CatBoostClassifier, tr1, va1, self.stage1_n_trials, is_stage2=False
            )
        else:
            self.model1_ = CatBoostClassifier(**s1_params)
            self.model1_.fit(tr1, eval_set=va1, verbose=False)
            s1_used_params = s1_params

        s1_va = self.model1_.predict_proba(va1)[:, 1]
        self.threshold1_ = self._find_threshold(y_va, s1_va)
        self.stage1_recall_ = float(recall_score(
            y_va, (s1_va >= self.threshold1_).astype(int), zero_division=0
        ))
        candidate_mask_va = s1_va >= self.threshold1_
        self.stage2_coverage_ = float(candidate_mask_va.mean())
        logger.info('[Cascade] Stage 1  threshold=%.4f  recall=%.4f  coverage=%.3f',
                    self.threshold1_, self.stage1_recall_, self.stage2_coverage_)

        # ── Stage 2: train только на прошедших Stage 1 (по OOF-скорам) ──────
        s1_tr = self._stage1_oof_scores(X_train[feats], y_tr, s1_used_params, va1)
        candidate_mask_tr = s1_tr >= self.threshold1_

        n_pos = int(y_tr[candidate_mask_tr].sum())
        n_neg = int((y_tr[candidate_mask_tr] == 0).sum())
        logger.info('[Cascade] Stage 2: %d позитивов, %d трудных негативов в train',
                    n_pos, n_neg)

        if candidate_mask_tr.sum() < 10 or n_pos == 0 or n_neg == 0:
            logger.warning('[Cascade] Stage 2 train содержит один класс — снижаем threshold до p30')
            self.threshold1_ = float(np.percentile(s1_tr, 30))
            candidate_mask_tr = s1_tr >= self.threshold1_
            candidate_mask_va = s1_va >= self.threshold1_
            # Диагностика пересчитывается под новый порог
            self.stage1_recall_ = float(recall_score(
                y_va, (s1_va >= self.threshold1_).astype(int), zero_division=0
            ))
            self.stage2_coverage_ = float(candidate_mask_va.mean())
            logger.info('[Cascade] Fallback  threshold=%.4f  recall=%.4f  coverage=%.3f',
                        self.threshold1_, self.stage1_recall_, self.stage2_coverage_)

        X_tr2 = X_train[feats][candidate_mask_tr]
        y_tr2 = y_tr[candidate_mask_tr]
        X_va2 = X_valid[feats][candidate_mask_va]
        y_va2 = y_va[candidate_mask_va]

        tr2 = Pool(X_tr2, y_tr2, cat_features=self.cat_features_)
        # Не передаём eval_set если val-подмножество содержит только один класс
        _va2_ok = len(y_va2) > 1 and len(np.unique(y_va2)) > 1
        if not _va2_ok:
            logger.warning('[Cascade] Stage 2 val содержит один класс (%s) — обучаем без eval_set',
                           np.unique(y_va2).tolist() if len(y_va2) > 0 else '[]')
        va2 = Pool(X_va2, y_va2, cat_features=self.cat_features_) if _va2_ok else None

        s2_params = {**(self.stage2_params or self._default_stage2_params()), 'random_seed': self.random_seed}

        if self.stage2_n_trials > 0 and va2 is not None:
            self.model2_, _ = self._train_with_optuna(
                CatBoostClassifier, tr2, va2, self.stage2_n_trials, is_stage2=True
            )
        else:
            self.model2_ = CatBoostClassifier(**s2_params)
            if va2 is not None:
                self.model2_.fit(tr2, eval_set=va2, verbose=False)
            else:
                self.model2_.fit(tr2, verbose=False)

        # ── Итоговые предсказания ────────────────────────────────────────────
        self.valid_pred_ = self._cascade_score(X_valid[feats], s1_va, candidate_mask_va)
        s1_tr_full = self.model1_.predict_proba(
            Pool(X_train[feats], cat_features=self.cat_features_)
        )[:, 1]
        self.train_pred_ = self._cascade_score(X_train[feats], s1_tr_full, s1_tr_full >= self.threshold1_)

        self.best_params_ = {'threshold1': self.threshold1_, 'recall_target': self.recall_target}
        self._model = True

        logger.info('[Cascade] val PR-AUC=%.4f', average_precision_score(y_va, self.valid_pred_))
        return self

    def _cascade_score(
        self,
        X_feats: pd.DataFrame,
        s1_score: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> np.ndarray:
        """Непрерывный ранкинг: кандидаты в [threshold1, 1], остальные в [0, threshold1)."""
        from catboost import Pool
        final = s1_score.copy()
        if candidate_mask.any():
            X_cand = X_feats[candidate_mask]
            s2 = self.model2_.predict_proba(
                Pool(X_cand, cat_features=self.cat_features_)
            )[:, 1]
            # Отображаем Stage 2 score в [threshold1, 1.0]
            t = self.threshold1_
            final[candidate_mask] = t + (1.0 - t) * s2
        return final

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        s1 = self.model1_.predict_proba(
            Pool(X[self.selected_features_], cat_features=self.cat_features_)
        )[:, 1]
        candidates = s1 >= self.threshold1_
        return self._cascade_score(X[self.selected_features_], s1, candidates)
