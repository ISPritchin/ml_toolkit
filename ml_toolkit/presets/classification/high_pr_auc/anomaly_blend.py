"""AnomalyBlendClassifier: Isolation Forest + supervised CatBoost blending.

Принципиально другой взгляд: не учить классификатор на всех данных,
а комбинировать детекцию аномалий с supervised сигналом.

Идея:
  IsolationForest обучается только на известных позитивах.
  Чем более «нормальным» по меркам позитивов выглядит пример — тем выше его
  anomaly_score (высокий score = похож на позитивов).

  Итоговый скор: α * supervised_score + (1-α) * anomaly_score
  α ищется по val PR-AUC: полный перебор 50 значений [0, 1].

Когда аномальный сигнал помогает:
  - Позитивы образуют компактный кластер в feature space.
  - Supervised модель «переобучилась» на видимые негативы и теряет recall.
  - IsolationForest более устойчив к небольшому числу позитивов,
    потому что он учится ТОЛЬКО на позитивах (без дисбаланса).

Интерпретация alpha_ после fit:
  alpha_ ≈ 1.0 → аномальный сигнал почти не нужен (supervised достаточен).
  alpha_ ≈ 0.5 → оба источника дополняют друг друга.
  alpha_ ≈ 0.0 → только IsolationForest; supervised модель не информативна.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from ml_toolkit.presets.classification._base import BasePreset
from ml_toolkit.presets.classification._optuna_utils import (
    CatBoostPruningCallback,
    catboost_arch_space,
    make_pruner,
)

logger = logging.getLogger(__name__)

_DEFAULT_CBT_PARAMS: dict[str, Any] = {
    'iterations': 700,
    'max_depth': 5,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'min_data_in_leaf': 10,
    'early_stopping_rounds': 100,
    'loss_function': 'Logloss',
    'eval_metric': 'PRAUC',
    'random_seed': 42,
    'verbose': 0,
}


def _minmax_normalize(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    span = hi - lo
    if span < 1e-12:
        return np.full_like(arr, 0.5)
    return np.clip((arr - lo) / span, 0.0, 1.0)


class AnomalyBlendClassifier(BasePreset):
    """Blend Isolation Forest (обученного на позитивах) и CatBoost.

    Parameters
    ----------
    n_if_estimators:
        Число деревьев Isolation Forest. Больше → стабильнее, медленнее.
    supervised_params:
        Параметры CatBoost. None → дефолтные. Игнорируется, если n_optuna_trials > 0.
    n_optuna_trials:
        Если > 0, параметры supervised CatBoost подбираются через Optuna по val
        PR-AUC (до alpha-блендинга с IF) вместо supervised_params/дефолтных.
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
    n_alpha_steps:
        Количество значений alpha при поиске [0, 1]. Дефолт: 51.
    optuna_pruner:
        None/строковый алиас ('median'/'hyperband'/'percentile'/
        'successive_halving'/'none')/готовый optuna.pruners.BasePruner —
        см. ml_toolkit.models model_settings.md. 'none' (по умолчанию) —
        прунинг выключен.
    random_seed:
        Зерно для IF, CatBoost и Optuna sampler'а.

    Атрибуты после fit::

        alpha_          — оптимальное alpha = вес supervised сигнала
        if_pr_auc_      — val PR-AUC аномального сигнала (α=0)
        sup_pr_auc_     — val PR-AUC supervised сигнала (α=1)
        blend_pr_auc_   — val PR-AUC итогового blend
        alpha_scan_df_  — pd.DataFrame (alpha, pr_auc) для всех точек поиска

    Пример::

        model = AnomalyBlendClassifier(n_if_estimators=200)
        model.fit(X_train, y_train, X_valid, y_valid)
        print(f'alpha={model.alpha_:.2f}: sup={model.sup_pr_auc_:.4f}  '
              f'IF={model.if_pr_auc_:.4f}  blend={model.blend_pr_auc_:.4f}')

    """

    def __init__(
        self,
        n_if_estimators: int = 200,
        supervised_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        param_space: Callable[[Any], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        optuna_pruner: str | object | None = 'none',
        n_alpha_steps: int = 51,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(params=supervised_params, n_optuna_trials=n_optuna_trials)
        self.n_if_estimators = n_if_estimators
        self.supervised_params = supervised_params
        self.param_space = param_space
        self.optuna_timeout = optuna_timeout
        self.optuna_verbose = optuna_verbose
        self.optuna_pruner = optuna_pruner
        self.n_alpha_steps = n_alpha_steps
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.alpha_: float = 1.0
        self.if_pr_auc_: float = 0.0
        self.sup_pr_auc_: float = 0.0
        self.blend_pr_auc_: float = 0.0
        self.alpha_scan_df_: pd.DataFrame | None = None

        self._if_model: IsolationForest | None = None
        self._sup_model: Any = None
        self._if_lo: float = 0.0
        self._if_hi: float = 1.0
        self.best_supervised_params_: dict[str, Any] | None = None

    # ── Isolation Forest ──────────────────────────────────────────────────────

    def _fit_isolation_forest(
        self, X_pos: np.ndarray
    ) -> IsolationForest:
        return IsolationForest(
            n_estimators=self.n_if_estimators,
            contamination='auto',
            random_state=self.random_seed,
        ).fit(X_pos)

    def _if_score(self, X: np.ndarray) -> np.ndarray:
        # score_samples: более высокий → менее аномальный (= более похож на позитивов IF)
        raw = self._if_model.score_samples(X)
        return _minmax_normalize(raw, self._if_lo, self._if_hi)

    # ── Supervised ────────────────────────────────────────────────────────────

    def _fit_catboost(
        self,
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
        X_va: pd.DataFrame,
        y_va: np.ndarray,
    ) -> Any:
        from catboost import CatBoostClassifier, Pool

        tr_pool = Pool(X_tr, y_tr, cat_features=self.cat_features_)
        va_pool = Pool(X_va, y_va, cat_features=self.cat_features_)
        if self.n_optuna_trials > 0:
            params = self._tune(tr_pool, va_pool, y_va)
        else:
            params = {**(self.supervised_params or _DEFAULT_CBT_PARAMS), 'random_seed': self.random_seed}
        model = CatBoostClassifier(**params)
        model.fit(tr_pool, eval_set=va_pool, verbose=False)
        self.best_supervised_params_ = params
        return model

    def _tune(self, tr_pool: Any, va_pool: Any, y_va: np.ndarray) -> dict[str, Any]:
        from catboost import CatBoostClassifier
        import optuna

        _optuna_prev_verbosity = optuna.logging.get_verbosity()
        if not self.optuna_verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            tunable = self.param_space(trial) if self.param_space is not None else catboost_arch_space(trial)
            params = {
                'loss_function': 'Logloss',
                'eval_metric': 'PRAUC',
                'early_stopping_rounds': 100,
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

        logger.info('[AnomalyBlend] Optuna: %d trials (supervised CatBoost)', self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed),
                                    pruner=make_pruner(self.optuna_pruner))
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return dict(study.best_trial.user_attrs['cb_params'])

    def _sup_score(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        pool = Pool(X[self.selected_features_], cat_features=self.cat_features_)
        return self._sup_model.predict_proba(pool)[:, 1]

    # ── Alpha search ──────────────────────────────────────────────────────────

    def _find_alpha(
        self, sup_va: np.ndarray, if_va: np.ndarray, y_va: np.ndarray
    ) -> tuple[float, pd.DataFrame]:
        alphas = np.linspace(0.0, 1.0, self.n_alpha_steps)
        scores = np.array([
            average_precision_score(y_va, a * sup_va + (1 - a) * if_va)
            for a in alphas
        ])
        best_idx = int(np.argmax(scores))
        scan_df = pd.DataFrame({'alpha': alphas, 'pr_auc': scores})
        return float(alphas[best_idx]), scan_df

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> AnomalyBlendClassifier:
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values
        X_tr_feats = X_train[feats]
        X_va_feats = X_valid[feats]

        # Числовые признаки для IF (no cats)
        num_feats = [f for f in feats if f not in self.cat_features_]

        n_pos = int(y_tr.sum())
        logger.info(
            '[AnomalyBlend] n_pos_train=%d  n_if_estimators=%d  num_feats_for_IF=%d',
            n_pos, self.n_if_estimators, len(num_feats),
        )

        # ── 1. Isolation Forest на train-позитивах ─────────────────────────
        X_pos_np = X_tr_feats[num_feats].values[y_tr == 1]
        self._if_model = self._fit_isolation_forest(X_pos_np)

        # Калибруем диапазон нормализации по train (позитивы + негативы)
        all_if_raw = self._if_model.score_samples(X_tr_feats[num_feats].values)
        self._if_lo = float(all_if_raw.min())
        self._if_hi = float(all_if_raw.max())

        if_va = self._if_score(X_va_feats[num_feats].values)
        self.if_pr_auc_ = float(average_precision_score(y_va, if_va))

        # ── 2. CatBoost supervised ─────────────────────────────────────────
        self._sup_model = self._fit_catboost(X_tr_feats, y_tr, X_va_feats, y_va)
        sup_va = self._sup_score(X_valid)
        self.sup_pr_auc_ = float(average_precision_score(y_va, sup_va))

        logger.info(
            '[AnomalyBlend] IF PR-AUC=%.4f  supervised PR-AUC=%.4f',
            self.if_pr_auc_, self.sup_pr_auc_,
        )

        # ── 3. Поиск оптимального alpha ────────────────────────────────────
        self.alpha_, self.alpha_scan_df_ = self._find_alpha(sup_va, if_va, y_va)
        blend_va = self.alpha_ * sup_va + (1 - self.alpha_) * if_va
        self.blend_pr_auc_ = float(average_precision_score(y_va, blend_va))

        logger.info(
            '[AnomalyBlend] alpha=%.3f (sup_w=%.3f  if_w=%.3f)  blend PR-AUC=%.4f',
            self.alpha_, self.alpha_, 1 - self.alpha_, self.blend_pr_auc_,
        )

        self.valid_pred_ = blend_va

        from catboost import Pool as _Pool

        if_tr = self._if_score(X_tr_feats[num_feats].values)
        sup_tr = self._sup_model.predict_proba(
            _Pool(X_tr_feats, y_tr, cat_features=self.cat_features_)
        )[:, 1]
        self.train_pred_ = self.alpha_ * sup_tr + (1 - self.alpha_) * if_tr

        self.best_params_ = {
            'alpha': self.alpha_,
            'n_if_estimators': self.n_if_estimators,
            'if_pr_auc': self.if_pr_auc_,
            'sup_pr_auc': self.sup_pr_auc_,
            'supervised_params': self.best_supervised_params_,
        }
        self._model = True
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        num_feats = [f for f in self.selected_features_ if f not in self.cat_features_]
        if_s = self._if_score(X[num_feats].values)
        sup_s = self._sup_score(X)
        return self.alpha_ * sup_s + (1 - self.alpha_) * if_s

    def plot_alpha_scan(self, ax: Any = None, path: str | None = None) -> None:
        """График PR-AUC по всем значениям alpha с отмеченным оптимумом."""
        import matplotlib.pyplot as plt

        if self.alpha_scan_df_ is None:
            raise RuntimeError('Вызовите .fit() перед plot_alpha_scan()')

        df = self.alpha_scan_df_
        fig, ax_ = (plt.subplots(figsize=(8, 4)) if ax is None else (ax.get_figure(), ax))
        ax_.plot(df['alpha'], df['pr_auc'], color='steelblue', lw=1.5)
        ax_.axvline(self.alpha_, color='red', linestyle='--', lw=1.5,
                    label=f'optimal α={self.alpha_:.3f}  PR-AUC={self.blend_pr_auc_:.4f}')
        ax_.axhline(self.sup_pr_auc_, color='gray', linestyle=':', alpha=0.7,
                    label=f'supervised only={self.sup_pr_auc_:.4f}')
        ax_.axhline(self.if_pr_auc_, color='orange', linestyle=':', alpha=0.7,
                    label=f'IF only={self.if_pr_auc_:.4f}')
        ax_.set_xlabel('alpha (weight of supervised)')
        ax_.set_ylabel('val PR-AUC')
        ax_.set_title('AnomalyBlend: alpha search')
        ax_.legend()
        plt.tight_layout()
        if path:
            fig.savefig(path, dpi=150, bbox_inches='tight')
        elif ax is None:
            plt.show()
        if ax is None:
            plt.close(fig)
