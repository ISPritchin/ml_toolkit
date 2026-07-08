"""BoostedEnsemble: ансамбль CatBoost-моделей с разными функциями потерь.

Стратегии усреднения:
  'mean'     — простое среднее вероятностей.
  'rank'     — среднее нормализованных рангов (устойчиво к разным масштабам).
  'weighted' — веса оптимизированы по PR-AUC на val через Optuna (100 триалов).
  'power'    — generalized mean с alpha, подобранным по val.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.losses import FocalLoss as _FocalLoss
from ml_toolkit.models._utils import fit_rank_reference, rank_transform
from ml_toolkit.presets.classification._base import BasePreset
from ml_toolkit.presets.classification._optuna_utils import (
    CatBoostPruningCallback,
    catboost_arch_space,
    make_pruner,
)

logger = logging.getLogger(__name__)


# ─── Стратегии усреднения ────────────────────────────────────────────────────

def _optimize_weights(probas: list[np.ndarray], y_val: np.ndarray, random_seed: int = 42) -> np.ndarray:
    import optuna
    _optuna_prev_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n = len(probas)

    def objective(trial: optuna.Trial) -> float:
        raw = np.array([trial.suggest_float(f'w{i}', 0.0, 1.0) for i in range(n)])
        w = raw / (raw.sum() + 1e-9)
        blend = sum(wi * pi for wi, pi in zip(w, probas))
        return float(average_precision_score(y_val, blend))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=100, show_progress_bar=False)
    raw = np.array([study.best_params[f'w{i}'] for i in range(n)])
    optuna.logging.set_verbosity(_optuna_prev_verbosity)
    return raw / (raw.sum() + 1e-9)


def _fit_power_alpha(probas: list[np.ndarray], y_val: np.ndarray, random_seed: int = 42) -> float:
    import optuna
    _optuna_prev_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float('alpha', 0.1, 5.0, log=True)
        clipped = [np.clip(p, 1e-9, 1.0 - 1e-9) for p in probas]
        blend = np.mean([p ** alpha for p in clipped], axis=0) ** (1.0 / alpha)
        return float(average_precision_score(y_val, blend))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    optuna.logging.set_verbosity(_optuna_prev_verbosity)
    return float(study.best_params['alpha'])


def _get_proba(model, pool) -> np.ndarray:
    """predict_proba с fallback на sigmoid(raw) для кастомных лоссов."""
    from catboost import CatBoostError

    try:
        raw = model.predict_proba(pool)
        if raw.ndim == 2:
            return raw[:, 1]
        return 1.0 / (1.0 + np.exp(-raw.astype(np.float64)))
    except CatBoostError:
        # Модели с кастомным Python-лоссом не поддерживают predict_proba
        raw = model.predict(pool, prediction_type='RawFormulaVal')
        return 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float64)))


# ─── Дефолтные конфиги ──────────────────────────────────────────────────────

def _default_loss_configs():
    return [
        {'loss_function': 'Logloss', 'scale_pos_weight': 1.0, 'random_seed': 42},
        {'loss_function': 'Logloss', 'scale_pos_weight': 4.0, 'random_seed': 123},
        {'loss_function': _FocalLoss(gamma=1.0, alpha=0.5), 'eval_metric': 'AUC', 'random_seed': 42},
        {'loss_function': _FocalLoss(gamma=2.5, alpha=0.25), 'eval_metric': 'AUC', 'random_seed': 456},
    ]


DEFAULT_BASE_PARAMS: dict[str, Any] = {
    'iterations': 700,
    'max_depth': 5,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'min_data_in_leaf': 10,
    'early_stopping_rounds': 100,
    'eval_metric': 'PRAUC',
    'verbose': 0,
}


# ─── Класс ──────────────────────────────────────────────────────────────────

class BoostedEnsemble(BasePreset):
    """Ансамбль CatBoost с разными loss-функциями и умным усреднением.

    Parameters
    ----------
    loss_configs:
        Список словарей; каждый мёрджится с base_params перед обучением модели.
        По умолчанию — 4 конфига: Logloss×2 + FocalLoss×2.
    averaging:
        'mean', 'rank', 'weighted' (Optuna по val), 'power' (alpha по val).
    base_params:
        Общие параметры CatBoost (iterations, max_depth и т.д.).
        Конфиги из loss_configs переопределяют эти параметры. Игнорируется, если
        n_optuna_trials > 0.
    n_optuna_trials:
        Если > 0, общая часть архитектуры (base_params: iterations, max_depth,
        learning_rate, l2_leaf_reg, subsample, min_data_in_leaf) подбирается через
        Optuna по val PR-AUC на первом конфиге ансамбля (Logloss, scale_pos_weight=1.0),
        вместо дефолтных base_params. per-конфиг разнообразие (loss_function,
        scale_pos_weight, random_seed из loss_configs) не затрагивается.
    param_space:
        Кастомная функция `f(trial) -> dict` — search space для общей части
        base_params вместо дефолтного. Может как включать только часть
        тюнящихся параметров (недостающие из early_stopping_rounds/
        loss_function/eval_metric/scale_pos_weight/random_seed/verbose
        подставляются дефолтами), так и переопределять любой из них для
        ЦЕЛЕЙ САМОГО ПОИСКА (например, оценивать архитектуру под другой
        loss_function). loss_function/scale_pos_weight/random_seed при этом
        всё равно НЕ попадают в возвращаемый base_params — это per-конфиг
        diversity-оси (см. loss_configs), и пропускать их дальше опасно:
        scale_pos_weight, например, несовместим с кастомным Python-лоссом
        (FocalLoss и т.п.) в части loss_configs. eval_metric/early_stopping_
        rounds/verbose и вся архитектура — пропускаются в base_params как есть.
        Действует только при n_optuna_trials > 0. None → дефолтный search space.
    optuna_timeout:
        Ограничение по времени (сек) на весь Optuna-поиск. None — без ограничения.
    optuna_verbose:
        Если True — не глушит логи Optuna. Если False (по умолчанию) —
        форсирует WARNING на время поиска.
    random_seed:
        Зерно Optuna sampler'а для averaging='weighted'/'power' (подбор весов/alpha
        блендинга). Отдельные модели ансамбля намеренно используют разные seed'ы
        (см. loss_configs) — это создаёт разнообразие внутри ансамбля, а не
        несогласованность.

    Пример::

        model = BoostedEnsemble(averaging='rank')
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=[...])
        proba = model.predict_proba(X_test)

    """

    def __init__(
        self,
        loss_configs: list[dict] | None = None,
        averaging: str = 'rank',
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        param_space: Callable[[Any], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ):
        super().__init__(params=None, n_optuna_trials=n_optuna_trials)
        self.loss_configs = loss_configs  # None → ленивый дефолт в fit()
        self.averaging = averaging
        self.base_params = base_params or dict(DEFAULT_BASE_PARAMS)
        self.param_space = param_space
        self.optuna_timeout = optuna_timeout
        self.optuna_verbose = optuna_verbose
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.models_: list = []
        self._weights: np.ndarray | None = None
        self._power_alpha: float = 1.0
        self._rank_refs_: list[np.ndarray] = []

    def _tune_base_params(self, tr_pool: Any, va_pool: Any, y_va: np.ndarray) -> dict[str, Any]:
        from catboost import CatBoostClassifier
        import optuna

        _optuna_prev_verbosity = optuna.logging.get_verbosity()
        if not self.optuna_verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            tunable = self.param_space(trial) if self.param_space is not None else catboost_arch_space(trial)
            params = {
                'early_stopping_rounds': 100,
                'loss_function': 'Logloss',
                'eval_metric': 'PRAUC',
                'scale_pos_weight': 1.0,
                'random_seed': 42,
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

        logger.info('[BoostedEnsemble] Optuna: %d trials (общая часть base_params для всех конфигов)',
                    self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed),
                                    pruner=make_pruner())
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        best = dict(study.best_trial.user_attrs['cb_params'])
        # loss_function/scale_pos_weight/random_seed остаются per-конфиг diversity-осями
        # (см. loss_configs) — не пропускаем их в base_params, иначе, например,
        # scale_pos_weight=1.0 просочится в конфиги с кастомным Python-лоссом
        # (FocalLoss и т.п.), для которых CatBoost его не поддерживает вовсе.
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return {k: v for k, v in best.items() if k not in ('loss_function', 'scale_pos_weight', 'random_seed')}

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> BoostedEnsemble:
        from catboost import CatBoostClassifier, Pool

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values

        va_pool = Pool(X_valid[feats], y_va, cat_features=self.cat_features_)
        loss_configs = self.loss_configs or _default_loss_configs()

        base_params = self.base_params
        if self.n_optuna_trials > 0:
            tr_pool_full = Pool(X_train[feats], y_tr, cat_features=self.cat_features_)
            base_params = self._tune_base_params(tr_pool_full, va_pool, y_va)

        tr_probas, va_probas = [], []
        self.models_ = []

        for i, cfg in enumerate(loss_configs):
            params = {**base_params, **cfg}
            # CatBoost требует eval_metric при кастомном loss_function
            if not isinstance(params.get('loss_function', 'Logloss'), str):
                params.setdefault('eval_metric', 'AUC')
            logger.info('[BoostedEnsemble] Модель %d/%d  loss_cfg=%s', i + 1, len(loss_configs), cfg)

            tr_pool = Pool(X_train[feats], y_tr, cat_features=self.cat_features_)
            m = CatBoostClassifier(**params)
            m.fit(tr_pool, eval_set=va_pool, verbose=False)

            va_p = _get_proba(m, va_pool)
            tr_p = _get_proba(m, Pool(X_train[feats], cat_features=self.cat_features_))
            logger.info('[BoostedEnsemble] Модель %d  val PR-AUC=%.4f', i + 1,
                        average_precision_score(y_va, va_p))

            self.models_.append(m)
            va_probas.append(va_p)
            tr_probas.append(tr_p)

        # Референсы rank-усреднения — train-вероятности каждой модели; predict
        # использует их же, поэтому скор не зависит от состава батча.
        self._rank_refs_ = [fit_rank_reference(p) for p in tr_probas]

        self.valid_pred_ = self._blend(va_probas, y_va, fit_blend=True)
        self.train_pred_ = self._blend(tr_probas, fit_blend=False)
        self.best_params_ = {'averaging': self.averaging, 'n_models': len(self.models_), 'base_params': base_params}
        self._model = True  # sentinel for _check_fitted

        logger.info('[BoostedEnsemble] Ансамбль val PR-AUC=%.4f',
                    average_precision_score(y_va, self.valid_pred_))
        return self

    def _blend(
        self,
        probas: list[np.ndarray],
        y_val: np.ndarray | None = None,
        fit_blend: bool = False,
    ) -> np.ndarray:
        if self.averaging == 'mean':
            return np.mean(probas, axis=0)

        if self.averaging == 'rank':
            return np.mean(
                [rank_transform(p, ref) for p, ref in zip(probas, self._rank_refs_)],
                axis=0,
            )

        if self.averaging == 'weighted':
            if fit_blend and y_val is not None:
                self._weights = _optimize_weights(probas, y_val, random_seed=self.random_seed)
                logger.info('[BoostedEnsemble] Оптимальные веса: %s', np.round(self._weights, 3))
            w = self._weights if self._weights is not None else np.ones(len(probas)) / len(probas)
            return sum(wi * pi for wi, pi in zip(w, probas))

        if self.averaging == 'power':
            if fit_blend and y_val is not None:
                self._power_alpha = _fit_power_alpha(probas, y_val, random_seed=self.random_seed)
                logger.info('[BoostedEnsemble] Power alpha=%.3f', self._power_alpha)
            alpha = self._power_alpha
            clipped = [np.clip(p, 1e-9, 1.0 - 1e-9) for p in probas]
            return np.mean([p ** alpha for p in clipped], axis=0) ** (1.0 / alpha)

        raise ValueError(f"Неизвестный averaging={self.averaging!r}. "
                         "Допустимые: 'mean', 'rank', 'weighted', 'power'.")

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        probas = [
            _get_proba(m, Pool(X[self.selected_features_], cat_features=self.cat_features_))
            for m in self.models_
        ]
        return self._blend(probas, fit_blend=False)
