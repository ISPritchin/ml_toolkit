"""StabilitySelectionClassifier: отбор признаков по устойчивости важности на бутстрэпах.

Идея (Stability Selection, Meinshausen & Buhlmann, адаптировано под важности
градиентного бустинга вместо коэффициентов лассо): обучаем n_bootstrap лёгких
CatBoost на стратифицированных бутстрэп-подвыборках train, на каждой смотрим
топ-top_k признаков по PredictionValuesChange importance. Признак считается
стабильным, если он попадает в топ хотя бы в freq_threshold доле бутстрэпов.
Финальная модель обучается один раз на полном train, но только по стабильному
ядру признаков.

Когда: важность признаков заметно скачет между запусками (типично при сотнях
коррелированных инженерных признаков после feature generation), а бизнесу нужен
воспроизводимый, интерпретируемый набор фичей — а не максимум PR-AUC любой ценой.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.models._base import XInput, YInput
from ml_toolkit.models._utils import fit_calibrator
from ml_toolkit.presets.classification._base import BasePreset
from ml_toolkit.presets.classification._optuna_utils import (
    CatBoostPruningCallback,
    catboost_arch_space,
    make_pruner,
)

if TYPE_CHECKING:
    from catboost import Pool

logger = logging.getLogger(__name__)

_DEFAULT_BOOTSTRAP_PARAMS: dict[str, Any] = {
    'iterations': 300,
    'max_depth': 5,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'min_data_in_leaf': 10,
    'loss_function': 'Logloss',
    'verbose': 0,
}

_DEFAULT_FINAL_PARAMS: dict[str, Any] = {
    'iterations': 700,
    'max_depth': 5,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'min_data_in_leaf': 10,
    'early_stopping_rounds': 100,
    'loss_function': 'Logloss',
    'eval_metric': 'PRAUC',
    'verbose': 0,
}


class StabilitySelectionClassifier(BasePreset):
    """Отбор устойчивого ядра признаков через бутстрэп-важности + финальный CatBoost.

    Parameters
    ----------
    n_bootstrap:
        Число бутстрэп-повторов для оценки важности признаков.
    top_k:
        Сколько признаков (по убыванию важности) считаются «топовыми» в каждом
        бутстрэпе. Обрезается до len(selected_features), если задан больше.
    freq_threshold:
        Минимальная доля бутстрэпов, в топ-k которых должен попасть признак,
        чтобы попасть в стабильное ядро.
    bootstrap_params:
        Параметры CatBoost для быстрых бутстрэп-моделей (используются только для
        важностей). None → дефолтные, меньше iterations и без early stopping —
        здесь важна стабильность важности, а не итоговое качество вероятностей.
    final_params:
        Параметры финальной CatBoost-модели на стабильном ядре. None → дефолтные.
        Игнорируется, если n_optuna_trials > 0.
    n_optuna_trials:
        Если > 0, параметры финальной модели (на стабильном ядре признаков)
        подбираются через Optuna по val PR-AUC вместо final_params/дефолтных.
        bootstrap_params не затрагиваются — там важна скорость и стабильность
        importance-оценок, а не итоговое качество вероятностей.
    param_space:
        Кастомная функция `f(trial) -> dict` — search space для Optuna (только
        для финальной модели) вместо дефолтного. Может как включать только
        часть тюнящихся параметров (недостающие из loss_function/eval_metric/
        early_stopping_rounds/random_seed/verbose подставляются дефолтами),
        так и переопределять любой из них — то, что вернула param_space,
        имеет приоритет над дефолтами. Действует только при n_optuna_trials > 0.
        None → дефолтный search space.
    optuna_timeout:
        Ограничение по времени (сек) на весь Optuna-поиск. None — без ограничения.
    optuna_verbose:
        Если True — не глушит логи Optuna. Если False (по умолчанию) —
        форсирует WARNING на время поиска.
    calibrate:
        Применять ли изотоническую калибровку к финальным вероятностям.
    optuna_pruner:
        None/строковый алиас ('median'/'hyperband'/'percentile'/
        'successive_halving'/'none')/готовый optuna.pruners.BasePruner —
        см. ml_toolkit.models model_settings.md. 'none' (по умолчанию) —
        прунинг выключен.
    random_seed:
        Начальное зерно. Бутстрэп i использует seed + i. Также сид Optuna sampler'а.

    Атрибуты после fit::

        selection_freq_   — pd.Series: доля бутстрэпов, где признак попал в топ-k,
                             по убыванию; индекс — все признаки из selected_features
        stable_features_  — список признаков, прошедших freq_threshold

    Пример::

        model = StabilitySelectionClassifier(n_bootstrap=50, top_k=20, freq_threshold=0.6)
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=[...])
        print(model.stable_features_)
        proba = model.predict_proba(X_test)

    """

    def __init__(
        self,
        n_bootstrap: int = 50,
        top_k: int = 20,
        freq_threshold: float = 0.6,
        bootstrap_params: dict[str, Any] | None = None,
        final_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        param_space: Callable[[Any], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        optuna_pruner: str | object | None = 'none',
        calibrate: bool = True,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        if not 0.0 < freq_threshold <= 1.0:
            raise ValueError(f'freq_threshold должен быть в (0, 1], получено {freq_threshold}')
        super().__init__(params=final_params, n_optuna_trials=n_optuna_trials)
        self.n_bootstrap = n_bootstrap
        self.top_k = top_k
        self.freq_threshold = freq_threshold
        self.bootstrap_params = bootstrap_params
        self.final_params = final_params
        self.param_space = param_space
        self.optuna_timeout = optuna_timeout
        self.optuna_verbose = optuna_verbose
        self.optuna_pruner = optuna_pruner
        self.calibrate = calibrate
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.selection_freq_: pd.Series | None = None
        self.stable_features_: list[str] = []

    def _tune(self, tr_pool: Pool, va_pool: Pool, y_va: np.ndarray) -> dict[str, Any]:
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

        logger.info('[StabilitySelection] Optuna: %d trials (финальная модель на стабильном ядре)',
                    self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed),
                                    pruner=make_pruner(self.optuna_pruner))
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return dict(study.best_trial.user_attrs['cb_params'])

    def _stratified_bootstrap(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Бутстрэп (с возвратом) отдельно по каждому классу.

        Иначе редкий позитивный класс рискует полностью выпасть из подвыборки
        при сильном дисбалансе.
        """
        parts = [
            rng.choice(np.where(y == cls)[0], size=int((y == cls).sum()), replace=True)
            for cls in np.unique(y)
        ]
        return np.concatenate(parts)

    def fit(
        self,
        X_train: XInput,
        y_train: YInput,
        X_valid: XInput,
        y_valid: YInput,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> StabilitySelectionClassifier:
        from catboost import CatBoostClassifier, Pool

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values

        top_k = min(self.top_k, len(feats))
        boot_params = self.bootstrap_params or _DEFAULT_BOOTSTRAP_PARAMS
        selection_counts = pd.Series(0.0, index=feats)

        logger.info('[StabilitySelection] n_bootstrap=%d  top_k=%d/%d  freq_threshold=%.2f',
                    self.n_bootstrap, top_k, len(feats), self.freq_threshold)

        log_every = max(1, self.n_bootstrap // 5)
        for i in range(self.n_bootstrap):
            rng = np.random.default_rng(self.random_seed + i)
            boot_idx = self._stratified_bootstrap(y_tr, rng)

            boot_pool = Pool(
                X_train[feats].iloc[boot_idx], y_tr[boot_idx], cat_features=self.cat_features_
            )
            m = CatBoostClassifier(**{**boot_params, 'random_seed': self.random_seed + i})
            m.fit(boot_pool, verbose=False)

            importances = pd.Series(m.get_feature_importance(boot_pool), index=feats)
            top_feats = importances.sort_values(ascending=False).index[:top_k]
            selection_counts.loc[top_feats] += 1.0

            if (i + 1) % log_every == 0:
                logger.info('[StabilitySelection] бутстрэп %d/%d готов', i + 1, self.n_bootstrap)

        self.selection_freq_ = (selection_counts / self.n_bootstrap).sort_values(ascending=False)
        self.stable_features_ = [
            f for f in feats if self.selection_freq_[f] >= self.freq_threshold
        ]
        if not self.stable_features_:
            raise ValueError(
                f'Ни один признак не набрал freq_threshold={self.freq_threshold} '
                f'(top_k={top_k}, n_bootstrap={self.n_bootstrap}). '
                'Снизьте freq_threshold или увеличьте top_k.'
            )
        preview = ', '.join(self.stable_features_[:10])
        if len(self.stable_features_) > 10:
            preview += ', ...'
        logger.info('[StabilitySelection] стабильное ядро: %d/%d признаков (%s)',
                    len(self.stable_features_), len(feats), preview)

        stable_cat = [c for c in self.cat_features_ if c in self.stable_features_]
        tr_pool = Pool(X_train[self.stable_features_], y_tr, cat_features=stable_cat)
        va_pool = Pool(X_valid[self.stable_features_], y_va, cat_features=stable_cat)

        if self.n_optuna_trials > 0:
            final_params = self._tune(tr_pool, va_pool, y_va)
        else:
            final_params = {**(self.final_params or _DEFAULT_FINAL_PARAMS), 'random_seed': self.random_seed}
        self._model = CatBoostClassifier(**final_params)
        self._model.fit(tr_pool, eval_set=va_pool, verbose=False)

        raw_va = self._model.predict_proba(va_pool)[:, 1]
        self.train_pred_ = self._model.predict_proba(tr_pool)[:, 1]
        if self.calibrate:
            self.calibrator_ = fit_calibrator(raw_va, y_va)
            self.valid_pred_ = self.calibrator_.predict(raw_va)
        else:
            self.valid_pred_ = raw_va

        self.best_params_ = {
            'n_bootstrap': self.n_bootstrap, 'top_k': top_k,
            'freq_threshold': self.freq_threshold,
            'n_stable_features': len(self.stable_features_),
            'final_params': final_params,
        }
        logger.info('[StabilitySelection] финал val PR-AUC=%.4f',
                    average_precision_score(y_va, self.valid_pred_))
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        stable_cat = [c for c in self.cat_features_ if c in self.stable_features_]
        pool = Pool(X[self.stable_features_], cat_features=stable_cat)
        raw = self._model.predict_proba(pool)[:, 1]
        if self.calibrate and self.calibrator_ is not None:
            return self.calibrator_.predict(raw)
        return raw
