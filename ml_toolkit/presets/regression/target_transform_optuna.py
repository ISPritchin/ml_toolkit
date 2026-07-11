"""TargetTransformOptunaRegressor: Optuna выбирает трансформ таргета вместе с гиперпараметрами CatBoost.

Масштаб/скошенность таргета часто неочевидны заранее (деньги, обороты, счётные
величины с тяжёлым хвостом) — вместо того чтобы гадать identity/log1p/box-cox/
yeo-johnson/quantile руками, Optuna перебирает трансформ как ещё один
гиперпараметр в общем пространстве поиска вместе с архитектурой CatBoost.

Ключевой момент корректности: и подбор (objective trial), и итоговый predict()
всегда сравниваются/возвращаются в ИСХОДНОМ масштабе таргета — модель обучается
в трансформированном пространстве (CatBoost видит g(y)), но score считается на
g^{-1}(f(x)) против сырых y_valid, иначе трансформы с разным масштабом ошибки
(например, log-пространство vs исходное) были бы несравнимы между собой и выбор
Optuna был бы бессмысленным.

Для log1p — единственного трансформа с систематическим смещением при обратном
преобразовании (g^{-1}=expm1 выпукла, из неравенства Йенсена g^{-1}(E[g(y)]) <
E[y]) — применяется smearing-поправка Duan: g^{-1}(f(x) + log(mean(exp(e_i)))),
где e_i = g(y_i) - f(x_i) — остатки обучающей выборки в трансформированном
пространстве. Без неё predict() систематически недооценивал бы y.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ml_toolkit.models._base import XInput, YInput
from ml_toolkit.presets.regression._base import BasePreset
from ml_toolkit.presets.regression._optuna_utils import (
    CatBoostPruningCallback,
    catboost_arch_space,
    make_pruner,
)

if TYPE_CHECKING:
    from catboost import CatBoostRegressor

logger = logging.getLogger(__name__)

_DEFAULT_ARCH_PARAMS: dict[str, Any] = {
    'iterations': 800,
    'max_depth': 6,
    'learning_rate': 0.03,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'min_data_in_leaf': 5,
    'early_stopping_rounds': 80,
    'verbose': 0,
}

ALL_TRANSFORMS = ('identity', 'log1p', 'box-cox', 'yeo-johnson', 'quantile')


# ── трансформы ───────────────────────────────────────────────────────────────

class _IdentityTransform:
    def fit(self, y: np.ndarray) -> None:
        pass

    def transform(self, y: np.ndarray) -> np.ndarray:
        return y

    def inverse_transform(self, yt: np.ndarray) -> np.ndarray:
        return yt


class _Log1pTransform:
    """inverse_transform применяет Duan smearing (см. докстринг модуля)."""

    def __init__(self) -> None:
        self.smear_ = 1.0

    def fit(self, y: np.ndarray) -> None:
        pass

    def transform(self, y: np.ndarray) -> np.ndarray:
        return np.log1p(y)

    def fit_smear(self, residuals: np.ndarray) -> None:
        self.smear_ = float(np.mean(np.exp(residuals)))

    def inverse_transform(self, yt: np.ndarray) -> np.ndarray:
        return np.expm1(yt + np.log(self.smear_))


class _BoxCoxTransform:
    def __init__(self) -> None:
        self.lmbda_: float = 0.0

    def fit(self, y: np.ndarray) -> None:
        from scipy.stats import boxcox
        _, self.lmbda_ = boxcox(y)

    def transform(self, y: np.ndarray) -> np.ndarray:
        from scipy.stats import boxcox
        return boxcox(y, lmbda=self.lmbda_)

    def inverse_transform(self, yt: np.ndarray) -> np.ndarray:
        from scipy.special import inv_boxcox
        return inv_boxcox(yt, self.lmbda_)


class _YeoJohnsonTransform:
    def __init__(self) -> None:
        from sklearn.preprocessing import PowerTransformer
        self._pt = PowerTransformer(method='yeo-johnson')

    def fit(self, y: np.ndarray) -> None:
        self._pt.fit(y.reshape(-1, 1))

    def transform(self, y: np.ndarray) -> np.ndarray:
        return self._pt.transform(y.reshape(-1, 1)).ravel()

    def inverse_transform(self, yt: np.ndarray) -> np.ndarray:
        return self._pt.inverse_transform(yt.reshape(-1, 1)).ravel()


class _QuantileTransform:
    def __init__(self, n_train: int) -> None:
        from sklearn.preprocessing import QuantileTransformer
        self._qt = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=min(1000, n_train),
        )

    def fit(self, y: np.ndarray) -> None:
        self._qt.fit(y.reshape(-1, 1))

    def transform(self, y: np.ndarray) -> np.ndarray:
        return self._qt.transform(y.reshape(-1, 1)).ravel()

    def inverse_transform(self, yt: np.ndarray) -> np.ndarray:
        return self._qt.inverse_transform(yt.reshape(-1, 1)).ravel()


_Transform = (
    _IdentityTransform | _Log1pTransform | _BoxCoxTransform | _YeoJohnsonTransform | _QuantileTransform
)


def _build_transform(name: str, n_train: int) -> _Transform:
    if name == 'identity':
        return _IdentityTransform()
    if name == 'log1p':
        return _Log1pTransform()
    if name == 'box-cox':
        return _BoxCoxTransform()
    if name == 'yeo-johnson':
        return _YeoJohnsonTransform()
    if name == 'quantile':
        return _QuantileTransform(n_train)
    raise ValueError(f'Неизвестный transform={name!r}. Доступные: {ALL_TRANSFORMS}')


def _valid_transforms(transforms: list[str], y: np.ndarray) -> list[str]:
    """Отфильтровывает трансформы, несовместимые с диапазоном y_train (иначе.

    Optuna рано или поздно выберет box-cox/log1p на данных с y <= 0/-1 и trial
    упадёт с исключением scipy/numpy на середине поиска; в прямом режиме
    (без Optuna) фильтр даёт «первый совместимый» вместо жёсткого transforms[0]).
    """
    y_min = float(np.min(y))
    out = []
    for name in transforms:
        if name == 'box-cox' and y_min <= 0:
            logger.warning('[TargetTransformOptuna] box-cox исключён: min(y_train)=%.4g <= 0', y_min)
            continue
        if name == 'log1p' and y_min <= -1:
            logger.warning('[TargetTransformOptuna] log1p исключён: min(y_train)=%.4g <= -1', y_min)
            continue
        out.append(name)
    if not out:
        raise ValueError(
            f'Ни один из transforms={transforms} не совместим с диапазоном y_train '
            f'(min={y_min:.4g}). Используйте identity/yeo-johnson/quantile.'
        )
    return out


class TargetTransformOptunaRegressor(BasePreset):
    """CatBoost, где Optuna выбирает трансформ таргета вместе с гиперпараметрами.

    Parameters
    ----------
    transforms:
        Список трансформов-кандидатов для Optuna (`n_optuna_trials > 0`) —
        подмножество/перестановка ALL_TRANSFORMS = ('identity', 'log1p',
        'box-cox', 'yeo-johnson', 'quantile'). Несовместимые с диапазоном
        y_train (box-cox/log1p при y <= 0 / y <= -1) отбрасываются автоматически
        с предупреждением в лог. При `n_optuna_trials == 0` используется первый
        совместимый трансформ из списка (без поиска) — задайте список из одного
        имени, если нужен конкретный трансформ без права на автозамену.
    base_params:
        Параметры CatBoost для прямого режима (`n_optuna_trials == 0`).
    n_optuna_trials:
        Число Optuna trials. 0 → прямой режим с transforms[0]/base_params.
    param_space:
        Кастомная функция `f(trial) -> dict`, может переопределить ключ
        'transform' (иначе выбирается trial.suggest_categorical по transforms)
        и/или любой архитектурный ключ CatBoost. Действует только при
        n_optuna_trials > 0.
    optuna_timeout / optuna_verbose / optuna_pruner / random_seed:
        См. другие Optuna-пресеты пакета.

    Атрибуты после fit::

        transform_name_  — выбранное имя трансформа
        transform_       — обученный объект трансформа (нужен predict())

    Пример::

        model = TargetTransformOptunaRegressor(n_optuna_trials=20)
        model.fit(X_train, y_train, X_valid, y_valid)
        pred = model.predict(X_test)               # уже в исходном масштабе y
        print(model.transform_name_, model.best_params_)

    """

    def __init__(
        self,
        transforms: list[str] | None = None,
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        param_space: Callable[[Any], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        optuna_pruner: str | object | None = 'none',
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(params=None, n_optuna_trials=n_optuna_trials)
        self.transforms = list(transforms) if transforms else list(ALL_TRANSFORMS)
        self.base_params = base_params
        self.param_space = param_space
        self.optuna_timeout = optuna_timeout
        self.optuna_verbose = optuna_verbose
        self.optuna_pruner = optuna_pruner
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.transform_name_: str | None = None
        self.transform_: Any = None

    # ── обучение одной модели с фиксированным трансформом ──────────────────

    def _fit_one(
        self,
        X_train: pd.DataFrame, y_tr: np.ndarray,
        X_valid: pd.DataFrame, y_va: np.ndarray,
        feats: list[str], transform_name: str, arch_params: dict,
        callbacks: list | None = None,
    ) -> tuple[CatBoostRegressor, _Transform, np.ndarray]:
        from catboost import CatBoostRegressor, Pool

        transform = _build_transform(transform_name, len(y_tr))
        transform.fit(y_tr)
        y_tr_t = transform.transform(y_tr)
        y_va_t = transform.transform(y_va)

        tr_pool = Pool(X_train[feats], y_tr_t, cat_features=self.cat_features_)
        va_pool = Pool(X_valid[feats], y_va_t, cat_features=self.cat_features_)

        model = CatBoostRegressor(loss_function='MAE', eval_metric='MAE', **arch_params)
        model.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=callbacks)

        if isinstance(transform, _Log1pTransform):
            train_pred_t = model.predict(tr_pool)
            transform.fit_smear(y_tr_t - train_pred_t)

        pred_va_t = model.predict(va_pool)
        pred_va_orig = transform.inverse_transform(pred_va_t)
        return model, transform, pred_va_orig

    def _tune(
        self,
        X_train: pd.DataFrame,
        y_tr: np.ndarray,
        X_valid: pd.DataFrame,
        y_va: np.ndarray,
        feats: list[str],
    ) -> tuple[CatBoostRegressor, _Transform, dict]:
        import optuna

        _optuna_prev_verbosity = optuna.logging.get_verbosity()
        if not self.optuna_verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        candidates = _valid_transforms(self.transforms, y_tr)
        esr = _DEFAULT_ARCH_PARAMS['early_stopping_rounds']

        def objective(trial: optuna.Trial) -> float:
            custom = self.param_space(trial) if self.param_space is not None else {}
            transform_name = custom.get(
                'transform',
                trial.suggest_categorical('transform', candidates) if len(candidates) > 1 else candidates[0],
            )
            arch_p = {
                **catboost_arch_space(trial, custom),
                'early_stopping_rounds': custom.get('early_stopping_rounds', esr),
                'random_seed': custom.get('random_seed', self.random_seed),
                'verbose': custom.get('verbose', 0),
            }
            trial.set_user_attr('transform_name', transform_name)
            trial.set_user_attr('arch_p', arch_p)

            pruning_cb = CatBoostPruningCallback(trial, 'MAE')
            try:
                _, _, pred_va_orig = self._fit_one(
                    X_train, y_tr, X_valid, y_va, feats, transform_name, arch_p, callbacks=[pruning_cb],
                )
            except Exception as err:
                raise optuna.TrialPruned(f'transform={transform_name} failed: {err}') from err
            pruning_cb.check_pruned()
            return float(mean_absolute_error(y_va, pred_va_orig))

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            pruner=make_pruner(self.optuna_pruner),
        )
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)

        best = study.best_trial
        transform_name = best.user_attrs['transform_name']
        arch_p = dict(best.user_attrs['arch_p'])
        model, transform, _ = self._fit_one(X_train, y_tr, X_valid, y_va, feats, transform_name, arch_p)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return model, transform, {'transform': transform_name, **arch_p}

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: XInput,
        y_train: YInput,
        X_valid: XInput,
        y_valid: YInput,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> TargetTransformOptunaRegressor:
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values

        if self.n_optuna_trials > 0:
            self._model, self.transform_, best = self._tune(X_train, y_tr, X_valid, y_va, feats)
            self.best_params_ = best
        else:
            transform_name = _valid_transforms(self.transforms, y_tr)[0]
            arch_params = {**(self.base_params or _DEFAULT_ARCH_PARAMS), 'random_seed': self.random_seed}
            self._model, self.transform_, _ = self._fit_one(
                X_train, y_tr, X_valid, y_va, feats, transform_name, arch_params,
            )
            self.best_params_ = {'transform': transform_name, **arch_params}

        self.transform_name_ = self.best_params_['transform']

        from catboost import Pool
        tr_pool = Pool(X_train[feats], cat_features=self.cat_features_)
        va_pool = Pool(X_valid[feats], cat_features=self.cat_features_)
        self.train_pred_ = self.transform_.inverse_transform(self._model.predict(tr_pool))
        self.valid_pred_ = self.transform_.inverse_transform(self._model.predict(va_pool))

        mae = float(mean_absolute_error(y_va, self.valid_pred_))
        logger.info(
            '[TargetTransformOptuna] transform=%s  val MAE=%.4f', self.transform_name_, mae,
        )
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool

        pool = Pool(X[self.selected_features_], cat_features=self.cat_features_)
        pred_t = self._model.predict(pool)
        return self.transform_.inverse_transform(pred_t)
