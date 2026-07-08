"""NGBoostPreset: параметрический прогноз распределения (Normal/LogNormal/Gamma) через NGBoost.

В отличие от остальных пресетов пакета (точечный прогноз ± пост-хок интервал),
NGBoost учится напрямую предсказывать ПАРАМЕТРЫ распределения на каждую строку
(natural gradient boosting, Duan et al., 2020) — predict() возвращает среднее
распределения (совпадает с NGBoost's `pred.mean()`), а полная плотность/квантили
доступны через predict_dist()/predict_interval(). Полезно, когда важны не
только точечные прогнозы, но и корректно откалиброванная неопределённость
(в отличие от split-conformal обёрток вроде ConformalRegressionWrapper —
интервалы здесь получаются из самой формы распределения, а не из остатков).

LogNormal/Gamma определены только на строго положительной полуоси — fit()
поднимает ValueError, если в y_train/y_valid есть y <= 0.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ml_toolkit.presets.regression._base import BasePreset

logger = logging.getLogger(__name__)

_DIST_NAMES = ('Normal', 'LogNormal', 'Gamma')
_POSITIVE_DISTS = ('LogNormal', 'Gamma')


def _import_ngboost():
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Gamma, LogNormal, Normal
    except ImportError as err:
        raise ImportError('NGBoost not installed. Run: pip install ngboost') from err
    return NGBRegressor, {'Normal': Normal, 'LogNormal': LogNormal, 'Gamma': Gamma}


class NGBoostPreset(BasePreset):
    """NGBoost-регрессор с параметрическим прогнозом распределения.

    Parameters
    ----------
    dist:
        'Normal' | 'LogNormal' (по умолчанию) | 'Gamma'. LogNormal/Gamma
        требуют строго положительный таргет.
    n_estimators:
        Верхняя граница числа boosting-раундов (реальное число может быть
        меньше — обучение всегда идёт с early stopping по X_valid/y_valid).
    learning_rate / base_max_depth / minibatch_frac:
        Остальные гиперпараметры NGBRegressor (base_max_depth — глубина
        дерева базового learner'а).
    n_optuna_trials:
        Если > 0, n_estimators/learning_rate/base_max_depth/minibatch_frac
        тюнятся Optuna по MAE на валидации (mean предсказанного распределения
        против y_valid). 0 → прямой режим с параметрами конструктора.
    param_space / optuna_timeout / optuna_verbose / random_seed:
        См. другие Optuna-пресеты пакета.

    Атрибуты после fit::

        best_iteration_  — число деревьев, использованных после early stopping

    Пример::

        model = NGBoostPreset(dist='LogNormal', n_optuna_trials=20)
        model.fit(X_train, y_train, X_valid, y_valid)
        mean_pred = model.predict(X_test)
        interval = model.predict_interval(X_test, alpha=0.1)   # (lower, upper)

    """

    def __init__(
        self,
        dist: str = 'LogNormal',
        n_estimators: int = 500,
        learning_rate: float = 0.01,
        base_max_depth: int = 3,
        minibatch_frac: float = 1.0,
        n_optuna_trials: int = 0,
        param_space: Callable[[Any], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        early_stopping_rounds: int = 50,
        random_seed: int = 42,
        selected_features: list[str] | None = None,
    ) -> None:
        if dist not in _DIST_NAMES:
            raise ValueError(f'dist должен быть одним из {_DIST_NAMES}, получено {dist!r}')
        super().__init__(params=None, n_optuna_trials=n_optuna_trials)
        self.dist = dist
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_max_depth = base_max_depth
        self.minibatch_frac = minibatch_frac
        self.param_space = param_space
        self.optuna_timeout = optuna_timeout
        self.optuna_verbose = optuna_verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.random_seed = random_seed
        self.selected_features = selected_features or []
        self.best_iteration_: int | None = None

    def _build_model(self, Dist, n_estimators, learning_rate, base_max_depth, minibatch_frac):
        NGBRegressor, _ = _import_ngboost()
        from sklearn.tree import DecisionTreeRegressor

        base = DecisionTreeRegressor(criterion='squared_error', max_depth=base_max_depth)
        return NGBRegressor(
            Dist=Dist, Base=base, n_estimators=n_estimators, learning_rate=learning_rate,
            minibatch_frac=minibatch_frac, natural_gradient=True,
            random_state=self.random_seed, verbose=False,
        )

    def _tune(self, X_tr, y_tr, X_va, y_va, Dist):
        import optuna

        _optuna_prev_verbosity = optuna.logging.get_verbosity()
        if not self.optuna_verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            custom = self.param_space(trial) if self.param_space is not None else {}

            def val(key: str, suggest: Callable[[], Any]) -> Any:
                return custom[key] if key in custom else suggest()

            n_estimators = val('n_estimators', lambda: trial.suggest_int('n_estimators', 100, 500, step=50))
            learning_rate = val('learning_rate', lambda: trial.suggest_float('learning_rate', 0.005, 0.2, log=True))
            base_max_depth = val('base_max_depth', lambda: trial.suggest_int('base_max_depth', 2, 5))
            minibatch_frac = val('minibatch_frac', lambda: trial.suggest_float('minibatch_frac', 0.5, 1.0))

            params = {
                'n_estimators': n_estimators, 'learning_rate': learning_rate,
                'base_max_depth': base_max_depth, 'minibatch_frac': minibatch_frac,
            }
            trial.set_user_attr('ngb_params', params)

            model = self._build_model(Dist, n_estimators, learning_rate, base_max_depth, minibatch_frac)
            model.fit(X_tr, y_tr, X_val=X_va, Y_val=y_va,
                     early_stopping_rounds=self.early_stopping_rounds)
            pred = model.predict(X_va)
            return float(mean_absolute_error(y_va, pred))

        study = optuna.create_study(
            direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.random_seed),
        )
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        best = dict(study.best_trial.user_attrs['ngb_params'])
        model = self._build_model(Dist, **best)
        model.fit(X_tr, y_tr, X_val=X_va, Y_val=y_va, early_stopping_rounds=self.early_stopping_rounds)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return model, best

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> NGBoostPreset:
        if cat_features:
            raise ValueError(
                'NGBoostPreset не поддерживает категориальные признаки нативно — '
                'закодируйте их заранее (см. ml_toolkit.models._utils.encode_cat_features)'
            )
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = []

        y_tr = y_train.values
        y_va = y_valid.values
        if self.dist in _POSITIVE_DISTS:
            for name, y in (('y_train', y_tr), ('y_valid', y_va)):
                if (y <= 0).any():
                    raise ValueError(
                        f'NGBoostPreset(dist={self.dist!r}) требует строго положительный таргет: '
                        f'{name} содержит y <= 0'
                    )

        _, dist_map = _import_ngboost()
        Dist = dist_map[self.dist]
        X_tr = X_train[feats].values
        X_va = X_valid[feats].values

        if self.n_optuna_trials > 0:
            self._model, best = self._tune(X_tr, y_tr, X_va, y_va, Dist)
            self.best_params_ = {'dist': self.dist, **best}
        else:
            self._model = self._build_model(
                Dist, self.n_estimators, self.learning_rate, self.base_max_depth, self.minibatch_frac,
            )
            self._model.fit(X_tr, y_tr, X_val=X_va, Y_val=y_va,
                            early_stopping_rounds=self.early_stopping_rounds)
            self.best_params_ = {
                'dist': self.dist, 'n_estimators': self.n_estimators, 'learning_rate': self.learning_rate,
                'base_max_depth': self.base_max_depth, 'minibatch_frac': self.minibatch_frac,
            }

        self.best_iteration_ = getattr(self._model, 'best_val_loss_itr', None)
        self.train_pred_ = self._model.predict(X_tr)
        self.valid_pred_ = self._model.predict(X_va)
        mae = float(mean_absolute_error(y_va, self.valid_pred_))
        logger.info('[NGBoostPreset] dist=%s  val MAE=%.4f  best_iteration=%s',
                   self.dist, mae, self.best_iteration_)
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict_dist(self, X: Any) -> Any:
        """Полное предсказанное распределение (ngboost distribution object) на X."""
        self._check_fitted()
        from ml_toolkit.models._base import _to_pandas
        Xp = _to_pandas(X)
        return self._model.pred_dist(Xp[self.selected_features_].values)

    def predict_interval(self, X: Any, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Предиктивный интервал уровня (1 - alpha) через квантили предсказанного распределения."""
        if not 0.0 < alpha < 1.0:
            raise ValueError(f'alpha должен быть в (0, 1), получено {alpha}')
        dist = self.predict_dist(X)
        return dist.ppf(alpha / 2.0), dist.ppf(1.0 - alpha / 2.0)

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X[self.selected_features_].values)
