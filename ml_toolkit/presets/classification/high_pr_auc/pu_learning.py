"""PULearningClassifier: коррекция Элкана–Ното для Positive-Unlabeled данных.

Предположение: часть истинных позитивов в «негативном» классе не помечена
(undetected positives). Стандартный классификатор обучается на s (labeled),
а не на y (true positive), что приводит к систематическому занижению P(y=1|x).

Коррекция Элкана–Ното:
  P(y=1|x) = P(s=1|x) / c,  где c = P(s=1|y=1)

c оценивается как среднее raw-вероятностей по известным позитивам:
c ≈ mean(model.predict_proba(X_val_c[y==1])), где X_val_c — половина
валидации, НЕ использованная для early stopping/Optuna (иначе оценка c
смещена вверх: модель подгонялась под эту же выборку).

Важно: коррекция raw/c монотонна и НЕ меняет ранжирование — PR-AUC «до» и
«после» могут расходиться только из-за клипа в [0, 1] (все raw > c
схлопываются в 1.0 и образуют ties). Смысл коррекции — калибровка
абсолютных значений вероятностей, а не рост ранговых метрик.

Когда использовать:
  - Клиенты в «Крупные» могут быть незамечены (сегментация менялась во времени).
  - Time split: в val больше «настоящих» позитивов, чем в train.
  - Консистентно низкий recall при хорошем ROC-AUC.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.presets.classification._base import BasePreset
from ml_toolkit.presets.classification._optuna_utils import CatBoostPruningCallback, make_pruner

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
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


class PULearningClassifier(BasePreset):
    """CatBoost с коррекцией Элкана–Ното для positive-unlabeled данных.

    Parameters
    ----------
    base_params:
        Параметры CatBoost. None → дефолтные.
    n_optuna_trials:
        Число Optuna-триалов. 0 → использовать base_params без поиска.
    param_space:
        Кастомная функция `f(trial) -> dict` — переопределяет search space для
        Optuna. Любой отсутствующий в возвращённом словаре ключ (iterations/
        max_depth/learning_rate/l2_leaf_reg/subsample/min_data_in_leaf или
        loss_function/eval_metric/early_stopping_rounds/random_seed/verbose)
        тюнится/подставляется дефолтным способом — можно переопределить как
        ни одного параметра, так и часть, так и все сразу, включая
        loss_function/eval_metric. Действует только при n_optuna_trials > 0.
        None → дефолтный search space.
    optuna_verbose:
        Если True — не глушит логи Optuna. Если False (по умолчанию) —
        форсирует WARNING на время поиска.
    c_lower_bound:
        Минимально допустимое значение c (защита от деления на очень малое).
        При c < c_lower_bound выдаётся предупреждение.
    c_estimation_frac:
        Доля валидации, откладываемая под оценку c (stratified). Остальная
        часть используется для early stopping/Optuna. При < 2 позитивах в
        валидации сплит невозможен — c оценивается по всей валидации
        с предупреждением о смещении.

    Атрибуты после fit::

        c_               — оценённая P(s=1|y=1) по c-holdout позитивам
        raw_pr_auc_      — PR-AUC без коррекции (на c-holdout)
        corrected_pr_auc_ — PR-AUC после коррекции (на c-holdout; отличается
                            от raw только эффектом клипа, см. докстринг модуля)

    Пример::

        model = PULearningClassifier()
        model.fit(X_train, y_train, X_valid, y_valid)
        proba = model.predict_proba(X_test)  # уже откорректированные вероятности
    """

    def __init__(
        self,
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        param_space: Callable[[Any], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        c_lower_bound: float = 0.1,
        c_estimation_frac: float = 0.5,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(params=None, n_optuna_trials=n_optuna_trials)
        if not 0.0 < c_estimation_frac < 1.0:
            raise ValueError(f'c_estimation_frac должен быть в (0, 1), получено {c_estimation_frac}')
        self.optuna_timeout = optuna_timeout
        self.param_space = param_space
        self.optuna_verbose = optuna_verbose
        self.base_params = base_params
        self.c_lower_bound = c_lower_bound
        self.c_estimation_frac = c_estimation_frac
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.c_: float = 1.0
        self.raw_pr_auc_: float = 0.0
        self.corrected_pr_auc_: float = 0.0

    # ── Optuna ────────────────────────────────────────────────────────────────

    def _tune(self, tr_pool: Any, va_pool: Any) -> Any:
        import optuna
        from catboost import CatBoostClassifier

        if not self.optuna_verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            custom = self.param_space(trial) if self.param_space is not None else {}

            def val(key: str, suggest: Callable[[], Any]) -> Any:
                return custom[key] if key in custom else suggest()

            params = {
                'iterations': val('iterations', lambda: trial.suggest_int('iterations', 300, 1000, step=100)),
                'max_depth': val('max_depth', lambda: trial.suggest_int('max_depth', 3, 7)),
                'learning_rate': val('learning_rate',
                    lambda: trial.suggest_float('learning_rate', 0.005, 0.3, log=True)),
                'l2_leaf_reg': val('l2_leaf_reg',
                    lambda: trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True)),
                'subsample': val('subsample', lambda: trial.suggest_float('subsample', 0.5, 1.0)),
                'min_data_in_leaf': val('min_data_in_leaf',
                    lambda: trial.suggest_int('min_data_in_leaf', 1, 30)),
                'loss_function': custom.get('loss_function', 'Logloss'),
                'eval_metric': custom.get('eval_metric', 'PRAUC'),
                'early_stopping_rounds': custom.get('early_stopping_rounds', 80),
                'random_seed': custom.get('random_seed', self.random_seed),
                'verbose': custom.get('verbose', 0),
            }
            trial.set_user_attr('cb_params', params)
            pruning_cb = CatBoostPruningCallback(trial, params['eval_metric'])
            m = CatBoostClassifier(**params)
            m.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=[pruning_cb])
            pruning_cb.check_pruned()
            p = m.predict_proba(va_pool)[:, 1]
            return float(average_precision_score(va_pool.get_label(), p))

        study = optuna.create_study(
            direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            pruner=make_pruner(),
        )
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        best_params = dict(study.best_trial.user_attrs['cb_params'])
        model = CatBoostClassifier(**best_params)
        model.fit(tr_pool, eval_set=va_pool, verbose=False)
        self.best_params_ = best_params
        return model

    # ── Оценка c ──────────────────────────────────────────────────────────────

    def _estimate_c(self, raw_c: np.ndarray, y_c: np.ndarray) -> None:
        """Точечная оценка c = mean(raw score) по c-holdout позитивам, с клипом

        снизу по c_lower_bound. Выделено в отдельный метод, чтобы
        ElkanNotoHoldoutPU (029) мог переопределить его и добавить bootstrap-CI,
        не дублируя остальной fit().
        """
        pos_mask = y_c == 1
        if pos_mask.sum() == 0:
            logger.warning('[PULearning] Нет позитивов в c-holdout — c оставлен = 1.0')
            self.c_ = 1.0
        else:
            self.c_ = float(raw_c[pos_mask].mean())

        if self.c_ < self.c_lower_bound:
            logger.warning(
                '[PULearning] c=%.4f < lower_bound=%.4f. '
                'Возможно, модель недостаточно различает классы. '
                'Используем c=%.4f.',
                self.c_, self.c_lower_bound, self.c_lower_bound,
            )
            self.c_ = self.c_lower_bound

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> 'PULearningClassifier':
        from catboost import CatBoostClassifier, Pool

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values

        tr_pool = Pool(X_train[feats], y_tr, cat_features=self.cat_features_)
        va_pool = Pool(X_valid[feats], y_va, cat_features=self.cat_features_)

        # ── Сплит валидации: es-часть (early stopping/Optuna) и c-holdout ────
        # Оценка c по той же выборке, на которую делался early stopping,
        # смещена вверх — модель под неё подгонялась.
        n_pos_va = int((y_va == 1).sum())
        if n_pos_va >= 2 and len(y_va) >= 10:
            from sklearn.model_selection import train_test_split
            idx_es, idx_c = train_test_split(
                np.arange(len(y_va)),
                test_size=self.c_estimation_frac,
                stratify=y_va,
                random_state=self.random_seed,
            )
        else:
            logger.warning(
                '[PULearning] В валидации %d позитивов — сплит под оценку c невозможен, '
                'c оценивается по всей валидации (оценка смещена вверх)', n_pos_va,
            )
            idx_es = np.arange(len(y_va))
            idx_c = np.arange(len(y_va))

        va_pool_es = Pool(
            X_valid[feats].iloc[idx_es], y_va[idx_es], cat_features=self.cat_features_
        )

        if self.n_optuna_trials > 0:
            self._model = self._tune(tr_pool, va_pool_es)
        else:
            params = {**(self.base_params or _DEFAULT_PARAMS), 'random_seed': self.random_seed}
            self._model = CatBoostClassifier(**params)
            self._model.fit(tr_pool, eval_set=va_pool_es, verbose=False)
            self.best_params_ = params

        # ── Оценка c на c-holdout ────────────────────────────────────────────
        raw_va = self._model.predict_proba(va_pool)[:, 1]
        raw_c = raw_va[idx_c]
        y_c = y_va[idx_c]
        self.raw_pr_auc_ = float(average_precision_score(y_c, raw_c))

        self._estimate_c(raw_c, y_c)

        corrected_c = np.clip(raw_c / self.c_, 0.0, 1.0)
        self.corrected_pr_auc_ = float(average_precision_score(y_c, corrected_c))

        logger.info(
            '[PULearning] c=%.4f (по %d c-holdout позитивам)  '
            'raw PR-AUC=%.4f  corrected PR-AUC=%.4f (различие — только эффект клипа)',
            self.c_, int((y_c == 1).sum()), self.raw_pr_auc_, self.corrected_pr_auc_,
        )

        self.valid_pred_ = np.clip(raw_va / self.c_, 0.0, 1.0)
        raw_tr = self._model.predict_proba(tr_pool)[:, 1]
        self.train_pred_ = np.clip(raw_tr / self.c_, 0.0, 1.0)

        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool

        pool = Pool(X[self.selected_features_], cat_features=self.cat_features_)
        raw = self._model.predict_proba(pool)[:, 1]
        return np.clip(raw / self.c_, 0.0, 1.0)
