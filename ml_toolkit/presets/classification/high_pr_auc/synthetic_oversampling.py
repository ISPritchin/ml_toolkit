"""SyntheticOversamplingClassifier: SMOTE/ADASYN для экстремального дисбаланса.

Противоположный угол к undersampling: вместо выбрасывания негативов —
генерация синтетических позитивов через интерполяцию в пространстве признаков.

Методы:
  'smote'      — интерполяция между k ближайшими соседями-позитивами.
  'adasyn'     — ADASYN: концентрирует генерацию на «трудных» позитивах
                 (у которых среди k соседей больше негативов).
  'borderline' — BorderlineSMOTE: только позитивы на границе с негативами.
  'smoteenn'   — SMOTE + Edited Nearest Neighbours: удаляет зашумлённые примеры
                 после oversampling. Иногда лучшее соотношение качество/скорость.

Требует: pip install imbalanced-learn

Когда использовать вместо undersampling (EasyEnsemble):
  - train positives < 200: мало позитивов → SMOTE создаёт их больше,
    undersampling не поможет если негативов уже мало.
  - Позитивы образуют кластер в feature space: SMOTE интерполирует внутри него,
    что работает хорошо.
  - Когда нужна совместимость с sklearn Pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.presets.classification._base import BasePreset
from ml_toolkit.presets.classification._optuna_utils import CatBoostPruningCallback, make_pruner

logger = logging.getLogger(__name__)

_DEFAULT_CBT_PARAMS: dict[str, Any] = {
    'iterations': 700,
    'max_depth': 5,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'min_data_in_leaf': 5,
    'early_stopping_rounds': 80,
    'loss_function': 'Logloss',
    'eval_metric': 'PRAUC',
    'random_seed': 42,
    'verbose': 0,
}

_DEFAULT_LGB_PARAMS: dict[str, Any] = {
    'n_estimators': 600,
    'max_depth': 5,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
}


def _make_sampler(method: str, sampling_strategy: float, random_seed: int) -> Any:
    try:
        from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
        from imblearn.combine import SMOTEENN
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn не установлен. Установите: uv add imbalanced-learn"
        ) from exc

    kwargs = {'sampling_strategy': sampling_strategy, 'random_state': random_seed, 'k_neighbors': 5}
    if method == 'smote':
        return SMOTE(**kwargs)
    elif method == 'adasyn':
        return ADASYN(
            sampling_strategy=sampling_strategy,
            random_state=random_seed,
            n_neighbors=5,
        )
    elif method == 'borderline':
        return BorderlineSMOTE(**kwargs)
    elif method == 'smoteenn':
        smote = SMOTE(**kwargs)
        from imblearn.under_sampling import EditedNearestNeighbours
        enn = EditedNearestNeighbours()
        return SMOTEENN(smote=smote, enn=enn, random_state=random_seed)
    else:
        raise ValueError(
            f"method должен быть 'smote', 'adasyn', 'borderline' или 'smoteenn', получено {method!r}"
        )


class SyntheticOversamplingClassifier(BasePreset):
    """SMOTE/ADASYN oversampling с CatBoost или LightGBM.

    Parameters
    ----------
    method:
        Метод генерации синтетических примеров: 'smote', 'adasyn', 'borderline', 'smoteenn'.
    sampling_strategy:
        Целевое соотношение minority/majority ПОСЛЕ oversampling.
        0.1 → 1:10 (один позитив на 10 негативов).
        'auto' → 1:1 (полная балансировка — обычно слишком агрессивно при < 1%).
    base:
        'catboost' (по умолчанию) или 'lightgbm'.
    base_params:
        Гиперпараметры базовой модели. None → дефолтные. Игнорируется, если
        n_optuna_trials > 0.
    n_optuna_trials:
        Если > 0, архитектура базовой модели подбирается через Optuna по val
        PR-AUC на уже аугментированном (после SMOTE/ADASYN) train.
    optuna_timeout:
        Ограничение по времени (сек) на весь Optuna-поиск. None — без ограничения.
    random_seed:
        Зерно для sampler, модели и Optuna sampler'а.

    Замечание о cat_features:
        SMOTE интерполирует непрерывные значения — для категориальных признаков
        результат может быть не валидным. Категориальные исключаются из SMOTE;
        синтетическим примерам их значения копируются от случайного исходного
        позитива. Комбинация method='smoteenn' + категориальные признаки не
        поддерживается (ENN удаляет строки, соответствие восстановить нельзя) —
        поднимается ValueError.

    Атрибуты после fit::

        n_synthetic_     — количество сгенерированных синтетических примеров
        augmented_ratio_ — реальное соотношение minority/majority после oversampling
    """

    def __init__(
        self,
        method: str = 'smote',
        sampling_strategy: float | str = 0.1,
        base: str = 'catboost',
        base_params: dict[str, Any] | None = None,
        n_optuna_trials: int = 0,
        optuna_timeout: int | None = None,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(params=base_params, n_optuna_trials=n_optuna_trials)
        if base not in ('catboost', 'lightgbm'):
            raise ValueError(f"base должен быть 'catboost' или 'lightgbm', получено {base!r}")
        self.method = method
        self.sampling_strategy = sampling_strategy
        self.base = base
        self.base_params = base_params
        self.optuna_timeout = optuna_timeout
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.n_synthetic_: int = 0
        self.augmented_ratio_: float = 0.0

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> 'SyntheticOversamplingClassifier':
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values
        X_tr_feats = X_train[feats]

        n_pos_before = int(y_tr.sum())
        n_neg_before = int((y_tr == 0).sum())
        logger.info(
            '[SyntheticOversampling] method=%s  strategy=%s  before: n_pos=%d  n_neg=%d',
            self.method, self.sampling_strategy, n_pos_before, n_neg_before,
        )

        # Категориальные признаки исключаем из SMOTE, применяем только к числовым
        num_feats = [f for f in feats if f not in self.cat_features_]
        cat_feats_in_feats = [f for f in feats if f in self.cat_features_]

        if self.method == 'smoteenn' and cat_feats_in_feats:
            raise ValueError(
                "method='smoteenn' несовместим с категориальными признаками: ENN удаляет "
                "строки и не сохраняет порядок, поэтому восстановить соответствие "
                "категориальных значений исходным строкам невозможно. Уберите cat_features "
                "из selected_features или используйте method='smote'/'adasyn'/'borderline'."
            )

        if cat_feats_in_feats:
            logger.warning(
                '[SyntheticOversampling] SMOTE будет применён только к %d числовым признакам. '
                '%d cat_features (%s...) будут взяты от случайного исходного позитива.',
                len(num_feats), len(cat_feats_in_feats), cat_feats_in_feats[:3],
            )

        sampler = _make_sampler(self.method, self.sampling_strategy, self.random_seed)

        X_num_resampled, y_resampled = sampler.fit_resample(
            X_tr_feats[num_feats].values if num_feats else np.zeros((len(y_tr), 1)),
            y_tr,
        )
        if self.method == 'smoteenn':
            # ENN удаляет строки: инвариант «хвост результата = синтетика» не
            # выполняется, считаем чистый прирост позитивов.
            n_generated = 0
            self.n_synthetic_ = int(max(0, int(y_resampled.sum()) - int(y_tr.sum())))
        else:
            n_generated = len(y_resampled) - len(y_tr)
            self.n_synthetic_ = int((y_resampled[len(y_tr):] == 1).sum())
        n_pos_after = int(y_resampled.sum())
        n_neg_after = int((y_resampled == 0).sum())
        self.augmented_ratio_ = n_pos_after / max(n_neg_after, 1)

        logger.info(
            '[SyntheticOversampling] after: n_pos=%d (+%d synthetic)  n_neg=%d  ratio=%.3f',
            n_pos_after, self.n_synthetic_, n_neg_after, self.augmented_ratio_,
        )

        # Собираем итоговый DataFrame для обучения
        if num_feats:
            X_aug_num = pd.DataFrame(X_num_resampled, columns=num_feats)
        else:
            X_aug_num = pd.DataFrame(index=range(len(y_resampled)))

        # Категориальные: для новых синтетических примеров копируем из ближайшего оригинального позитива
        if cat_feats_in_feats:
            X_orig_cat = X_tr_feats[cat_feats_in_feats].values
            orig_pos_idx = np.where(y_tr == 1)[0]
            cat_aug_rows = [X_orig_cat]
            if n_generated > 0:
                rng = np.random.default_rng(self.random_seed)
                fill_idx = rng.choice(orig_pos_idx, size=n_generated)
                cat_aug_rows.append(X_orig_cat[fill_idx])
            cat_aug = np.vstack(cat_aug_rows)
            X_aug_cat = pd.DataFrame(cat_aug, columns=cat_feats_in_feats)
            X_aug = pd.concat([X_aug_num, X_aug_cat], axis=1)[feats]
        else:
            X_aug = X_aug_num[feats] if num_feats else X_tr_feats.iloc[:len(y_resampled)]

        X_va_feats = X_valid[feats]

        tuned_params = None
        if self.n_optuna_trials > 0:
            tuned_params = (
                self._tune_cbt(X_aug, y_resampled, X_va_feats, y_va) if self.base == 'catboost'
                else self._tune_lgb(X_aug, y_resampled, X_va_feats, y_va)
            )

        if self.base == 'catboost':
            self._model, self.train_pred_, self.valid_pred_ = self._fit_catboost(
                X_aug, y_resampled, X_va_feats, y_va, tuned_params
            )
        else:
            self._model, self.train_pred_, self.valid_pred_ = self._fit_lgb(
                X_aug, y_resampled, X_va_feats, y_va, tuned_params
            )

        pr_auc = float(average_precision_score(y_va, self.valid_pred_))
        logger.info('[SyntheticOversampling] val PR-AUC=%.4f', pr_auc)

        self.best_params_ = {
            'method': self.method,
            'sampling_strategy': self.sampling_strategy,
            'base': self.base,
            'n_synthetic': self.n_synthetic_,
            'base_params': tuned_params or (self.base_params or (
                _DEFAULT_CBT_PARAMS if self.base == 'catboost' else _DEFAULT_LGB_PARAMS
            )),
        }
        return self

    def _fit_catboost(
        self,
        X_aug: pd.DataFrame,
        y_aug: np.ndarray,
        X_va: pd.DataFrame,
        y_va: np.ndarray,
        params: dict[str, Any] | None = None,
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        from catboost import CatBoostClassifier, Pool

        p = {**(params or self.base_params or _DEFAULT_CBT_PARAMS), 'random_seed': self.random_seed}
        model = CatBoostClassifier(**p)
        tr_pool = Pool(X_aug, y_aug, cat_features=self.cat_features_)
        va_pool = Pool(X_va, y_va, cat_features=self.cat_features_)
        model.fit(tr_pool, eval_set=va_pool, verbose=False)
        return (
            model,
            model.predict_proba(tr_pool)[:, 1],
            model.predict_proba(va_pool)[:, 1],
        )

    def _fit_lgb(
        self,
        X_aug: pd.DataFrame,
        y_aug: np.ndarray,
        X_va: pd.DataFrame,
        y_va: np.ndarray,
        params: dict[str, Any] | None = None,
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        import lightgbm as lgb

        p = {**(params or self.base_params or _DEFAULT_LGB_PARAMS), 'random_state': self.random_seed}
        model = lgb.LGBMClassifier(**p)
        model.fit(
            X_aug, y_aug,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        return (
            model,
            model.predict_proba(X_aug)[:, 1],
            model.predict_proba(X_va)[:, 1],
        )

    def _tune_cbt(self, X_aug: pd.DataFrame, y_aug: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray) -> dict[str, Any]:
        import optuna
        from catboost import CatBoostClassifier, Pool

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        tr_pool = Pool(X_aug, y_aug, cat_features=self.cat_features_)
        va_pool = Pool(X_va, y_va, cat_features=self.cat_features_)

        def objective(trial: optuna.Trial) -> float:
            params = {
                'iterations': trial.suggest_int('iterations', 300, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
                'loss_function': 'Logloss',
                'eval_metric': 'PRAUC',
                'early_stopping_rounds': 80,
                'random_seed': self.random_seed,
                'verbose': 0,
            }
            pruning_cb = CatBoostPruningCallback(trial, 'PRAUC')
            m = CatBoostClassifier(**params)
            m.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=[pruning_cb])
            pruning_cb.check_pruned()
            p = m.predict_proba(va_pool)[:, 1]
            return float(average_precision_score(y_va, p))

        logger.info('[SyntheticOversampling] Optuna (catboost): %d trials', self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed),
                                    pruner=make_pruner())
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        return {
            **study.best_params,
            'loss_function': 'Logloss', 'eval_metric': 'PRAUC',
            'early_stopping_rounds': 80, 'random_seed': self.random_seed, 'verbose': 0,
        }

    def _tune_lgb(self, X_aug: pd.DataFrame, y_aug: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray) -> dict[str, Any]:
        import optuna
        import lightgbm as lgb

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'random_state': self.random_seed,
                'verbose': -1,
                'n_jobs': -1,
            }
            m = lgb.LGBMClassifier(**params)
            m.fit(
                X_aug, y_aug,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            p = m.predict_proba(X_va)[:, 1]
            return float(average_precision_score(y_va, p))

        logger.info('[SyntheticOversampling] Optuna (lightgbm): %d trials', self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed))
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        return {**study.best_params, 'random_state': self.random_seed, 'verbose': -1, 'n_jobs': -1}

    # ── predict ───────────────────────────────────────────────────────────────

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_feats = X[self.selected_features_]
        if self.base == 'catboost':
            from catboost import Pool
            return self._model.predict_proba(Pool(X_feats, cat_features=self.cat_features_))[:, 1]
        return self._model.predict_proba(X_feats)[:, 1]
