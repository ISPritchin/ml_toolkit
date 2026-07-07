"""HeterogeneousStacking: стек CatBoost + LightGBM + XGBoost + линейной модели.

Отличие от SubsampleStacking: та строит diversity через РАЗНЫЕ конфиги ОДНОГО
алгоритма (CatBoost) на stratified подвыборках; здесь diversity — из РАЗНЫХ
семейств алгоритмов (разные индуктивные смещения: два разных бустинга + линейная
модель поверх one-hot/scaled признаков). Полезно, когда потолок одного
алгоритма уже достигнут — семейства ошибаются по-разному именно потому, что
устроены принципиально иначе, а не только имеют разные гиперпараметры.

OOF-механика идентична SubsampleStacking (честный K-fold, без утечки): каждая
строка train получает предсказание каждого члена зоопарка, полученное на
фолде, где эта строка не участвовала в обучении.

xgboost — опциональная тяжёлая зависимость (не входит в основные зависимости
пакета, как и в функциональном API ml_toolkit.models). Если недоступен —
пропускается из зоопарка с предупреждением в логах, а не падением; base_zoo
должен содержать хотя бы 2 доступных члена после фильтрации.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.models._utils import fit_calibrator, prep_cat_features
from ml_toolkit.presets.classification._base import BasePreset

logger = logging.getLogger(__name__)

_CBT_PARAMS: dict[str, Any] = {
    'iterations': 500, 'max_depth': 5, 'learning_rate': 0.05, 'l2_leaf_reg': 3.0,
    'subsample': 0.8, 'min_data_in_leaf': 10, 'loss_function': 'Logloss',
    'eval_metric': 'PRAUC', 'verbose': 0,
}
_LGB_PARAMS: dict[str, Any] = {
    'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05, 'num_leaves': 31,
    'min_child_samples': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'verbose': -1, 'n_jobs': -1,
}
_XGB_PARAMS: dict[str, Any] = {
    'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
    'eval_metric': 'aucpr', 'n_jobs': -1, 'use_label_encoder': False,
}

_DEFAULT_ZOO = ['catboost', 'lightgbm', 'xgboost', 'logistic']

_MEMBER_FIXED_EXTRAS: dict[str, dict[str, Any]] = {
    'catboost': {'loss_function': 'Logloss', 'eval_metric': 'PRAUC', 'verbose': 0},
    'lightgbm': {'verbose': -1, 'n_jobs': -1},
    'xgboost': {'eval_metric': 'aucpr', 'n_jobs': -1, 'use_label_encoder': False},
    'logistic': {},
}


class HeterogeneousStacking(BasePreset):
    """Стек CatBoost + LightGBM + XGBoost + LogisticRegression на честном OOF.

    Parameters
    ----------
    base_zoo:
        Список библиотек-членов зоопарка: подмножество
        ['catboost', 'lightgbm', 'xgboost', 'logistic'].
    meta:
        Мета-модель: 'logistic', 'weighted', 'catboost' (см. SubsampleStacking
        — то же семейство мета-моделей).
    n_folds:
        Число фолдов честного OOF.
    n_optuna_trials:
        Если > 0, архитектура КАЖДОГО члена зоопарка подбирается через Optuna
        отдельно (свой search space на семейство алгоритмов: CatBoost/LightGBM/
        XGBoost — бустинг-гиперпараметры, LogisticRegression — только C) вместо
        фиксированных _CBT_PARAMS/_LGB_PARAMS/_XGB_PARAMS. Каждый member тюнится
        независимо на полном train, с оценкой по val PR-AUC.
    optuna_timeout:
        Ограничение по времени (сек) на Optuna-поиск ОДНОГО члена зоопарка
        (суммарное время растёт с числом членов). None — без ограничения.
    calibrate:
        Применять ли изотоническую калибровку к финальным предсказаниям.
    random_seed:
        Зерно всех членов зоопарка, StratifiedKFold, мета-модели и Optuna sampler'ов.

    Атрибуты после fit::

        zoo_used_       — фактически использованные члены (после фильтрации недоступных)
        base_models_    — dict {имя: обученная финальная модель}
        meta_model_     — обученная мета-модель
        oob_pr_aucs_    — {имя: OOF PR-AUC на train}
        valid_pr_auc_   — PR-AUC ансамбля на val

    Пример::

        model = HeterogeneousStacking(base_zoo=['catboost', 'lightgbm', 'logistic'])
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=[...], cat_features=[...])
    """

    def __init__(
        self,
        base_zoo: list[str] | None = None,
        meta: str = 'logistic',
        n_folds: int = 5,
        n_optuna_trials: int = 0,
        optuna_timeout: int | None = None,
        calibrate: bool = True,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        super().__init__(params=None, n_optuna_trials=n_optuna_trials)
        base_zoo = base_zoo or list(_DEFAULT_ZOO)
        unknown = set(base_zoo) - {'catboost', 'lightgbm', 'xgboost', 'logistic'}
        if unknown:
            raise ValueError(f'Неизвестные члены base_zoo: {unknown}')
        if len(base_zoo) < 2:
            raise ValueError(f'base_zoo должен содержать >= 2 членов, получено {base_zoo}')
        if meta not in ('logistic', 'weighted', 'catboost'):
            raise ValueError("meta должен быть 'logistic', 'weighted' или 'catboost'")
        if n_folds < 2:
            raise ValueError(f'n_folds должен быть >= 2, получено {n_folds}')
        self.base_zoo = base_zoo
        self.meta = meta
        self.n_folds = n_folds
        self.optuna_timeout = optuna_timeout
        self.calibrate = calibrate
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

        self.zoo_used_: list[str] = []
        self.base_models_: dict[str, Any] = {}
        self.meta_model_: Any = None
        self.oob_pr_aucs_: dict[str, float] = {}
        self.valid_pr_auc_: float = 0.0

    # ── Члены зоопарка ────────────────────────────────────────────────────────

    def _fit_catboost(self, X, y, X_num, seed, params: dict[str, Any] | None = None):
        from catboost import CatBoostClassifier, Pool
        m = CatBoostClassifier(**{**(params or _CBT_PARAMS), 'random_seed': seed})
        m.fit(Pool(X, y, cat_features=self.cat_features_), verbose=False)
        return m

    def _predict_catboost(self, m, X, X_num):
        from catboost import Pool
        return m.predict_proba(Pool(X, cat_features=self.cat_features_))[:, 1]

    def _fit_lightgbm(self, X, y, X_num, seed, params: dict[str, Any] | None = None):
        import lightgbm as lgb
        m = lgb.LGBMClassifier(**{**(params or _LGB_PARAMS), 'random_state': seed})
        cat_idx = [c for c in self.cat_features_ if c in X.columns]
        Xp = prep_cat_features(X, list(X.columns), self.cat_features_)
        m.fit(Xp, y, categorical_feature=cat_idx or 'auto')
        return m

    def _predict_lightgbm(self, m, X, X_num):
        Xp = prep_cat_features(X, list(X.columns), self.cat_features_)
        return m.predict_proba(Xp)[:, 1]

    def _fit_xgboost(self, X, y, X_num, seed, params: dict[str, Any] | None = None):
        import xgboost as xgb
        m = xgb.XGBClassifier(**{**(params or _XGB_PARAMS), 'random_state': seed, 'enable_categorical': True})
        Xp = prep_cat_features(X, list(X.columns), self.cat_features_)
        return m.fit(Xp, y)

    def _predict_xgboost(self, m, X, X_num):
        Xp = prep_cat_features(X, list(X.columns), self.cat_features_)
        return m.predict_proba(Xp)[:, 1]

    def _build_linear_pipeline(self, C: float = 1.0):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        num_cols = [c for c in self.selected_features_ if c not in self.cat_features_]
        cat_cols = [c for c in self.selected_features_ if c in self.cat_features_]
        transformers = []
        if num_cols:
            transformers.append(('num', Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('scale', StandardScaler()),
            ]), num_cols))
        if cat_cols:
            transformers.append(('cat', Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]), cat_cols))
        pre = ColumnTransformer(transformers)
        return Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=2000, C=C))])

    def _fit_logistic(self, X, y, X_num, seed, params: dict[str, Any] | None = None):
        pipe = self._build_linear_pipeline(C=(params or {}).get('C', 1.0))
        pipe.fit(X, y)
        return pipe

    def _predict_logistic(self, m, X, X_num):
        return m.predict_proba(X)[:, 1]

    _FIT_DISPATCH = {
        'catboost': '_fit_catboost', 'lightgbm': '_fit_lightgbm',
        'xgboost': '_fit_xgboost', 'logistic': '_fit_logistic',
    }
    _PREDICT_DISPATCH = {
        'catboost': '_predict_catboost', 'lightgbm': '_predict_lightgbm',
        'xgboost': '_predict_xgboost', 'logistic': '_predict_logistic',
    }

    def _fit_member(self, name: str, X, y, seed: int, params: dict[str, Any] | None = None):
        return getattr(self, self._FIT_DISPATCH[name])(X, y, None, seed, params)

    def _predict_member(self, name: str, model: Any, X) -> np.ndarray:
        return getattr(self, self._PREDICT_DISPATCH[name])(model, X, None)

    # ── Optuna (по семейству алгоритмов) ─────────────────────────────────────

    def _suggest_member_params(self, name: str, trial: Any) -> dict[str, Any]:
        if name == 'catboost':
            search = {
                'iterations': trial.suggest_int('iterations', 300, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
            }
        elif name == 'lightgbm':
            search = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            }
        elif name == 'xgboost':
            search = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            }
        else:  # logistic
            search = {'C': trial.suggest_float('C', 1e-3, 1e2, log=True)}
        return {**search, **_MEMBER_FIXED_EXTRAS[name]}

    def _tune_member(
        self, name: str, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray,
    ) -> dict[str, Any]:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_member_params(name, trial)
            m = self._fit_member(name, X_tr, y_tr, self.random_seed, params)
            p = self._predict_member(name, m, X_va)
            return float(average_precision_score(y_va, p))

        logger.info('[HeteroStacking] Optuna (%s): %d trials', name, self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed))
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        return {**study.best_params, **_MEMBER_FIXED_EXTRAS[name]}

    # ── Мета-модели (то же семейство, что в SubsampleStacking) ──────────────

    def _fit_meta_logistic(self, X_meta, y_meta):
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=self.random_seed)
        m.fit(X_meta, y_meta)
        return m

    def _fit_meta_weighted(self, X_meta, y_meta):
        from scipy.optimize import minimize
        n = X_meta.shape[1]
        y = y_meta.astype(float)
        eps = 1e-7

        def neg_log_likelihood(raw_w):
            w = np.exp(raw_w) / np.exp(raw_w).sum()
            blend = np.clip(X_meta @ w, eps, 1.0 - eps)
            return -float(np.mean(y * np.log(blend) + (1.0 - y) * np.log(1.0 - blend)))

        res = minimize(neg_log_likelihood, np.zeros(n), method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-12})
        weights = np.exp(res.x) / np.exp(res.x).sum()
        logger.info('[HeteroStacking] Мета-веса (BCE): %s', dict(zip(self.zoo_used_, np.round(weights, 3))))
        return weights

    def _fit_meta_catboost(self, X_meta, y_meta):
        from catboost import CatBoostClassifier, Pool
        params = {'iterations': 200, 'max_depth': 3, 'learning_rate': 0.05,
                  'loss_function': 'Logloss', 'eval_metric': 'PRAUC',
                  'random_seed': self.random_seed, 'verbose': 0}
        m = CatBoostClassifier(**params)
        m.fit(Pool(X_meta, y_meta))
        return m

    def _meta_predict(self, X_meta):
        if self.meta == 'logistic':
            return self.meta_model_.predict_proba(X_meta)[:, 1]
        if self.meta == 'weighted':
            return X_meta @ self.meta_model_
        from catboost import Pool
        return self.meta_model_.predict_proba(Pool(X_meta))[:, 1]

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> 'HeterogeneousStacking':
        from sklearn.model_selection import StratifiedKFold

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        zoo = list(self.base_zoo)
        if 'xgboost' in zoo:
            try:
                import xgboost  # noqa: F401
            except ImportError:
                logger.warning('[HeteroStacking] xgboost не установлен — исключён из зоопарка')
                zoo = [z for z in zoo if z != 'xgboost']
        if len(zoo) < 2:
            raise ValueError(f'После фильтрации доступных библиотек в зоопарке < 2 членов: {zoo}')
        self.zoo_used_ = zoo

        y_tr = y_train.values
        y_va = y_valid.values
        n_train = len(y_tr)
        X_tr_feats = X_train[feats]
        X_va_feats = X_valid[feats]

        min_class = int(min(np.bincount(y_tr.astype(int))))
        if min_class < 2:
            raise ValueError(f'Нужно >= 2 примеров каждого класса в train, минимальный класс={min_class}')
        n_splits = min(self.n_folds, min_class)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        folds = list(skf.split(np.zeros(n_train), y_tr))

        oof_matrix = np.zeros((n_train, len(zoo)))
        va_matrix = np.zeros((len(y_va), len(zoo)))
        self.base_models_ = {}
        self.oob_pr_aucs_ = {}

        tuned_member_params: dict[str, dict[str, Any] | None] = {
            name: (self._tune_member(name, X_tr_feats, y_tr, X_va_feats, y_va) if self.n_optuna_trials > 0 else None)
            for name in zoo
        }

        for i, name in enumerate(zoo):
            logger.info('[HeteroStacking] Член зоопарка %d/%d: %s', i + 1, len(zoo), name)
            member_params = tuned_member_params[name]
            for tr_idx_f, te_idx_f in folds:
                m_f = self._fit_member(
                    name, X_tr_feats.iloc[tr_idx_f], y_tr[tr_idx_f], self.random_seed, member_params
                )
                oof_matrix[te_idx_f, i] = self._predict_member(name, m_f, X_tr_feats.iloc[te_idx_f])

            oof_auc = float(average_precision_score(y_tr, oof_matrix[:, i]))
            self.oob_pr_aucs_[name] = oof_auc
            logger.info('[HeteroStacking] %s  OOF PR-AUC=%.4f', name, oof_auc)

            m_final = self._fit_member(name, X_tr_feats, y_tr, self.random_seed, member_params)
            self.base_models_[name] = m_final
            va_matrix[:, i] = self._predict_member(name, m_final, X_va_feats)

        if self.meta == 'logistic':
            self.meta_model_ = self._fit_meta_logistic(oof_matrix, y_tr)
        elif self.meta == 'weighted':
            self.meta_model_ = self._fit_meta_weighted(oof_matrix, y_tr)
        else:
            self.meta_model_ = self._fit_meta_catboost(oof_matrix, y_tr)

        raw_va = self._meta_predict(va_matrix)
        if self.calibrate:
            self.calibrator_ = fit_calibrator(raw_va, y_va)
            self.valid_pred_ = self.calibrator_.predict(raw_va)
        else:
            self.valid_pred_ = raw_va
        self.train_pred_ = self._meta_predict(oof_matrix)

        default_params = {'catboost': _CBT_PARAMS, 'lightgbm': _LGB_PARAMS, 'xgboost': _XGB_PARAMS, 'logistic': {'C': 1.0}}
        self.valid_pr_auc_ = float(average_precision_score(y_va, self.valid_pred_))
        self.best_params_ = {
            'meta': self.meta, 'zoo': zoo, 'n_folds': n_splits,
            'member_params': {
                name: (tuned_member_params[name] or default_params[name]) for name in zoo
            },
        }
        self._model = True

        logger.info('[HeteroStacking] val PR-AUC=%.4f  OOF PR-AUCs=%s',
                    self.valid_pr_auc_, {k: f'{v:.3f}' for k, v in self.oob_pr_aucs_.items()})
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_feats = X[self.selected_features_]
        preds = np.stack([
            self._predict_member(name, self.base_models_[name], X_feats) for name in self.zoo_used_
        ], axis=1)
        raw = self._meta_predict(preds)
        if self.calibrate and self.calibrator_ is not None:
            return self.calibrator_.predict(raw)
        return raw
