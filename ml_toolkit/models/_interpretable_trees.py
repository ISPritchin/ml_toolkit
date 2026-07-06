"""Интерпретируемые древесные модели: Soft Decision Tree и Locally Linear Forest.

Soft Decision Tree (Irsoy et al. 2012): дифференцируемое дерево с мягкими разбиениями.
Вместо жёстких порогов — сигмоиды; предсказание = взвешенная сумма значений листьев.
Обучается через backpropagation (PyTorch, уже установлен).

Locally Linear Forest: RandomForest proximity weights + локальная Ridge регрессия.
Для каждой точки инференса: ближайшие соседи по RF-proximity → взвешенная Ridge.

Поддерживаемые имена (model_settings['name']): 'soft_decision_tree', 'locally_linear_forest'
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import CLS_METRICS, REG_METRICS, calibrate_proba, fit_calibrator, resolve_metric_fn

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

_MAX_LLF_TRAIN_ROWS = 2000


def _num_features(selected_features: list[str], cat_features: list[str]) -> list[str]:
    cat_set = set(cat_features)
    return [f for f in selected_features if f not in cat_set]


def _fit_prep(
    X_train: pd.DataFrame, X_valid: pd.DataFrame | None, num_feats: list[str],
) -> tuple[np.ndarray, np.ndarray | None, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(imputer.fit_transform(X_train[num_feats].to_numpy(dtype=float)))
    X_va = None
    if X_valid is not None:
        X_va = scaler.transform(imputer.transform(X_valid[num_feats].to_numpy(dtype=float)))
    return X_tr, X_va, imputer, scaler


# ── Soft Decision Tree ────────────────────────────────────────────────────────

class _SoftDecisionTree:
    """Soft Decision Tree с дифференцируемыми разбиениями (PyTorch)."""

    def __init__(self, depth: int = 4, lr: float = 0.01, n_epochs: int = 200, patience: int = 20) -> None:
        self.depth = depth
        self.lr = lr
        self.n_epochs = n_epochs
        self.patience = patience
        self._net: Any = None
        self._is_cls = False

    def _build_net(self, n_features: int, is_cls: bool) -> Any:
        import torch.nn as nn

        depth = self.depth
        n_inner = 2**depth - 1
        n_leaves = 2**depth

        class _SDT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner_w = nn.Linear(n_features, n_inner)
                self.leaf_vals = nn.Parameter(
                    __import__('torch').randn(n_leaves, 1) * 0.1
                )
                self._is_cls = is_cls

            def _path_probs(self, x: Any) -> Any:
                import torch
                batch = x.shape[0]
                probs = torch.ones(batch, 1, device=x.device)
                inner = torch.sigmoid(self.inner_w(x))
                for level in range(depth):
                    n_nodes = 2**level
                    offset = n_nodes - 1
                    left = inner[:, offset:offset + n_nodes]
                    right = 1.0 - left
                    probs = torch.cat([probs * left, probs * right], dim=1)
                return probs

            def forward(self, x: Any) -> Any:
                path_probs = self._path_probs(x)
                out = (path_probs @ self.leaf_vals).squeeze(-1)
                if self._is_cls:
                    return __import__('torch').sigmoid(out)
                return out

        return _SDT()

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, is_cls: bool = False) -> None:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        self._is_cls = is_cls
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = self._build_net(X.shape[1], is_cls).to(device)
        opt = optim.Adam(net.parameters(), lr=self.lr)

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_v = torch.tensor(y_val, dtype=torch.float32, device=device)

        loss_fn = nn.BCELoss() if is_cls else nn.L1Loss()
        best_val, best_state, no_improve = float('inf'), None, 0

        for _ in range(self.n_epochs):
            net.train()
            opt.zero_grad()
            loss_fn(net(X_t), y_t).backward()
            opt.step()
            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(X_v), y_v).item()
            if val_loss < best_val - 1e-7:
                best_val, best_state, no_improve = val_loss, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self._net = net

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        device = next(self._net.parameters()).device
        with torch.no_grad():
            return self._net(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()


# ── Locally Linear Forest ─────────────────────────────────────────────────────

class _LocallyLinearForest:
    """RF proximity weights + локальная Ridge регрессия для каждой точки инференса."""

    def __init__(
        self, n_estimators: int = 100, max_depth: int | None = None,
        n_neighbors: int = 100, ridge_alpha: float = 1.0, random_state: int = 42,
    ) -> None:
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1,
        )
        self.n_neighbors = n_neighbors
        self.ridge_alpha = ridge_alpha
        self._X_tr: np.ndarray | None = None
        self._y_tr: np.ndarray | None = None
        self._leaves_tr: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> '_LocallyLinearForest':
        if len(X) > _MAX_LLF_TRAIN_ROWS:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), size=_MAX_LLF_TRAIN_ROWS, replace=False)
            X, y = X[idx], y[idx]
        self.rf.fit(X, y)
        self._X_tr = X
        self._y_tr = y
        self._leaves_tr = self.rf.apply(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        leaves_pred = self.rf.apply(X)
        preds = np.zeros(len(X))
        for i, leaves in enumerate(leaves_pred):
            proximity = (self._leaves_tr == leaves).mean(axis=1)
            top_idx = np.argsort(proximity)[-self.n_neighbors:]
            weights = proximity[top_idx]
            if weights.sum() < 1e-10:
                preds[i] = self.rf.predict(X[i:i + 1])[0]
                continue
            weights = weights / weights.sum()
            ridge = Ridge(alpha=self.ridge_alpha)
            ridge.fit(self._X_tr[top_idx], self._y_tr[top_idx], sample_weight=weights)
            preds[i] = ridge.predict(X[i:i + 1])[0]
        return preds


# ── Классы (новый API) ────────────────────────────────────────────────────────

class InterpretableTreeRegressor(BaseModel):
    """SoftDecisionTree или LocallyLinearForest для регрессии с подбором через Optuna.

    Dispatch по model_settings['name']: 'soft_decision_tree' | 'locally_linear_forest'.
    Категориальные признаки исключаются. Хранит _imputer, _scaler, _num_feats_.
    params=None → Optuna; params=dict → прямое обучение без тюнинга.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> 'InterpretableTreeRegressor':
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        name = ms.get('name', 'soft_decision_tree')

        self._num_feats_ = _num_features(self.selected_features_, self.cat_features_)
        logger.info('[%s Reg] features=%d', name.upper(), len(self._num_feats_))

        X_tr, X_va, self._imputer, self._scaler = _fit_prep(X_train, X_valid, self._num_feats_)
        y_tr = y_train.to_numpy(dtype=float)
        y_va = y_valid.to_numpy(dtype=float) if y_valid is not None else None

        metric_fn, direction = resolve_metric_fn(ms, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS)

        if name == 'soft_decision_tree':
            if self.params is not None:
                fitted = _SoftDecisionTree(**self.params)
                fitted.fit(X_tr, y_tr, X_va if X_va is not None else X_tr, y_va if y_va is not None else y_tr)
                self.best_params_ = self.params
            else:
                if X_va is None:
                    raise ValueError('X_valid обязателен при params=None (режим Optuna)')

                def objective(trial: optuna.Trial) -> float:
                    sdt = _SoftDecisionTree(
                        depth=trial.suggest_int('depth', 2, 6),
                        lr=trial.suggest_float('lr', 1e-3, 0.1, log=True),
                        n_epochs=trial.suggest_int('n_epochs', 100, 500, step=100),
                    )
                    sdt.fit(X_tr, y_tr, X_va, y_va, is_cls=False)
                    return metric_fn(y_va, sdt.predict(X_va))

                study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(objective, n_trials=max(1, self.n_optuna_trials), show_progress_bar=False)
                self.best_params_ = study.best_params
                logger.info('[%s Reg] Best score=%.4f params=%s', name.upper(), study.best_value, self.best_params_)
                fitted = _SoftDecisionTree(**self.best_params_)
                fitted.fit(X_tr, y_tr, X_va, y_va, is_cls=False)

        else:  # locally_linear_forest
            if self.params is not None:
                fitted = _LocallyLinearForest(**self.params)
                fitted.fit(X_tr, y_tr)
                self.best_params_ = self.params
            else:
                if X_va is None:
                    raise ValueError('X_valid обязателен при params=None (режим Optuna)')

                def objective(trial: optuna.Trial) -> float:  # type: ignore[misc]
                    llf = _LocallyLinearForest(
                        n_estimators=trial.suggest_int('n_estimators', 50, 300, step=50),
                        max_depth=trial.suggest_int('max_depth', 3, 15),
                        n_neighbors=trial.suggest_int('n_neighbors', 20, 200, step=20),
                        ridge_alpha=trial.suggest_float('ridge_alpha', 0.01, 100.0, log=True),
                    )
                    llf.fit(X_tr, y_tr)
                    return metric_fn(y_va, llf.predict(X_va))

                study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(objective, n_trials=max(1, self.n_optuna_trials), show_progress_bar=False)
                self.best_params_ = study.best_params
                logger.info('[%s Reg] Best score=%.4f params=%s', name.upper(), study.best_value, self.best_params_)
                fitted = _LocallyLinearForest(**self.best_params_)
                fitted.fit(X_tr, y_tr)

        self._model = fitted
        self.train_pred_ = self._model.predict(X_tr)
        if X_va is not None:
            self.valid_pred_ = self._model.predict(X_va)
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_t = self._scaler.transform(self._imputer.transform(X[self._num_feats_].to_numpy(dtype=float)))
        return np.asarray(self._model.predict(X_t))


class InterpretableTreeClassifier(BaseModel):
    """SoftDecisionTree или LocallyLinearForest (RF-based) для классификации.

    Dispatch по model_settings['name']: 'soft_decision_tree' | 'locally_linear_forest'.
    Категориальные признаки исключаются. Вероятности калибруются изотонической регрессией.
    params=None → Optuna; params=dict → прямое обучение без тюнинга.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> 'InterpretableTreeClassifier':
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        name = ms.get('name', 'soft_decision_tree')

        self._num_feats_ = _num_features(self.selected_features_, self.cat_features_)
        logger.info('[%s Cls] features=%d', name.upper(), len(self._num_feats_))

        X_tr, X_va, self._imputer, self._scaler = _fit_prep(X_train, X_valid, self._num_feats_)
        y_tr = y_train.to_numpy(dtype=int)
        y_va = y_valid.to_numpy(dtype=int) if y_valid is not None else None

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

        if name == 'soft_decision_tree':
            if self.params is not None:
                fitted = _SoftDecisionTree(**self.params)
                fitted.fit(X_tr, y_tr.astype(float), X_va if X_va is not None else X_tr,
                           y_va.astype(float) if y_va is not None else y_tr.astype(float), is_cls=True)
                self.best_params_ = self.params
            else:
                if X_va is None:
                    raise ValueError('X_valid обязателен при params=None (режим Optuna)')

                def objective(trial: optuna.Trial) -> float:
                    sdt = _SoftDecisionTree(
                        depth=trial.suggest_int('depth', 2, 6),
                        lr=trial.suggest_float('lr', 1e-3, 0.1, log=True),
                        n_epochs=trial.suggest_int('n_epochs', 100, 500, step=100),
                    )
                    sdt.fit(X_tr, y_tr.astype(float), X_va, y_va.astype(float), is_cls=True)
                    return metric_fn(y_va, np.clip(sdt.predict(X_va), 0.0, 1.0))

                study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(objective, n_trials=max(1, self.n_optuna_trials), show_progress_bar=False)
                self.best_params_ = study.best_params
                logger.info('[%s Cls] Best score=%.4f params=%s', name.upper(), study.best_value, self.best_params_)
                fitted = _SoftDecisionTree(**self.best_params_)
                fitted.fit(X_tr, y_tr.astype(float), X_va, y_va.astype(float), is_cls=True)

            self._model = fitted
            self._is_sdt = True
            self.train_pred_ = np.clip(self._model.predict(X_tr), 0.0, 1.0)
            if X_va is not None:
                self.valid_pred_ = np.clip(self._model.predict(X_va), 0.0, 1.0)

        else:  # locally_linear_forest → RF classifier
            if self.params is not None:
                fitted = RandomForestClassifier(**self.params)
                fitted.fit(X_tr, y_tr)
                self.best_params_ = self.params
            else:
                if X_va is None:
                    raise ValueError('X_valid обязателен при params=None (режим Optuna)')

                def objective(trial: optuna.Trial) -> float:  # type: ignore[misc]
                    m = RandomForestClassifier(
                        n_estimators=trial.suggest_int('n_estimators', 50, 300, step=50),
                        max_depth=trial.suggest_int('max_depth', 3, 15),
                        min_samples_leaf=trial.suggest_int('min_samples_leaf', 5, 50),
                        class_weight='balanced', random_state=42, n_jobs=-1,
                    )
                    m.fit(X_tr, y_tr)
                    return metric_fn(y_va, m.predict_proba(X_va)[:, 1])

                study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(objective, n_trials=max(1, self.n_optuna_trials), show_progress_bar=False)
                self.best_params_ = {**study.best_params, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}
                logger.info('[%s Cls] Best score=%.4f params=%s', name.upper(), study.best_value, self.best_params_)
                fitted = RandomForestClassifier(**self.best_params_)
                fitted.fit(X_tr, y_tr)

            self._model = fitted
            self._is_sdt = False
            self.train_pred_ = self._model.predict_proba(X_tr)[:, 1]
            if X_va is not None:
                self.valid_pred_ = self._model.predict_proba(X_va)[:, 1]

        if X_va is not None:
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_t = self._scaler.transform(self._imputer.transform(X[self._num_feats_].to_numpy(dtype=float)))
        if self._is_sdt:
            raw = np.clip(self._model.predict(X_t), 0.0, 1.0)
        else:
            raw = self._model.predict_proba(X_t)[:, 1]
        return self.calibrator_.predict(raw) if self.calibrator_ is not None else raw


# ── Backward-compat functional wrappers ──────────────────────────────────────

def train_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_inference: pd.DataFrame,
    selected_features: list[str],
    cat_features: list[str],
    model_settings: dict[str, Any],
    n_optuna_trials: int,
    postprocess_fn: Callable[[pd.DataFrame, np.ndarray], np.ndarray] | None = None,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict]:
    model = InterpretableTreeRegressor(n_optuna_trials=n_optuna_trials, model_settings=model_settings)
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    _pp = postprocess_fn or (lambda _X, p: p)
    name = model_settings.get('name', 'soft_decision_tree')
    train_pred = _pp(X_train, model.train_pred_)
    valid_pred = _pp(X_valid, model.valid_pred_)
    infer_pred = _pp(X_inference, model.predict(X_inference))
    logger.info('[%s Reg] Final MAE: %.3f', name.upper(), mean_absolute_error(y_valid, valid_pred))
    return (model._model, model._imputer, model._scaler, model._num_feats_), train_pred, valid_pred, infer_pred, model.best_params_


def train_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_inference: pd.DataFrame,
    selected_features: list[str],
    cat_features: list[str],
    n_optuna_trials: int,
    model_settings: dict[str, Any] | None = None,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict]:
    ms = model_settings or {}
    name = ms.get('name', 'soft_decision_tree')
    model = InterpretableTreeClassifier(n_optuna_trials=n_optuna_trials, model_settings=ms)
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    infer_proba = model.predict_proba(X_inference)
    logger.info('[%s Cls] Final PR-AUC: %.3f', name.upper(), average_precision_score(y_valid, model.valid_pred_))
    return (model._model, model._imputer, model._scaler, model._num_feats_), model.train_pred_, model.valid_pred_, infer_proba, model.best_params_


def make_predict_fn(model: Any, task: str, selected_features: list[str]) -> Any:
    """Возвращает callable (X → np.ndarray) с imputer+scaler препроцессингом для permutation importance."""
    import numpy as _np  # noqa: PLC0415
    _m, _imp, _sc, _nf = model
    if task == 'regression':
        return lambda X: _np.asarray(_m.predict(_sc.transform(_imp.transform(X[_nf].to_numpy(dtype=float)))))
    if hasattr(_m, 'predict_proba'):
        return lambda X: _m.predict_proba(_sc.transform(_imp.transform(X[_nf].to_numpy(dtype=float))))[:, 1]
    return lambda X: _np.clip(_m.predict(_sc.transform(_imp.transform(X[_nf].to_numpy(dtype=float)))), 0.0, 1.0)
