"""ModelExplainer — единый интерфейс интерпретации обученных моделей.

Принимает любую модель, унаследованную от BaseModel (LightGBMClassifier,
CatBoostRegressor, DecisionTreeClassifier и т.д.), и предоставляет единый
API для получения важности признаков, SHAP-значений и визуализаций.

Пример использования:
    from ml_toolkit.model_explainer import ModelExplainer

    explainer = ModelExplainer(model, X_valid, y_valid, task='classification')

    imp = explainer.feature_importance()           # pd.Series, важность по убыванию
    sv  = explainer.shap_values()                  # np.ndarray (n_samples, n_features)
    exp = explainer.explain_row(X_valid.iloc[[0]]) # pd.Series, вклад каждого признака

    fig = explainer.plot_importance()
    fig = explainer.plot_shap_beeswarm()
    fig = explainer.plot_shap_waterfall()
    fig = explainer.plot_partial_dependence()
    explainer.plot_intrinsic()                     # EBM/GAM/правила/дерево

    paths = explainer.report('reports/my_model/', prefix='lgbm_cls')
"""

from __future__ import annotations

import io
import logging
import warnings
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_toolkit.model_explainer.intrinsic_visualization import ALL_INTERPRETABLE
from ml_toolkit.models._base import BaseModel

# Определяем наборы имён локально, чтобы не тянуть ml_toolkit.models.__init__
# (который может require optuna/torch при eager-import всех адаптеров).
_TREE_NAMES: frozenset[str] = frozenset({
    'catboost', 'xgboost', 'lightgbm',
    'random_forest', 'extra_trees', 'hist_gbm',
    'quantile_forest', 'oblique_forest', 'mondrian', 'decision_tree',
})
_LINEAR_NAMES: frozenset[str] = frozenset({
    'ridge', 'elasticnet', 'huber', 'tweedie', 'quantile', 'bayesian_ridge',
})
_LIGHTGBM_VARIANTS: frozenset[str] = frozenset({'lightgbm'})
_SKLEARN_TREE_NAMES: frozenset[str] = frozenset({
    'random_forest', 'extra_trees', 'hist_gbm', 'quantile_forest', 'oblique_forest', 'decision_tree',
})

logger = logging.getLogger(__name__)

_SHAP_SUPPORTED: frozenset[str] = _TREE_NAMES - {'mondrian'}

# ── Model name detection ───────────────────────────────────────────────────────

_CLASS_TO_NAME: dict[str, str] = {
    'LightGBMRegressor': 'lightgbm',
    'LightGBMClassifier': 'lightgbm',
    'CatBoostRegressor': 'catboost',
    'CatBoostClassifier': 'catboost',
    'XGBoostRegressor': 'xgboost',
    'XGBoostClassifier': 'xgboost',
    'RandomForestRegressor': 'random_forest',
    'RandomForestClassifier': 'random_forest',
    'ExtraTreesRegressor': 'extra_trees',
    'ExtraTreesClassifier': 'extra_trees',
    'HistGBMRegressor': 'hist_gbm',
    'HistGBMClassifier': 'hist_gbm',
    'DecisionTreeRegressor': 'decision_tree',
    'DecisionTreeClassifier': 'decision_tree',
    'QuantileForestRegressor': 'quantile_forest',
    'QuantileForestClassifier': 'quantile_forest',
    'ObliqueForestRegressor': 'oblique_forest',
    'ObliqueForestClassifier': 'oblique_forest',
    'MondrianForestRegressor': 'mondrian',
    'MondrianForestClassifier': 'mondrian',
    'LinearRegressor': 'ridge',
    'LinearClassifier': 'ridge',
    'EBMRegressor': 'ebm',
    'EBMClassifier': 'ebm',
    'PyGAMRegressor': 'pygam',
    'PyGAMClassifier': 'pygam',
    'MARSRegressor': 'mars',
    'MARSClassifier': 'mars',
    'RuleFitRegressor': 'rulefit',
    'RuleFitClassifier': 'rulefit',
    'IModelsRegressor': 'figs',
    'IModelsClassifier': 'figs',
    'LinearTreeRegressor': 'linear_tree',
    'LinearTreeClassifier': 'linear_tree',
    'InterpretableTreeRegressor': 'soft_decision_tree',
    'InterpretableTreeClassifier': 'soft_decision_tree',
    'InterpretableNeuralRegressor': 'gaminet',
    'InterpretableNeuralClassifier': 'gaminet',
    'LAMARegressor': 'lama',
    'LAMAClassifier': 'lama',
    'TabMRegressor': 'tabm',
    'TabMClassifier': 'tabm',
}


def _detect_name(model: BaseModel) -> str:
    cls_name = type(model).__name__
    name = _CLASS_TO_NAME.get(cls_name, 'unknown')

    if cls_name == 'IModelsClassifier':
        name = getattr(model, 'model_settings', {}).get('name', 'figs')

    if cls_name in ('InterpretableTreeRegressor', 'InterpretableTreeClassifier'):
        inner = type(getattr(model, '_model', None)).__name__
        name = 'locally_linear_forest' if ('Locally' in inner or 'Linear' in inner) else 'soft_decision_tree'

    return name


def _compute_linear_importance(model: BaseModel, feature_names: list[str]) -> np.ndarray | None:
    raw = model._model
    if hasattr(raw, 'named_steps'):
        raw = raw.named_steps.get('estimator', raw[-1])
    if not hasattr(raw, 'coef_'):
        return None
    coef = np.abs(raw.coef_).flatten()
    num_feats = getattr(model, '_num_feats_', feature_names)
    nf_idx = {f: i for i, f in enumerate(num_feats)}
    return np.array([
        coef[nf_idx[f]] if f in nf_idx and nf_idx[f] < len(coef) else 0.0
        for f in feature_names
    ])


def _model_for_intrinsic(model: BaseModel, model_name: str) -> Any:
    """Возвращает аргумент `model` для plot_interpretable_extra в нужном формате."""
    m = model
    if model_name == 'decision_tree':
        return m._model
    if model_name == 'linear_tree':
        return (m._model, m._imputer, m._scaler, m._num_feats_)
    if model_name == 'ebm':
        return m._model
    if model_name == 'pygam':
        return (m._model, m._prep, m._num_feats_)
    if model_name == 'mars':
        if hasattr(m, '_clf'):
            return (m._model, m._imputer, m._clf, m._num_feats_)
        return (m._model, m._imputer, m._num_feats_)
    if model_name in ('rulefit', 'figs', 'skope_rules', 'brl', 'ripper'):
        return (m._model, m._prep, m._num_feats_)
    if model_name in ('soft_decision_tree', 'locally_linear_forest'):
        return (m._model, m._imputer, m._scaler, m._num_feats_)
    if model_name == 'gaminet':
        return (m._model, m._imputer, m._qt, m._num_feats_)
    return m._model


def _save_fig(fig: plt.Figure, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    logger.info('→ %s', path)


# ── ModelExplainer ─────────────────────────────────────────────────────────────

class ModelExplainer:
    """Единый интерфейс для интерпретации обученных моделей.

    Attributes:
        model_name_:         Строковый идентификатор типа модели.
        feature_names_:      Список признаков, использованных при обучении.
        supports_shap_:      True, если для модели доступен SHAP TreeExplainer.
        supports_intrinsic_: True, если доступна нативная интерпретация (EBM/GAM/rules/tree).
    """

    def __init__(
        self,
        model: BaseModel,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        task: str = 'classification',
    ) -> None:
        if not isinstance(model, BaseModel):
            raise TypeError(
                f'Ожидается BaseModel, получено {type(model).__name__}. '
                'Передавайте обученный экземпляр класса из ml_toolkit.models.'
            )
        model._check_fitted()
        if model.selected_features_ is None:
            raise ValueError('Модель не обучена: selected_features_ is None. Вызовите model.fit() сначала.')

        self.model = model
        self.task = task
        self.model_name_: str = _detect_name(model)
        self.feature_names_: list[str] = list(model.selected_features_)

        self.X_valid: pd.DataFrame = X_valid[self.feature_names_].copy()
        self.y_valid: pd.Series = y_valid.copy()

        self._predict_fn: Callable = (
            model.predict_proba if task == 'classification' else model.predict
        )

        self.supports_shap_: bool = self.model_name_ in _SHAP_SUPPORTED
        self.supports_intrinsic_: bool = self.model_name_ in ALL_INTERPRETABLE

        self._importance_cache: dict[str, pd.Series] = {}
        self._shap_cache: tuple[float, np.ndarray] | None = None  # (base_value, shap_values)

    # ── core: importance ────────────────────────────────────────────────────────

    def feature_importance(self, method: str = 'auto', n_repeats: int = 5) -> pd.Series:
        """Важность признаков в виде pd.Series (по убыванию).

        Args:
            method: 'auto' | 'gain' | 'shap' | 'permutation' | 'coef'
                    'auto' — выбирает лучший доступный метод автоматически.
            n_repeats: Число повторений для permutation importance.

        Returns:
            pd.Series с именами признаков в индексе, отсортированный по убыванию.
        """
        cache_key = f'{method}_{n_repeats}'
        if cache_key in self._importance_cache:
            return self._importance_cache[cache_key]
        result = self._compute_importance(method, n_repeats)
        self._importance_cache[cache_key] = result
        return result

    def _compute_importance(self, method: str, n_repeats: int) -> pd.Series:
        feats = self.feature_names_
        name = self.model_name_

        from ml_toolkit.model_explainer.feature_importance import (  # noqa: PLC0415
            _permutation_importance, _tree_importance,
        )

        # gain / встроенная важность для tree-моделей
        if method in ('gain', 'auto') and name in _TREE_NAMES:
            raw = _tree_importance(self.model._model, name, feats)
            if raw is not None:
                return pd.Series(raw, index=feats, name='importance').sort_values(ascending=False)
            if method == 'gain':
                raise ValueError(f'Gain importance недоступен для {name}')

        # |coef| для линейных моделей
        if method in ('coef', 'auto') and name in _LINEAR_NAMES:
            raw = _compute_linear_importance(self.model, feats)
            if raw is not None:
                return pd.Series(raw, index=feats, name='importance').sort_values(ascending=False)
            if method == 'coef':
                raise ValueError(f'Coef importance недоступен для {name}')

        # SHAP mean(|sv|) — точнее permutation
        if method in ('shap', 'auto') and self.supports_shap_:
            try:
                _, sv = self._compute_shap(self.X_valid)
                raw = np.abs(sv).mean(axis=0)
                return pd.Series(raw, index=feats, name='importance').sort_values(ascending=False)
            except Exception:
                if method == 'shap':
                    raise

        # permutation importance — работает для любой модели
        if method in ('permutation', 'auto'):
            raw = _permutation_importance(
                self._predict_fn, feats, self.X_valid, self.y_valid, self.task, n_repeats
            )
            return pd.Series(raw, index=feats, name='importance').sort_values(ascending=False)

        raise ValueError(f'Неизвестный method={method!r}. Выберите: auto|gain|shap|permutation|coef')

    # ── core: shap ─────────────────────────────────────────────────────────────

    def shap_values(self, X: pd.DataFrame | None = None, max_samples: int = 500) -> np.ndarray:
        """SHAP values — массив (n_samples, n_features).

        Поддерживается только для tree-моделей (lightgbm, catboost, xgboost, sklearn-деревья).
        Для остальных моделей используйте feature_importance(method='permutation').

        Args:
            X:           DataFrame для вычисления. None → X_valid.
            max_samples: Максимальное число строк (сэмплирование при превышении).

        Returns:
            numpy array (n_samples, n_features) SHAP values.
        """
        if not self.supports_shap_:
            raise ValueError(
                f'SHAP недоступен для {self.model_name_}. '
                'Используйте feature_importance(method="permutation").'
            )
        X_use = self.X_valid if X is None else X[self.feature_names_].copy()
        if len(X_use) > max_samples:
            X_use = X_use.sample(max_samples, random_state=42)
        _, sv = self._compute_shap(X_use)
        return sv

    def _compute_shap(self, X: pd.DataFrame) -> tuple[float, np.ndarray]:
        """Вычисляет (base_value, shap_values) для TreeExplainer."""
        import shap

        name = self.model_name_
        raw = self.model._model

        if name == 'catboost':
            from catboost import Pool
            cat_idx = list(raw.get_cat_feature_indices())
            ex = shap.TreeExplainer(raw)
            sv = ex.shap_values(Pool(X, cat_features=cat_idx))
            ev = float(np.asarray(ex.expected_value).ravel()[-1])

        elif name in _LIGHTGBM_VARIANTS:
            ex = shap.TreeExplainer(raw)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                sv = ex.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            ev_arr = np.asarray(ex.expected_value).ravel()
            ev = float(ev_arr[1] if len(ev_arr) > 1 else ev_arr[0])

        elif name == 'xgboost':
            ex = shap.TreeExplainer(raw)
            sv = ex.shap_values(X)
            ev = float(ex.expected_value)

        else:  # sklearn-деревья (random_forest, extra_trees, hist_gbm, ...)
            actual = raw[0] if isinstance(raw, tuple) else raw
            X_t = X
            if hasattr(actual, 'named_steps'):
                X_t = pd.DataFrame(actual[:-1].transform(X), columns=X.columns)
                actual = actual.named_steps.get('estimator', actual[-1])
            ex = shap.TreeExplainer(actual)
            sv = ex.shap_values(X_t)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]
            ev_arr = np.asarray(ex.expected_value).ravel()
            ev = float(ev_arr[1] if len(ev_arr) > 1 else ev_arr[0])

        return ev, np.asarray(sv, dtype=float)

    # ── core: row explanation ───────────────────────────────────────────────────

    def explain_row(self, X_row: pd.DataFrame) -> pd.Series:
        """Вклад каждого признака в предсказание для одного наблюдения.

        Метод зависит от типа модели:
        - Tree-модели:   SHAP values (точный, аддитивный).
        - Линейные:      coef × (x_i − mean(X_valid)).
        - Все остальные: ΔMAE/ΔPR-AUC при замене признака на медиану.

        Args:
            X_row: DataFrame из одной строки.

        Returns:
            pd.Series — вклад каждого признака (по убыванию |вклада|).
        """
        X_row = X_row[self.feature_names_].head(1).copy()

        if self.supports_shap_:
            _, sv = self._compute_shap(X_row)
            return (
                pd.Series(sv[0], index=self.feature_names_, name='shap_contribution')
                .sort_values(ascending=False, key=abs)
            )

        if self.model_name_ in _LINEAR_NAMES:
            raw = _compute_linear_importance(self.model, self.feature_names_)
            if raw is not None:
                means = self.X_valid.mean()
                deviations = X_row.iloc[0] - means
                contribs = {f: float(raw[i] * deviations[f]) for i, f in enumerate(self.feature_names_)}
                return (
                    pd.Series(contribs, name='coef_contribution')
                    .sort_values(ascending=False, key=abs)
                )

        # local permutation: заменяем каждый признак медианой, смотрим Δpred
        medians = self.X_valid.median()
        base_pred = float(self._predict_fn(X_row)[0])
        contribs: dict[str, float] = {}
        for feat in self.feature_names_:
            X_tmp = X_row.copy()
            X_tmp[feat] = medians[feat]
            try:
                delta = base_pred - float(self._predict_fn(X_tmp)[0])
            except Exception:
                delta = 0.0
            contribs[feat] = delta
        return (
            pd.Series(contribs, name='local_contribution')
            .sort_values(ascending=False, key=abs)
        )

    # ── plots ───────────────────────────────────────────────────────────────────

    def plot_importance(
        self,
        top_n: int = 30,
        method: str = 'auto',
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Bar chart важности признаков + SHAP beeswarm (для tree-моделей).

        Args:
            top_n:     Число признаков в bar chart.
            method:    Метод расчёта (см. feature_importance).
            save_path: Если задан — сохраняет PNG.

        Returns:
            matplotlib Figure.
        """
        from ml_toolkit.model_explainer.feature_importance import _draw_bar, _try_shap_plot  # noqa: PLC0415

        imp = self.feature_importance(method=method)
        n_show = min(top_n, len(imp))
        fig_h = max(6.0, n_show * 0.36 + 2.5)
        imp_label = self._imp_label(method)

        if self.supports_shap_:
            fig, (ax_bar, ax_shap) = plt.subplots(1, 2, figsize=(22, fig_h))
        else:
            fig, ax_bar = plt.subplots(1, 1, figsize=(11, fig_h))
            ax_shap = None

        _draw_bar(ax_bar, imp, self.model_name_, self.task, top_n, len(imp), imp_label)

        if ax_shap is not None:
            ok = _try_shap_plot(
                ax_shap, self.model._model, self.model_name_,
                self.feature_names_, self.X_valid, self.task, top_n,
            )
            if not ok:
                ax_shap.set_visible(False)

        fig.suptitle(
            f'Feature Importance — {self.model_name_.upper()} [{self.task.capitalize()}]',
            fontsize=11, y=1.01,
        )
        fig.tight_layout()
        if save_path:
            _save_fig(fig, save_path)
        return fig

    def _imp_label(self, method: str) -> str:
        if method == 'auto':
            if self.model_name_ in _TREE_NAMES:
                return 'Feature Importance (gain)'
            if self.model_name_ in _LINEAR_NAMES:
                return '|Coefficient| (scaled)'
            return 'Permutation Importance'
        return {
            'gain': 'Feature Importance (gain)',
            'shap': 'Mean |SHAP|',
            'permutation': 'Permutation Importance (ΔMAE)' if self.task == 'regression'
                           else 'Permutation Importance (ΔPR-AUC)',
            'coef': '|Coefficient| (scaled)',
        }.get(method, 'Importance')

    def plot_shap_beeswarm(
        self,
        X: pd.DataFrame | None = None,
        max_samples: int = 500,
        top_n: int = 20,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """SHAP beeswarm — распределение SHAP values по всей выборке.

        Показывает, как каждый признак влияет на предсказание: цвет = значение признака,
        разброс по оси X = сила влияния.

        Args:
            X:           DataFrame для анализа. None → X_valid.
            max_samples: Максимальное число строк.
            top_n:       Число признаков на графике.
            save_path:   Если задан — сохраняет PNG.
        """
        if not self.supports_shap_:
            raise ValueError(
                f'SHAP недоступен для {self.model_name_}. '
                'Используйте plot_importance(method="permutation").'
            )
        from ml_toolkit.model_explainer.feature_importance import _try_shap_plot  # noqa: PLC0415

        X_use = self.X_valid if X is None else X[self.feature_names_].copy()
        if len(X_use) > max_samples:
            X_use = X_use.sample(max_samples, random_state=42)

        fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.42 + 2)))
        ok = _try_shap_plot(
            ax, self.model._model, self.model_name_,
            self.feature_names_, X_use, self.task, top_n,
        )
        if not ok:
            ax.text(0.5, 0.5, 'SHAP beeswarm недоступен', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)

        fig.suptitle(
            f'SHAP Beeswarm — {self.model_name_.upper()} [{self.task.capitalize()}]  '
            f'(n={len(X_use)})',
            fontsize=11, y=1.01,
        )
        fig.tight_layout()
        if save_path:
            _save_fig(fig, save_path)
        return fig

    def plot_shap_waterfall(
        self,
        X: pd.DataFrame | None = None,
        n_show: int = 9,
        top_n: int = 15,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Waterfall-диаграммы SHAP для отдельных наблюдений.

        Каждый waterfall показывает, как признаки «смещают» предсказание от базового значения
        к конкретному предсказанию для этого наблюдения.

        Args:
            X:        DataFrame для анализа (берёт первые n_show строк). None → X_valid.
            n_show:   Число наблюдений на итоговом графике (сетка ceil(n/3) × 3).
            top_n:    Число признаков в каждом waterfall.
            save_path: Если задан — сохраняет PNG.
        """
        if not self.supports_shap_:
            raise ValueError(f'SHAP недоступен для {self.model_name_}')

        import shap

        X_use = (self.X_valid if X is None else X[self.feature_names_].copy()).head(n_show)
        ev, sv = self._compute_shap(X_use)

        shap_exp = shap.Explanation(
            values=sv,
            base_values=np.full(len(X_use), ev),
            data=X_use.values,
            feature_names=list(self.feature_names_),
        )

        n_actual = len(X_use)
        n_cols = min(3, n_actual)
        n_rows = int(np.ceil(n_actual / n_cols))
        images: list[np.ndarray] = []

        for i in range(n_actual):
            shap.waterfall_plot(shap_exp[i], max_display=top_n, show=False)
            buf = io.BytesIO()
            plt.gcf().savefig(buf, format='png', dpi=180, bbox_inches='tight')
            buf.seek(0)
            plt.close('all')
            images.append(plt.imread(buf))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 4.5))
        axes_flat = np.array(axes).flatten() if n_actual > 1 else np.array([axes])
        for i, img in enumerate(images):
            axes_flat[i].imshow(img, aspect='auto')
            axes_flat[i].axis('off')
        for ax in axes_flat[n_actual:]:
            ax.set_visible(False)

        fig.suptitle(
            f'SHAP Waterfall — {self.model_name_.upper()} [{self.task.capitalize()}]  '
            f'· {n_actual} observations',
            fontsize=11, y=1.01,
        )
        fig.tight_layout()
        if save_path:
            _save_fig(fig, save_path)
        return fig

    def plot_partial_dependence(
        self,
        features: list[str] | None = None,
        top_n: int = 9,
        grid_points: int = 50,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Partial Dependence Plot (PDP) для топ признаков.

        Показывает, как среднее предсказание меняется при изменении одного признака
        (все остальные фиксированы на своих реальных значениях).
        Работает для ЛЮБОЙ модели через predict_fn.

        Args:
            features:    Список признаков. None → топ top_n по feature_importance.
            top_n:       Число признаков, если features=None.
            grid_points: Число точек сетки вдоль оси признака.
            save_path:   Если задан — сохраняет PNG.
        """
        if features is None:
            imp = self.feature_importance()
            features = list(imp.head(top_n).index)

        n = len(features)
        n_cols = min(3, n)
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.8))
        axes_flat = np.array(axes).flatten() if n > 1 else np.array([axes])

        X_ref = self.X_valid.copy()

        for feat, ax in zip(features, axes_flat[:n]):
            lo = float(X_ref[feat].quantile(0.02))
            hi = float(X_ref[feat].quantile(0.98))
            grid = np.linspace(lo, hi, grid_points)
            preds: list[float] = []
            for val in grid:
                X_tmp = X_ref.copy()
                X_tmp[feat] = val
                try:
                    preds.append(float(np.mean(self._predict_fn(X_tmp))))
                except Exception:
                    preds.append(np.nan)

            preds_arr = np.array(preds)
            mean_pred = float(np.nanmean(preds_arr))

            ax.plot(grid, preds_arr, lw=2.2, color='steelblue')
            ax.axhline(mean_pred, color='gray', lw=0.9, linestyle='--', alpha=0.6,
                       label=f'mean={mean_pred:.3f}')
            ax.set_title(feat, fontsize=8.5, pad=4)
            ax.set_xlabel('feature value', fontsize=7.5)
            ax.set_ylabel('mean prediction', fontsize=7.5)
            ax.legend(fontsize=7, framealpha=0.5)
            ax.grid(alpha=0.25)
            ax.spines[['top', 'right']].set_visible(False)

        for ax in axes_flat[n:]:
            ax.set_visible(False)

        fig.suptitle(
            f'Partial Dependence — {self.model_name_.upper()} [{self.task.capitalize()}]  '
            f'(top {n} features)',
            fontsize=11,
        )
        fig.tight_layout()
        if save_path:
            _save_fig(fig, save_path)
        return fig

    def plot_intrinsic(self, save_path: str | Path | None = None) -> bool:
        """Нативная интерпретация модели (shape functions, правила, структура дерева и т.д.).

        Доступна только для интерпретируемых моделей:
        decision_tree, linear_tree, ebm, pygam, mars, rulefit, figs,
        skope_rules, brl, ripper, soft_decision_tree, locally_linear_forest, gaminet.

        Args:
            save_path: Путь для сохранения PNG. Если None — сохраняет во временный файл
                       и пытается показать в Jupyter через IPython.display.

        Returns:
            True если визуализация успешно создана.
        """
        if not self.supports_intrinsic_:
            logger.info('plot_intrinsic недоступен для %s', self.model_name_)
            return False

        raw = _model_for_intrinsic(self.model, self.model_name_)
        cleanup_path: Path | None = None

        if save_path is None:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            save_path = Path(tmp.name)
            tmp.close()
            cleanup_path = save_path

        from ml_toolkit.model_explainer.intrinsic_visualization import plot_interpretable_extra  # noqa: PLC0415
        ok = plot_interpretable_extra(
            raw, self.model_name_, self.feature_names_,
            self.X_valid, save_path=save_path, task=self.task,
        )

        if cleanup_path is not None and cleanup_path.exists():
            if ok:
                try:
                    from IPython.display import Image, display
                    display(Image(str(cleanup_path)))
                except Exception:
                    pass
            try:
                cleanup_path.unlink()
            except Exception:
                pass

        return bool(ok)

    # ── full report ─────────────────────────────────────────────────────────────

    def report(
        self,
        out_dir: str | Path,
        prefix: str = 'model',
    ) -> list[Path]:
        """Генерирует все доступные визуализации и сохраняет их в out_dir.

        Набор графиков зависит от типа модели:
        - Всегда: importance, partial_dependence.
        - Tree-модели: + shap_beeswarm, shap_waterfall.
        - Интерпретируемые: + intrinsic.

        Args:
            out_dir: Директория для сохранения файлов (создаётся автоматически).
            prefix:  Префикс имён файлов.

        Returns:
            Список Path сохранённых PNG-файлов.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []

        def _try(name: str, fn: Callable, **kwargs) -> None:
            path = out_dir / f'{prefix}_{name}.png'
            try:
                fig = fn(save_path=path, **kwargs)
                if fig is not None:
                    plt.close(fig)
                    saved.append(path)
            except Exception:
                logger.warning('report: %s не удался', name, exc_info=True)

        _try('importance', self.plot_importance)
        _try('partial_dependence', self.plot_partial_dependence)

        if self.supports_shap_:
            _try('shap_beeswarm', self.plot_shap_beeswarm)
            _try('shap_waterfall', self.plot_shap_waterfall)

        if self.supports_intrinsic_:
            path = out_dir / f'{prefix}_intrinsic.png'
            try:
                from ml_toolkit.model_explainer.intrinsic_visualization import plot_interpretable_extra  # noqa: PLC0415
                raw = _model_for_intrinsic(self.model, self.model_name_)
                ok = plot_interpretable_extra(
                    raw, self.model_name_, self.feature_names_,
                    self.X_valid, save_path=path, task=self.task,
                )
                if ok:
                    saved.append(path)
            except Exception:
                logger.warning('report: intrinsic не удался', exc_info=True)

        logger.info('report: %d файлов → %s', len(saved), out_dir)
        return saved

    # ── dunder ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        shap_str = 'SHAP✓' if self.supports_shap_ else 'SHAP✗'
        intr_str = 'intrinsic✓' if self.supports_intrinsic_ else 'intrinsic✗'
        return (
            f'ModelExplainer({self.model_name_}, task={self.task}, '
            f'features={len(self.feature_names_)}, {shap_str}, {intr_str})'
        )
