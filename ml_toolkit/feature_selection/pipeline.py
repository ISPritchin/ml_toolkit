"""FeatureSelectionPipeline: единый пайплайн отбора признаков.

Три стадии применяются последовательно:
  1. Структурный скрининг  — NaN, константы, квазиконстанты, дубликаты.
  2. Релевантность          — однофакторный ROC-AUC по y_train.
  3. Drift-фильтр           — adversarial validation train vs valid.

Каждая стадия опциональна. transform() применяется к любому датасету (X_train,
X_valid, X_test) одинаково.

Пример::

    from ml_toolkit.feature_selection.pipeline import FeatureSelectionPipeline

    pipeline = FeatureSelectionPipeline(
        min_univariate_auc=0.52,
        adversarial_target_auc=0.55,
    )
    pipeline.fit(X_train, y_train, X_valid, y_valid)

    X_train_clean = pipeline.transform(X_train)
    X_valid_clean = pipeline.transform(X_valid)
    X_test_clean  = pipeline.transform(X_test)

    print(pipeline.summary())
    pipeline.report().to_csv('feature_selection_report.csv', index=False)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ml_toolkit.feature_selection.drift_filter import (
    AdversarialDriftFilter,
    compute_psi,
)

logger = logging.getLogger(__name__)


# ─── Структурный скрининг (pandas-native) ────────────────────────────────────

def _null_rate(s: pd.Series) -> float:
    return float(s.isna().mean())


def _variance(s: pd.Series) -> float:
    vals = s.dropna()
    if len(vals) < 2:
        return 0.0
    try:
        return float(vals.astype(float).var(ddof=0))
    except (TypeError, ValueError):
        return 0.0


def _quasi_constant_rate(s: pd.Series) -> float:
    vals = s.dropna()
    if len(vals) == 0:
        return 1.0
    return float(vals.value_counts(normalize=True).iloc[0])


def _arrays_equal_numpy(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    try:
        a = a.astype(float)
        b = b.astype(float)
    except (TypeError, ValueError):
        return bool(np.array_equal(a, b))
    return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def _univariate_auc(col: pd.Series, y: np.ndarray, subsample: int | None) -> float:
    from sklearn.metrics import roc_auc_score

    mask = ~col.isna()
    if mask.sum() < 10:
        return 0.5

    vals = col[mask].values
    try:
        vals = vals.astype(float)
    except (TypeError, ValueError):
        return 0.5

    target = y[mask.values]

    if len(np.unique(target)) < 2:
        return 0.5

    if subsample and len(vals) > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(vals), size=subsample, replace=False)
        vals, target = vals[idx], target[idx]
        if len(np.unique(target)) < 2:
            return 0.5

    try:
        auc = roc_auc_score(target, vals)
        return float(max(auc, 1.0 - auc))
    except Exception:
        return 0.5


# ─── Pipeline ────────────────────────────────────────────────────────────────

class FeatureSelectionPipeline:
    """Трёхэтапный пайплайн отбора признаков для задач классификации.

    Этапы (каждый опционален):

    1. **Структурный скрининг** — удаляет признаки с высокой долей NaN,
       нулевой/малой дисперсией, квазиконстанты и точные дубликаты.

    2. **Релевантность** — удаляет признаки с низким однофакторным ROC-AUC
       относительно целевой переменной `y_train`.

    3. **Drift-фильтр** — adversarial validation: обучает CatBoost отличать
       train от valid; итеративно удаляет признаки с высокой adversarial-
       важностью до достижения целевого AUC ≤ target.

    Parameters
    ----------
    max_null_rate:
        Порог доли NaN (этап 1).
    min_variance:
        Минимальная дисперсия (этап 1).
    max_quasi_constant_rate:
        Максимальная доля доминирующего значения (этап 1).
    drop_duplicates:
        Удалять ли точные дубликаты признаков (этап 1).
    min_univariate_auc:
        Минимальный однофакторный ROC-AUC (этап 2).
        0.0 → этап 2 отключён.
    auc_subsample:
        Подвыборка строк для расчёта AUC (None → все строки).
    use_drift_filter:
        Включить ли adversarial drift-фильтр (этап 3).
    adversarial_target_auc:
        Целевой adversarial AUC. Чем ближе к 0.5, тем жёстче фильтрация.
    max_drift_features_to_drop:
        Максимум удаляемых drift-признаков (None = без ограничения).
    remove_per_step:
        Признаков удаляемых за одну adversarial итерацию.
    cat_features:
        Категориальные признаки для adversarial CatBoost.
    compute_psi_report:
        Вычислить PSI для drift-отчёта (информационно, не влияет на отбор).

    Атрибуты после fit::

        selected_features_          — итоговый список признаков
        stage1_removed_             — dict {feature: причина} этап 1
        stage2_removed_             — список признаков этап 2
        stage3_removed_             — список признаков этап 3 (drift)
        adversarial_auc_initial_    — adversarial AUC до drift-фильтра
        adversarial_auc_final_      — adversarial AUC после drift-фильтра
        psi_report_                 — DataFrame с PSI (если compute_psi_report=True)

    Пример::

        pipeline = FeatureSelectionPipeline(
            min_univariate_auc=0.52,
            adversarial_target_auc=0.55,
            remove_per_step=3,
        )
        pipeline.fit(X_train, y_train, X_valid, y_valid)

        print(pipeline.summary())
        print(pipeline.report())

        X_train_clean = pipeline.transform(X_train)
        X_valid_clean = pipeline.transform(X_valid)
        X_test_clean  = pipeline.transform(X_test)

    """

    def __init__(
        self,
        # Этап 1: структурный скрининг
        max_null_rate: float = 0.95,
        min_variance: float = 1e-5,
        max_quasi_constant_rate: float = 0.95,
        drop_duplicates: bool = False,
        # Этап 2: релевантность
        min_univariate_auc: float = 0.52,
        auc_subsample: int | None = 50_000,
        # Этап 3: drift
        use_drift_filter: bool = True,
        adversarial_target_auc: float = 0.55,
        max_drift_features_to_drop: int | None = None,
        remove_per_step: int = 1,
        cat_features: list[str] | None = None,
        compute_psi_report: bool = True,
        # Adversarial model params
        cb_iterations: int = 300,
        cb_max_depth: int = 4,
    ):
        self.max_null_rate = max_null_rate
        self.min_variance = min_variance
        self.max_quasi_constant_rate = max_quasi_constant_rate
        self.drop_duplicates = drop_duplicates
        self.min_univariate_auc = min_univariate_auc
        self.auc_subsample = auc_subsample
        self.use_drift_filter = use_drift_filter
        self.adversarial_target_auc = adversarial_target_auc
        self.max_drift_features_to_drop = max_drift_features_to_drop
        self.remove_per_step = remove_per_step
        self.cat_features = cat_features or []
        self.compute_psi_report = compute_psi_report
        self.cb_iterations = cb_iterations
        self.cb_max_depth = cb_max_depth

        self.selected_features_: list[str] = []
        self.stage1_removed_: dict[str, str] = {}
        self.stage2_removed_: list[str] = []
        self.stage3_removed_: list[str] = []
        self.adversarial_auc_initial_: float | None = None
        self.adversarial_auc_final_: float | None = None
        self.psi_report_: pd.DataFrame | None = None
        self._report_rows: list[dict] = []
        self._fitted = False

    # ── Этап 1: структурный скрининг ────────────────────────────────────────

    def _stage1(self, X: pd.DataFrame) -> list[str]:
        kept = []
        removed: dict[str, str] = {}

        for col in X.columns:
            s = X[col]
            nr = _null_rate(s)
            if nr > self.max_null_rate:
                removed[col] = 'high_null_rate'
                continue

            var = _variance(s)
            if var < self.min_variance:
                removed[col] = 'low_variance'
                continue

            qcr = _quasi_constant_rate(s)
            if qcr > self.max_quasi_constant_rate:
                removed[col] = 'quasi_constant'
                continue

            kept.append(col)

        if self.drop_duplicates and len(kept) > 1:
            seen: list[tuple[str, np.ndarray]] = []
            deduped: list[str] = []
            for col in kept:
                arr = X[col].values
                is_dup = any(_arrays_equal_numpy(arr, prev) for _, prev in seen)
                if is_dup:
                    removed[col] = 'duplicate'
                else:
                    seen.append((col, arr))
                    deduped.append(col)
            kept = deduped

        self.stage1_removed_ = removed
        n_rem = len(removed)
        logger.info('[Pipeline] Этап 1 (структурный): удалено %d / %d  → осталось %d',
                    n_rem, len(X.columns), len(kept))
        return kept

    # ── Этап 2: релевантность ────────────────────────────────────────────────

    def _stage2(self, X: pd.DataFrame, y: np.ndarray, candidates: list[str]) -> list[str]:
        if self.min_univariate_auc <= 0.0:
            logger.info('[Pipeline] Этап 2 (AUC) отключён')
            self.stage2_removed_ = []
            return candidates

        kept, removed = [], []
        for col in candidates:
            auc = _univariate_auc(X[col], y, self.auc_subsample)
            if auc < self.min_univariate_auc:
                removed.append(col)
            else:
                kept.append(col)

        self.stage2_removed_ = removed
        logger.info('[Pipeline] Этап 2 (AUC ≥ %.2f): удалено %d → осталось %d',
                    self.min_univariate_auc, len(removed), len(kept))
        return kept

    # ── Этап 3: drift-фильтр ────────────────────────────────────────────────

    def _stage3(
        self,
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        candidates: list[str],
    ) -> list[str]:
        if not self.use_drift_filter:
            logger.info('[Pipeline] Этап 3 (drift) отключён')
            self.stage3_removed_ = []
            return candidates

        if self.compute_psi_report:
            try:
                self.psi_report_ = compute_psi(X_train[candidates], X_valid[candidates])
                high_psi = self.psi_report_[self.psi_report_['drift_level'] == 'high']
                if len(high_psi) > 0:
                    logger.info('[Pipeline] PSI > 0.25 у %d признаков: %s',
                                len(high_psi), high_psi['feature'].tolist()[:10])
            except Exception as e:
                logger.warning('[Pipeline] PSI не удалось вычислить: %s', e)

        adf = AdversarialDriftFilter(
            target_auc=self.adversarial_target_auc,
            max_features_to_drop=self.max_drift_features_to_drop,
            remove_per_step=self.remove_per_step,
            cat_features=self.cat_features,
            cb_iterations=self.cb_iterations,
            cb_max_depth=self.cb_max_depth,
        )
        adf.fit(X_train[candidates], X_valid[candidates])

        self._adf = adf
        self.stage3_removed_ = adf.removed_features_
        self.adversarial_auc_initial_ = adf.adversarial_auc_history_[0] if adf.adversarial_auc_history_ else None
        self.adversarial_auc_final_ = adf.adversarial_auc_history_[-1] if adf.adversarial_auc_history_ else None

        kept = adf.selected_features_
        logger.info('[Pipeline] Этап 3 (drift AUC target≤%.2f): удалено %d → осталось %d',
                    self.adversarial_target_auc, len(self.stage3_removed_), len(kept))
        return kept

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Any,
        X_valid: pd.DataFrame,
        y_valid: Any = None,
        selected_features: list[str] | None = None,
    ) -> FeatureSelectionPipeline:
        """Обучить пайплайн на X_train/y_train с учётом X_valid.

        Args:
            X_train: Обучающая выборка (pandas DataFrame).
            y_train: Целевая переменная (бинарная, для AUC-фильтра).
            X_valid: Валидационная выборка (для drift-фильтра).
            y_valid: Не используется в фильтрации, только для совместимости.
            selected_features: Если задан — сначала ограничиваем X_train этим
                списком, затем применяем фильтры.

        Returns:
            self

        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError('X_train должен быть pandas DataFrame')
        if not isinstance(X_valid, pd.DataFrame):
            raise TypeError('X_valid должен быть pandas DataFrame')

        y_arr = np.asarray(y_train)

        if selected_features is not None:
            X_train = X_train[[f for f in selected_features if f in X_train.columns]]
            X_valid = X_valid[[f for f in selected_features if f in X_valid.columns]]

        logger.info('[Pipeline] Старт: %d признаков', len(X_train.columns))

        # Этапы 1 и 2 только на train
        candidates = self._stage1(X_train)
        candidates = self._stage2(X_train, y_arr, candidates)

        # Этап 3: drift между train и valid
        X_va_common = X_valid[[c for c in candidates if c in X_valid.columns]]
        if len(candidates) > 0 and len(X_va_common.columns) > 0:
            candidates = self._stage3(X_train, X_va_common, candidates)
        else:
            logger.warning('[Pipeline] Нет общих признаков train/valid для drift-фильтра')
            self.stage3_removed_ = []

        self.selected_features_ = candidates
        self._build_report(X_train, y_arr)
        self._fitted = True

        logger.info('[Pipeline] Итог: %d / %d признаков оставлено',
                    len(self.selected_features_), len(X_train.columns))
        return self

    # ── transform ───────────────────────────────────────────────────────────

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Выбрать отобранные признаки из X.

        Работает для X_train, X_valid и X_test.
        """
        if not self._fitted:
            raise RuntimeError('FeatureSelectionPipeline не обучен. Вызовите fit() первым.')
        missing = [f for f in self.selected_features_ if f not in X.columns]
        if missing:
            raise ValueError(f'В X отсутствуют признаки: {missing}')
        return X[self.selected_features_]

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: Any,
        X_valid: pd.DataFrame,
        y_valid: Any = None,
    ) -> pd.DataFrame:
        """fit() + transform(X_train)."""
        return self.fit(X_train, y_train, X_valid, y_valid).transform(X_train)

    # ── Отчёты ──────────────────────────────────────────────────────────────

    def _build_report(self, X_train: pd.DataFrame, y_arr: np.ndarray) -> None:
        stage1_map = self.stage1_removed_
        stage2_set = set(self.stage2_removed_)
        stage3_set = set(self.stage3_removed_)
        kept_set = set(self.selected_features_)

        rows = []
        for col in X_train.columns:
            if col in stage1_map:
                stage = 'stage1_structural'
                reason = stage1_map[col]
            elif col in stage2_set:
                stage = 'stage2_relevance'
                reason = 'low_auc'
            elif col in stage3_set:
                stage = 'stage3_drift'
                reason = 'adversarial_importance'
            else:
                stage = None
                reason = None

            # Основные статистики
            s = X_train[col]
            auc = _univariate_auc(s, y_arr, self.auc_subsample) if col not in stage1_map else float('nan')

            rows.append({
                'feature': col,
                'null_rate': _null_rate(s),
                'variance': _variance(s),
                'quasi_constant_rate': _quasi_constant_rate(s),
                'univariate_auc': auc,
                'kept': col in kept_set,
                'stage_removed': stage,
                'removed_by': reason,
            })

        self._report_rows = rows

    def report(self) -> pd.DataFrame:
        """Полный отчёт: статистики и причина удаления по каждому признаку.

        Columns: feature, null_rate, variance, quasi_constant_rate,
                 univariate_auc, kept, stage_removed, removed_by.
        """
        if not self._fitted:
            raise RuntimeError('FeatureSelectionPipeline не обучен.')
        return pd.DataFrame(self._report_rows)

    def summary(self) -> str:
        """Краткая сводка результатов фильтрации."""
        if not self._fitted:
            raise RuntimeError('FeatureSelectionPipeline не обучен.')

        total = len(self._report_rows)
        n1 = len(self.stage1_removed_)
        n2 = len(self.stage2_removed_)
        n3 = len(self.stage3_removed_)
        nk = len(self.selected_features_)

        lines = [
            f'FeatureSelectionPipeline  |  {total} признаков на входе',
            f'  Этап 1 (структурный):   удалено {n1:4d}',
            f'  Этап 2 (AUC ≥ {self.min_univariate_auc:.2f}):  удалено {n2:4d}',
        ]
        if self.use_drift_filter:
            auc_str = (
                f'{self.adversarial_auc_initial_:.4f} → {self.adversarial_auc_final_:.4f}'
                if self.adversarial_auc_initial_ is not None else 'N/A'
            )
            lines.append(f'  Этап 3 (drift AUC={auc_str}):  удалено {n3:4d}')
        lines.append(f'  ИТОГО оставлено:        {nk:4d}')
        return '\n'.join(lines)

    def removal_summary(self) -> pd.DataFrame:
        """Агрегированная сводка по причинам удаления."""
        if not self._fitted:
            raise RuntimeError('FeatureSelectionPipeline не обучен.')
        rows = [r for r in self._report_rows if not r['kept']]
        if not rows:
            return pd.DataFrame(columns=['причина', 'признаков'])
        df = pd.DataFrame(rows)
        counts = df['removed_by'].value_counts().reset_index()
        counts.columns = ['причина', 'признаков']
        total = pd.DataFrame([{'причина': 'ИТОГО удалено', 'признаков': len(rows)}])
        return pd.concat([counts, total], ignore_index=True)
