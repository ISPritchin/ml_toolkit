"""Adversarial Validation для обнаружения и устранения смещения распределений.

AdversarialDriftFilter:
    Обучает CatBoost-классификатор отличать train от valid.
    Adversarial AUC ≈ 0.5 → смещения нет; > target_auc → смещение есть.
    Итеративно удаляет признаки с наибольшей adversarial-важностью до
    достижения target_auc.

compute_psi(X_train, X_valid):
    Population Stability Index — быстрая диагностика без модели.
    PSI < 0.10: стабильно, 0.10–0.25: умеренное смещение, > 0.25: критическое.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── PSI ─────────────────────────────────────────────────────────────────────

def _psi_one_feature(
    train_vals: np.ndarray,
    valid_vals: np.ndarray,
    n_bins: int,
    min_bin_count: int,
) -> float:
    eps = 1e-6
    _, edges = np.histogram(train_vals, bins=n_bins)
    edges[0] = -np.inf
    edges[-1] = np.inf

    tr_cnt = np.histogram(train_vals, bins=edges)[0]
    va_cnt = np.histogram(valid_vals, bins=edges)[0]

    mask = (tr_cnt >= min_bin_count) & (va_cnt >= min_bin_count)
    if mask.sum() < 2:
        return 0.0

    tr_p = tr_cnt[mask] / len(train_vals)
    va_p = va_cnt[mask] / len(valid_vals)
    return float(np.sum((va_p - tr_p) * np.log((va_p + eps) / (tr_p + eps))))


def compute_psi(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    n_bins: int = 10,
    min_bin_count: int = 5,
) -> pd.DataFrame:
    """Population Stability Index по каждому числовому признаку.

    Быстрая диагностика смещения без обучения модели. Для категориальных
    признаков PSI считается по частотам уникальных значений (вместо бинов).

    Args:
        X_train: Обучающая выборка (pandas DataFrame).
        X_valid: Валидационная выборка (pandas DataFrame).
        n_bins: Число бинов для числовых признаков.
        min_bin_count: Минимальное число наблюдений в бине с каждой стороны
            — бины с меньшим числом наблюдений исключаются из расчёта PSI.

    Returns:
        DataFrame с колонками ``feature``, ``psi``, ``drift_level``,
        отсортированный по убыванию PSI.

    Interpretation::

        PSI < 0.10   → stable (нет значимого смещения)
        PSI 0.10–0.25 → moderate (умеренное смещение)
        PSI > 0.25   → high (критическое смещение, признак под подозрением)

    Пример::

        from ml_toolkit.feature_selection.drift_filter import compute_psi
        psi_report = compute_psi(X_train, X_valid)
        print(psi_report[psi_report['drift_level'] == 'high'])

    """
    common = [c for c in X_train.columns if c in X_valid.columns]
    rows = []
    for col in common:
        tr = X_train[col].dropna().values
        va = X_valid[col].dropna().values
        if len(tr) < min_bin_count or len(va) < min_bin_count:
            rows.append({'feature': col, 'psi': 0.0})
            continue

        # Категориальные — PSI по частотам значений
        if not np.issubdtype(tr.dtype, np.number):
            tr_freq = pd.Series(tr).value_counts(normalize=True)
            va_freq = pd.Series(va).value_counts(normalize=True)
            all_vals = set(tr_freq.index) | set(va_freq.index)
            eps = 1e-6
            psi = sum(
                (va_freq.get(v, eps) - tr_freq.get(v, eps))
                * np.log((va_freq.get(v, eps) + eps) / (tr_freq.get(v, eps) + eps))
                for v in all_vals
            )
            rows.append({'feature': col, 'psi': float(psi)})
        else:
            rows.append({'feature': col, 'psi': _psi_one_feature(tr, va, n_bins, min_bin_count)})

    df = pd.DataFrame(rows).sort_values('psi', ascending=False).reset_index(drop=True)
    df['drift_level'] = pd.cut(
        df['psi'],
        bins=[-np.inf, 0.10, 0.25, np.inf],
        labels=['stable', 'moderate', 'high'],
    )
    return df


# ─── Adversarial Drift Filter ─────────────────────────────────────────────────

class AdversarialDriftFilter:
    """Удаляет признаки, вызывающие смещение train/valid, через adversarial validation.

    Алгоритм::

        1. Объединяем X_train (label=0) + X_valid (label=1) → adversarial датасет.
        2. Обучаем CatBoost (70% train / 30% test) и оцениваем ROC-AUC.
        3. Если AUC > target_auc → удаляем remove_per_step признаков
           с наибольшей adversarial-важностью.
        4. Повторяем до AUC ≤ target_auc или исчерпания лимита.

    После fit::

        selected_features_      — признаки без значимого drift
        removed_features_       — список удалённых (в порядке удаления)
        adversarial_auc_history_ — AUC на каждой итерации
        feature_importances_    — важность признаков из последнего adversarial run

    Parameters
    ----------
    target_auc:
        Целевой adversarial AUC. 0.55 — мягкий порог, 0.50 — максимальный
        (полное отсутствие дискриминации).
    max_features_to_drop:
        Максимальное число удаляемых признаков (None = без ограничения).
    remove_per_step:
        Сколько признаков удалять за одну итерацию. > 1 ускоряет работу при
        большом числе дрейфующих признаков.
    cat_features:
        Категориальные признаки для CatBoost adversarial модели.
    cb_iterations:
        Число итераций CatBoost adversarial модели.
    cb_max_depth:
        Глубина деревьев adversarial модели (мелкая → меньше переобучения).

    Пример::

        from ml_toolkit.feature_selection.drift_filter import AdversarialDriftFilter

        adf = AdversarialDriftFilter(target_auc=0.55, remove_per_step=2)
        adf.fit(X_train, X_valid)
        print(f"Удалено {len(adf.removed_features_)} признаков: {adf.removed_features_}")
        print(f"AUC: {adf.adversarial_auc_history_[0]:.3f} → {adf.adversarial_auc_history_[-1]:.3f}")

        X_train_clean = adf.transform(X_train)
        X_valid_clean = adf.transform(X_valid)
        X_test_clean  = adf.transform(X_test)

    """

    def __init__(
        self,
        target_auc: float = 0.55,
        max_features_to_drop: int | None = None,
        remove_per_step: int = 1,
        cat_features: list[str] | None = None,
        cb_iterations: int = 300,
        cb_max_depth: int = 4,
        cb_learning_rate: float = 0.05,
        random_seed: int = 42,
    ):
        self.target_auc = target_auc
        self.max_features_to_drop = max_features_to_drop
        self.remove_per_step = remove_per_step
        self.cat_features = cat_features or []
        self.cb_iterations = cb_iterations
        self.cb_max_depth = cb_max_depth
        self.cb_learning_rate = cb_learning_rate
        self.random_seed = random_seed

        self.selected_features_: list[str] = []
        self.removed_features_: list[str] = []
        self.adversarial_auc_history_: list[float] = []
        self.feature_importances_: pd.Series | None = None
        self._fitted = False

    # ── Обучение adversarial модели ──────────────────────────────────────────

    def _train_adversarial(
        self,
        X_combined: pd.DataFrame,
        y_combined: np.ndarray,
        current_features: list[str],
    ) -> tuple[float, pd.Series]:
        from catboost import CatBoostClassifier, Pool
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        X_sub = X_combined[current_features]
        cat_f = [f for f in self.cat_features if f in current_features]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sub, y_combined, test_size=0.30, stratify=y_combined,
            random_state=self.random_seed,
        )

        params: dict[str, Any] = {
            'iterations': self.cb_iterations,
            'max_depth': self.cb_max_depth,
            'learning_rate': self.cb_learning_rate,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'early_stopping_rounds': 30,
            'random_seed': self.random_seed,
            'verbose': 0,
        }

        tr_pool = Pool(X_tr, y_tr, cat_features=cat_f)
        te_pool = Pool(X_te, y_te, cat_features=cat_f)

        model = CatBoostClassifier(**params)
        model.fit(tr_pool, eval_set=te_pool, verbose=False)

        proba = model.predict_proba(te_pool)[:, 1]
        auc = float(roc_auc_score(y_te, proba))

        importances = pd.Series(
            model.get_feature_importance(tr_pool),
            index=current_features,
        )
        return auc, importances

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
    ) -> AdversarialDriftFilter:
        """Выявить и устранить drift между X_train и X_valid.

        Args:
            X_train: Обучающая выборка.
            X_valid: Валидационная выборка.

        Returns:
            self

        """
        common = [c for c in X_train.columns if c in X_valid.columns]
        if not common:
            raise ValueError('X_train и X_valid не имеют общих признаков.')

        X_combined = pd.concat(
            [X_train[common].reset_index(drop=True),
             X_valid[common].reset_index(drop=True)],
            ignore_index=True,
        )
        y_combined = np.array([0] * len(X_train) + [1] * len(X_valid))

        current_features = list(common)
        self.removed_features_ = []
        self.adversarial_auc_history_ = []

        max_drop = self.max_features_to_drop or len(current_features)
        n_dropped = 0

        logger.info(
            '[ADF] Старт adversarial validation: %d признаков, target_auc=%.3f',
            len(current_features), self.target_auc,
        )

        while True:
            auc, importances = self._train_adversarial(X_combined, y_combined, current_features)
            self.adversarial_auc_history_.append(auc)
            self.feature_importances_ = importances

            logger.info(
                '[ADF] Итерация %d  AUC=%.4f  признаков=%d  удалено=%d',
                len(self.adversarial_auc_history_), auc, len(current_features), n_dropped,
            )

            if auc <= self.target_auc:
                logger.info('[ADF] AUC=%.4f ≤ target=%.3f — drift устранён', auc, self.target_auc)
                break

            if len(current_features) <= 1:
                logger.warning('[ADF] Остался 1 признак — останавливаемся')
                break

            if n_dropped >= max_drop:
                logger.warning(
                    '[ADF] Достигнут лимит удалений (%d) — AUC=%.4f всё ещё выше target',
                    max_drop, auc,
                )
                break

            n_remove = min(self.remove_per_step, len(current_features) - 1, max_drop - n_dropped)
            to_remove = importances.nlargest(n_remove).index.tolist()

            logger.info('[ADF] Удаляем %s (importance %s)',
                        to_remove, [f'{importances[f]:.2f}' for f in to_remove])

            self.removed_features_.extend(to_remove)
            current_features = [f for f in current_features if f not in set(to_remove)]
            n_dropped += n_remove

        self.selected_features_ = current_features
        logger.info(
            '[ADF] Итог: %d признаков оставлено, %d удалено  AUC: %.4f → %.4f',
            len(self.selected_features_), len(self.removed_features_),
            self.adversarial_auc_history_[0], self.adversarial_auc_history_[-1],
        )
        self._fitted = True
        return self

    # ── transform ───────────────────────────────────────────────────────────

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Вернуть X без drift-признаков."""
        if not self._fitted:
            raise RuntimeError('AdversarialDriftFilter не обучен. Вызовите fit() первым.')
        missing = [f for f in self.selected_features_ if f not in X.columns]
        if missing:
            raise ValueError(f'В X отсутствуют признаки: {missing}')
        return X[self.selected_features_]

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
    ) -> pd.DataFrame:
        """fit(X_train, X_valid) + transform(X_train)."""
        return self.fit(X_train, X_valid).transform(X_train)

    # ── Отчёт ───────────────────────────────────────────────────────────────

    def report(self) -> pd.DataFrame:
        """Детальный отчёт по adversarial важности всех проверенных признаков.

        Returns:
            DataFrame с колонками ``feature``, ``adversarial_importance``,
            ``removed_by_drift``, отсортированный по убыванию важности.

        """
        if not self._fitted:
            raise RuntimeError('AdversarialDriftFilter не обучен.')
        removed_set = set(self.removed_features_)
        imp = self.feature_importances_ if self.feature_importances_ is not None else pd.Series(dtype=float)
        rows = [
            {
                'feature': f,
                'adversarial_importance': float(imp.get(f, 0.0)),
                'removed_by_drift': f in removed_set,
            }
            for f in (list(self.selected_features_) + self.removed_features_)
        ]
        return (
            pd.DataFrame(rows)
            .sort_values('adversarial_importance', ascending=False)
            .reset_index(drop=True)
        )
