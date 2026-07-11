"""ThresholdMovingCV: оптимальный порог через поиск по валидации.

При < 1% позитивов дефолтный порог 0.5 почти всегда неоптимален:
большинство позитивов получает score << 0.5, и модель ничего не находит.

Этот пресет обучает любой base_preset, затем ищет порог на валидации,
максимизирующий выбранную метрику. predict() использует найденный порог
вместо дефолтного 0.5.

Режимы оптимизации:
  'f1'                  — стандартный F1 (precision и recall одинаково важны)
  'f2'                  — F-beta с beta=2 (recall важнее в 2 раза)
  'f0.5'                — F-beta с beta=0.5 (precision важнее в 2 раза)
  'precision_at_recall' — макс. precision при recall ≥ min_recall (требует min_recall)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, precision_recall_curve

from ml_toolkit.models._base import XInput, YInput
from ml_toolkit.presets.classification._base import BasePreset

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

_BETA_MAP = {'f1': 1.0, 'f2': 2.0, 'f0.5': 0.5}


class ThresholdMovingCV(BasePreset):
    """Оптимальный порог по валидации поверх любого base_preset.

    Parameters
    ----------
    base_preset:
        Любой необученный экземпляр BasePreset.
    optimize:
        Метрика для оптимизации порога: 'f1', 'f2', 'f0.5' или 'precision_at_recall'.
    min_recall:
        Минимальный recall (обязателен при optimize='precision_at_recall').
    n_thresholds:
        Количество точек в поиске порога.

    Атрибуты после fit::

        base_          — обученный base_preset
        threshold_     — найденный оптимальный порог
        threshold_metric_ — значение метрики при оптимальном пороге
        scan_df_       — pd.DataFrame со всеми (threshold, metric) точками

    Пример::

        from ml_toolkit.presets.classification.high_pr_auc.cascade import TwoStageCascade
        from ml_toolkit.presets.classification.high_pr_auc.threshold_moving import ThresholdMovingCV

        model = ThresholdMovingCV(TwoStageCascade(), optimize='f2')
        model.fit(X_train, y_train, X_valid, y_valid)
        labels = model.predict(X_test)            # использует threshold_
        proba  = model.predict_proba(X_test)      # сырые вероятности base

    """

    def __init__(
        self,
        base_preset: BasePreset,
        optimize: str = 'f2',
        min_recall: float | None = None,
        n_thresholds: int = 500,
    ) -> None:
        super().__init__(params=None, n_optuna_trials=0)
        if optimize not in ('f1', 'f2', 'f0.5', 'precision_at_recall'):
            raise ValueError(
                f"optimize должен быть 'f1', 'f2', 'f0.5' или 'precision_at_recall', "
                f"получено {optimize!r}"
            )
        if optimize == 'precision_at_recall' and min_recall is None:
            raise ValueError("min_recall обязателен при optimize='precision_at_recall'")
        self.base_preset = base_preset
        self.optimize = optimize
        self.min_recall = min_recall
        self.n_thresholds = n_thresholds

        self.base_: BasePreset | None = None
        self.threshold_: float = 0.5
        self.threshold_metric_: float = 0.0
        self.scan_df_: pd.DataFrame | None = None

    # ── Поиск порога ──────────────────────────────────────────────────────────

    def _find_threshold(self, y_va: np.ndarray, proba: np.ndarray) -> tuple[float, float, pd.DataFrame]:
        if self.optimize == 'precision_at_recall':
            return self._threshold_precision_at_recall(y_va, proba)
        return self._threshold_fbeta(y_va, proba)

    def _threshold_fbeta(
        self, y_va: np.ndarray, proba: np.ndarray
    ) -> tuple[float, float, pd.DataFrame]:
        beta = _BETA_MAP[self.optimize]
        ts = np.linspace(proba.min() + 1e-6, proba.max() - 1e-6, self.n_thresholds)
        scores = np.array([
            fbeta_score(y_va, (proba >= t).astype(int), beta=beta, zero_division=0)
            for t in ts
        ])
        best_idx = int(np.argmax(scores))
        scan_df = pd.DataFrame({'threshold': ts, f'f{self.optimize[1:]}': scores})
        return float(ts[best_idx]), float(scores[best_idx]), scan_df

    def _threshold_precision_at_recall(
        self, y_va: np.ndarray, proba: np.ndarray
    ) -> tuple[float, float, pd.DataFrame]:
        assert self.min_recall is not None
        prec_curve, rec_curve, t_curve = precision_recall_curve(y_va, proba)
        # prec_curve[:-1], rec_curve[:-1] соответствуют t_curve
        valid_mask = rec_curve[:-1] >= self.min_recall
        if not valid_mask.any():
            logger.warning(
                '[ThresholdMoving] Не удалось достичь recall=%.2f. '
                'Используем наименьший доступный threshold (максимальный recall).',
                self.min_recall,
            )
            # thresholds возрастают, recall убывает → индекс 0 даёт max recall
            best_idx = 0
        else:
            # Среди порогов с recall >= min_recall — макс precision
            best_idx = int(np.where(valid_mask)[0][np.argmax(prec_curve[:-1][valid_mask])])

        best_t = float(t_curve[best_idx])
        best_prec = float(prec_curve[best_idx])
        actual_rec = float(rec_curve[best_idx])

        scan_df = pd.DataFrame({
            'threshold': t_curve,
            'precision': prec_curve[:-1],
            'recall': rec_curve[:-1],
        })
        logger.info(
            '[ThresholdMoving] precision@recall≥%.2f: t=%.4f  prec=%.4f  rec=%.4f',
            self.min_recall, best_t, best_prec, actual_rec,
        )
        return best_t, best_prec, scan_df

    def plot_threshold_scan(self, ax: Axes | None = None, path: str | None = None) -> None:
        """График метрики по всем порогам с вертикальной линией на оптимуме."""
        import matplotlib.pyplot as plt

        if self.scan_df_ is None:
            raise RuntimeError('Вызовите .fit() перед plot_threshold_scan()')

        df = self.scan_df_
        metric_col = [c for c in df.columns if c != 'threshold'][0]

        fig, ax_ = (plt.subplots(figsize=(8, 4)) if ax is None else (ax.get_figure(), ax))
        ax_.plot(df['threshold'], df[metric_col], color='steelblue', lw=1.5)
        ax_.axvline(self.threshold_, color='red', linestyle='--', lw=1.5,
                    label=f'opt threshold={self.threshold_:.4f}  {metric_col}={self.threshold_metric_:.4f}')
        ax_.set_xlabel('Threshold')
        ax_.set_ylabel(metric_col)
        ax_.set_title(f'Threshold scan — optimize={self.optimize}')
        ax_.legend()
        plt.tight_layout()
        if path:
            fig.savefig(path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        if ax is None:
            plt.close(fig)

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: XInput,
        y_train: YInput,
        X_valid: XInput,
        y_valid: YInput,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> ThresholdMovingCV:
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        y_va = y_valid.values

        logger.info('[ThresholdMoving] Обучаем base_preset: %s', type(self.base_preset).__name__)
        self.base_preset.fit(
            X_train, y_train, X_valid, y_valid,
            selected_features=selected_features,
            cat_features=cat_features,
        )
        self.base_ = self.base_preset

        proba_va = (
            self.base_.valid_pred_
            if self.base_.valid_pred_ is not None
            else self.base_.predict_proba(X_valid)
        )

        self.threshold_, self.threshold_metric_, self.scan_df_ = self._find_threshold(y_va, proba_va)

        pr_auc = float(average_precision_score(y_va, proba_va))
        logger.info(
            '[ThresholdMoving] optimize=%s  threshold=%.4f  metric=%.4f  PR-AUC=%.4f',
            self.optimize, self.threshold_, self.threshold_metric_, pr_auc,
        )

        self.valid_pred_ = proba_va
        self.train_pred_ = self.base_.train_pred_
        self.selected_features_ = self.base_.selected_features_
        self.cat_features_ = self.base_.cat_features_
        self.best_params_ = {
            'base': type(self.base_).__name__,
            'optimize': self.optimize,
            'threshold': self.threshold_,
            **(self.base_.best_params_ or {}),
        }
        self._model = True
        return self

    # ── predict / predict_proba ───────────────────────────────────────────────

    def predict(self, X: XInput, threshold: float | None = None) -> np.ndarray:
        """Возвращает бинарные метки. Использует self.threshold_ если threshold не задан."""
        t = threshold if threshold is not None else self.threshold_
        return (self.predict_proba(X) >= t).astype(int)

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_.predict_proba(X)
