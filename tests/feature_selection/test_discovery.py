"""Тесты `ml_toolkit.feature_selection.discovery` — discover -> select -> artifact -> replay workflow.

Никаких бизнес-терминов: сущность/таймстемп называются `entity_id`/`ts_key`, как
и в `tests/test_feature_generation.py`. `select_features_by_model_feedback` /
`run_feature_discovery` используют лёгкую заглушку `model_factory` вместо
реального CatBoost+Optuna, чтобы тесты оставались быстрыми и детерминированными.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml_toolkit.feature_selection import (
    FeatureSelectionArtifact,
    materialize_feature_selection_artifact,
    merge_feature_selection_artifacts,
    run_feature_discovery,
    sample_entities,
    select_features_by_model_feedback,
)

SLOPE_PRESET = {'slope': {'windows': [6, 12]}}


class _StubImportanceModel:
    """Заглушка вместо CatBoostClassifier: держит те же атрибуты/контракт.

    Без реального обучения — все переданные признаки получают importance=1,
    кроме тех, что в drop_features (importance=0), чтобы протестировать фильтрацию.
    """

    def __init__(self, drop_features: list[str] | None = None):
        self._drop_features = set(drop_features or [])
        self.selected_features_: list[str] = []
        self._model = None

    def fit(self, X_train, y_train, X_valid, y_valid, selected_features):  # noqa: ARG002
        self.selected_features_ = list(selected_features)
        importances = np.array([0.0 if f in self._drop_features else 1.0 for f in selected_features])
        self._model = _FittedStub(importances)


class _FittedStub:
    def __init__(self, importances: np.ndarray):
        self.feature_importances_ = importances


def _growth_and_decline_df(n_entities: int = 20) -> pl.DataFrame:
    months = list(range(1, 13))
    entities, ts, values, labels = [], [], [], []
    for e in range(n_entities):
        growing = e % 2 == 0
        series = [10.0 * i for i in months] if growing else [10.0 * (13 - i) for i in months]
        entities += [e] * len(months)
        ts += months
        values += series
        labels += [1 if growing else 0] * len(months)
    return pl.DataFrame({
        'entity_id': entities,
        'ts_key': ts,
        'value': values,
        'label': labels,
    })


# ── sample_entities ──────────────────────────────────────────────────────────

def test_sample_entities_unstratified_keeps_full_history():
    df = _growth_and_decline_df(n_entities=20)
    sampled = sample_entities(df, entity_column_name='entity_id', n_entities=5, seed=0).collect()

    assert sampled['entity_id'].n_unique() == 5
    # полная история 12 месяцев на каждую сущность, а не по одной строке
    assert sampled.height == 5 * 12


def test_sample_entities_stratified_preserves_label_presence():
    df = _growth_and_decline_df(n_entities=20)
    sampled = sample_entities(
        df, entity_column_name='entity_id', label_column_name='label', n_entities=10, seed=0,
    ).collect()

    labels_present = set(sampled.select('entity_id', 'label').unique()['label'].to_list())
    assert labels_present == {0, 1}


def test_sample_entities_requires_exactly_one_of_n_entities_or_frac():
    df = _growth_and_decline_df(n_entities=5)
    with pytest.raises(ValueError, match='ровно один'):
        sample_entities(df, entity_column_name='entity_id')
    with pytest.raises(ValueError, match='ровно один'):
        sample_entities(df, entity_column_name='entity_id', n_entities=2, frac=0.5)


# ── FeatureSelectionArtifact save/load ───────────────────────────────────────

def test_artifact_save_and_load_roundtrip(tmp_path):
    artifact = FeatureSelectionArtifact(
        product_cols=['value'],
        preset=SLOPE_PRESET,
        accepted_cols=['value__slope__w6'],
        meta={'n_sample_entities': 5},
    )
    path = tmp_path / 'nested' / 'artifact.json'
    artifact.save(path)

    loaded = FeatureSelectionArtifact.load(path)
    assert loaded == artifact


def test_artifact_normalizes_path_preset_to_str(tmp_path):
    from pathlib import Path

    preset_path = tmp_path / 'preset.yaml'
    artifact = FeatureSelectionArtifact(
        product_cols=['value'], preset=Path(preset_path), accepted_cols=[],
    )
    assert artifact.preset == str(preset_path)


# ── merge_feature_selection_artifacts ────────────────────────────────────────

def test_merge_unions_accepted_cols_preserving_order_and_dedup():
    a = FeatureSelectionArtifact(product_cols=['value'], preset=SLOPE_PRESET, accepted_cols=['c1', 'c2'])
    b = FeatureSelectionArtifact(product_cols=['value'], preset=SLOPE_PRESET, accepted_cols=['c2', 'c3'])

    merged = merge_feature_selection_artifacts([a, b], meta={'source': 'test'})

    assert merged.accepted_cols == ['c1', 'c2', 'c3']
    assert merged.product_cols == ['value']
    assert merged.preset == SLOPE_PRESET
    assert merged.meta == {'source': 'test'}


def test_merge_rejects_empty_list():
    with pytest.raises(ValueError, match='пустой список'):
        merge_feature_selection_artifacts([])


def test_merge_rejects_mismatched_product_cols():
    a = FeatureSelectionArtifact(product_cols=['value'], preset=SLOPE_PRESET, accepted_cols=['c1'])
    b = FeatureSelectionArtifact(product_cols=['other'], preset=SLOPE_PRESET, accepted_cols=['c2'])
    with pytest.raises(ValueError, match='product_cols'):
        merge_feature_selection_artifacts([a, b])


def test_merge_rejects_mismatched_preset():
    a = FeatureSelectionArtifact(product_cols=['value'], preset=SLOPE_PRESET, accepted_cols=['c1'])
    b = FeatureSelectionArtifact(product_cols=['value'], preset={'slope': {'windows': [24]}}, accepted_cols=['c2'])
    with pytest.raises(ValueError, match='preset'):
        merge_feature_selection_artifacts([a, b])


# ── materialize_feature_selection_artifact ───────────────────────────────────

def test_materialize_replays_only_accepted_cols(tmp_path):
    df = _growth_and_decline_df(n_entities=4)
    artifact = FeatureSelectionArtifact(
        product_cols=['value'], preset=SLOPE_PRESET, accepted_cols=['value__slope__w6'],
    )
    out_path = tmp_path / 'full.parquet'

    materialize_feature_selection_artifact(
        df, entity_column_name='entity_id', ts_column_name='ts_key',
        artifact=artifact, out_path=out_path,
    )

    result = pl.read_parquet(out_path)
    assert 'value__slope__w6' in result.columns
    assert 'value__slope__w12' not in result.columns  # не в accepted_cols - не материализуется


# ── select_features_by_model_feedback ────────────────────────────────────────

def test_select_features_by_model_feedback_drops_zero_importance():
    rng = np.random.default_rng(0)
    n = 200
    y = (rng.random(n) > 0.5).astype(int)
    X = pd.DataFrame({
        'good': rng.normal(0, 1, n) + y * 3.0,
        'useless': rng.normal(0, 1, n) + y * 3.0,  # хороший AUC, но importance=0 у заглушки
        'const': np.ones(n),  # отсеивается ещё на структурном скрининге
    })
    X_train, X_valid = X.iloc[:150], X.iloc[150:]
    y_train, y_valid = y[:150], y[150:]

    selected = select_features_by_model_feedback(
        X_train, y_train, X_valid, y_valid,
        # drift-фильтр отключён: 'good'/'useless' статистически неразличимы,
        # так что adversarial validation могла бы отсеять любую из них по шуму —
        # здесь важна только фильтрация по model importance, не сам drift-этап.
        pipeline_kwargs={'min_univariate_auc': 0.55, 'use_drift_filter': False},
        model_factory=lambda: _StubImportanceModel(drop_features=['useless']),
    )

    assert selected == ['good']


def test_select_features_by_model_feedback_empty_screening_returns_empty():
    n = 50
    y = np.array([0, 1] * (n // 2))
    X = pd.DataFrame({'const': np.ones(n)})

    selected = select_features_by_model_feedback(
        X.iloc[:40], y[:40], X.iloc[40:], y[40:],
        pipeline_kwargs={'min_univariate_auc': 0.55},
        model_factory=_StubImportanceModel,
    )

    assert selected == []


# ── run_feature_discovery (end-to-end, лёгкая заглушка модели) ──────────────

def test_run_feature_discovery_returns_artifact_ready_to_replay(tmp_path):
    df = _growth_and_decline_df(n_entities=20)

    artifact = run_feature_discovery(
        df, entity_column_name='entity_id', ts_column_name='ts_key',
        product_cols=['value'], label_column_name='label',
        preset=SLOPE_PRESET, out_dir=tmp_path / 'discovery',
        n_sample_entities=20,
        corr_threshold=None,
        pipeline_kwargs={'min_univariate_auc': 0.5},
        model_factory=_StubImportanceModel,
    )

    assert artifact.product_cols == ['value']
    assert artifact.preset == SLOPE_PRESET
    assert artifact.accepted_cols  # slope хорошо разделяет растущие/падающие серии
    assert artifact.meta['n_sample_entities'] == 20

    out_path = tmp_path / 'replayed.parquet'
    materialize_feature_selection_artifact(
        df, entity_column_name='entity_id', ts_column_name='ts_key',
        artifact=artifact, out_path=out_path,
    )
    result = pl.read_parquet(out_path)
    for col in artifact.accepted_cols:
        assert col in result.columns
