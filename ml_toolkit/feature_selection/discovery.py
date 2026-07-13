"""Discover-on-sample -> select -> artifact -> replay-on-full-data workflow.

Motivation: running a large "kitchen sink" transformer preset (many transformers,
many windows) over a full dataset is wasteful when most of the resulting columns
will be thrown away anyway. This module lets you generate that large preset on a
small sample of entities, train a quick model, keep only the columns that survive
structural/relevance screening and show non-zero model importance, and freeze that
decision as a `FeatureSelectionArtifact` (which transformers/params + which
resulting column names). The artifact can then be replayed on the full dataset via
`materialize_feature_selection_artifact` (a thin wrapper over
`ml_toolkit.feature_generation.apply_feature_groups`) so the expensive full-data
pass only ever computes the columns known to matter.

No business terms anywhere here — works for any dataset shaped as
(entity_column_name, ts_column_name, product_cols) with an optional per-entity
label, same convention as `ml_toolkit.feature_generation`.

Example::

    from ml_toolkit.feature_selection import run_feature_discovery, materialize_feature_selection_artifact

    artifact = run_feature_discovery(
        df, entity_column_name='entity_id', ts_column_name='ts_key',
        product_cols=['value'], label_column_name='label',
        preset='discount_sensitivity_full', out_dir='discovery/',
        n_sample_entities=300,
    )
    artifact.save('discovery/artifact.json')

    materialize_feature_selection_artifact(
        full_df, entity_column_name='entity_id', ts_column_name='ts_key',
        artifact=artifact, out_path='full_features.parquet',
    )
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from ml_toolkit.feature_generation import apply_feature_groups, select_features
from ml_toolkit.feature_selection.pipeline import FeatureSelectionPipeline
from ml_toolkit.models import CatBoostClassifier

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_VALID_SIZE = 0.25


@dataclass
class FeatureSelectionArtifact:
    """Frozen record of a feature-discovery decision: what to compute, and where.

    `product_cols` + `preset` together are exactly the `feature_spec` entry
    (`(product_cols, preset)`) that generated `accepted_cols` as candidates —
    passing them straight to `ml_toolkit.feature_generation.apply_feature_groups`
    (or via `materialize_feature_selection_artifact`) recomputes only
    `accepted_cols` on a new dataset of the same schema.

    Attributes:
        product_cols: Product columns the preset was applied to.
        preset: Exactly what was passed to `select_features`/`run_feature_discovery`
            (a preset name, a full path, or an inline `{transformer: params}` dict).
            Stored as given except `Path` is normalized to `str` for JSON safety.
        accepted_cols: Final selected column names.
        meta: Free-form bookkeeping (e.g. `{"n_sample_entities": ..., "label": ...}`).
            Not used for correctness — informational only.

    """

    product_cols: list[str]
    preset: str | dict[str, dict]
    accepted_cols: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.preset, Path):
            self.preset = str(self.preset)

    def save(self, path: Path | str) -> None:
        """Write this artifact as JSON, creating parent directories if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2))
        logger.info('FeatureSelectionArtifact: сохранён в %s (%d колонок)', path, len(self.accepted_cols))

    @classmethod
    def load(cls, path: Path | str) -> FeatureSelectionArtifact:
        """Load an artifact previously written by `save`."""
        data = json.loads(Path(path).read_text())
        return cls(**data)


def merge_feature_selection_artifacts(
    artifacts: list[FeatureSelectionArtifact],
    meta: dict[str, Any] | None = None,
) -> FeatureSelectionArtifact:
    """Union several artifacts' `accepted_cols` into one, e.g. across per-target discovery runs.

    Requires identical `product_cols` (as sets) and identical `preset` across all
    inputs — merging only makes sense when every artifact's `accepted_cols` are
    candidates of the exact same `feature_spec`; a genuine mismatch is a caller
    error, not something to silently paper over (same philosophy as
    `ml_toolkit.feature_generation._resolve_feature_spec`'s conflict detection).

    Args:
        artifacts: Non-empty list of artifacts sharing the same `product_cols`/`preset`.
        meta: Metadata for the merged artifact (independent of the inputs' own `meta`).

    Returns:
        A new `FeatureSelectionArtifact` with the union of `accepted_cols`
        (first-seen order preserved, duplicates removed).

    Raises:
        ValueError: If `artifacts` is empty, or `product_cols`/`preset` disagree
            across artifacts.

    """
    if not artifacts:
        raise ValueError('merge_feature_selection_artifacts: пустой список артефактов')

    first = artifacts[0]
    first_cols = set(first.product_cols)
    for other in artifacts[1:]:
        if set(other.product_cols) != first_cols:
            raise ValueError(
                f'merge_feature_selection_artifacts: несовпадающие product_cols '
                f'({sorted(first_cols)} vs {sorted(other.product_cols)}) — артефакты нельзя объединить'
            )
        if other.preset != first.preset:
            raise ValueError(
                f'merge_feature_selection_artifacts: несовпадающие preset '
                f'({first.preset!r} vs {other.preset!r}) — артефакты нельзя объединить'
            )

    merged_cols: list[str] = []
    seen: set[str] = set()
    for artifact in artifacts:
        for col in artifact.accepted_cols:
            if col not in seen:
                seen.add(col)
                merged_cols.append(col)

    logger.info(
        'merge_feature_selection_artifacts: %d артефактов объединены в %d колонок',
        len(artifacts), len(merged_cols),
    )
    return FeatureSelectionArtifact(
        product_cols=list(first.product_cols),
        preset=first.preset,
        accepted_cols=merged_cols,
        meta=meta or {},
    )


def materialize_feature_selection_artifact(
    df: pl.DataFrame | pl.LazyFrame,
    entity_column_name: str,
    ts_column_name: str,
    artifact: FeatureSelectionArtifact,
    out_path: Path | str,
    **kwargs: Any,
) -> None:
    """Replay a `FeatureSelectionArtifact` on `df`, computing only `accepted_cols`.

    Thin wrapper over `ml_toolkit.feature_generation.apply_feature_groups` — no
    correlation filter runs here, since selection already happened when the
    artifact was built.

    Args:
        df: Dataset (eager or lazy) at (entity_column_name, ts_column_name) grain.
        entity_column_name: Entity identifier column name.
        ts_column_name: Timestamp column name.
        artifact: Artifact produced by `run_feature_discovery` (or `.load(...)`/
            `merge_feature_selection_artifacts`).
        out_path: Output parquet path.
        **kwargs: Forwarded to `apply_feature_groups` (e.g. `min_output_ts_key`,
            `max_output_ts_key`, `tmp_dir`, `name`).

    """
    apply_feature_groups(
        df,
        entity_column_name=entity_column_name,
        ts_column_name=ts_column_name,
        feature_spec=[(artifact.product_cols, artifact.preset)],
        accepted_cols=artifact.accepted_cols,
        out_path=out_path,
        **kwargs,
    )


def sample_entities(
    df: pl.DataFrame | pl.LazyFrame,
    entity_column_name: str,
    label_column_name: str | None = None,
    n_entities: int | None = None,
    frac: float | None = None,
    seed: int = 42,
) -> pl.LazyFrame:
    """Return the full history of a sampled subset of entities.

    If `label_column_name` is given, sampling is stratified by it (assumes the
    label is constant per entity — one row per entity in the (entity, label)
    projection — proportions of each class in the full data are preserved in the
    sample, as evenly as `n_entities`/`frac` allow). Otherwise a plain random
    sample of entity ids is taken.

    Args:
        df: Dataset (eager or lazy) with an `entity_column_name` column.
        entity_column_name: Entity identifier column name.
        label_column_name: Optional per-entity label column to stratify by.
        n_entities: Absolute number of entities to sample. Mutually exclusive with `frac`.
        frac: Fraction of entities to sample (0 < frac <= 1). Mutually exclusive with `n_entities`.
        seed: Random seed for reproducibility.

    Returns:
        A `pl.LazyFrame` — every row of `df` belonging to a sampled entity (full
        history kept, not just one row per entity).

    Raises:
        ValueError: If neither or both of `n_entities`/`frac` are given.

    """
    if (n_entities is None) == (frac is None):
        raise ValueError('sample_entities: ровно один из n_entities/frac должен быть задан')

    lazy_df = df.lazy() if isinstance(df, pl.DataFrame) else df

    if label_column_name is None:
        entities = lazy_df.select(entity_column_name).unique().collect()
        sampled = (
            entities.sample(fraction=frac, seed=seed)
            if frac is not None
            else entities.sample(n=min(n_entities, entities.height), seed=seed)
        )
        sampled_ids = sampled.lazy()
    else:
        entity_labels = lazy_df.select(entity_column_name, label_column_name).unique().collect()
        total = entity_labels.height
        parts: list[pl.DataFrame] = []
        for group_df in entity_labels.partition_by(label_column_name):
            if frac is not None:
                part = group_df.sample(fraction=frac, seed=seed)
            else:
                n_for_group = min(len(group_df), max(1, round(n_entities * len(group_df) / total)))
                part = group_df.sample(n=n_for_group, seed=seed)
            parts.append(part)
        sampled_ids = pl.concat(parts).select(entity_column_name).lazy()

    return lazy_df.join(sampled_ids, on=entity_column_name, how='semi')


def select_features_by_model_feedback(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: pd.Series | np.ndarray,
    pipeline_kwargs: dict[str, Any] | None = None,
    model_factory: Callable[[], Any] | None = None,
    n_optuna_trials: int = 10,
) -> list[str]:
    """Screen features structurally/statistically, then keep only non-zero-importance ones.

    Two stages: `ml_toolkit.feature_selection.FeatureSelectionPipeline` (structural
    + univariate AUC + adversarial drift, all generic/pandas-based) narrows the
    candidate set first — much cheaper than training a full model on every
    candidate — then a quick classifier (`model_factory`, default
    `CatBoostClassifier` with a small Optuna budget and `undersample_majority=True`)
    is trained on the survivors and keeps only features with non-zero
    `feature_importances_`. Binary vs multiclass is handled automatically by
    `CatBoostClassifier` based on `y_train`'s cardinality.

    Args:
        X_train: Training candidate features.
        y_train: Training labels.
        X_valid: Validation candidate features (used by the drift stage).
        y_valid: Validation labels.
        pipeline_kwargs: Forwarded to `FeatureSelectionPipeline(...)`.
        model_factory: Zero-arg callable returning an unfitted model exposing the
            same `fit`/`selected_features_`/`_model.feature_importances_` contract
            as `ml_toolkit.models.CatBoostClassifier`. Defaults to `CatBoostClassifier`.
        n_optuna_trials: Optuna trial budget for the default model (ignored if
            `model_factory` is given).

    Returns:
        Feature names surviving both stages, in `X_train`'s column order.

    """
    pipeline = FeatureSelectionPipeline(**(pipeline_kwargs or {}))
    pipeline.fit(X_train, y_train, X_valid, y_valid)
    screened = pipeline.selected_features_
    if not screened:
        logger.warning('select_features_by_model_feedback: скрининг не оставил ни одного признака')
        return []

    model = model_factory() if model_factory is not None else CatBoostClassifier(
        n_optuna_trials=n_optuna_trials,
        model_settings={'undersample_majority': True, 'optuna_pruner': 'none'},
    )
    model.fit(
        X_train=X_train[screened], y_train=y_train,
        X_valid=X_valid[screened], y_valid=y_valid,
        selected_features=screened,
    )
    importances = model._model.feature_importances_
    selected = [f for f, imp in zip(model.selected_features_, importances, strict=True) if imp != 0]
    logger.info(
        'select_features_by_model_feedback: %d кандидатов -> %d после скрининга -> %d после importance',
        X_train.shape[1], len(screened), len(selected),
    )
    return selected


def run_feature_discovery(
    df: pl.DataFrame | pl.LazyFrame,
    entity_column_name: str,
    ts_column_name: str,
    product_cols: list[str],
    label_column_name: str,
    preset: Path | str | dict,
    out_dir: Path | str,
    n_sample_entities: int | None = None,
    frac: float | None = None,
    corr_threshold: float | None = 0.9,
    n_optuna_trials: int = 10,
    seed: int = 42,
    tmp_dir: Path | str | None = None,
    pipeline_kwargs: dict[str, Any] | None = None,
    model_factory: Callable[[], Any] | None = None,
) -> FeatureSelectionArtifact:
    """Discover useful features on a sample and freeze the decision as an artifact.

    One-call orchestrator: `sample_entities` (stratified by `label_column_name`) ->
    `select_features` (Phase A candidates + Phase B correlation filter on the
    sample, big `preset` included) -> reduce to one row per entity (last
    `ts_column_name` row, since window features already summarize history) joined
    with the label -> stratified train/valid split -> `select_features_by_model_feedback`.

    Assumes `label_column_name` is constant per entity (whole-entity
    classification, e.g. one label per time series) — it is read once per entity
    via `(entity_column_name, label_column_name).unique()` before feature
    generation and re-joined afterwards (feature-generation output never carries
    arbitrary extra columns, only `entity_column_name`/`ts_column_name`/`product_cols`/
    engineered features).

    Args:
        df: Dataset (eager or lazy) at (entity_column_name, ts_column_name) grain,
            with `product_cols` and a per-entity `label_column_name`.
        entity_column_name: Entity identifier column name.
        ts_column_name: Timestamp column name.
        product_cols: Product columns to generate features for.
        label_column_name: Per-entity classification label column.
        preset: Preset to try on the sample — name, path, or inline dict. Passed
            straight through to `select_features` and stored as-is in the returned
            artifact.
        out_dir: Directory for the sample's intermediate engineered parquet.
        n_sample_entities: Absolute number of entities to sample. Mutually
            exclusive with `frac`.
        frac: Fraction of entities to sample. Mutually exclusive with `n_sample_entities`.
        corr_threshold: Correlation filter threshold for the sample's Phase B
            (same as `select_features`). `None` disables it.
        n_optuna_trials: Optuna budget for the default importance model.
        seed: Random seed (sampling + train/valid split).
        tmp_dir: Temp directory for Phase A candidate parquet files.
        pipeline_kwargs: Forwarded to `FeatureSelectionPipeline`.
        model_factory: Forwarded to `select_features_by_model_feedback`.

    Returns:
        `FeatureSelectionArtifact` with `product_cols`/`preset` (ready to replay via
        `materialize_feature_selection_artifact`) and the selected `accepted_cols`.

    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lazy_df = df.lazy() if isinstance(df, pl.DataFrame) else df
    entity_labels = lazy_df.select(entity_column_name, label_column_name).unique()

    sample_lazy = sample_entities(
        lazy_df,
        entity_column_name=entity_column_name,
        label_column_name=label_column_name,
        n_entities=n_sample_entities,
        frac=frac,
        seed=seed,
    )

    sample_out_path = out_dir / 'discovery_sample_features.parquet'
    candidate_cols = select_features(
        sample_lazy,
        entity_column_name=entity_column_name,
        ts_column_name=ts_column_name,
        product_cols=product_cols,
        out_path=sample_out_path,
        corr_threshold=corr_threshold,
        preset=preset,
        tmp_dir=tmp_dir,
        name='discovery_sample',
    )

    entity_level = (
        pl.scan_parquet(sample_out_path)
        .filter(pl.col(ts_column_name) == pl.col(ts_column_name).max().over(entity_column_name))
        .unique(subset=[entity_column_name], keep='first')
        .join(entity_labels, on=entity_column_name, how='inner')
        .collect()
    )

    y = entity_level[label_column_name].to_numpy()
    X = entity_level.select(candidate_cols).to_pandas()

    stratify = y if len(np.unique(y)) > 1 and np.min(np.unique(y, return_counts=True)[1]) > 1 else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=_DEFAULT_VALID_SIZE, random_state=seed, stratify=stratify,
    )

    selected = select_features_by_model_feedback(
        X_train, y_train, X_valid, y_valid,
        pipeline_kwargs=pipeline_kwargs, model_factory=model_factory, n_optuna_trials=n_optuna_trials,
    )

    return FeatureSelectionArtifact(
        product_cols=list(product_cols),
        preset=preset,
        accepted_cols=selected,
        meta={'n_sample_entities': entity_level.height, 'n_candidates': len(candidate_cols)},
    )
