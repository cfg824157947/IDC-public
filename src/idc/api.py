"""High-level, full-data IDC API composed from the tested scientific units."""

from __future__ import annotations

from dataclasses import replace
import time
from typing import Any, Callable

import numpy as np

from .association import AssociationError, fit_association
from .backend import BackendError, resolve_device
from .config import IDCConfig
from .projection import (
    SPARSE_EXECUTION_MODES,
    CellScores,
    ProjectionError,
    apply_cell_score_normalization,
    extract_expression,
    fit_control_reference,
    fit_cell_score_normalization,
    fit_projection,
    transform_projection,
)
from .results import IDCFit, build_provenance


class IDCAPIError(ValueError):
    """Raised when high-level IDC inputs or fitted-state use are invalid."""


def fit_idc(
    data: Any,
    *,
    sample_key: str,
    condition_key: str,
    config: IDCConfig,
    execution_mode: str = "dense",
    profile_callback: Callable[[dict[str, Any]], None] | None = None,
) -> IDCFit:
    """Fit IDC using exactly the cells supplied by the caller.

    For validation, the fold runner passes a training-only AnnData object.
    This function has no access to a source object other than ``data``.
    """

    if execution_mode not in {"dense", *SPARSE_EXECUTION_MODES}:
        raise IDCAPIError("unsupported execution_mode")
    try:
        resolve_device(config)
    except BackendError as exc:
        raise IDCAPIError(str(exc)) from exc
    obs = _obs(data)
    sample_ids = _column(obs, sample_key, "sample")
    conditions = _column(obs, condition_key, "condition")
    reference_donors = (
        _column(obs, config.data.donor_key, "donor")
        if config.data.donor_key is not None
        else sample_ids
    )

    stage_runtime: dict[str, float] = {}
    def profile(event: str, stage: str, **details: Any) -> None:
        if profile_callback is not None:
            profile_callback({"event": event, "stage": stage, **details})

    profile("stage_start", "input_extraction")
    stage_start = time.perf_counter()
    expression, gene_names = extract_expression(data, config=config, execution_mode=execution_mode)
    stage_runtime["input_extraction_transform"] = time.perf_counter() - stage_start
    profile("stage_end", "input_extraction", elapsed_seconds=stage_runtime["input_extraction_transform"])
    control_label = config.data.control_label
    if control_label is None:
        raise IDCAPIError("data.control_label is required")
    reference_mask = np.asarray(conditions, dtype=object) == control_label
    profile("stage_start", "sparse_reference_statistics")
    stage_start = time.perf_counter()
    reference_fit = fit_control_reference(
        expression,
        gene_names,
        reference_mask=reference_mask,
        reference_donor_ids=reference_donors,
        config=config,
        execution_mode=execution_mode,
    )
    stage_runtime["reference_statistics_filter"] = time.perf_counter() - stage_start
    profile(
        "stage_end", "sparse_reference_statistics",
        elapsed_seconds=stage_runtime["reference_statistics_filter"],
        retained_gene_count=len(reference_fit.gene_names),
        excluded_gene_count=len(reference_fit.excluded_gene_ids),
        reference_donor_count=len(reference_fit.reference_donor_ids),
    )
    retained = np.asarray(reference_fit.retained_indices, dtype=int)
    stage_start = time.perf_counter()
    association = fit_association(
        expression[:, retained],
        sample_ids=sample_ids,
        conditions=conditions,
        gene_names=reference_fit.gene_names,
        config=config,
        execution_mode=execution_mode,
        profile_callback=profile_callback,
    )
    if association.execution_runtime_seconds:
        stage_runtime.update(association.execution_runtime_seconds)
    else:
        stage_runtime["sample_covariance_eigendecomposition"] = time.perf_counter() - stage_start

    profile("stage_start", "projection_during_fit")
    stage_start = time.perf_counter()
    projection = fit_projection(
        data,
        gene_theta=association.gene_theta,
        reference_mask=reference_mask,
        reference_donor_ids=reference_donors,
        config=config,
        reference_fit=reference_fit,
        execution_mode=execution_mode,
    )
    all_projected = transform_projection(projection, data, config=config, execution_mode=execution_mode)
    projection_seconds = time.perf_counter() - stage_start
    stage_runtime["projection"] = projection_seconds
    profile("stage_end", "projection_during_fit", elapsed_seconds=projection_seconds)
    scored_mask = _scored_population_mask(data, config)
    if not scored_mask.any():
        raise IDCAPIError("configured scored population contains no cells")
    profile("stage_start", "cell_score_normalization_during_fit")
    normalization_start = time.perf_counter()
    normalization = fit_cell_score_normalization(
        all_projected.cell_theta[scored_mask], config=config
    )
    stage_runtime["cell_score_normalization"] = time.perf_counter() - normalization_start
    profile("stage_end", "cell_score_normalization_during_fit", elapsed_seconds=stage_runtime["cell_score_normalization"])

    # Sample-level association matrices are needed to calculate the contrast,
    # but retaining S dense p-by-p matrices in every fitted object is not
    # required to reproduce transforms and scales poorly for validation folds.
    compact_association = replace(
        association,
        sample_associations=np.empty(
            (0, len(reference_fit.gene_names), len(reference_fit.gene_names))
        ),
    )

    ids = _cell_ids(data)
    selected_samples = tuple(map(str, np.asarray(sample_ids, dtype=object)[scored_mask]))
    selected_conditions = tuple(map(str, np.asarray(conditions, dtype=object)[scored_mask]))
    return IDCFit(
        association_fit=compact_association,
        projection_fit=projection,
        cell_score_normalization=normalization,
        sample_key=sample_key,
        condition_key=condition_key,
        scored_cell_ids=tuple(np.asarray(ids, dtype=object)[scored_mask].astype(str)),
        scored_sample_ids=selected_samples,
        scored_conditions=selected_conditions,
        provenance=build_provenance(
            data,
            config,
            sample_ids=association.sample_ids,
            gene_names=projection.gene_names,
            execution_mode=execution_mode,
            stage_runtime_seconds=stage_runtime,
        ),
        execution_mode=execution_mode,
    )


def transform_cells(
    fit: IDCFit,
    data: Any,
    *,
    config: IDCConfig,
    profile_callback: Callable[[dict[str, Any]], None] | None = None,
) -> CellScores:
    """Apply frozen projection and cell-score state without refitting."""

    try:
        resolve_device(config)
    except BackendError as exc:
        raise IDCAPIError(str(exc)) from exc
    if config.to_dict() != fit.provenance.resolved_config:
        raise IDCAPIError("transform config differs from the fitted resolved config")
    def profile(event: str, stage: str, **details: Any) -> None:
        if profile_callback is not None:
            profile_callback({"event": event, "stage": stage, **details})

    profile("stage_start", "projection_for_cell_scores")
    stage_start = time.perf_counter()
    projected = transform_projection(fit.projection_fit, data, config=config, execution_mode=fit.execution_mode)
    profile("stage_end", "projection_for_cell_scores", elapsed_seconds=time.perf_counter() - stage_start)
    scored_mask = _scored_population_mask(data, config)
    if not scored_mask.any():
        raise IDCAPIError("configured scored population contains no cells")
    profile("stage_start", "cell_score_normalization_for_cell_scores")
    stage_start = time.perf_counter()
    normalized = apply_cell_score_normalization(
        projected.cell_theta[scored_mask], fit.cell_score_normalization, config=config
    )
    profile("stage_end", "cell_score_normalization_for_cell_scores", elapsed_seconds=time.perf_counter() - stage_start)
    obs = _obs(data)
    sample_ids = _column(obs, fit.sample_key, "sample")
    conditions = _column(obs, fit.condition_key, "condition")
    ids = _cell_ids(data)
    provenance = {
        "configuration_sha256": fit.provenance.configuration_sha256,
        "analysis_class": fit.provenance.analysis_class,
        "git_revision": fit.provenance.git_revision,
    }
    return CellScores(
        cell_score=normalized.cell_score,
        normalization=fit.cell_score_normalization,
        cell_theta=projected.cell_theta[scored_mask],
        sample_ids=tuple(np.asarray(sample_ids, dtype=object)[scored_mask].astype(str)),
        conditions=tuple(np.asarray(conditions, dtype=object)[scored_mask].astype(str)),
        cell_ids=tuple(np.asarray(ids, dtype=object)[scored_mask].astype(str)),
        gene_names=projected.gene_names,
        sample_key=fit.sample_key,
        condition_key=fit.condition_key,
        provenance=provenance,
    )


def _obs(data: Any):
    obs = getattr(data, "obs", None)
    if obs is None or not hasattr(obs, "columns"):
        raise IDCAPIError("data must expose AnnData-like .obs metadata")
    return obs


def _column(obs, key: str, label: str):
    if key not in obs.columns:
        raise IDCAPIError(f"{label} key {key!r} is absent from data.obs")
    values = np.asarray(obs[key], dtype=object)
    if values.ndim != 1 or any(_missing(value) for value in values):
        raise IDCAPIError(f"{label} metadata {key!r} contains missing values")
    return values


def _scored_population_mask(data: Any, config: IDCConfig) -> np.ndarray:
    n = len(data.obs)
    target = config.data.target_cell_type
    key = config.data.cell_type_key
    if target is None:
        return np.ones(n, dtype=bool)
    if key is None:
        raise IDCAPIError("target_cell_type requires data.cell_type_key")
    labels = _column(_obs(data), key, "cell-type")
    return labels == target


def _cell_ids(data: Any) -> np.ndarray:
    names = getattr(data, "obs_names", None)
    if names is None:
        raise IDCAPIError("data must expose unique obs_names")
    result = np.asarray(names, dtype=object)
    if len(set(map(str, result))) != len(result):
        raise IDCAPIError("cell identifiers in obs_names must be unique")
    return result


def _missing(value: object) -> bool:
    try:
        return bool(value is None or (isinstance(value, float) and np.isnan(value)))
    except (TypeError, ValueError):
        return False
