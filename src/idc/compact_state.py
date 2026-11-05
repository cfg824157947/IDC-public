"""Compact, pickle-free IDC+PU inference state serialization.

The artifact contains only frozen parameters needed to project new cells and
apply the final PU classifier. Training-only association matrices, PU
membership arrays, and cell-level initialization diagnostics are omitted.
Prediction delegates to the same projection, normalization, and PU functions
used by the ordinary API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import IDCConfig
from .projection import (
    CellScoreNormalization,
    ProjectionFit,
    apply_cell_score_normalization,
    transform_projection,
)
from .pu import PUFit, PUFitStatus, predict_pu
from .canonical import canonical_config_hash


SCHEMA_VERSION = "idc_compact_prediction_state_v1"


@dataclass(frozen=True)
class CompactPredictionState:
    projection_fit: ProjectionFit
    cell_score_normalization: CellScoreNormalization
    pu_scaling_mean: np.ndarray
    pu_scaling_sd: np.ndarray
    pu_coefficient: np.ndarray
    pu_intercept: float
    pu_status: PUFitStatus
    pu_categorical_threshold: float
    pu_disease_like_label: str
    pu_control_like_label: str
    pu_final_refit_performed: bool
    pu_cycles_completed: int
    sample_key: str
    condition_key: str
    execution_mode: str
    scientific_config_sha256: str
    execution_config_sha256: str | None
    resolved_config: dict[str, Any]
    provenance: dict[str, Any]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(x) for x in value]
    return value


def compact_state_from_fits(idc_fit, pu_fit) -> CompactPredictionState:
    """Copy only prediction-required values out of ordinary fitted objects."""
    if pu_fit.status is PUFitStatus.FAILED or pu_fit.coefficient is None:
        raise ValueError("cannot export a failed or incomplete PU model")
    if pu_fit.scaling_mean is None or pu_fit.scaling_sd is None or pu_fit.intercept is None:
        raise ValueError("PU prediction parameters are incomplete")
    execution = dict(idc_fit.provenance.execution)
    provenance = {
        "configuration_sha256": idc_fit.provenance.configuration_sha256,
        "method_version": idc_fit.provenance.method_version,
        "git_revision": idc_fit.provenance.git_revision,
        "input_identifier": idc_fit.provenance.input_identifier,
        "analysis_class": idc_fit.provenance.analysis_class,
        "software": dict(idc_fit.provenance.software),
        "execution": execution,
        "reference_donor_ids": list(idc_fit.projection_fit.reference_donor_ids),
        "reference_cells_per_donor": list(idc_fit.projection_fit.reference_cells_per_donor),
        "reference_cell_count": idc_fit.projection_fit.reference_cell_count,
    }
    return CompactPredictionState(
        projection_fit=idc_fit.projection_fit,
        cell_score_normalization=idc_fit.cell_score_normalization,
        pu_scaling_mean=np.asarray(pu_fit.scaling_mean, dtype=np.float64).copy(),
        pu_scaling_sd=np.asarray(pu_fit.scaling_sd, dtype=np.float64).copy(),
        pu_coefficient=np.asarray(pu_fit.coefficient, dtype=np.float64).copy(),
        pu_intercept=float(pu_fit.intercept),
        pu_status=pu_fit.status,
        pu_categorical_threshold=float(pu_fit.categorical_threshold),
        pu_disease_like_label=str(pu_fit.disease_like_label),
        pu_control_like_label=str(pu_fit.control_like_label),
        pu_final_refit_performed=bool(pu_fit.final_refit_performed),
        pu_cycles_completed=len(pu_fit.iteration_history),
        sample_key=str(idc_fit.sample_key),
        condition_key=str(idc_fit.condition_key),
        execution_mode=str(idc_fit.execution_mode),
        scientific_config_sha256=str(idc_fit.provenance.configuration_sha256),
        execution_config_sha256=execution.get("execution_config_sha256"),
        resolved_config=dict(idc_fit.provenance.resolved_config),
        provenance=provenance,
    )


def save_compact_state(state: CompactPredictionState, path: str | Path) -> dict[str, Any]:
    """Write JSON metadata and compressed numeric arrays; refuse overwrite."""
    base = Path(path)
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = base.with_suffix(".json")
    arrays_path = base.with_suffix(".npz")
    if metadata_path.exists() or arrays_path.exists():
        raise FileExistsError(f"compact prediction state already exists at {base}")
    p = state.projection_fit
    n = state.cell_score_normalization
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "sample_key": state.sample_key,
        "condition_key": state.condition_key,
        "execution_mode": state.execution_mode,
        "scientific_config_sha256": state.scientific_config_sha256,
        "execution_config_sha256": state.execution_config_sha256,
        "resolved_config": _jsonable(state.resolved_config),
        "provenance": _jsonable(state.provenance),
        "projection": {
            "retained_gene_ids": list(p.gene_names),
            "input_gene_ids": list(p.input_gene_names),
            "retained_gene_indices": list(p.retained_gene_indices),
            "genes_before_reference": list(p.genes_before_reference),
            "excluded_gene_ids": list(p.excluded_gene_ids),
            "excluded_gene_reasons": [list(x) for x in p.excluded_gene_reasons],
            "reference_cell_count": p.reference_cell_count,
            "reference_donor_ids": list(p.reference_donor_ids),
            "reference_cells_per_donor": list(p.reference_cells_per_donor),
            "reference_sd_ddof": p.reference_sd_ddof,
            "zero_reference_sd_policy": p.zero_reference_sd_policy,
            "expression_source": p.expression_source,
            "upstream_transform": p.upstream_transform,
            "upstream_target_sum": p.upstream_target_sum,
            "batch_reference_mode": p.batch_reference_mode,
            "batch_key": p.batch_key,
        },
        "cell_score_normalization": {
            "ddof": n.ddof,
            "fitted_cell_count": n.fitted_cell_count,
            "normalization": n.normalization,
        },
        "pu": {
            "status": state.pu_status.value,
            "intercept": state.pu_intercept,
            "categorical_threshold": state.pu_categorical_threshold,
            "disease_like_label": state.pu_disease_like_label,
            "control_like_label": state.pu_control_like_label,
            "final_refit_performed": state.pu_final_refit_performed,
            "cycles_completed": state.pu_cycles_completed,
        },
    }
    array_tmp = arrays_path.with_name(arrays_path.name + f".tmp.{os.getpid()}")
    metadata_tmp = metadata_path.with_name(metadata_path.name + f".tmp.{os.getpid()}")
    with array_tmp.open("wb") as stream:
        np.savez_compressed(
            stream,
            gene_theta=p.gene_theta,
            reference_mean=p.reference_mean,
            reference_sd=p.reference_sd,
            cell_score_mean=n.mean,
            cell_score_sd=n.sd,
            cell_score_degenerate=n.degenerate_components,
            pu_scaling_mean=state.pu_scaling_mean,
            pu_scaling_sd=state.pu_scaling_sd,
            pu_coefficient=state.pu_coefficient,
        )
    metadata["arrays_sha256"] = _sha256(array_tmp)
    metadata_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
    array_tmp.replace(arrays_path)
    metadata_tmp.replace(metadata_path)
    return {
        "metadata_path": str(metadata_path),
        "arrays_path": str(arrays_path),
        "metadata_sha256": _sha256(metadata_path),
        "arrays_sha256": metadata["arrays_sha256"],
        "total_bytes": metadata_path.stat().st_size + arrays_path.stat().st_size,
    }


def load_compact_state(path: str | Path) -> CompactPredictionState:
    """Load the pickle-free state and verify its array payload checksum."""
    base = Path(path)
    if base.suffix:
        base = base.with_suffix("")
    metadata_path, arrays_path = base.with_suffix(".json"), base.with_suffix(".npz")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported compact prediction state schema")
    if _sha256(arrays_path) != metadata.get("arrays_sha256"):
        raise ValueError("compact prediction state array checksum mismatch")
    with np.load(arrays_path, allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    projection = metadata["projection"]
    p = ProjectionFit(
        gene_names=tuple(projection["retained_gene_ids"]),
        input_gene_names=tuple(projection["input_gene_ids"]),
        retained_gene_indices=tuple(map(int, projection["retained_gene_indices"])),
        genes_before_reference=tuple(projection["genes_before_reference"]),
        excluded_gene_ids=tuple(projection["excluded_gene_ids"]),
        excluded_gene_reasons=tuple(tuple(x) for x in projection["excluded_gene_reasons"]),
        gene_theta=arrays["gene_theta"],
        reference_mean=arrays["reference_mean"],
        reference_sd=arrays["reference_sd"],
        reference_cell_count=int(projection["reference_cell_count"]),
        reference_donor_ids=tuple(projection["reference_donor_ids"]),
        reference_cells_per_donor=tuple(map(int, projection["reference_cells_per_donor"])),
        reference_sd_ddof=int(projection["reference_sd_ddof"]),
        zero_reference_sd_policy=projection["zero_reference_sd_policy"],
        expression_source=projection["expression_source"],
        upstream_transform=projection["upstream_transform"],
        upstream_target_sum=projection["upstream_target_sum"],
        batch_reference_mode=projection["batch_reference_mode"],
        batch_key=projection["batch_key"],
    )
    if (len(p.retained_gene_indices) != len(p.gene_names)
            or any(i < 0 or i >= len(p.input_gene_names) for i in p.retained_gene_indices)
            or tuple(p.input_gene_names[i] for i in p.retained_gene_indices) != p.gene_names):
        raise ValueError("compact retained-gene IDs/order do not agree with input IDs/indices")
    if canonical_config_hash(metadata["resolved_config"]) != metadata["scientific_config_sha256"]:
        raise ValueError("compact state scientific configuration checksum mismatch")
    norm_meta = metadata["cell_score_normalization"]
    normalization = CellScoreNormalization(
        mean=arrays["cell_score_mean"],
        sd=arrays["cell_score_sd"],
        degenerate_components=arrays["cell_score_degenerate"],
        ddof=int(norm_meta["ddof"]),
        fitted_cell_count=int(norm_meta["fitted_cell_count"]),
        normalization=norm_meta["normalization"],
    )
    if (len(set(p.gene_names)) != len(p.gene_names)
            or len(set(p.input_gene_names)) != len(p.input_gene_names)
            or p.gene_theta.ndim != 2
            or p.gene_theta.shape[0] != len(p.gene_names)
            or p.reference_mean.shape != (len(p.gene_names),)
            or p.reference_sd.shape != (len(p.gene_names),)
            or not np.isfinite(p.gene_theta).all()
            or not np.isfinite(p.reference_mean).all()
            or not np.isfinite(p.reference_sd).all()
            or np.any(p.reference_sd <= 0)):
        raise ValueError("compact projection state has inconsistent shapes or invalid values")
    if (normalization.mean.shape != normalization.sd.shape
            or normalization.mean.shape != normalization.degenerate_components.shape
            or not np.isfinite(normalization.mean).all()
            or not np.isfinite(normalization.sd).all()
            or arrays["pu_coefficient"].shape != normalization.mean.shape):
        raise ValueError("compact score-normalization/PU classifier dimensions are invalid")
    pu = metadata["pu"]
    if (arrays["pu_scaling_mean"].shape != arrays["pu_coefficient"].shape
            or arrays["pu_scaling_sd"].shape != arrays["pu_coefficient"].shape
            or not np.isfinite(arrays["pu_scaling_mean"]).all()
            or not np.isfinite(arrays["pu_scaling_sd"]).all()
            or np.any(arrays["pu_scaling_sd"] <= 0)
            or not np.isfinite(arrays["pu_coefficient"]).all()
            or not np.isfinite(float(pu["intercept"]))):
        raise ValueError("compact PU state has inconsistent shapes or invalid values")
    return CompactPredictionState(
        projection_fit=p,
        cell_score_normalization=normalization,
        pu_scaling_mean=arrays["pu_scaling_mean"],
        pu_scaling_sd=arrays["pu_scaling_sd"],
        pu_coefficient=arrays["pu_coefficient"],
        pu_intercept=float(pu["intercept"]),
        pu_status=PUFitStatus(pu["status"]),
        pu_categorical_threshold=float(pu["categorical_threshold"]),
        pu_disease_like_label=pu["disease_like_label"],
        pu_control_like_label=pu["control_like_label"],
        pu_final_refit_performed=bool(pu["final_refit_performed"]),
        pu_cycles_completed=int(pu["cycles_completed"]),
        sample_key=metadata["sample_key"],
        condition_key=metadata["condition_key"],
        execution_mode=metadata["execution_mode"],
        scientific_config_sha256=metadata["scientific_config_sha256"],
        execution_config_sha256=metadata["execution_config_sha256"],
        resolved_config=metadata["resolved_config"],
        provenance=metadata["provenance"],
    )


def predict_compact_state(state: CompactPredictionState, data: Any, *, config: IDCConfig):
    """Apply frozen IDC projection, normalization, scaler, and PU model."""
    if config.to_dict() != state.resolved_config:
        raise ValueError("prediction config differs from compact state config")
    projected = transform_projection(
        state.projection_fit, data, config=config, execution_mode=state.execution_mode
    )
    normalized = apply_cell_score_normalization(
        projected.cell_theta, state.cell_score_normalization, config=config
    )
    minimal_pu = PUFit(
        status=state.pu_status,
        initialization=None,
        scaling_mean=state.pu_scaling_mean,
        scaling_sd=state.pu_scaling_sd,
        coefficient=state.pu_coefficient,
        intercept=state.pu_intercept,
        final_effective_membership=None,
        iteration_history=(),
        logistic_fit_count=0,
        final_refit_attempted=state.pu_final_refit_performed,
        final_refit_performed=state.pu_final_refit_performed,
        categorical_threshold=state.pu_categorical_threshold,
        disease_like_label=state.pu_disease_like_label,
        control_like_label=state.pu_control_like_label,
        resolved_config=state.resolved_config,
    )
    return predict_pu(minimal_pu, normalized), normalized, projected
