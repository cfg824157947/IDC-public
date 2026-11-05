"""Control-reference standardization and projection into ``gene_theta``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np
from scipy import sparse

from .config import IDCConfig


SPARSE_EXECUTION_MODES = {
    "sparse_safe",
    "sparse_safe_condition_weighted",
    "sparse_safe_gpu_condition_weighted",
}
from .backend import (
    BackendError,
    apply_cell_score as backend_apply_cell_score,
    fit_cell_score_stats,
    projection as backend_projection,
    reference_mean_sd,
    resolve_device,
)


class ProjectionFailureStatus(str, Enum):
    INVALID_INPUT = "invalid_input"
    INSUFFICIENT_REFERENCE = "insufficient_reference"
    NONFINITE_REFERENCE = "nonfinite_reference"
    ZERO_REFERENCE_SD = "zero_reference_sd"
    CONFIG_MISMATCH = "config_mismatch"


@dataclass(frozen=True)
class ProjectionReferenceFit:
    """Training-fitted control reference and its frozen gene-validity filter."""

    input_gene_names: tuple[str, ...]
    gene_names: tuple[str, ...]
    retained_indices: tuple[int, ...]
    reference_mean: np.ndarray
    reference_sd: np.ndarray
    genes_before_reference: tuple[str, ...]
    excluded_gene_ids: tuple[str, ...]
    excluded_gene_reasons: tuple[tuple[str, str], ...]
    reference_cell_count: int
    reference_donor_ids: tuple[str, ...]
    reference_cells_per_donor: tuple[int, ...]
    reference_sd_ddof: int
    zero_reference_sd_policy: str


class ProjectionError(ValueError):
    """Typed projection failure with affected gene diagnostics when available."""

    def __init__(
        self,
        status: ProjectionFailureStatus,
        message: str,
        *,
        genes: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.status = status
        self.genes = tuple(genes)


@dataclass(frozen=True)
class ProjectionFit:
    """Fitted state sufficient to transform cells without refitting."""

    gene_names: tuple[str, ...]
    input_gene_names: tuple[str, ...]
    retained_gene_indices: tuple[int, ...]
    genes_before_reference: tuple[str, ...]
    excluded_gene_ids: tuple[str, ...]
    excluded_gene_reasons: tuple[tuple[str, str], ...]
    gene_theta: np.ndarray
    reference_mean: np.ndarray
    reference_sd: np.ndarray
    reference_cell_count: int
    reference_donor_ids: tuple[str, ...]
    reference_cells_per_donor: tuple[int, ...]
    reference_sd_ddof: int
    zero_reference_sd_policy: str
    expression_source: str
    upstream_transform: str
    upstream_target_sum: float | None
    batch_reference_mode: str
    batch_key: str | None

    @property
    def exclusion_count(self) -> int:
        return len(self.excluded_gene_ids)


@dataclass(frozen=True)
class ProjectionScores:
    """Projected cell coordinates and the gene order used to produce them."""

    cell_theta: np.ndarray
    gene_names: tuple[str, ...]


@dataclass(frozen=True)
class CellScoreNormalization:
    """Fitted component-wise embedding normalization state."""

    mean: np.ndarray
    sd: np.ndarray
    degenerate_components: np.ndarray
    ddof: int
    fitted_cell_count: int
    normalization: str


@dataclass(frozen=True)
class CellScores:
    """Normalized cell embedding and the frozen normalization state."""

    cell_score: np.ndarray
    normalization: CellScoreNormalization
    cell_theta: np.ndarray | None = None
    sample_ids: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    cell_ids: tuple[str, ...] = ()
    gene_names: tuple[str, ...] = ()
    sample_key: str | None = None
    condition_key: str | None = None
    provenance: dict[str, object] | None = None


def extract_expression(
    data: Any,
    *,
    config: IDCConfig,
    execution_mode: str = "dense",
) -> tuple[Any, tuple[str, ...]]:
    """Read the configured AnnData expression object in its stored gene order."""

    source = config.data.expression_source
    transform = config.data.upstream_transform
    if source is None:
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "data.expression_source must be explicitly configured",
        )
    if transform is None:
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "data.upstream_transform must be explicitly configured",
        )

    try:
        if source == "X":
            matrix = data.X
            names = data.var_names
        elif source == "raw":
            if data.raw is None:
                raise ProjectionError(
                    ProjectionFailureStatus.INVALID_INPUT,
                    "expression_source='raw' but data.raw is absent",
                )
            matrix = data.raw.X
            names = data.raw.var_names
        else:
            layer_name = source.removeprefix("layer:")
            if not layer_name:
                raise ProjectionError(
                    ProjectionFailureStatus.INVALID_INPUT,
                    "expression_source layer name is empty",
                )
            matrix = data.layers[layer_name]
            names = data.var_names
    except ProjectionError:
        raise
    except (AttributeError, KeyError, TypeError) as exc:
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            f"could not read configured expression source {source!r}: {exc}",
        ) from exc

    if execution_mode not in {"dense", *SPARSE_EXECUTION_MODES}:
        raise ProjectionError(ProjectionFailureStatus.INVALID_INPUT, "unsupported execution_mode")
    if execution_mode in SPARSE_EXECUTION_MODES and not sparse.issparse(matrix):
        raise ProjectionError(ProjectionFailureStatus.INVALID_INPUT, "sparse_safe execution requires sparse expression input")
    if config.compute.backend == "gpu" and sparse.issparse(matrix):
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "GPU backend does not currently support sparse expression input; "
            "convert to a supported dense input explicitly",
        )
    try:
        resolve_device(config)
    except BackendError as exc:
        raise ProjectionError(ProjectionFailureStatus.INVALID_INPUT, str(exc)) from exc
    is_sparse = execution_mode in SPARSE_EXECUTION_MODES
    values = _sparse_float_matrix(matrix, config) if is_sparse else _dense_float_matrix(matrix)
    gene_names = tuple(str(name) for name in names)
    if values.shape[1] != len(gene_names):
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "expression columns do not match configured gene names",
        )
    return (apply_upstream_transform_sparse(values, config) if is_sparse else apply_upstream_transform(values, config)), gene_names


def _sparse_float_matrix(values: Any, config: IDCConfig):
    """Validate/copy sparse input without materializing its dense shape."""
    if not sparse.issparse(values) or getattr(values, "ndim", None) != 2:
        raise ProjectionError(ProjectionFailureStatus.INVALID_INPUT, "sparse_safe requires a two-dimensional sparse matrix")
    matrix = values.tocsr(copy=False).astype(np.float64, copy=True)
    if not np.isfinite(matrix.data).all():
        raise ProjectionError(ProjectionFailureStatus.INVALID_INPUT, "expression contains non-finite stored values before upstream transform")
    return matrix


def apply_upstream_transform_sparse(values, config: IDCConfig):
    """Exact sparse equivalent of the configured transform."""
    transform = config.data.upstream_transform
    # ``_sparse_float_matrix`` already owns this CSR copy, so in-place transforms
    # avoid an additional nnz-sized buffer without mutating the source AnnData.
    matrix = values.tocsr(copy=False)
    if transform == "identity":
        pass
    elif transform == "log1p":
        if np.any(matrix.data <= -1):
            raise ProjectionError(ProjectionFailureStatus.INVALID_INPUT, "log1p requires every expression value to be greater than -1")
        matrix.data = np.log1p(matrix.data)
    elif transform == "normalize_total_log1p":
        target = config.data.upstream_target_sum
        if target is None or np.any(matrix.data < 0):
            raise ProjectionError(ProjectionFailureStatus.INVALID_INPUT, "normalize_total_log1p requires a target and nonnegative expression")
        totals = np.asarray(matrix.sum(axis=1), dtype=np.float64).reshape(-1)
        if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
            raise ProjectionError(ProjectionFailureStatus.INVALID_INPUT, "normalize_total_log1p requires positive finite cell totals")
        matrix = sparse.diags(float(target) / totals, format="csr") @ matrix
        matrix = matrix.tocsr()
        matrix.data = np.log1p(matrix.data)
    else:
        raise ProjectionError(ProjectionFailureStatus.INVALID_INPUT, f"unsupported upstream transform: {transform!r}")
    if not np.isfinite(matrix.data).all():
        raise ProjectionError(ProjectionFailureStatus.INVALID_INPUT, "upstream expression transform produced non-finite values")
    return matrix


def apply_upstream_transform(values: np.ndarray, config: IDCConfig) -> np.ndarray:
    """Apply the explicitly named, serializable expression transform."""

    transform = config.data.upstream_transform
    if transform is None:
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "data.upstream_transform must be explicitly configured",
        )
    matrix = _dense_float_matrix(values)
    if not np.isfinite(matrix).all():
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "expression contains non-finite values before upstream transform",
        )
    if transform == "identity":
        result = matrix
    elif transform == "log1p":
        if np.any(matrix <= -1):
            raise ProjectionError(
                ProjectionFailureStatus.INVALID_INPUT,
                "log1p requires every expression value to be greater than -1",
            )
        result = np.log1p(matrix)
    elif transform == "normalize_total_log1p":
        target = config.data.upstream_target_sum
        if target is None:
            raise ProjectionError(
                ProjectionFailureStatus.INVALID_INPUT,
                "normalize_total_log1p requires data.upstream_target_sum",
            )
        if np.any(matrix < 0):
            raise ProjectionError(
                ProjectionFailureStatus.INVALID_INPUT,
                "normalize_total_log1p requires nonnegative expression",
            )
        totals = matrix.sum(axis=1)
        if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
            raise ProjectionError(
                ProjectionFailureStatus.INVALID_INPUT,
                "normalize_total_log1p requires positive finite cell totals",
            )
        result = np.log1p(matrix * (float(target) / totals[:, np.newaxis]))
    else:
        # IDCConfig validates this, but retain a defensive check for mutated
        # or manually constructed config state.
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            f"unsupported upstream transform: {transform!r}",
        )
    if not np.isfinite(result).all():
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "upstream expression transform produced non-finite values",
        )
    return result


def fit_projection(
    data: Any,
    *,
    gene_theta: np.ndarray,
    reference_mask: Sequence[bool],
    reference_donor_ids: Sequence[object],
    config: IDCConfig,
    reference_fit: ProjectionReferenceFit | None = None,
    execution_mode: str = "dense",
) -> ProjectionFit:
    """Fit projection state, excluding invalid reference-SD genes explicitly.

    When ``reference_fit`` is supplied, ``gene_theta`` must already be fitted
    on its retained gene universe. This allows the high-level workflow to
    determine validity from training controls before fitting the association
    basis.
    """

    if config.projection.batch_key is not None or config.projection.batch_reference_mode != "global":
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "batch-aware projection reference is unsupported; use global mode "
            "with batch_key=None",
        )
    expression, gene_names = extract_expression(data, config=config, execution_mode=execution_mode)
    theta = np.asarray(gene_theta, dtype=np.float64)
    if theta.ndim != 2:
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "gene_theta must have shape genes-by-components in expression gene order",
        )
    if not np.isfinite(theta).all():
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "gene_theta contains non-finite values",
        )
    if reference_fit is None:
        reference_fit = fit_control_reference(
            expression,
            gene_names,
            reference_mask=reference_mask,
            reference_donor_ids=reference_donor_ids,
            config=config,
        )
        if theta.shape[0] != len(reference_fit.input_gene_names):
            raise ProjectionError(
                ProjectionFailureStatus.INVALID_INPUT,
                "gene_theta must have one row per input expression gene",
            )
        theta = theta[np.asarray(reference_fit.retained_indices, dtype=int), :]
    elif gene_names != reference_fit.input_gene_names:
        raise ProjectionError(
            ProjectionFailureStatus.CONFIG_MISMATCH,
            "provided reference fit was learned with a different input gene order",
        )
    elif theta.shape[0] != len(reference_fit.gene_names):
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "gene_theta must be fitted on the retained reference gene universe",
        )
    if not len(reference_fit.gene_names):
        raise ProjectionError(
            ProjectionFailureStatus.ZERO_REFERENCE_SD,
            "no genes remain after control-reference SD filtering",
            genes=reference_fit.excluded_gene_ids,
        )
    return ProjectionFit(
        gene_names=reference_fit.gene_names,
        input_gene_names=reference_fit.input_gene_names,
        retained_gene_indices=reference_fit.retained_indices,
        genes_before_reference=reference_fit.genes_before_reference,
        excluded_gene_ids=reference_fit.excluded_gene_ids,
        excluded_gene_reasons=reference_fit.excluded_gene_reasons,
        gene_theta=theta.copy(),
        reference_mean=reference_fit.reference_mean.copy(),
        reference_sd=reference_fit.reference_sd.copy(),
        reference_cell_count=reference_fit.reference_cell_count,
        reference_donor_ids=reference_fit.reference_donor_ids,
        reference_cells_per_donor=reference_fit.reference_cells_per_donor,
        reference_sd_ddof=reference_fit.reference_sd_ddof,
        zero_reference_sd_policy=config.projection.zero_reference_sd_policy,
        expression_source=config.data.expression_source or "",
        upstream_transform=config.data.upstream_transform or "",
        upstream_target_sum=config.data.upstream_target_sum,
        batch_reference_mode=config.projection.batch_reference_mode,
        batch_key=config.projection.batch_key,
    )


def fit_control_reference(
    expression: np.ndarray,
    gene_names: Sequence[object],
    *,
    reference_mask: Sequence[bool],
    reference_donor_ids: Sequence[object],
    config: IDCConfig,
    execution_mode: str = "dense",
) -> ProjectionReferenceFit:
    """Fit the global donor-balanced control reference and freeze valid genes."""

    if execution_mode in SPARSE_EXECUTION_MODES:
        if not sparse.issparse(expression):
            raise ProjectionError(ProjectionFailureStatus.INVALID_INPUT, "sparse_safe reference fitting requires sparse expression")
        expression = expression.tocsr(copy=False)
    else:
        expression = _dense_float_matrix(expression)
    gene_names = tuple(str(name) for name in gene_names)
    if expression.ndim != 2 or expression.shape[1] != len(gene_names):
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "expression and gene_names must have matching columns",
        )
    expression_finite = (
        np.isfinite(expression.data).all()
        if sparse.issparse(expression)
        else np.isfinite(expression).all()
    )
    if not expression_finite:
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "expression contains non-finite values before reference fitting",
        )
    if len(set(gene_names)) != len(gene_names):
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "input gene names must be unique and ordered",
        )
    mask = np.asarray(reference_mask)
    if mask.ndim != 1 or len(mask) != expression.shape[0] or mask.dtype.kind != "b":
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "reference_mask must be a boolean vector with one entry per cell",
        )
    donor_array = np.asarray(reference_donor_ids, dtype=object)
    if donor_array.ndim != 1 or len(donor_array) != expression.shape[0]:
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "reference_donor_ids must have one entry per cell",
        )
    reference_donors = donor_array[mask]
    if any(_is_missing(value) for value in reference_donors):
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "reference donor identifiers cannot be missing",
        )
    donor_order: list[object] = []
    donor_indices: dict[object, list[int]] = {}
    try:
        for idx, donor in enumerate(reference_donors):
            if donor not in donor_indices:
                donor_order.append(donor)
                donor_indices[donor] = []
            donor_indices[donor].append(idx)
    except TypeError as exc:
        raise ProjectionError(ProjectionFailureStatus.INVALID_INPUT, "reference donor identifiers must be hashable scalar values") from exc
    if not donor_order:
        raise ProjectionError(
            ProjectionFailureStatus.INSUFFICIENT_REFERENCE,
            "control reference contains no eligible donors",
        )
    reference = expression[mask, :]
    ddof = config.projection.reference_sd_ddof
    donor_count = len(donor_order)
    weights = np.zeros(reference.shape[0], dtype=np.float64)
    counts = []
    for donor in donor_order:
        indices = donor_indices[donor]
        counts.append(len(indices))
        weights[indices] = 1.0 / (donor_count * len(indices))
    correction = 1.0 - ddof * np.sum(weights * weights)
    if correction <= 0:
        raise ProjectionError(
            ProjectionFailureStatus.INSUFFICIENT_REFERENCE,
            "donor-balanced reference has insufficient effective support for "
            f"ddof={ddof}",
        )
    # Compute the weighted mean as the arithmetic mean of donor cell means.
    # Besides matching the scientific definition directly, this avoids a
    # floating-point summation residue for exactly constant genes.
    try:
        if execution_mode in SPARSE_EXECUTION_MODES:
            mean, sd = _sparse_reference_mean_sd(reference, [donor_indices[d] for d in donor_order], ddof, config.compute.chunk_size)
        else:
            mean, sd = reference_mean_sd(reference, [donor_indices[donor] for donor in donor_order], ddof, config=config)
    except BackendError as exc:
        raise ProjectionError(ProjectionFailureStatus.INVALID_INPUT, str(exc)) from exc
    bad_sd = (~np.isfinite(sd)) | (sd == 0)
    if np.any(~np.isfinite(mean)):
        bad = ~np.isfinite(mean)
        raise ProjectionError(
            ProjectionFailureStatus.NONFINITE_REFERENCE,
            "control-reference mean is non-finite",
            genes=np.asarray(gene_names)[bad].tolist(),
        )
    if np.any(bad_sd) and config.projection.zero_reference_sd_policy == "fail":
        bad_names = np.asarray(gene_names)[bad_sd].tolist()
        raise ProjectionError(
            ProjectionFailureStatus.ZERO_REFERENCE_SD,
            "control-reference SD is zero or undefined for genes: "
            + ", ".join(bad_names),
            genes=bad_names,
        )
    retained = np.flatnonzero(~bad_sd)
    excluded = np.flatnonzero(bad_sd)
    reasons = tuple(
        (
            gene_names[index],
            "nonfinite_control_reference_sd"
            if not np.isfinite(sd[index])
            else "zero_control_reference_sd",
        )
        for index in excluded
    )
    return ProjectionReferenceFit(
        input_gene_names=gene_names,
        gene_names=tuple(gene_names[index] for index in retained),
        retained_indices=tuple(int(index) for index in retained),
        reference_mean=mean[retained].copy(),
        reference_sd=sd[retained].copy(),
        genes_before_reference=gene_names,
        excluded_gene_ids=tuple(gene_names[index] for index in excluded),
        excluded_gene_reasons=reasons,
        reference_cell_count=reference.shape[0],
        reference_donor_ids=tuple(str(donor) for donor in donor_order),
        reference_cells_per_donor=tuple(counts),
        reference_sd_ddof=ddof,
        zero_reference_sd_policy=config.projection.zero_reference_sd_policy,
    )


def _sparse_reference_mean_sd(expression, donor_indices, ddof: int, chunk_rows: int):
    """Exact donor-balanced reference statistics with bounded dense row blocks.

    The weighted mean is the mean of donor means. Variance uses the same
    per-cell donor weights and weighted ddof correction as the dense backend.
    Only a bounded row block is converted to dense for stable residual sums.
    """
    n_donors = len(donor_indices)
    if n_donors == 0:
        raise BackendError("control reference contains no eligible donors")
    n_cells, n_genes = expression.shape
    counts = [len(rows) for rows in donor_indices]
    if any(count <= 0 for count in counts):
        raise BackendError("control reference contains an empty donor")
    donor_means = np.empty((n_donors, n_genes), dtype=np.float64)
    for index, rows in enumerate(donor_indices):
        donor_means[index] = np.asarray(expression[rows].sum(axis=0), dtype=np.float64).reshape(-1) / len(rows)
    mean = donor_means.mean(axis=0)
    correction = 1.0 - ddof * sum(1.0 / (n_donors * n_donors * count) for count in counts)
    if correction <= 0:
        return mean, np.full(n_genes, np.nan, dtype=np.float64)
    numerator = np.zeros(n_genes, dtype=np.float64)
    # Map each donor's reference-local row indices into bounded chunks. Donor
    # loops mirror the dense weighting semantics without iterating cells.
    max_rows = max(1, min(int(chunk_rows), max(1, (64 * 1024 * 1024) // (8 * n_genes))))
    for rows, donor_mean, count in zip(donor_indices, donor_means, counts):
        weight = 1.0 / (n_donors * count)
        for start in range(0, count, max_rows):
            local_rows = rows[start : start + max_rows]
            block = expression[local_rows].toarray().astype(np.float64, copy=False)
            block -= mean
            numerator += weight * np.sum(block * block, axis=0)
            del block
    return mean, np.sqrt(numerator / correction)


def _sparse_projection(expression, mean, sd, theta):
    """Project CSR rows using the exact frozen reference standardization."""
    scaled_theta = np.asarray(theta, dtype=np.float64) / np.asarray(sd, dtype=np.float64)[:, None]
    offset = (np.asarray(mean, dtype=np.float64) / np.asarray(sd, dtype=np.float64)) @ theta
    projected = expression @ scaled_theta
    return np.asarray(projected, dtype=np.float64) - offset


def transform_projection(
    fit: ProjectionFit,
    data: Any,
    *,
    config: IDCConfig,
    execution_mode: str = "dense",
) -> ProjectionScores:
    """Apply frozen reference and gene basis; no statistic is refit."""

    expected = (
        fit.expression_source,
        fit.upstream_transform,
        fit.upstream_target_sum,
        fit.batch_reference_mode,
        fit.batch_key,
        fit.zero_reference_sd_policy,
    )
    current = (
        config.data.expression_source,
        config.data.upstream_transform,
        config.data.upstream_target_sum,
        config.projection.batch_reference_mode,
        config.projection.batch_key,
        config.projection.zero_reference_sd_policy,
    )
    if current != expected:
        raise ProjectionError(
            ProjectionFailureStatus.CONFIG_MISMATCH,
            "transform configuration differs from the fitted projection state",
        )
    expression, gene_names = extract_expression(data, config=config, execution_mode=execution_mode)
    if gene_names != fit.input_gene_names:
        raise ProjectionError(
            ProjectionFailureStatus.CONFIG_MISMATCH,
            "transform input gene names/order differ from the fitted input universe",
        )
    expression = expression[:, np.asarray(fit.retained_gene_indices, dtype=int)]
    try:
        if execution_mode in SPARSE_EXECUTION_MODES:
            cell_theta = _sparse_projection(expression, fit.reference_mean, fit.reference_sd, fit.gene_theta)
        else:
            cell_theta = backend_projection(expression, fit.reference_mean, fit.reference_sd, fit.gene_theta, config=config)
    except BackendError as exc:
        raise ProjectionError(ProjectionFailureStatus.INVALID_INPUT, str(exc)) from exc
    if not np.isfinite(cell_theta).all():
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "projection produced non-finite cell_theta values",
        )
    return ProjectionScores(cell_theta=cell_theta, gene_names=fit.gene_names)


def fit_cell_score_normalization(
    cell_theta: np.ndarray,
    *,
    config: IDCConfig,
) -> CellScoreNormalization:
    """Fit the primary cell-level arithmetic mean/sample-SD normalizer.

    The caller supplies exactly the configured scored population (all supplied
    scored cells for full-data analysis, training scored cells for validation).
    """

    values = _dense_float_matrix(cell_theta)
    if values.shape[0] == 0:
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "cell_score normalization requires at least one scored cell",
        )
    if not np.isfinite(values).all():
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "cell_theta contains non-finite values before cell_score fitting",
        )
    if config.projection.cell_score_normalization != "cell_weighted":
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "donor-balanced cell_score normalization is a sensitivity whose exact "
            "estimator is not specified",
        )
    try:
        means, raw_sd = fit_cell_score_stats(
            values, config.projection.cell_score_ddof, config=config
        )
    except BackendError as exc:
        raise ProjectionError(ProjectionFailureStatus.INVALID_INPUT, str(exc)) from exc
    degenerate = (~np.isfinite(raw_sd)) | (raw_sd == 0)
    sd = np.where(degenerate, 0.0, raw_sd)
    return CellScoreNormalization(
        mean=means,
        sd=sd,
        degenerate_components=degenerate,
        ddof=config.projection.cell_score_ddof,
        fitted_cell_count=values.shape[0],
        normalization=config.projection.cell_score_normalization,
    )


def apply_cell_score_normalization(
    cell_theta: np.ndarray,
    normalization: CellScoreNormalization,
    *,
    config: IDCConfig,
) -> CellScores:
    """Apply saved embedding mean/SD; degenerate coordinates map to zero."""

    values = _dense_float_matrix(cell_theta)
    if not np.isfinite(values).all():
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "cell_theta contains non-finite values before cell_score transform",
        )
    if values.shape[1] != normalization.mean.size:
        raise ProjectionError(
            ProjectionFailureStatus.CONFIG_MISMATCH,
            "cell_theta component count differs from fitted cell_score state",
        )
    try:
        scores = backend_apply_cell_score(
            values,
            normalization.mean,
            normalization.sd,
            normalization.degenerate_components,
            config=config,
        )
    except BackendError as exc:
        raise ProjectionError(ProjectionFailureStatus.INVALID_INPUT, str(exc)) from exc
    if not np.isfinite(scores).all():
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "cell_score transform produced non-finite values",
        )
    return CellScores(cell_score=scores, normalization=normalization)


def _dense_float_matrix(values: Any) -> np.ndarray:
    if sparse.issparse(values):
        matrix = values.toarray()
    else:
        matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "expression must be a two-dimensional cells-by-genes matrix",
        )
    try:
        return np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ProjectionError(
            ProjectionFailureStatus.INVALID_INPUT,
            "expression cannot be converted to a numeric matrix",
        ) from exc


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(np.asarray(value != value).item())
    except (TypeError, ValueError):
        return True
