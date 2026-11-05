"""Sample-level association matrices and case-control contrasts."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Sequence

import numpy as np
from scipy import sparse
from scipy.linalg.blas import dgemm

from .config import IDCConfig
from .backend import BackendError, condition_contrast, sample_association, symmetric_eigh


class AssociationError(ValueError):
    """Raised when association inputs cannot define the requested contrast."""


@dataclass(frozen=True)
class AssociationFit:
    """Fitted sample-level association contrast and gene basis."""

    gene_names: tuple[str, ...]
    sample_ids: tuple[str, ...]
    sample_conditions: tuple[str, ...]
    sample_associations: np.ndarray
    control_association: np.ndarray
    case_association: np.ndarray
    contrast: np.ndarray
    eigenvalues: np.ndarray
    gene_theta: np.ndarray
    orientation_indices: tuple[int, ...]
    orientation_signs: tuple[int, ...]
    association: str
    execution_runtime_seconds: dict[str, float] | None = None


def fit_association(
    expression,
    *,
    sample_ids: Sequence[object],
    conditions: Sequence[object],
    gene_names: Sequence[object],
    config: IDCConfig,
    execution_mode: str = "dense",
    profile_callback: Callable[[dict[str, Any]], None] | None = None,
) -> AssociationFit:
    """Fit equal-sample-weighted condition association and top eigenvectors.

    ``expression`` is cells by genes in the caller-provided, fixed gene order.
    Every row must have one sample and one of the configured control/case
    labels. Covariance uses the sample divisor (n_cells - 1); correlation uses
    conventional per-sample Pearson correlation. Constant genes fail explicitly
    in correlation mode rather than producing undefined entries.
    """

    if execution_mode in {"sparse_safe", "sparse_safe_condition_weighted", "sparse_safe_gpu_condition_weighted"}:
        if execution_mode == "sparse_safe_condition_weighted":
            return _fit_association_cpu_condition_weighted(
                expression, sample_ids=sample_ids, conditions=conditions,
                gene_names=gene_names, config=config, profile_callback=profile_callback,
            )
        if execution_mode == "sparse_safe_gpu_condition_weighted":
            return _fit_association_gpu_condition_weighted(
                expression, sample_ids=sample_ids, conditions=conditions,
                gene_names=gene_names, config=config, profile_callback=profile_callback,
            )
        return _fit_association_sparse_safe(
            expression, sample_ids=sample_ids, conditions=conditions,
            gene_names=gene_names, config=config,
            profile_callback=profile_callback,
        )
    if execution_mode != "dense":
        raise AssociationError("unsupported execution_mode")
    matrix = np.asarray(expression, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise AssociationError("expression must be a non-empty cells-by-genes matrix")
    if not np.isfinite(matrix).all():
        raise AssociationError("expression contains non-finite values")

    n_cells, n_genes = matrix.shape
    sample_array = np.asarray(sample_ids, dtype=object)
    condition_array = np.asarray(conditions, dtype=object)
    names = tuple(str(name) for name in gene_names)
    if sample_array.ndim != 1 or len(sample_array) != n_cells:
        raise AssociationError("sample_ids must have one entry per expression row")
    if condition_array.ndim != 1 or len(condition_array) != n_cells:
        raise AssociationError("conditions must have one entry per expression row")
    if len(names) != n_genes:
        raise AssociationError("gene_names must have one entry per expression column")
    if len(set(names)) != len(names):
        raise AssociationError("gene_names must be unique and ordered with expression")
    if pd_missing(sample_array) or pd_missing(condition_array):
        raise AssociationError("sample_ids and conditions cannot be missing")
    if config.method.n_components > n_genes:
        raise AssociationError("n_components cannot exceed the selected gene count")

    data_config = config.data
    if data_config.control_label is None or data_config.case_label is None:
        raise AssociationError("data.control_label and data.case_label are required")
    order: list[object] = []
    row_indices: dict[object, list[int]] = {}
    donor_conditions: dict[object, object] = {}
    for row, (sample, condition) in enumerate(zip(sample_array, condition_array)):
        try:
            unseen = sample not in row_indices
        except TypeError as exc:
            raise AssociationError("sample_ids must be hashable scalar values") from exc
        if unseen:
            order.append(sample)
            row_indices[sample] = []
            donor_conditions[sample] = condition
        elif donor_conditions[sample] != condition:
            raise AssociationError(
                f"sample {sample!r} has more than one condition label"
            )
        row_indices[sample].append(row)

    expected_conditions = {data_config.control_label, data_config.case_label}
    observed_conditions = set(donor_conditions.values())
    unexpected = observed_conditions - expected_conditions
    if unexpected:
        raise AssociationError(f"unexpected condition labels: {sorted(map(str, unexpected))}")
    if not expected_conditions.issubset(observed_conditions):
        raise AssociationError("at least one control and one case sample are required")

    matrices: list[np.ndarray] = []
    sample_conditions: list[str] = []
    for sample in order:
        rows = row_indices[sample]
        if len(rows) < config.method.minimum_cells_per_sample:
            raise AssociationError(
                f"sample {sample!r} has {len(rows)} cells; minimum is "
                f"{config.method.minimum_cells_per_sample}"
            )
        x = matrix[rows, :]
        if config.method.association not in {"covariance", "correlation"}:
            raise AssociationError(
                f"unsupported association estimator: {config.method.association!r}"
            )
        if config.method.association == "correlation":
            centered = x - x.mean(axis=0, keepdims=True)
            constant = np.sum(centered * centered, axis=0) == 0
            if np.any(constant):
                bad_names = np.asarray(names)[constant].tolist()
                raise AssociationError(
                    f"sample {sample!r} has constant genes in correlation mode: "
                    + ", ".join(bad_names)
                )
        try:
            sample_matrix = sample_association(
                x,
                correlation=config.method.association == "correlation",
                config=config,
            )
        except BackendError as exc:
            raise AssociationError(str(exc)) from exc
        if not np.isfinite(sample_matrix).all():
            raise AssociationError(f"sample {sample!r} association is non-finite")
        matrices.append(sample_matrix)
        sample_conditions.append(str(donor_conditions[sample]))

    sample_associations = np.stack(matrices, axis=0)
    try:
        control_mean, case_mean, contrast = condition_contrast(
            sample_associations,
            sample_conditions,
            data_config.control_label,
            data_config.case_label,
            config=config,
        )
    except BackendError as exc:
        raise AssociationError(str(exc)) from exc

    try:
        eigenvalues_all, eigenvectors_all = symmetric_eigh(contrast, config=config)
    except BackendError as exc:
        raise AssociationError(str(exc)) from exc
    if not np.isfinite(eigenvalues_all).all() or not np.isfinite(eigenvectors_all).all():
        raise AssociationError("symmetric eigendecomposition returned non-finite values")
    selected = np.argsort(eigenvalues_all, kind="stable")[::-1][
        : config.method.n_components
    ]
    eigenvalues = eigenvalues_all[selected]
    theta = eigenvectors_all[:, selected].copy()

    # A deterministic orientation convention: the largest-magnitude loading
    # in each component is positive; np.argmax gives a stable first-index tie.
    orientation_indices: list[int] = []
    orientation_signs: list[int] = []
    for component in range(theta.shape[1]):
        pivot = int(np.argmax(np.abs(theta[:, component])))
        sign = 1 if theta[pivot, component] >= 0 else -1
        theta[:, component] *= sign
        orientation_indices.append(pivot)
        orientation_signs.append(sign)

    return AssociationFit(
        gene_names=names,
        sample_ids=tuple(str(sample) for sample in order),
        sample_conditions=tuple(sample_conditions),
        sample_associations=sample_associations,
        control_association=control_mean,
        case_association=case_mean,
        contrast=contrast,
        eigenvalues=eigenvalues,
        gene_theta=theta,
        orientation_indices=tuple(orientation_indices),
        orientation_signs=tuple(orientation_signs),
        association=config.method.association,
    )


def _fit_association_sparse_safe(expression, *, sample_ids, conditions, gene_names, config, profile_callback=None):
    """Sparse-safe exact implementation of the current dense association path.

    The association definition is covariance/correlation of each sample's
    *retained raw expression values*. Reference mean/SD are not applied here.
    Bounded row blocks are densified only to form centered cross-products.
    """
    if not sparse.issparse(expression):
        raise AssociationError("sparse_safe association requires sparse expression input")
    matrix = expression.tocsr(copy=False)
    if len(matrix.shape) != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise AssociationError("expression must be a non-empty cells-by-genes matrix")
    if not np.isfinite(matrix.data).all():
        raise AssociationError("expression contains non-finite values")
    n_cells, n_genes = matrix.shape
    sample_array = np.asarray(sample_ids, dtype=object)
    condition_array = np.asarray(conditions, dtype=object)
    names = tuple(str(name) for name in gene_names)
    if sample_array.ndim != 1 or len(sample_array) != n_cells:
        raise AssociationError("sample_ids must have one entry per expression row")
    if condition_array.ndim != 1 or len(condition_array) != n_cells:
        raise AssociationError("conditions must have one entry per expression row")
    if len(names) != n_genes or len(set(names)) != len(names):
        raise AssociationError("gene_names must be unique and ordered with expression")
    if pd_missing(sample_array) or pd_missing(condition_array):
        raise AssociationError("sample_ids and conditions cannot be missing")
    if config.method.n_components > n_genes:
        raise AssociationError("n_components cannot exceed the selected gene count")
    data_config = config.data
    if data_config.control_label is None or data_config.case_label is None:
        raise AssociationError("data.control_label and data.case_label are required")
    try:
        import pandas as pd
        codes, sample_order = pd.factorize(sample_array, sort=False)
    except (TypeError, ValueError) as exc:
        raise AssociationError("sample_ids must be hashable scalar values") from exc
    if np.any(codes < 0):
        raise AssociationError("sample_ids cannot be missing")
    order_rows = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(sample_order))
    boundaries = np.concatenate(([0], np.cumsum(counts)))
    sample_conditions = []
    condition_codes = np.empty(len(sample_order), dtype=object)
    for i in range(len(sample_order)):
        rows = order_rows[boundaries[i] : boundaries[i + 1]]
        group_conditions = condition_array[rows]
        first = group_conditions[0]
        if np.any(group_conditions != first):
            raise AssociationError(f"sample {sample_order[i]!r} has more than one condition label")
        condition_codes[i] = first
        sample_conditions.append(str(first))
    expected = {data_config.control_label, data_config.case_label}
    observed = set(condition_codes.tolist())
    unexpected = observed - expected
    if unexpected:
        raise AssociationError(f"unexpected condition labels: {sorted(map(str, unexpected))}")
    if not expected.issubset(observed):
        raise AssociationError("at least one control and one case sample are required")
    if config.method.association not in {"covariance", "correlation"}:
        raise AssociationError(f"unsupported association estimator: {config.method.association!r}")

    # Each p×p accumulator is required by the existing fitted-state schema.
    # Centered cross-products use bounded dense row blocks and BLAS accumulation;
    # only gene×gene workspaces are dense, never the full cell×gene input.
    control_sum = np.zeros((n_genes, n_genes), dtype=np.float64)
    case_sum = np.zeros_like(control_sum)
    sample_count_control = sample_count_case = 0
    max_rows = max(1, min(config.compute.chunk_size, (64 * 1024 * 1024) // (8 * n_genes)))
    def profile(event, stage, **details):
        if profile_callback is not None:
            profile_callback({"event": event, "stage": stage, **details})

    covariance_start = time.perf_counter()
    profile("stage_start", "per_sample_covariance")
    sample_seconds = 0.0
    for i, sample in enumerate(sample_order):
        sample_start = time.perf_counter()
        rows = order_rows[boundaries[i] : boundaries[i + 1]]
        n_sample = len(rows)
        if n_sample < config.method.minimum_cells_per_sample:
            raise AssociationError(f"sample {sample!r} has {n_sample} cells; minimum is {config.method.minimum_cells_per_sample}")
        sample_sum = np.asarray(matrix[rows].sum(axis=0), dtype=np.float64).reshape(-1)
        sample_mean = sample_sum / n_sample
        target = control_sum if condition_codes[i] == data_config.control_label else case_sum
        sample_cross = np.zeros((n_genes, n_genes), dtype=np.float64, order="F")
        for start in range(0, n_sample, max_rows):
            block = matrix[rows[start : start + max_rows]].toarray().astype(np.float64, copy=False)
            block -= sample_mean
            sample_cross = dgemm(
                1.0, block, block, beta=1.0, c=sample_cross,
                trans_a=True, overwrite_c=1,
            )
            del block
        # This remains the dense backend definition: centered covariance of
        # raw retained expression, divided by sample cell count minus one.
        sample_cross /= n_sample - 1
        sample_cross += sample_cross.T
        sample_cross *= 0.5
        if config.method.association == "correlation":
            ss = np.diag(sample_cross).copy()
            if np.any(ss == 0):
                raise AssociationError(f"sample {sample!r} has constant genes in correlation mode")
            scales = np.sqrt(ss)
            sample_cross /= scales[:, None]
            sample_cross /= scales[None, :]
            sample_cross += sample_cross.T
            sample_cross *= 0.5
        target += sample_cross
        del sample_cross
        if condition_codes[i] == data_config.control_label:
            sample_count_control += 1
        else:
            sample_count_case += 1
        elapsed = time.perf_counter() - sample_start
        sample_seconds += elapsed
        profile(
            "progress", "per_sample_covariance", sample_index=i + 1,
            sample_count=len(sample_order), sample_id=str(sample),
            sample_elapsed_seconds=elapsed, covariance_elapsed_seconds=time.perf_counter() - covariance_start,
        )
    covariance_seconds = time.perf_counter() - covariance_start
    profile("stage_end", "per_sample_covariance", elapsed_seconds=covariance_seconds,
            sample_count=len(sample_order), sample_compute_seconds=sample_seconds)
    profile("stage_start", "condition_level_association_aggregation")
    aggregation_start = time.perf_counter()
    control_mean = control_sum / sample_count_control
    case_mean = case_sum / sample_count_case
    contrast = case_mean - control_mean
    contrast = (contrast + contrast.T) * 0.5
    aggregation_seconds = time.perf_counter() - aggregation_start
    profile("stage_end", "condition_level_association_aggregation",
            elapsed_seconds=aggregation_seconds, control_sample_count=sample_count_control,
            case_sample_count=sample_count_case)
    profile("stage_start", "eigendecomposition")
    eig_start = time.perf_counter()
    try:
        eigenvalues_all, eigenvectors_all = symmetric_eigh(contrast, config=config)
    except BackendError as exc:
        raise AssociationError(str(exc)) from exc
    if not np.isfinite(eigenvalues_all).all() or not np.isfinite(eigenvectors_all).all():
        raise AssociationError("symmetric eigendecomposition returned non-finite values")
    selected = np.argsort(eigenvalues_all, kind="stable")[::-1][: config.method.n_components]
    eigenvalues = eigenvalues_all[selected]
    theta = eigenvectors_all[:, selected].copy()
    orientation_indices, orientation_signs = [], []
    for component in range(theta.shape[1]):
        pivot = int(np.argmax(np.abs(theta[:, component])))
        sign = 1 if theta[pivot, component] >= 0 else -1
        theta[:, component] *= sign
        orientation_indices.append(pivot)
        orientation_signs.append(sign)
    eig_seconds = time.perf_counter() - eig_start
    profile("stage_end", "eigendecomposition", elapsed_seconds=eig_seconds)
    return AssociationFit(
        gene_names=names,
        sample_ids=tuple(str(sample) for sample in sample_order),
        sample_conditions=tuple(sample_conditions),
        sample_associations=np.empty((0, n_genes, n_genes), dtype=np.float64),
        control_association=control_mean,
        case_association=case_mean,
        contrast=contrast,
        eigenvalues=eigenvalues,
        gene_theta=theta,
        orientation_indices=tuple(orientation_indices),
        orientation_signs=tuple(orientation_signs),
        association=config.method.association,
        execution_runtime_seconds={
            "sample_covariance": covariance_seconds,
            "condition_level_association_aggregation": aggregation_seconds,
            "eigendecomposition": eig_seconds,
        },
    )


def _sparse_association_layout(expression, sample_ids, conditions, gene_names, config):
    """Validate sparse association input and return stable per-sample row groups."""
    if not sparse.issparse(expression):
        raise AssociationError("sparse association execution requires sparse expression input")
    matrix = expression.tocsr(copy=False)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise AssociationError("expression must be a non-empty cells-by-genes matrix")
    if not np.isfinite(matrix.data).all():
        raise AssociationError("expression contains non-finite values")
    n_cells, n_genes = matrix.shape
    sample_array = np.asarray(sample_ids, dtype=object)
    condition_array = np.asarray(conditions, dtype=object)
    names = tuple(str(name) for name in gene_names)
    if sample_array.ndim != 1 or len(sample_array) != n_cells:
        raise AssociationError("sample_ids must have one entry per expression row")
    if condition_array.ndim != 1 or len(condition_array) != n_cells:
        raise AssociationError("conditions must have one entry per expression row")
    if len(names) != n_genes or len(set(names)) != len(names):
        raise AssociationError("gene_names must be unique and ordered with expression")
    if pd_missing(sample_array) or pd_missing(condition_array):
        raise AssociationError("sample_ids and conditions cannot be missing")
    if config.method.n_components > n_genes:
        raise AssociationError("n_components cannot exceed the selected gene count")
    data_config = config.data
    if data_config.control_label is None or data_config.case_label is None:
        raise AssociationError("data.control_label and data.case_label are required")
    try:
        import pandas as pd
        codes, sample_order = pd.factorize(sample_array, sort=False)
    except (TypeError, ValueError) as exc:
        raise AssociationError("sample_ids must be hashable scalar values") from exc
    if np.any(codes < 0):
        raise AssociationError("sample_ids cannot be missing")
    order_rows = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(sample_order))
    boundaries = np.concatenate(([0], np.cumsum(counts)))
    sample_conditions = []
    condition_codes = np.empty(len(sample_order), dtype=object)
    sample_rows = []
    for i in range(len(sample_order)):
        rows = order_rows[boundaries[i]:boundaries[i + 1]]
        labels = condition_array[rows]
        if np.any(labels != labels[0]):
            raise AssociationError(f"sample {sample_order[i]!r} has more than one condition label")
        condition_codes[i] = labels[0]
        sample_conditions.append(str(labels[0]))
        if len(rows) < config.method.minimum_cells_per_sample:
            raise AssociationError(f"sample {sample_order[i]!r} has {len(rows)} cells; minimum is {config.method.minimum_cells_per_sample}")
    expected = {data_config.control_label, data_config.case_label}
    observed = set(condition_codes.tolist())
    unexpected = observed - expected
    if unexpected:
        raise AssociationError(f"unexpected condition labels: {sorted(map(str, unexpected))}")
    if not expected.issubset(observed):
        raise AssociationError("at least one control and one case sample are required")
    if config.method.association not in {"covariance", "correlation"}:
        raise AssociationError(f"unsupported association estimator: {config.method.association!r}")
    return matrix, names, sample_order, order_rows, boundaries, condition_codes, sample_conditions


def _finish_condition_weighted_association(control_mean, case_mean, sample_order, sample_conditions,
                                           gene_names, config, *, eigensolver, profile_callback=None,
                                           runtime_seconds=None, gpu_metrics=None):
    """Build contrast and frozen top-component basis from condition matrices."""
    def profile(event, stage, **details):
        if profile_callback is not None:
            profile_callback({"event": event, "stage": stage, **details})

    data_config = config.data
    profile("stage_start", "condition_level_association_aggregation")
    agg_start = time.perf_counter()
    contrast = case_mean - control_mean
    contrast = (contrast + contrast.T) * 0.5
    aggregation_seconds = time.perf_counter() - agg_start
    profile("stage_end", "condition_level_association_aggregation", elapsed_seconds=aggregation_seconds)
    profile("stage_start", "eigendecomposition")
    eig_start = time.perf_counter()
    try:
        eigenvalues_all, eigenvectors_all = eigensolver(contrast)
    except BackendError as exc:
        raise AssociationError(str(exc)) from exc
    if not np.isfinite(eigenvalues_all).all() or not np.isfinite(eigenvectors_all).all():
        raise AssociationError("symmetric eigendecomposition returned non-finite values")
    selected = np.argsort(eigenvalues_all, kind="stable")[::-1][:config.method.n_components]
    eigenvalues = eigenvalues_all[selected]
    theta = eigenvectors_all[:, selected].copy()
    orientation_indices, orientation_signs = [], []
    for component in range(theta.shape[1]):
        pivot = int(np.argmax(np.abs(theta[:, component])))
        sign = 1 if theta[pivot, component] >= 0 else -1
        theta[:, component] *= sign
        orientation_indices.append(pivot)
        orientation_signs.append(sign)
    eig_seconds = time.perf_counter() - eig_start
    profile("stage_end", "eigendecomposition", elapsed_seconds=eig_seconds,
            **(gpu_metrics() if gpu_metrics else {}))
    timings = dict(runtime_seconds or {})
    timings.update({"condition_level_association_aggregation": aggregation_seconds,
                    "eigendecomposition": eig_seconds})
    return AssociationFit(
        gene_names=gene_names,
        sample_ids=tuple(str(s) for s in sample_order),
        sample_conditions=tuple(sample_conditions),
        sample_associations=np.empty((0, len(gene_names), len(gene_names)), dtype=np.float64),
        control_association=control_mean,
        case_association=case_mean,
        contrast=contrast,
        eigenvalues=eigenvalues,
        gene_theta=theta,
        orientation_indices=tuple(orientation_indices),
        orientation_signs=tuple(orientation_signs),
        association=config.method.association,
        execution_runtime_seconds=timings,
    )


def _fit_association_cpu_condition_weighted(expression, *, sample_ids, conditions, gene_names,
                                            config, profile_callback=None):
    """Exact CPU condition-weighted sparse sufficient-statistic association.

    For each condition this computes one weighted Gram, then subtracts the
    exact per-sample centering corrections. CSR remains the input storage;
    bounded dense row chunks feed BLAS, never a full cell×gene array.
    """
    if config.method.association != "covariance":
        raise AssociationError("condition-weighted exact execution currently supports covariance only")
    matrix, names, sample_order, order_rows, boundaries, condition_codes, sample_conditions = _sparse_association_layout(
        expression, sample_ids, conditions, gene_names, config
    )
    data_config = config.data
    sizes = np.diff(boundaries).astype(np.int64)
    sample_sums = np.vstack([
        np.asarray(matrix[order_rows[boundaries[i]:boundaries[i+1]]].sum(axis=0), dtype=np.float64).reshape(-1)
        for i in range(len(sample_order))
    ])
    n_genes = matrix.shape[1]
    control_mean = np.zeros((n_genes, n_genes), dtype=np.float64, order="F")
    case_mean = np.zeros_like(control_mean, order="F")
    runtime = {}
    def profile(event, stage, **details):
        if profile_callback is not None:
            profile_callback({"event": event, "stage": stage, **details})

    max_rows = max(1, min(config.compute.chunk_size, (64 * 1024 * 1024) // (8 * n_genes)))
    for condition, target in ((data_config.control_label, control_mean), (data_config.case_label, case_mean)):
        sample_indices = np.flatnonzero(condition_codes == condition)
        n_samples = len(sample_indices)
        condition_rows = np.concatenate([order_rows[boundaries[i]:boundaries[i+1]] for i in sample_indices])
        sqrt_weights = np.concatenate([
            np.full(sizes[i], np.sqrt(1.0 / (n_samples * (sizes[i] - 1))), dtype=np.float64)
            for i in sample_indices
        ])
        profile("stage_start", "condition_weighted_sparse_gram")
        gram_start = time.perf_counter()
        for start in range(0, len(condition_rows), max_rows):
            stop = min(start + max_rows, len(condition_rows))
            rows = condition_rows[start:stop]
            block = matrix[rows].toarray().astype(np.float64, copy=False)
            block *= sqrt_weights[start:stop, None]
            target[:] = dgemm(1.0, block, block, beta=1.0, c=target,
                              trans_a=True, overwrite_c=1)
            del block
        runtime[f"{str(condition).lower()}_weighted_gram"] = time.perf_counter() - gram_start
        profile("stage_end", "condition_weighted_sparse_gram", elapsed_seconds=runtime[f"{str(condition).lower()}_weighted_gram"],
                condition=str(condition), sample_count=n_samples)
        profile("stage_start", "condition_sample_centering_correction")
        correction_start = time.perf_counter()
        coefficients = 1.0 / (n_samples * sizes[sample_indices] * (sizes[sample_indices] - 1))
        weighted_sums = sample_sums[sample_indices] * np.sqrt(coefficients)[:, None]
        correction = dgemm(1.0, weighted_sums, weighted_sums, beta=0.0, trans_a=True)
        target -= correction
        if config.method.association == "correlation":
            diagonal = np.diag(target).copy()
            if np.any(diagonal <= 0):
                raise AssociationError(f"condition {condition!r} has constant genes in correlation mode")
            scales = np.sqrt(diagonal)
            target /= scales[:, None]
            target /= scales[None, :]
        target[:] = (target + target.T) * 0.5
        runtime[f"{str(condition).lower()}_centering_correction"] = time.perf_counter() - correction_start
        profile("stage_end", "condition_sample_centering_correction",
                elapsed_seconds=runtime[f"{str(condition).lower()}_centering_correction"],
                condition=str(condition), sample_count=n_samples)
        del correction, weighted_sums
    runtime["sample_covariance"] = sum(
        value for key, value in runtime.items()
        if key.endswith("weighted_gram") or key.endswith("centering_correction")
    )
    return _finish_condition_weighted_association(
        control_mean, case_mean, sample_order, sample_conditions, names, config,
        eigensolver=lambda value: symmetric_eigh(value, config=config),
        profile_callback=profile_callback, runtime_seconds=runtime,
    )


def _fit_association_gpu_condition_weighted(expression, *, sample_ids, conditions, gene_names,
                                            config, profile_callback=None):
    """Exact PyTorch CUDA condition-weighted Gram and device eigensolve.

    Bounded CSR row blocks are densified, transferred to CUDA, weighted, and
    accumulated with float64 GEMM. CUDA CSR sparse-sparse SpGEMM can fail for
    this wide workload; the bounded block formulation is algebraically the
    same exact sufficient-statistic computation as the CPU backend and never
    materializes the complete cell-by-gene matrix.
    """
    if config.method.association != "covariance":
        raise AssociationError("GPU condition-weighted exact execution currently supports covariance only")
    try:
        import torch
    except ImportError as exc:
        raise AssociationError("GPU condition-weighted execution requires installed PyTorch") from exc
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise AssociationError("GPU condition-weighted execution found no CUDA device")
    except RuntimeError as exc:
        raise AssociationError(f"GPU condition-weighted execution cannot access CUDA: {exc}") from exc
    matrix, names, sample_order, order_rows, boundaries, condition_codes, sample_conditions = _sparse_association_layout(
        expression, sample_ids, conditions, gene_names, config
    )
    data_config = config.data
    sizes = np.diff(boundaries).astype(np.int64)
    sample_sums = np.vstack([
        np.asarray(matrix[order_rows[i0:i1]].sum(axis=0), dtype=np.float64).reshape(-1)
        for i0, i1 in zip(boundaries[:-1], boundaries[1:])
    ])
    n_genes = matrix.shape[1]
    profile = profile_callback or (lambda event: None)
    device = torch.device("cuda", torch.cuda.current_device())
    def gpu_memory():
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        return {"gpu_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "gpu_free_bytes": int(free), "gpu_total_bytes": int(total)}
    try:
        torch.cuda.reset_peak_memory_stats(device)
        control_gpu = torch.zeros((n_genes, n_genes), dtype=torch.float64, device=device)
        case_gpu = torch.zeros_like(control_gpu)
        runtime = {}
        for condition, target in ((data_config.control_label, control_gpu), (data_config.case_label, case_gpu)):
            sample_indices = np.flatnonzero(condition_codes == condition)
            n_samples = len(sample_indices)
            rows = np.concatenate([order_rows[boundaries[i]:boundaries[i+1]] for i in sample_indices])
            row_sqrt_weights = np.concatenate([
                np.full(sizes[i], np.sqrt(1.0 / (n_samples * (sizes[i] - 1))), dtype=np.float64)
                for i in sample_indices
            ])
            profile({"event":"stage_start","stage":"gpu_condition_weighted_gram",**gpu_memory()})
            gram_start = time.perf_counter()
            max_rows = max(1, min(config.compute.chunk_size, (64 * 1024 * 1024) // (8 * n_genes)))
            for start in range(0, len(rows), max_rows):
                stop = min(start + max_rows, len(rows))
                block = matrix[rows[start:stop]].toarray().astype(np.float64, copy=False)
                block *= row_sqrt_weights[start:stop, None]
                block_gpu = torch.as_tensor(block, dtype=torch.float64, device=device)
                gram_block = block_gpu.T @ block_gpu
                target.add_(gram_block)
                del block, block_gpu, gram_block
            runtime[f"{str(condition).lower()}_weighted_gram"] = time.perf_counter() - gram_start
            profile({"event":"stage_end","stage":"gpu_condition_weighted_gram",
                     "elapsed_seconds":runtime[f"{str(condition).lower()}_weighted_gram"],
                     "condition":str(condition),"sample_count":n_samples,
                     "row_chunk_size":max_rows,**gpu_memory()})
            profile({"event":"stage_start","stage":"gpu_condition_centering_correction",**gpu_memory()})
            correction_start = time.perf_counter()
            coefficients = 1.0 / (n_samples * sizes[sample_indices] * (sizes[sample_indices] - 1))
            sums_gpu = torch.as_tensor(sample_sums[sample_indices], dtype=torch.float64, device=device)
            weighted_sums = sums_gpu * torch.sqrt(torch.as_tensor(coefficients, dtype=torch.float64, device=device))[:, None]
            correction = weighted_sums.T @ weighted_sums
            target.sub_(correction)
            target.copy_((target + target.T) * 0.5)
            torch.cuda.synchronize(device)
            runtime[f"{str(condition).lower()}_centering_correction"] = time.perf_counter() - correction_start
            profile({"event":"stage_end","stage":"gpu_condition_centering_correction",
                     "elapsed_seconds":runtime[f"{str(condition).lower()}_centering_correction"],
                     "condition":str(condition),**gpu_memory()})
            del correction, weighted_sums, sums_gpu
        runtime["sample_covariance"] = sum(
            value for key, value in runtime.items()
            if key.endswith("weighted_gram") or key.endswith("centering_correction")
        )
        # Transfer the condition matrices required by AssociationFit as one
        # final handoff; include it in condition-level aggregation timing.
        profile({"event":"stage_start","stage":"condition_level_association_aggregation",**gpu_memory()})
        aggregation_start = time.perf_counter()
        control_mean = control_gpu.cpu().numpy()
        case_mean = case_gpu.cpu().numpy()
        contrast_gpu = case_gpu - control_gpu
        contrast_gpu = (contrast_gpu + contrast_gpu.T) * 0.5
        contrast = contrast_gpu.cpu().numpy()
        torch.cuda.synchronize(device)
        runtime["condition_level_association_aggregation"] = time.perf_counter() - aggregation_start
        profile({"event":"stage_end","stage":"condition_level_association_aggregation",
                 "elapsed_seconds":runtime["condition_level_association_aggregation"],**gpu_memory()})
        # Keep the full symmetric solve on device, and transfer only selected
        # eigenpairs back to host after deterministic component selection.
        profile({"event":"stage_start","stage":"eigendecomposition",**gpu_memory()})
        eig_start = time.perf_counter()
        eigenvalues_all, eigenvectors_all = torch.linalg.eigh(contrast_gpu)
        selected = torch.argsort(eigenvalues_all, stable=True).flip(0)[:config.method.n_components]
        eigenvalues = eigenvalues_all[selected].cpu().numpy()
        theta = eigenvectors_all[:, selected].cpu().numpy()
        torch.cuda.synchronize(device)
        runtime["eigendecomposition"] = time.perf_counter() - eig_start
        profile({"event":"stage_end","stage":"eigendecomposition",
                 "elapsed_seconds":runtime["eigendecomposition"],**gpu_memory()})
        orientation_indices, orientation_signs = [], []
        for component in range(theta.shape[1]):
            pivot = int(np.argmax(np.abs(theta[:, component])))
            sign = 1 if theta[pivot, component] >= 0 else -1
            theta[:, component] *= sign
            orientation_indices.append(pivot)
            orientation_signs.append(sign)
        control_host, case_host = control_mean, case_mean
        del control_gpu, case_gpu, contrast_gpu, eigenvalues_all, eigenvectors_all
        return AssociationFit(
            gene_names=names, sample_ids=tuple(str(s) for s in sample_order),
            sample_conditions=tuple(sample_conditions),
            sample_associations=np.empty((0,n_genes,n_genes),dtype=np.float64),
            control_association=control_host,case_association=case_host,contrast=contrast,
            eigenvalues=eigenvalues,gene_theta=theta,
            orientation_indices=tuple(orientation_indices),orientation_signs=tuple(orientation_signs),
            association=config.method.association,execution_runtime_seconds=runtime,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise AssociationError(f"CUDA out of memory in exact condition-weighted association: {exc}") from exc
    except RuntimeError as exc:
        raise AssociationError(f"CUDA failure in exact condition-weighted association: {exc}") from exc


def pd_missing(values: np.ndarray) -> bool:
    """Scalar missing-value check without requiring pandas as a dependency."""
    for value in values:
        if value is None:
            return True
        try:
            if bool(np.asarray(value != value).item()):
                return True
        except (TypeError, ValueError):
            return True
    return False
