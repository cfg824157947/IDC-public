"""Execution backend helpers shared by the CPU and optional CUDA paths."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import IDCConfig


class BackendError(RuntimeError):
    """Requested execution backend cannot perform the requested operation."""


def resolve_device(config: IDCConfig):
    """Return ``None`` for CPU or a validated torch CUDA device."""
    if config.compute.backend == "cpu":
        return None
    try:
        import torch
    except ImportError as exc:
        raise BackendError("GPU backend requested, but PyTorch is not installed") from exc
    try:
        device = torch.device(config.compute.device)
    except (TypeError, RuntimeError) as exc:
        raise BackendError(f"invalid GPU device {config.compute.device!r}") from exc
    if device.type != "cuda":
        raise BackendError("GPU backend requires a CUDA device")
    if not torch.cuda.is_available():
        raise BackendError("GPU backend requested, but CUDA is unavailable")
    if device.index is None:
        device = torch.device("cuda", 0)
    if device.index >= torch.cuda.device_count():
        raise BackendError(
            f"GPU device index {device.index} is invalid; only "
            f"{torch.cuda.device_count()} CUDA device(s) are available"
        )
    return device


def _torch(config: IDCConfig):
    resolve_device(config)
    import torch

    return torch


def sample_association(x: np.ndarray, *, correlation: bool, config: IDCConfig) -> np.ndarray:
    """Compute one centered sample covariance or Pearson matrix."""
    device = resolve_device(config)
    if device is None:
        centered = x - x.mean(axis=0, keepdims=True)
        cross = centered.T @ centered
        if correlation:
            ss = np.sum(centered * centered, axis=0)
            if np.any(ss == 0):
                raise BackendError("constant gene in Pearson correlation input")
            result = cross / np.sqrt(np.outer(ss, ss))
        else:
            result = cross / (x.shape[0] - 1)
        return (result + result.T) * 0.5

    torch = _torch(config)
    try:
        tx = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64, device=device)
        centered = tx - tx.mean(dim=0, keepdim=True)
        cross = centered.T @ centered
        if correlation:
            ss = torch.sum(centered * centered, dim=0)
            if bool(torch.any(ss == 0).item()):
                raise BackendError("constant gene in Pearson correlation input")
            result = cross / torch.sqrt(torch.outer(ss, ss))
        else:
            result = cross / (x.shape[0] - 1)
        result = (result + result.T) * 0.5
        return result.cpu().numpy()
    except torch.cuda.OutOfMemoryError as exc:
        raise BackendError("CUDA out of memory during sample association") from exc
    except BackendError:
        raise
    except RuntimeError as exc:
        raise BackendError(f"CUDA sample association failed: {exc}") from exc


def symmetric_eigh(matrix: np.ndarray, *, config: IDCConfig):
    """Use the symmetric eigensolver on the explicitly selected backend."""
    device = resolve_device(config)
    if device is None:
        return np.linalg.eigh(matrix)
    torch = _torch(config)
    try:
        value, vector = torch.linalg.eigh(
            torch.as_tensor(np.ascontiguousarray(matrix), dtype=torch.float64, device=device)
        )
        return value.cpu().numpy(), vector.cpu().numpy()
    except torch.cuda.OutOfMemoryError as exc:
        raise BackendError("CUDA out of memory during symmetric eigendecomposition") from exc
    except RuntimeError as exc:
        raise BackendError(f"CUDA symmetric eigendecomposition failed: {exc}") from exc


def condition_contrast(sample_matrices, sample_conditions, control_label, case_label, *, config):
    """Equal-sample condition means and symmetric case-control contrast.

    Aggregation is kept behind the backend dispatch just like the per-sample
    matrix operation.  The returned fitted state remains host NumPy arrays.
    """
    device = resolve_device(config)
    conditions = np.asarray(sample_conditions, dtype=object)
    matrices = np.asarray(sample_matrices, dtype=np.float64)
    control = conditions == control_label
    case = conditions == case_label
    if device is None:
        control_mean = matrices[control].mean(axis=0)
        case_mean = matrices[case].mean(axis=0)
        contrast = case_mean - control_mean
    else:
        torch = _torch(config)
        try:
            # Stream one matrix at a time so a second full sample×gene×gene
            # CUDA stack is never materialized.
            control_sum = torch.zeros(matrices.shape[1:], dtype=torch.float64, device=device)
            case_sum = torch.zeros_like(control_sum)
            for i, matrix in enumerate(matrices):
                target = control_sum if control[i] else case_sum
                target.add_(torch.as_tensor(np.ascontiguousarray(matrix), dtype=torch.float64, device=device))
            control_mean_t = control_sum / int(control.sum())
            case_mean_t = case_sum / int(case.sum())
            contrast_t = case_mean_t - control_mean_t
            contrast_t = (contrast_t + contrast_t.T) * 0.5
            control_mean = control_mean_t.cpu().numpy()
            case_mean = case_mean_t.cpu().numpy()
            contrast = contrast_t.cpu().numpy()
        except torch.cuda.OutOfMemoryError as exc:
            raise BackendError("CUDA out of memory aggregating condition associations") from exc
        except RuntimeError as exc:
            raise BackendError(f"CUDA condition association aggregation failed: {exc}") from exc
    contrast = (contrast + contrast.T) * 0.5
    return control_mean, case_mean, contrast


def projection(expression, mean, sd, theta, *, config: IDCConfig) -> np.ndarray:
    """Standardize and project in bounded row chunks on the selected backend."""
    device = resolve_device(config)
    if device is None:
        return ((expression - mean) / sd) @ theta
    torch = _torch(config)
    n, k = expression.shape[0], theta.shape[1]
    result = np.empty((n, k), dtype=np.float64)
    try:
        tmean = torch.as_tensor(mean, dtype=torch.float64, device=device)
        tsd = torch.as_tensor(sd, dtype=torch.float64, device=device)
        ttheta = torch.as_tensor(theta, dtype=torch.float64, device=device)
        for start in range(0, n, config.compute.chunk_size):
            end = min(start + config.compute.chunk_size, n)
            block = torch.as_tensor(
                np.ascontiguousarray(expression[start:end]),
                dtype=torch.float64,
                device=device,
            )
            projected = ((block - tmean) / tsd) @ ttheta
            result[start:end] = projected.cpu().numpy()
        return result
    except torch.cuda.OutOfMemoryError as exc:
        raise BackendError("CUDA out of memory during expression projection") from exc
    except RuntimeError as exc:
        raise BackendError(f"CUDA expression projection failed: {exc}") from exc


def reference_mean_sd(expression, donor_indices, ddof: int, *, config: IDCConfig):
    """Compute the approved donor-balanced mean and weighted sample SD."""
    device = resolve_device(config)
    if device is None:
        donor_means = np.stack([expression[idx].mean(axis=0) for idx in donor_indices])
        mean = donor_means.mean(axis=0)
        n_donors = len(donor_indices)
        weights = np.zeros(expression.shape[0], dtype=np.float64)
        for idx in donor_indices:
            weights[idx] = 1.0 / (n_donors * len(idx))
        correction = 1.0 - ddof * np.sum(weights * weights)
        variance = np.sum(weights[:, None] * (expression - mean) ** 2, axis=0) / correction
        return mean, np.sqrt(variance)

    torch = _torch(config)
    try:
        n_donors = len(donor_indices)
        donor_means = []
        # Transfer one donor block at a time; never create a dense second copy
        # of the complete control reference on the device.
        for idx in donor_indices:
            donor = torch.as_tensor(
                np.ascontiguousarray(expression[idx]), dtype=torch.float64, device=device
            )
            donor_means.append(donor.mean(dim=0))
        mean = torch.stack(donor_means).mean(dim=0)
        correction = 1.0
        variance_numerator = torch.zeros_like(mean)
        for idx in donor_indices:
            donor = torch.as_tensor(
                np.ascontiguousarray(expression[idx]), dtype=torch.float64, device=device
            )
            weight = 1.0 / (n_donors * len(idx))
            correction -= ddof * len(idx) * weight * weight
            variance_numerator += weight * torch.sum((donor - mean) ** 2, dim=0)
        if correction <= 0:
            return mean.cpu().numpy(), np.full(expression.shape[1], np.nan)
        sd = torch.sqrt(variance_numerator / correction)
        return mean.cpu().numpy(), sd.cpu().numpy()
    except torch.cuda.OutOfMemoryError as exc:
        raise BackendError("CUDA out of memory fitting control reference") from exc
    except RuntimeError as exc:
        raise BackendError(f"CUDA control-reference calculation failed: {exc}") from exc


def fit_cell_score_stats(cell_theta, ddof: int, *, config: IDCConfig):
    """Fit arithmetic cell mean and sample SD on the selected backend."""
    device = resolve_device(config)
    values = np.asarray(cell_theta, dtype=np.float64)
    if device is None:
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=ddof) if values.shape[0] > ddof else np.full(values.shape[1], np.nan)
        return mean, sd
    torch = _torch(config)
    try:
        tv = torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float64, device=device)
        mean = tv.mean(dim=0)
        sd = tv.std(dim=0, correction=ddof) if values.shape[0] > ddof else torch.full_like(mean, float("nan"))
        return mean.cpu().numpy(), sd.cpu().numpy()
    except torch.cuda.OutOfMemoryError as exc:
        raise BackendError("CUDA out of memory fitting cell_score normalization") from exc
    except RuntimeError as exc:
        raise BackendError(f"CUDA cell_score normalization failed: {exc}") from exc


def apply_cell_score(cell_theta, mean, sd, degenerate, *, config: IDCConfig):
    """Apply frozen cell_score state; degenerate dimensions map to zero."""
    device = resolve_device(config)
    values = np.asarray(cell_theta, dtype=np.float64)
    if device is None:
        safe_sd = np.where(degenerate, 1.0, sd)
        result = (values - mean) / safe_sd
        result[:, degenerate] = 0.0
        return result
    torch = _torch(config)
    try:
        tv = torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float64, device=device)
        tm = torch.as_tensor(mean, dtype=torch.float64, device=device)
        ts = torch.as_tensor(np.where(degenerate, 1.0, sd), dtype=torch.float64, device=device)
        result = (tv - tm) / ts
        mask = torch.as_tensor(degenerate, dtype=torch.bool, device=device)
        result[:, mask] = 0.0
        return result.cpu().numpy()
    except torch.cuda.OutOfMemoryError as exc:
        raise BackendError("CUDA out of memory applying cell_score normalization") from exc
    except RuntimeError as exc:
        raise BackendError(f"CUDA cell_score transform failed: {exc}") from exc


def execution_provenance(config: IDCConfig) -> dict[str, Any]:
    """Describe requested/actual execution without silently claiming CUDA use."""
    result: dict[str, Any] = {
        "backend": config.compute.backend,
        "device": config.compute.device or "cpu",
        "dtype": config.compute.dtype,
        "projection_chunk_size": config.compute.chunk_size,
        "stages": {
            "sample_association": config.compute.backend,
            "condition_association_aggregation": config.compute.backend,
            "symmetric_eigendecomposition": config.compute.backend,
            "control_reference_statistics": config.compute.backend,
            "projection": config.compute.backend,
            "cell_score_fit_and_transform": config.compute.backend,
            "pu_scaler_covariance_logistic_and_prediction": "cpu",
            "gene_score_backtransform_and_outputs": "cpu",
        },
        "gpu_model": None,
        "cuda_version": None,
        "pytorch_version": None,
    }
    if config.compute.backend == "gpu":
        device = resolve_device(config)
        import torch

        result.update(
            device=str(device),
            gpu_model=torch.cuda.get_device_name(device),
            cuda_version=torch.version.cuda,
            pytorch_version=torch.__version__,
        )
    return result
