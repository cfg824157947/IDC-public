"""Gene-space scores mapped from a fitted IDC+PU cell direction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import IDCConfig
from .projection import CellScoreNormalization
from .pu import PUFit, PUFitStatus


class GeneScoreError(ValueError):
    """Raised when fitted IDC/PU state cannot define a gene score."""


@dataclass(frozen=True)
class LegacyGeneDirection:
    """Secondary historical classifier direction for LEGACY_REPRODUCTION."""

    beta_historical: np.ndarray
    threshold: float
    n_cells: int
    n_high_probability: int
    n_low_probability: int
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    logistic_coefficient_standardized: np.ndarray
    logistic_intercept: float
    logistic_iterations: int


def fit_legacy_gene_direction(
    cell_score: np.ndarray,
    historical_probability: Sequence[float],
    *,
    threshold: float = 0.5,
    max_iter: int = 1000,
) -> LegacyGeneDirection:
    """Reproduce the historical second classifier, with supplied cells only.

    The historical notebooks thresholded the PU probability at 0.5, fit a
    ``StandardScaler`` + default L2 ``LogisticRegression(max_iter=1000)`` on
    the cell-score components, and divided the standardized coefficient by
    ``scaler.scale_`` before the absolute-component gene score. In validation,
    callers must pass training cells and their training-fit PU probabilities.
    """
    x = np.asarray(cell_score, dtype=np.float64)
    p = np.asarray(historical_probability, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != p.size or x.shape[0] == 0:
        raise GeneScoreError("legacy classifier requires aligned nonempty cell scores and probabilities")
    if not np.isfinite(x).all() or not np.isfinite(p).all():
        raise GeneScoreError("legacy classifier inputs must be finite")
    if not 0.0 <= threshold <= 1.0 or max_iter < 1:
        raise GeneScoreError("invalid legacy threshold or max_iter")
    y = (p >= threshold).astype(np.int8)
    if np.unique(y).size != 2:
        raise GeneScoreError("historical probability threshold produces only one class")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter),
    )
    model.fit(x, y)
    scaler = model.named_steps["standardscaler"]
    logistic = model.named_steps["logisticregression"]
    standardized_beta = np.asarray(logistic.coef_[0], dtype=np.float64)
    beta = standardized_beta / np.asarray(scaler.scale_, dtype=np.float64)
    if not np.isfinite(beta).all():
        raise GeneScoreError("legacy classifier produced a non-finite coefficient")
    return LegacyGeneDirection(
        beta_historical=beta,
        threshold=float(threshold),
        n_cells=int(x.shape[0]),
        n_high_probability=int(y.sum()),
        n_low_probability=int((y == 0).sum()),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        logistic_coefficient_standardized=standardized_beta,
        logistic_intercept=float(logistic.intercept_[0]),
        logistic_iterations=int(np.asarray(logistic.n_iter_).max()),
    )


@dataclass(frozen=True)
class GeneScores:
    """Distinct gene scores and the mapped direction used to calculate them."""

    gene_names: tuple[str, ...]
    cell_theta_direction: np.ndarray
    signed_gene_score: np.ndarray | None
    absolute_signed_gene_score: np.ndarray | None
    component_magnitude_gene_score: np.ndarray | None
    legacy_absolute_gene_score: np.ndarray | None
    degenerate_components: tuple[int, ...]
    resolved_config: dict[str, object]
    provenance: dict[str, object] | None = None


def score_genes(
    idc_fit: Any,
    pu_fit: PUFit,
    *,
    config: IDCConfig,
    legacy_beta_historical: Sequence[float] | None = None,
) -> GeneScores:
    """Map the final PU logistic direction through cell_theta into gene space.

    ``idc_fit`` is expected to expose ``gene_theta``, ``gene_names``, and
    ``cell_score_normalization``.  This small protocol lets the eventual
    aggregate ``IDCFit`` retain the projection state without coupling this
    module to its construction.

    The optional legacy coefficient must come from the historical secondary
    classifier, already expressed in the cell_theta coordinate system.  The
    legacy fit itself is not inferred from the current PU model.
    """

    if pu_fit.status == PUFitStatus.FAILED:
        raise GeneScoreError("cannot score genes from a FAILED PU fit")
    if pu_fit.coefficient is None or pu_fit.scaling_sd is None:
        raise GeneScoreError("PU fit does not contain a final logistic direction")

    try:
        theta = np.asarray(idc_fit.gene_theta, dtype=np.float64)
        names = tuple(str(value) for value in idc_fit.gene_names)
        normalization: CellScoreNormalization = idc_fit.cell_score_normalization
    except (AttributeError, TypeError) as exc:
        raise GeneScoreError(
            "idc_fit must expose gene_theta, gene_names, and "
            "cell_score_normalization"
        ) from exc

    if theta.ndim != 2 or theta.shape[0] != len(names):
        raise GeneScoreError("gene_theta must be genes-by-components")
    if len(set(names)) != len(names):
        raise GeneScoreError("gene_names must be unique and match gene_theta rows")
    if not np.isfinite(theta).all():
        raise GeneScoreError("gene_theta contains non-finite values")

    beta = np.asarray(pu_fit.coefficient, dtype=np.float64).reshape(-1)
    pu_sd = np.asarray(pu_fit.scaling_sd, dtype=np.float64).reshape(-1)
    score_sd = np.asarray(normalization.sd, dtype=np.float64).reshape(-1)
    degenerate = np.asarray(normalization.degenerate_components, dtype=bool).reshape(-1)
    k = theta.shape[1]
    if any(array.size != k for array in (beta, pu_sd, score_sd, degenerate)):
        raise GeneScoreError(
            "PU and cell_score fitted dimensions must match gene_theta components"
        )
    if not np.isfinite(beta).all() or not np.isfinite(pu_sd).all():
        raise GeneScoreError("PU direction and scales must be finite")
    if np.any(pu_sd <= 0):
        raise GeneScoreError("PU scaling SDs must be positive")
    if not np.isfinite(score_sd[~degenerate]).all() or np.any(score_sd[~degenerate] <= 0):
        raise GeneScoreError("nondegenerate cell_score SDs must be finite and positive")

    # z_j = (cell_score_j - mu_PU,j) / sd_PU,j and
    # cell_score_j = (cell_theta_j - mu_score,j) / sd_score,j.
    # Thus the cell_theta slope is beta_z,j / (sd_PU,j * sd_score,j).
    direction = np.zeros(k, dtype=np.float64)
    direction[~degenerate] = beta[~degenerate] / (
        pu_sd[~degenerate] * score_sd[~degenerate]
    )
    signed = theta @ direction

    requested = set(config.gene_score.outputs)
    signed_output = signed if "signed" in requested else None
    absolute_output = np.abs(signed) if "absolute_signed" in requested else None
    component_output = (
        np.abs(theta) @ np.abs(direction)
        if "component_magnitude" in requested
        else None
    )

    legacy_output: np.ndarray | None = None
    if config.gene_score.legacy_reproduction:
        if "legacy_absolute" not in requested:
            raise GeneScoreError(
                "legacy reproduction is enabled but legacy_absolute was not "
                "requested in gene_score.outputs"
            )
        if legacy_beta_historical is None:
            raise GeneScoreError(
                "legacy_absolute requires legacy_beta_historical from the "
                "historical secondary classifier"
            )
        legacy_beta = np.asarray(legacy_beta_historical, dtype=np.float64).reshape(-1)
        if legacy_beta.size != k or not np.isfinite(legacy_beta).all():
            raise GeneScoreError(
                "legacy_beta_historical must be a finite vector matching components"
            )
        legacy_output = np.abs(theta) @ np.abs(legacy_beta)
    elif legacy_beta_historical is not None:
        raise GeneScoreError(
            "legacy_beta_historical was supplied while legacy_reproduction is disabled"
        )

    return GeneScores(
        gene_names=names,
        cell_theta_direction=direction,
        signed_gene_score=signed_output,
        absolute_signed_gene_score=absolute_output,
        component_magnitude_gene_score=component_output,
        legacy_absolute_gene_score=legacy_output,
        degenerate_components=tuple(np.flatnonzero(degenerate).tolist()),
        resolved_config=config.to_dict(),
        provenance=(
            {
                "configuration_sha256": idc_fit.provenance.configuration_sha256,
                "analysis_class": idc_fit.provenance.analysis_class,
                "git_revision": idc_fit.provenance.git_revision,
            }
            if hasattr(idc_fit, "provenance")
            else None
        ),
    )
