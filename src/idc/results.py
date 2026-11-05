"""Fitted-state/result objects and lightweight reproducibility metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .association import AssociationFit
from .projection import CellScoreNormalization, ProjectionFit
from .backend import execution_provenance
from .canonical import canonical_config_hash


@dataclass(frozen=True)
class Provenance:
    resolved_config: dict[str, object]
    configuration_sha256: str
    method_version: str
    git_revision: str | None
    input_identifier: str | None
    sample_ids: tuple[str, ...]
    gene_names: tuple[str, ...]
    random_seed: int | None
    analysis_class: str
    software: dict[str, str | None]
    execution: dict[str, object]


@dataclass(frozen=True)
class IDCFit:
    """IDC learned state; contains no reference to the input AnnData object."""

    association_fit: AssociationFit
    projection_fit: ProjectionFit
    cell_score_normalization: CellScoreNormalization
    sample_key: str
    condition_key: str
    scored_cell_ids: tuple[str, ...]
    scored_sample_ids: tuple[str, ...]
    scored_conditions: tuple[str, ...]
    provenance: Provenance
    execution_mode: str = "dense"

    @property
    def gene_theta(self):
        return self.association_fit.gene_theta

    @property
    def eigenvalues(self):
        return self.association_fit.eigenvalues

    @property
    def gene_names(self):
        return self.association_fit.gene_names


def build_provenance(
    data: Any, config, *, sample_ids, gene_names, execution_mode: str = "dense",
    stage_runtime_seconds: dict[str, float] | None = None,
) -> Provenance:
    """Capture available provenance without depending on notebook globals."""

    import os
    import subprocess
    import sys

    resolved = config.to_dict()
    revision = None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = None
    try:
        import scipy
        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = None
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = None
    filename = getattr(data, "filename", None)
    input_id = os.fspath(filename) if filename else None
    execution = execution_provenance(config)
    execution_configuration = {
        "execution_mode": execution_mode,
        "backend": config.compute.backend,
        "device": config.compute.device or "cpu",
        "dtype": config.compute.dtype,
        "projection_chunk_size": config.compute.chunk_size,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }
    execution.update(execution_configuration)
    execution["execution_config_sha256"] = canonical_config_hash(execution_configuration)
    if stage_runtime_seconds is not None:
        execution["stage_runtime_seconds"] = dict(stage_runtime_seconds)
    return Provenance(
        resolved_config=resolved,
        configuration_sha256=canonical_config_hash(resolved),
        method_version="unreleased-source (no package version metadata)",
        git_revision=revision,
        input_identifier=input_id,
        sample_ids=tuple(map(str, sample_ids)),
        gene_names=tuple(map(str, gene_names)),
        random_seed=None,
        analysis_class=config.analysis_class,
        software={
            "python": sys.version.split()[0],
            "numpy": numpy_version,
            "scipy": scipy_version,
            "scikit_learn": sklearn_version,
        },
        execution=execution,
    )
