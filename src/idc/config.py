"""Typed, serializable IDC configuration.

Publication values are visible fields in this schema.  Inputs for which the
scientific documents do not name a publication value remain ``None`` and are
validated by the operation that needs them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import re
from typing import Any, Mapping, TypeVar

import yaml


T = TypeVar("T")


class ConfigurationError(ValueError):
    """Raised when a configuration is invalid or contains unknown fields."""


@dataclass(frozen=True)
class MethodConfig:
    association: str = "covariance"
    n_components: int = 2
    minimum_cells_per_sample: int = 2
    gene_filter: str | None = None

    def __post_init__(self) -> None:
        if self.association not in {"covariance", "correlation"}:
            raise ConfigurationError(
                "method.association must be 'covariance' or 'correlation'"
            )
        _positive_int(self.n_components, "method.n_components")
        if self.minimum_cells_per_sample < 2:
            raise ConfigurationError(
                "method.minimum_cells_per_sample must be >= 2 for sample covariance"
            )


@dataclass(frozen=True)
class DataConfig:
    expression_source: str | None = None
    upstream_transform: str | None = None
    upstream_target_sum: float | None = None
    control_label: str | None = None
    case_label: str | None = None
    donor_key: str | None = None
    cell_type_key: str | None = None
    target_cell_type: str | None = None

    def __post_init__(self) -> None:
        if self.expression_source is not None:
            source = self.expression_source
            if source not in {"X", "raw"} and not source.startswith("layer:"):
                raise ConfigurationError(
                    "data.expression_source must be 'X', 'raw', or 'layer:<name>'"
                )
            if source == "layer:":
                raise ConfigurationError("data.expression_source layer name is empty")
        if self.upstream_transform is not None and self.upstream_transform not in {
            "identity",
            "log1p",
            "normalize_total_log1p",
        }:
            raise ConfigurationError(
                "data.upstream_transform must be identity, log1p, "
                "or normalize_total_log1p"
            )
        if self.upstream_transform == "normalize_total_log1p":
            if self.upstream_target_sum is None:
                raise ConfigurationError(
                    "data.upstream_target_sum is required for "
                    "normalize_total_log1p"
                )
            _positive_float(self.upstream_target_sum, "data.upstream_target_sum")
        elif self.upstream_target_sum is not None:
            raise ConfigurationError(
                "data.upstream_target_sum is only used by normalize_total_log1p"
            )
        if self.control_label is not None and self.case_label is not None:
            if self.control_label == self.case_label:
                raise ConfigurationError("control_label and case_label must differ")


@dataclass(frozen=True)
class ProjectionConfig:
    # The expression source and transform are required when fitting; they are
    # intentionally not guessed from AnnData.X, .raw, or a named layer.
    reference_population: str | None = "controls"
    scored_population: str | None = None
    reference_weighting: str = "donor_balanced"
    batch_key: str | None = None
    batch_reference_mode: str = "global"
    reference_sd_ddof: int = 1
    zero_reference_sd_policy: str = "exclude"
    cell_score_normalization: str = "cell_weighted"
    cell_score_ddof: int = 1
    zero_cell_score_sd_policy: str = "set_zero"

    def __post_init__(self) -> None:
        if self.reference_weighting != "donor_balanced":
            raise ConfigurationError(
                "projection.reference_weighting must be 'donor_balanced'"
            )
        if self.cell_score_normalization != "cell_weighted":
            raise ConfigurationError(
                "only cell-level mean/sample-SD cell_score normalization is "
                "specified for this implementation; donor-balanced normalization "
                "is reserved for a future scientific specification"
            )
        if self.cell_score_ddof < 0:
            raise ConfigurationError("projection.cell_score_ddof must be >= 0")
        if self.zero_cell_score_sd_policy != "set_zero":
            raise ConfigurationError(
                "the approved zero cell-score SD policy is 'set_zero'"
            )
        if self.reference_sd_ddof < 0:
            raise ConfigurationError("projection.reference_sd_ddof must be >= 0")
        if self.zero_reference_sd_policy not in {"exclude", "fail"}:
            raise ConfigurationError(
                "projection.zero_reference_sd_policy must be 'exclude' or 'fail'"
            )
        if self.batch_reference_mode != "global":
            raise ConfigurationError(
                "only global projection reference is implemented; batch-aware "
                "reference is reserved for a future specified method"
            )
        if self.batch_key is not None:
            raise ConfigurationError(
                "projection.batch_key must be null while batch-aware reference "
                "is unsupported"
            )


@dataclass(frozen=True)
class LogisticConfig:
    penalty: str = "l2"
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 2000
    tol: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.penalty != "l2":
            raise ConfigurationError("only the approved L2 penalty is supported")
        if self.solver != "lbfgs":
            raise ConfigurationError("only the approved lbfgs solver is supported")
        _positive_float(self.C, "pu.logistic.C")
        _positive_int(self.max_iter, "pu.logistic.max_iter")
        _positive_float(self.tol, "pu.logistic.tol")


@dataclass(frozen=True)
class PUCovarianceConfig:
    estimator: str = "donor_weighted_population"
    ridge_trace_scaled: float = 1.0e-4
    maximum_condition_number: float = 1.0e8

    def __post_init__(self) -> None:
        if self.estimator != "donor_weighted_population":
            raise ConfigurationError(
                "pu.covariance.estimator must be 'donor_weighted_population'"
            )
        _positive_float(self.ridge_trace_scaled, "pu.covariance.ridge_trace_scaled")
        _positive_float(
            self.maximum_condition_number,
            "pu.covariance.maximum_condition_number",
        )


@dataclass(frozen=True)
class PUSupportConfig:
    minimum_cells_per_donor: int = 10
    minimum_donors_per_class: int = 3
    minimum_cells_per_class: int = 20
    minimum_positives_per_case_donor: int = 5

    def __post_init__(self) -> None:
        for item in fields(self):
            _positive_int(getattr(self, item.name), f"pu.support.{item.name}")


@dataclass(frozen=True)
class PUConfig:
    input: str = "cell_score"
    scaling_population: str = "training_controls_and_unlabeled_case_cells"
    quantile_init: float = 0.5
    membership_update: str = "fixed_threshold"
    membership_threshold: float = 0.5
    membership_quantile: float | None = None
    max_iter: int = 5
    early_stop: str = "unchanged_pseudo_positive_membership"
    final_refit: bool = True
    categorical_threshold: float = 0.5
    disease_like_label: str = "disease_like"
    control_like_label: str = "control_like"
    fraction_estimator: str = "mean_probability"
    weighting: str = "donor_balanced_equal_class"
    logistic: LogisticConfig = field(default_factory=LogisticConfig)
    covariance: PUCovarianceConfig = field(default_factory=PUCovarianceConfig)
    support: PUSupportConfig = field(default_factory=PUSupportConfig)

    def __post_init__(self) -> None:
        if self.input != "cell_score":
            raise ConfigurationError("pu.input must be 'cell_score'")
        if self.scaling_population != "training_controls_and_unlabeled_case_cells":
            raise ConfigurationError(
                "pu.scaling_population must be "
                "'training_controls_and_unlabeled_case_cells'"
            )
        _open_unit_interval(self.quantile_init, "pu.quantile_init")
        if self.membership_update not in {"fixed_threshold", "quantile"}:
            raise ConfigurationError(
                "pu.membership_update must be 'fixed_threshold' or 'quantile'"
            )
        _closed_unit_interval(self.membership_threshold, "pu.membership_threshold")
        if self.membership_quantile is not None:
            _open_unit_interval(self.membership_quantile, "pu.membership_quantile")
        if self.membership_update == "quantile" and self.membership_quantile is None:
            raise ConfigurationError(
                "pu.membership_quantile is required when membership_update='quantile'"
            )
        _positive_int(self.max_iter, "pu.max_iter")
        if self.early_stop != "unchanged_pseudo_positive_membership":
            raise ConfigurationError(
                "pu.early_stop must be 'unchanged_pseudo_positive_membership'"
            )
        _closed_unit_interval(
            self.categorical_threshold, "pu.categorical_threshold"
        )
        if not self.disease_like_label or not self.control_like_label:
            raise ConfigurationError("PU output labels must be non-empty")
        if self.disease_like_label == self.control_like_label:
            raise ConfigurationError("PU output labels must differ")
        if self.fraction_estimator != "mean_probability":
            raise ConfigurationError(
                "pu.fraction_estimator must be 'mean_probability'"
            )
        if self.weighting != "donor_balanced_equal_class":
            raise ConfigurationError(
                "pu.weighting must be 'donor_balanced_equal_class'"
            )


@dataclass(frozen=True)
class GeneScoreConfig:
    primary_direction: str = "pu_logistic"
    outputs: tuple[str, ...] = (
        "signed",
        "absolute_signed",
        "component_magnitude",
    )
    legacy_reproduction: bool = False

    def __post_init__(self) -> None:
        if self.primary_direction != "pu_logistic":
            raise ConfigurationError(
                "gene_score.primary_direction must be 'pu_logistic'"
            )
        allowed = {
            "signed",
            "absolute_signed",
            "component_magnitude",
            "legacy_absolute",
        }
        if len(set(self.outputs)) != len(self.outputs):
            raise ConfigurationError("gene_score.outputs contains duplicates")
        unknown = set(self.outputs) - allowed
        if unknown:
            raise ConfigurationError(
                f"unsupported gene_score.outputs: {sorted(unknown)}"
            )
        if "legacy_absolute" in self.outputs and not self.legacy_reproduction:
            raise ConfigurationError(
                "requesting legacy_absolute requires "
                "gene_score.legacy_reproduction=true"
            )


@dataclass(frozen=True)
class TechnicalConfig:
    dtype: str = "float64"
    eigen_solver: str = "symmetric_eigh"
    eigen_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.dtype != "float64":
            raise ConfigurationError(
                "technical.dtype is currently fixed to float64; lower precision "
                "is not implemented or validated"
            )
        if self.eigen_solver != "symmetric_eigh":
            raise ConfigurationError(
                "technical.eigen_solver is fixed to the unified symmetric solver"
            )
        _positive_float(self.eigen_tolerance, "technical.eigen_tolerance")


@dataclass(frozen=True)
class ComputeConfig:
    """Execution-only settings; these do not alter the scientific method."""

    backend: str = "cpu"
    device: str | None = None
    dtype: str = "float64"
    chunk_size: int = 4096

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "gpu"}:
            raise ConfigurationError("compute.backend must be 'cpu' or 'gpu'")
        if self.backend == "cpu" and self.device not in {None, "cpu"}:
            raise ConfigurationError("compute.device must be null or 'cpu' for CPU backend")
        if self.backend == "gpu":
            if not isinstance(self.device, str) or not self.device:
                raise ConfigurationError("compute.device is required for GPU backend")
            if re.fullmatch(r"cuda(?::[0-9]+)?", self.device) is None:
                raise ConfigurationError("compute.device must be 'cuda' or 'cuda:<index>'")
        if self.dtype != "float64":
            raise ConfigurationError(
                "compute.dtype is currently fixed to float64 to match the CPU reference"
            )
        _positive_int(self.chunk_size, "compute.chunk_size")


@dataclass(frozen=True)
class IDCConfig:
    method: MethodConfig = field(default_factory=MethodConfig)
    data: DataConfig = field(default_factory=DataConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    pu: PUConfig = field(default_factory=PUConfig)
    gene_score: GeneScoreConfig = field(default_factory=GeneScoreConfig)
    technical: TechnicalConfig = field(default_factory=TechnicalConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    analysis_class: str = "PRIMARY_PAPER_ANALYSIS"
    schema_version: int = 1

    def __post_init__(self) -> None:
        allowed = {
            "PRIMARY_PAPER_ANALYSIS",
            "SENSITIVITY",
            "ABLATION",
            "LEGACY_REPRODUCTION",
            "EXPLORATORY",
            "FULL_DATA_DESCRIPTIVE",
        }
        if self.analysis_class not in allowed:
            raise ConfigurationError(
                f"analysis_class must be one of {sorted(allowed)}"
            )
        if self.schema_version != 1:
            raise ConfigurationError("unsupported configuration schema_version")

    def to_dict(self) -> dict[str, Any]:
        """Return a fully resolved, YAML-compatible configuration mapping."""
        resolved = asdict(self)
        resolved["gene_score"]["outputs"] = list(resolved["gene_score"]["outputs"])
        return resolved

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "IDCConfig":
        return _construct_dataclass(cls, value, "config")

    @classmethod
    def from_yaml(cls, value: str | Path) -> "IDCConfig":
        """Load YAML text, or a path to a YAML file, then resolve all defaults."""
        if isinstance(value, Path):
            content = value.read_text(encoding="utf-8")
        elif "\n" not in value and Path(value).is_file():
            content = Path(value).read_text(encoding="utf-8")
        else:
            content = value
        try:
            parsed = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"invalid YAML: {exc}") from exc
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, Mapping):
            raise ConfigurationError("configuration YAML must contain a mapping")
        return cls.from_mapping(parsed)


def _construct_dataclass(cls: type[T], value: Mapping[str, Any], path: str) -> T:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a mapping")
    valid = {item.name for item in fields(cls)}  # type: ignore[arg-type]
    unknown = set(value) - valid
    if unknown:
        raise ConfigurationError(f"unknown {path} fields: {sorted(unknown)}")

    nested_types: dict[tuple[type[Any], str], type[Any]] = {
        (IDCConfig, "method"): MethodConfig,
        (IDCConfig, "data"): DataConfig,
        (IDCConfig, "projection"): ProjectionConfig,
        (IDCConfig, "pu"): PUConfig,
        (IDCConfig, "gene_score"): GeneScoreConfig,
        (IDCConfig, "technical"): TechnicalConfig,
        (IDCConfig, "compute"): ComputeConfig,
        (PUConfig, "logistic"): LogisticConfig,
        (PUConfig, "covariance"): PUCovarianceConfig,
        (PUConfig, "support"): PUSupportConfig,
    }
    kwargs: dict[str, Any] = {}
    for key, item in value.items():
        nested = nested_types.get((cls, key))
        child_path = f"{path}.{key}"
        if nested is not None:
            kwargs[key] = _construct_dataclass(nested, item, child_path)
        elif cls is GeneScoreConfig and key == "outputs":
            if not isinstance(item, (list, tuple)):
                raise ConfigurationError(f"{child_path} must be a sequence")
            kwargs[key] = tuple(item)
        else:
            kwargs[key] = item
    try:
        return cls(**kwargs)  # type: ignore[call-arg]
    except TypeError as exc:
        raise ConfigurationError(f"invalid {path}: {exc}") from exc


def _positive_int(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{path} must be a positive integer")


def _positive_float(value: Any, path: str) -> None:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path} must be a finite positive number")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ConfigurationError(f"{path} must be a finite positive number")


def _open_unit_interval(value: Any, path: str) -> None:
    _probability_number(value, path)
    if not 0 < value < 1:
        raise ConfigurationError(f"{path} must be strictly between 0 and 1")


def _closed_unit_interval(value: Any, path: str) -> None:
    _probability_number(value, path)
    if not 0 <= value <= 1:
        raise ConfigurationError(f"{path} must be between 0 and 1 inclusive")


def _probability_number(value: Any, path: str) -> None:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path} must be a finite number")
    if not math.isfinite(float(value)):
        raise ConfigurationError(f"{path} must be a finite number")
