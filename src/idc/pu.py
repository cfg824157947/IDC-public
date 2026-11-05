"""PU feature preprocessing and Mahalanobis initialization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .config import IDCConfig


class PUFailureStatus(str, Enum):
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_CLASS = "unsupported_class"
    NONFINITE_SCALER = "nonfinite_scaler"
    COVARIANCE_FAILURE = "covariance_failure"
    CONDITION_NUMBER_FAILURE = "condition_number_failure"
    NUMERICAL_FAILURE = "numerical_failure"
    LOGISTIC_FAILURE = "logistic_failure"
    MEMBERSHIP_CYCLE = "membership_cycle"


class PUFitStatus(str, Enum):
    CONVERGED = "CONVERGED"
    UNCONVERGED = "UNCONVERGED"
    FAILED = "FAILED"


class PUError(ValueError):
    """Typed PU fitting failure with serializable diagnostic context."""

    def __init__(
        self,
        status: PUFailureStatus,
        message: str,
        *,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class PUInitializationFit:
    """Frozen PU scaler, trusted-control covariance, and initial membership."""

    sample_ids: tuple[str, ...]
    sample_conditions: tuple[str, ...]
    eligible_cell_mask: np.ndarray
    eligible_donors: tuple[str, ...]
    excluded_donors: tuple[tuple[str, str], ...]
    scaling_mean: np.ndarray
    scaling_sd: np.ndarray
    zero_variance_features: np.ndarray
    scaled_scores: np.ndarray
    control_mean: np.ndarray
    control_covariance: np.ndarray
    ridge_value: float
    covariance_condition_number: float
    mahalanobis_squared: np.ndarray
    quantile_cutoff: float
    initial_candidate_mask: np.ndarray
    effective_pseudo_positive_mask: np.ndarray
    initial_support_excluded_donors: tuple[str, ...]
    scaling_cell_weights: np.ndarray


@dataclass(frozen=True)
class IterationRecord:
    cycle: int
    membership_before: tuple[int, ...]
    candidate_membership: tuple[int, ...]
    effective_membership: tuple[int, ...]
    candidate_cell_count: int
    effective_cell_count: int
    effective_donor_count: int
    unchanged: bool
    added_count: int = 0
    removed_count: int = 0
    retained_count: int = 0
    membership_jaccard: float = 0.0
    eligible_case_donor_count: int = 0
    candidate_donor_count: int = 0
    candidate_cells_removed_by_support: int = 0
    donors_removed_by_support: int = 0
    logistic_optimizer_iterations: int = 0
    logistic_converged: bool = True


@dataclass(frozen=True)
class LogisticState:
    coefficient: np.ndarray
    intercept: float
    optimizer_iterations: int = 0
    converged: bool = True


@dataclass(frozen=True)
class PUFit:
    status: PUFitStatus
    initialization: PUInitializationFit | None
    scaling_mean: np.ndarray | None
    scaling_sd: np.ndarray | None
    coefficient: np.ndarray | None
    intercept: float | None
    final_effective_membership: np.ndarray | None
    iteration_history: tuple[IterationRecord, ...]
    logistic_fit_count: int
    final_refit_attempted: bool
    final_refit_performed: bool
    categorical_threshold: float
    disease_like_label: str
    control_like_label: str
    final_refit_optimizer_iterations: int | None = None
    failure_status: PUFailureStatus | None = None
    failure_message: str | None = None
    failure_diagnostics: dict[str, object] | None = None
    resolved_config: dict[str, object] | None = None


@dataclass(frozen=True)
class CellProbabilities:
    disease_like_probability: np.ndarray | None
    categorical_label: np.ndarray | None
    status: PUFitStatus
    cell_ids: tuple[str, ...] = ()
    sample_ids: tuple[str, ...] = ()
    provenance: dict[str, object] | None = None


def fit_pu(
    scores: np.ndarray,
    *,
    sample_ids: Sequence[object] | None = None,
    conditions: Sequence[object] | None = None,
    sample_key: str | None = None,
    condition_key: str | None = None,
    config: IDCConfig,
) -> PUFit:
    """Fit the configured PU model and retain explicit convergence status."""

    score_object = scores if hasattr(scores, "cell_score") else None
    values = np.asarray(
        score_object.cell_score if score_object is not None else scores,
        dtype=np.float64,
    )
    if score_object is not None:
        if sample_ids is None:
            sample_ids = score_object.sample_ids
        if conditions is None:
            conditions = score_object.conditions
        if sample_key is not None and score_object.sample_key not in {None, sample_key}:
            raise PUError(PUFailureStatus.INVALID_INPUT, "sample_key differs from CellScores metadata")
        if condition_key is not None and score_object.condition_key not in {None, condition_key}:
            raise PUError(PUFailureStatus.INVALID_INPUT, "condition_key differs from CellScores metadata")
    if sample_ids is None or conditions is None:
        raise PUError(
            PUFailureStatus.INVALID_INPUT,
            "sample_ids and conditions are required, directly or through CellScores",
        )
    history: list[IterationRecord] = []
    init: PUInitializationFit | None = None
    fit_count = 0
    membership_state: np.ndarray | None = None
    final_refit_attempted = False
    final_refit_iterations: int | None = None
    try:
        init = fit_pu_initialization(
            values,
            sample_ids=sample_ids,
            conditions=conditions,
            config=config,
        )
        labels = np.asarray(conditions, dtype=object)
        samples = np.asarray(sample_ids, dtype=object)
        control_mask = init.eligible_cell_mask & (
            labels == config.data.control_label
        )
        case_mask = init.eligible_cell_mask & (labels == config.data.case_label)
        membership = init.effective_pseudo_positive_mask.copy()
        membership_state = membership.copy()
        seen: set[bytes] = {_membership_key(membership)}
        current_state: LogisticState | None = None
        status = PUFitStatus.UNCONVERGED

        for cycle in range(1, config.pu.max_iter + 1):
            fit_indices, fit_labels, weights = _fit_population(
                samples, control_mask, membership, config
            )
            fit_count += 1
            current_state = _fit_logistic(
                init.scaled_scores[fit_indices], fit_labels, weights, config
            )
            probabilities = _predict_from_state(current_state, init.scaled_scores)
            case_indices = np.flatnonzero(case_mask)
            candidate = np.zeros(len(values), dtype=bool)
            if config.pu.membership_update == "fixed_threshold":
                candidate[case_indices] = (
                    probabilities[case_indices] >= config.pu.membership_threshold
                )
            elif config.pu.membership_update == "quantile":
                if config.pu.membership_quantile is None:
                    raise PUError(
                        PUFailureStatus.INVALID_INPUT,
                        "membership_quantile is required in quantile update mode",
                    )
                cutoff = np.quantile(
                    probabilities[case_indices],
                    config.pu.membership_quantile,
                    method="linear",
                )
                candidate[case_indices] = probabilities[case_indices] >= cutoff
            else:
                raise PUError(
                    PUFailureStatus.INVALID_INPUT,
                    f"unsupported membership update: {config.pu.membership_update}",
                )
            updated = _filter_membership_support(candidate, samples, case_mask, config)
            _require_support(
                "updated pseudo-positive class",
                list(_donor_counts_for_mask(samples, updated)),
                int(updated.sum()),
                config,
            )
            unchanged = np.array_equal(updated, membership)
            before_set = set(np.flatnonzero(membership).tolist())
            candidate_set = set(np.flatnonzero(candidate).tolist())
            effective_set = set(np.flatnonzero(updated).tolist())
            before_donors = set(samples[np.flatnonzero(membership)].tolist())
            candidate_donors = set(samples[np.flatnonzero(candidate)].tolist())
            effective_donors = set(samples[np.flatnonzero(updated)].tolist())
            union_count = len(before_set | effective_set)
            history.append(
                IterationRecord(
                    cycle=cycle,
                    membership_before=tuple(np.flatnonzero(membership).tolist()),
                    candidate_membership=tuple(np.flatnonzero(candidate).tolist()),
                    effective_membership=tuple(np.flatnonzero(updated).tolist()),
                    candidate_cell_count=int(candidate.sum()),
                    effective_cell_count=int(updated.sum()),
                    effective_donor_count=len(_donor_counts_for_mask(samples, updated)),
                    unchanged=unchanged,
                    added_count=len(effective_set - before_set),
                    removed_count=len(before_set - effective_set),
                    retained_count=len(before_set & effective_set),
                    membership_jaccard=(
                        1.0 if union_count == 0 else len(before_set & effective_set) / union_count
                    ),
                    eligible_case_donor_count=len(set(samples[case_mask].tolist())),
                    candidate_donor_count=len(candidate_donors),
                    candidate_cells_removed_by_support=len(candidate_set - effective_set),
                    donors_removed_by_support=len(candidate_donors - effective_donors),
                    logistic_optimizer_iterations=current_state.optimizer_iterations,
                    logistic_converged=current_state.converged,
                )
            )
            if unchanged:
                membership = updated
                membership_state = membership.copy()
                status = PUFitStatus.CONVERGED
                break
            key = _membership_key(updated)
            membership_state = updated.copy()
            if key in seen:
                raise PUError(
                    PUFailureStatus.MEMBERSHIP_CYCLE,
                    "effective pseudo-positive membership entered a cycle",
                    diagnostics={"cycle": cycle},
                )
            seen.add(key)
            membership = updated

        if current_state is None:
            raise PUError(PUFailureStatus.LOGISTIC_FAILURE, "PU performed no logistic fits")

        if config.pu.final_refit:
            final_refit_attempted = True
            fit_indices, fit_labels, weights = _fit_population(
                samples, control_mask, membership, config
            )
            fit_count += 1
            current_state = _fit_logistic(
                init.scaled_scores[fit_indices], fit_labels, weights, config
            )
            final_refit_iterations = current_state.optimizer_iterations

        return PUFit(
            status=status,
            initialization=init,
            scaling_mean=init.scaling_mean.copy(),
            scaling_sd=init.scaling_sd.copy(),
            coefficient=current_state.coefficient.copy(),
            intercept=float(current_state.intercept),
            final_effective_membership=membership.copy(),
            iteration_history=tuple(history),
            logistic_fit_count=fit_count,
            final_refit_attempted=final_refit_attempted,
            final_refit_performed=config.pu.final_refit,
            final_refit_optimizer_iterations=final_refit_iterations,
            categorical_threshold=config.pu.categorical_threshold,
            disease_like_label=config.pu.disease_like_label,
            control_like_label=config.pu.control_like_label,
            resolved_config=config.to_dict(),
        )
    except PUError as exc:
        return _failed_fit(
            config,
            init,
            history,
            fit_count,
            membership_state,
            final_refit_attempted,
            exc.status,
            str(exc),
            exc.diagnostics,
        )
    except Exception as exc:
        return _failed_fit(
            config,
            init,
            history,
            fit_count,
            membership_state,
            final_refit_attempted,
            PUFailureStatus.NUMERICAL_FAILURE,
            f"unexpected numerical failure: {exc}",
        )


def predict_pu(fit: PUFit, scores: np.ndarray) -> CellProbabilities:
    """Apply the frozen PU model; failed fits emit no probabilities."""

    score_object = scores if hasattr(scores, "cell_score") else None
    values = np.asarray(
        score_object.cell_score if score_object is not None else scores,
        dtype=np.float64,
    )

    if fit.status is PUFitStatus.FAILED or fit.coefficient is None or fit.intercept is None:
        return CellProbabilities(
            None, None, PUFitStatus.FAILED,
            tuple(getattr(score_object, "cell_ids", ())),
            tuple(getattr(score_object, "sample_ids", ())),
            getattr(score_object, "provenance", None),
        )
    if values.ndim != 2 or values.shape[1] != fit.coefficient.size:
        raise PUError(PUFailureStatus.INVALID_INPUT, "prediction score dimensions mismatch")
    if not np.isfinite(values).all():
        raise PUError(PUFailureStatus.INVALID_INPUT, "prediction scores are non-finite")
    scaled = (values - fit.scaling_mean) / fit.scaling_sd
    probability = _predict_from_state(
        LogisticState(fit.coefficient, fit.intercept), scaled
    )
    labels = np.where(
        probability >= fit.categorical_threshold,
        fit.disease_like_label,
        fit.control_like_label,
    )
    return CellProbabilities(
        probability,
        labels,
        fit.status,
        tuple(getattr(score_object, "cell_ids", ())),
        tuple(getattr(score_object, "sample_ids", ())),
        getattr(score_object, "provenance", None),
    )


def _fit_population(
    sample_ids: np.ndarray,
    controls: np.ndarray,
    membership: np.ndarray,
    config: IDCConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(controls | membership)
    labels = np.where(membership[indices], 1, 0)
    included_samples = sample_ids[indices]
    donors_by_class: list[set[object]] = [set(), set()]
    for idx, y in zip(indices, labels):
        donors_by_class[int(y)].add(sample_ids[idx])
    weights = np.zeros(len(indices), dtype=np.float64)
    n_fit = len(indices)
    for local_idx, (sample, y) in enumerate(zip(included_samples, labels)):
        class_donors = donors_by_class[int(y)]
        count = int(np.sum((sample_ids == sample) & (membership if y else controls)))
        weights[local_idx] = n_fit / (2.0 * len(class_donors) * count)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise PUError(PUFailureStatus.NUMERICAL_FAILURE, "invalid donor/class logistic weights")
    return indices, labels, weights


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    config: IDCConfig,
) -> LogisticState:
    if set(np.unique(labels)) != {0, 1}:
        raise PUError(PUFailureStatus.UNSUPPORTED_CLASS, "logistic fit requires both classes")
    if not np.isfinite(features).all() or not np.isfinite(weights).all():
        raise PUError(PUFailureStatus.NUMERICAL_FAILURE, "logistic inputs are non-finite")
    model = LogisticRegression(
        penalty=config.pu.logistic.penalty,
        C=config.pu.logistic.C,
        solver=config.pu.logistic.solver,
        max_iter=config.pu.logistic.max_iter,
        tol=config.pu.logistic.tol,
        random_state=0,
    )
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(features, labels, sample_weight=weights)
        if any(issubclass(item.category, ConvergenceWarning) for item in captured):
            raise PUError(
                PUFailureStatus.LOGISTIC_FAILURE,
                "logistic regression did not converge",
            )
    except PUError:
        raise
    except Exception as exc:
        raise PUError(
            PUFailureStatus.LOGISTIC_FAILURE,
            f"logistic regression failed: {exc}",
        ) from exc
    coefficient = np.asarray(model.coef_[0], dtype=np.float64)
    intercept = float(model.intercept_[0])
    if not np.isfinite(coefficient).all() or not np.isfinite(intercept):
        raise PUError(
            PUFailureStatus.NUMERICAL_FAILURE,
            "logistic model parameters are non-finite",
        )
    return LogisticState(
        coefficient,
        intercept,
        int(np.asarray(model.n_iter_).max()),
        True,
    )


def _predict_from_state(state: LogisticState, scaled_features: np.ndarray) -> np.ndarray:
    linear = scaled_features @ state.coefficient + state.intercept
    probabilities = expit(linear)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise PUError(PUFailureStatus.NUMERICAL_FAILURE, "logistic probabilities are invalid")
    return probabilities


def _filter_membership_support(
    candidate: np.ndarray,
    sample_ids: np.ndarray,
    case_mask: np.ndarray,
    config: IDCConfig,
) -> np.ndarray:
    effective = candidate & case_mask
    case_donors = dict.fromkeys(sample_ids[case_mask].tolist())
    for donor in case_donors:
        donor_positive = (sample_ids == donor) & effective
        if int(donor_positive.sum()) < config.pu.support.minimum_positives_per_case_donor:
            effective[donor_positive] = False
    return effective


def _membership_key(membership: np.ndarray) -> bytes:
    return np.packbits(membership.astype(np.uint8)).tobytes()


def _failed_fit(
    config: IDCConfig,
    initialization: PUInitializationFit | None,
    history: Sequence[IterationRecord],
    fit_count: int,
    final_membership: np.ndarray | None,
    final_refit_attempted: bool,
    failure_status: PUFailureStatus,
    message: str,
    diagnostics: dict[str, object] | None = None,
) -> PUFit:
    return PUFit(
        status=PUFitStatus.FAILED,
        initialization=initialization,
        scaling_mean=None if initialization is None else initialization.scaling_mean.copy(),
        scaling_sd=None if initialization is None else initialization.scaling_sd.copy(),
        coefficient=None,
        intercept=None,
        final_effective_membership=(
            None if final_membership is None else final_membership.copy()
        ),
        iteration_history=tuple(history),
        logistic_fit_count=fit_count,
        final_refit_attempted=final_refit_attempted,
        final_refit_performed=False,
        categorical_threshold=config.pu.categorical_threshold,
        disease_like_label=config.pu.disease_like_label,
        control_like_label=config.pu.control_like_label,
        failure_status=failure_status,
        failure_message=message,
        failure_diagnostics=diagnostics or {},
        resolved_config=config.to_dict(),
    )


def fit_pu_initialization(
    scores: np.ndarray,
    *,
    sample_ids: Sequence[object],
    conditions: Sequence[object],
    config: IDCConfig,
) -> PUInitializationFit:
    """Fit donor/class-balanced scaler and initialize case pseudo-positives.

    ``scores`` is the already normalized ``cell_score`` matrix. Control/case
    labels are sample-level weak labels; no cell-state or gene-truth fields are
    accepted by this function.
    """

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise PUError(PUFailureStatus.INVALID_INPUT, "scores must be a non-empty matrix")
    if not np.isfinite(values).all():
        raise PUError(PUFailureStatus.INVALID_INPUT, "scores contain non-finite values")
    n_cells, n_features = values.shape
    samples = np.asarray(sample_ids, dtype=object)
    labels = np.asarray(conditions, dtype=object)
    if samples.ndim != 1 or len(samples) != n_cells:
        raise PUError(PUFailureStatus.INVALID_INPUT, "sample_ids length mismatch")
    if labels.ndim != 1 or len(labels) != n_cells:
        raise PUError(PUFailureStatus.INVALID_INPUT, "conditions length mismatch")
    if any(_is_missing(item) for item in samples) or any(_is_missing(item) for item in labels):
        raise PUError(PUFailureStatus.INVALID_INPUT, "sample_ids and conditions cannot be missing")
    data_config = config.data
    if data_config.control_label is None or data_config.case_label is None:
        raise PUError(PUFailureStatus.INVALID_INPUT, "control/case labels must be configured")

    donor_order: list[object] = []
    donor_rows: dict[object, list[int]] = {}
    donor_label: dict[object, object] = {}
    try:
        for i, (sample, label) in enumerate(zip(samples, labels)):
            if sample not in donor_rows:
                donor_order.append(sample)
                donor_rows[sample] = []
                donor_label[sample] = label
            elif donor_label[sample] != label:
                raise PUError(
                    PUFailureStatus.INVALID_INPUT,
                    f"sample {sample!r} has more than one condition label",
                )
            donor_rows[sample].append(i)
    except TypeError as exc:
        raise PUError(
            PUFailureStatus.INVALID_INPUT,
            "sample_ids must be hashable scalar values",
        ) from exc

    expected = {data_config.control_label, data_config.case_label}
    unexpected = set(donor_label.values()) - expected
    if unexpected:
        raise PUError(
            PUFailureStatus.INVALID_INPUT,
            f"unexpected condition labels: {sorted(map(str, unexpected))}",
        )

    eligible_cell_mask = np.zeros(n_cells, dtype=bool)
    eligible_donors: list[object] = []
    excluded: list[tuple[str, str]] = []
    for donor in donor_order:
        rows = donor_rows[donor]
        if len(rows) < config.pu.support.minimum_cells_per_donor:
            excluded.append((str(donor), "below_minimum_cells_per_donor"))
            continue
        eligible_cell_mask[rows] = True
        eligible_donors.append(donor)

    control_mask = labels == data_config.control_label
    case_mask = labels == data_config.case_label
    eligible_control = [d for d in eligible_donors if donor_label[d] == data_config.control_label]
    eligible_case = [d for d in eligible_donors if donor_label[d] == data_config.case_label]
    _require_support(
        "control scaling class", eligible_control,
        int(np.sum(eligible_cell_mask & control_mask)), config,
    )
    _require_support(
        "unlabeled case scaling class", eligible_case,
        int(np.sum(eligible_cell_mask & case_mask)), config,
    )

    scaling_mask = eligible_cell_mask & (control_mask | case_mask)
    scaling_indices = np.flatnonzero(scaling_mask)
    donor_counts = {d: int(np.sum(samples[scaling_indices] == d)) for d in eligible_donors}
    d_by_class = {
        data_config.control_label: len(eligible_control),
        data_config.case_label: len(eligible_case),
    }
    raw_weights = np.zeros(n_cells, dtype=np.float64)
    for idx in scaling_indices:
        donor = samples[idx]
        klass = donor_label[donor]
        raw_weights[idx] = n_cells / (
            2.0 * d_by_class[klass] * donor_counts[donor]
        )
    weight_sum = raw_weights[scaling_mask].sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise PUError(PUFailureStatus.NONFINITE_SCALER, "invalid PU scaling weights")
    weights = raw_weights / weight_sum
    scaling_values = values[scaling_mask]
    scaling_weights = weights[scaling_mask]
    scaler_mean = np.sum(scaling_weights[:, None] * scaling_values, axis=0)
    scaler_variance = np.sum(
        scaling_weights[:, None] * (scaling_values - scaler_mean) ** 2,
        axis=0,
    )
    if not np.isfinite(scaler_mean).all() or not np.isfinite(scaler_variance).all():
        raise PUError(PUFailureStatus.NONFINITE_SCALER, "PU scaler is non-finite")
    zero_variance = scaler_variance == 0
    scaler_sd = np.sqrt(scaler_variance)
    scaler_sd[zero_variance] = 1.0
    scaled = (values - scaler_mean) / scaler_sd
    if not np.isfinite(scaled).all():
        raise PUError(PUFailureStatus.NONFINITE_SCALER, "scaled PU scores are non-finite")

    eligible_control_mask = eligible_cell_mask & control_mask
    control_indices = np.flatnonzero(eligible_control_mask)
    control_weights = np.zeros(n_cells, dtype=np.float64)
    for idx in control_indices:
        donor = samples[idx]
        control_weights[idx] = 1.0 / (
            len(eligible_control) * donor_counts[donor]
        )
    control_mean = np.sum(control_weights[control_indices, None] * scaled[control_indices], axis=0)
    centered_controls = scaled[control_indices] - control_mean
    covariance = (centered_controls.T * control_weights[control_indices]) @ centered_controls
    covariance = (covariance + covariance.T) * 0.5
    if not np.isfinite(covariance).all():
        raise PUError(PUFailureStatus.COVARIANCE_FAILURE, "control covariance is non-finite")
    ridge = (
        config.pu.covariance.ridge_trace_scaled
        * float(np.trace(covariance))
        / n_features
    )
    stabilized = covariance + ridge * np.eye(n_features)
    try:
        condition_number = float(np.linalg.cond(stabilized))
    except np.linalg.LinAlgError as exc:
        raise PUError(
            PUFailureStatus.COVARIANCE_FAILURE,
            "could not calculate stabilized control covariance condition number",
        ) from exc
    if (
        not np.isfinite(condition_number)
        or condition_number > config.pu.covariance.maximum_condition_number
    ):
        raise PUError(
            PUFailureStatus.CONDITION_NUMBER_FAILURE,
            "stabilized control covariance exceeds the configured condition-number "
            "limit",
            diagnostics={
                "condition_number": condition_number,
                "maximum_condition_number": config.pu.covariance.maximum_condition_number,
            },
        )

    unlabeled_indices = np.flatnonzero(eligible_cell_mask & case_mask)
    centered_unlabeled = scaled[unlabeled_indices] - control_mean
    try:
        solved = np.linalg.solve(stabilized, centered_unlabeled.T).T
    except np.linalg.LinAlgError as exc:
        raise PUError(
            PUFailureStatus.COVARIANCE_FAILURE,
            "could not solve stabilized control covariance system",
        ) from exc
    mahalanobis = np.full(n_cells, np.nan, dtype=np.float64)
    distances = np.einsum("ij,ij->i", centered_unlabeled, solved)
    if not np.isfinite(distances).all():
        raise PUError(PUFailureStatus.NUMERICAL_FAILURE, "Mahalanobis scores are non-finite")
    mahalanobis[unlabeled_indices] = distances
    cutoff = float(np.quantile(distances, config.pu.quantile_init, method="linear"))
    initial_candidates = np.zeros(n_cells, dtype=bool)
    initial_candidates[unlabeled_indices] = distances >= cutoff

    effective = initial_candidates.copy()
    support_excluded: list[str] = []
    for donor in eligible_case:
        donor_mask = (samples == donor) & effective
        if int(donor_mask.sum()) < config.pu.support.minimum_positives_per_case_donor:
            effective[donor_mask] = False
            support_excluded.append(str(donor))
    effective_counts = _donor_counts_for_mask(samples, effective)
    _require_support(
        "initial pseudo-positive class", list(effective_counts),
        int(effective.sum()), config,
    )

    return PUInitializationFit(
        sample_ids=tuple(str(value) for value in samples),
        sample_conditions=tuple(str(value) for value in labels),
        eligible_cell_mask=eligible_cell_mask,
        eligible_donors=tuple(str(value) for value in eligible_donors),
        excluded_donors=tuple(excluded),
        scaling_mean=scaler_mean,
        scaling_sd=scaler_sd,
        zero_variance_features=zero_variance,
        scaled_scores=scaled,
        control_mean=control_mean,
        control_covariance=stabilized,
        ridge_value=ridge,
        covariance_condition_number=condition_number,
        mahalanobis_squared=mahalanobis,
        quantile_cutoff=cutoff,
        initial_candidate_mask=initial_candidates,
        effective_pseudo_positive_mask=effective,
        initial_support_excluded_donors=tuple(support_excluded),
        scaling_cell_weights=weights,
    )


def _require_support(
    label: str,
    donors: Sequence[object],
    cells: int,
    config: IDCConfig,
) -> None:
    if (
        len(donors) < config.pu.support.minimum_donors_per_class
        or cells < config.pu.support.minimum_cells_per_class
    ):
        raise PUError(
            PUFailureStatus.UNSUPPORTED_CLASS,
            f"{label} does not meet donor/cell support thresholds",
            diagnostics={
                "class": label,
                "donor_count": len(donors),
                "cell_count": cells,
                "minimum_donors": config.pu.support.minimum_donors_per_class,
                "minimum_cells": config.pu.support.minimum_cells_per_class,
            },
        )


def _donor_counts_for_mask(
    sample_ids: np.ndarray,
    mask: np.ndarray,
) -> dict[object, int]:
    result: dict[object, int] = {}
    for donor in sample_ids[mask]:
        result[donor] = result.get(donor, 0) + 1
    return result


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(np.asarray(value != value).item())
    except (TypeError, ValueError):
        return True
