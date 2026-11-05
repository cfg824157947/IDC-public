import numpy as np
import pytest

from idc.association import AssociationError, fit_association
from idc.config import IDCConfig


def _config(k=2, association="covariance"):
    return IDCConfig.from_mapping(
        {
            "method": {"association": association, "n_components": k},
            "data": {"control_label": "Healthy", "case_label": "Disease"},
        }
    )


def _data():
    # C1 has two points with covariance 2; C2 has three points with
    # covariance 1; D1 has covariance 5. Unequal cell counts make
    # equal-sample weighting distinguishable from pooled-cell weighting.
    expression = np.array(
        [
            [-1.0, 0.0], [1.0, 0.0],
            [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0],
            [-2.0, 0.0], [2.0, 0.0],
        ]
    )
    sample = ["C1", "C1", "C2", "C2", "C2", "D1", "D1"]
    condition = ["Healthy"] * 5 + ["Disease"] * 2
    return expression, sample, condition


def test_sample_covariances_and_equal_sample_contrast_are_exact():
    x, samples, conditions = _data()
    result = fit_association(
        x,
        sample_ids=samples,
        conditions=conditions,
        gene_names=["g1", "g2"],
        config=_config(k=1),
    )
    np.testing.assert_allclose(result.sample_associations[:, 0, 0], [2.0, 1.0, 8.0])
    np.testing.assert_allclose(result.control_association[0, 0], 1.5)
    np.testing.assert_allclose(result.case_association[0, 0], 8.0)
    np.testing.assert_allclose(result.contrast[0, 0], 6.5)
    # Pooled controls would yield 1.333..., so this also checks the weighting.
    assert result.control_association[0, 0] != pytest.approx(4.0 / 3.0)


def test_contrast_is_symmetric_and_eigenpairs_are_ordered_and_valid():
    x, samples, conditions = _data()
    result = fit_association(
        x,
        sample_ids=samples,
        conditions=conditions,
        gene_names=["g1", "g2"],
        config=_config(k=2),
    )
    np.testing.assert_allclose(result.contrast, result.contrast.T)
    assert np.all(np.diff(result.eigenvalues) <= 0)
    np.testing.assert_allclose(
        result.contrast @ result.gene_theta,
        result.gene_theta * result.eigenvalues[np.newaxis, :],
        atol=1e-12,
    )
    for j, pivot in enumerate(result.orientation_indices):
        assert result.gene_theta[pivot, j] >= 0


def test_k_one_uses_same_symmetric_eigensolver_path():
    x, samples, conditions = _data()
    result = fit_association(
        x,
        sample_ids=samples,
        conditions=conditions,
        gene_names=["g1", "g2"],
        config=_config(k=1),
    )
    assert result.gene_theta.shape == (2, 1)
    assert result.eigenvalues.shape == (1,)


def test_missing_conditions_or_labels_are_rejected():
    x, samples, conditions = _data()
    with pytest.raises(AssociationError, match="one control and one case"):
        fit_association(
            x[:5], sample_ids=samples[:5], conditions=conditions[:5],
            gene_names=["g1", "g2"], config=_config(k=1)
        )
    conditions[0] = None
    with pytest.raises(AssociationError, match="cannot be missing"):
        fit_association(
            x, sample_ids=samples, conditions=conditions,
            gene_names=["g1", "g2"], config=_config(k=1)
        )


def test_sample_must_have_consistent_condition_and_enough_cells():
    x, samples, conditions = _data()
    conditions[1] = "Disease"
    with pytest.raises(AssociationError, match="more than one condition"):
        fit_association(
            x, sample_ids=samples, conditions=conditions,
            gene_names=["g1", "g2"], config=_config(k=1)
        )
    x, samples, conditions = _data()
    with pytest.raises(AssociationError, match="minimum is 2"):
        fit_association(
            x[1:], sample_ids=samples[1:], conditions=conditions[1:],
            gene_names=["g1", "g2"], config=_config(k=1)
        )


def test_zero_variance_is_valid_covariance_and_correlation_fails_with_gene_name():
    x, samples, conditions = _data()
    result = fit_association(
        x, sample_ids=samples, conditions=conditions, gene_names=["g1", "g2"],
        config=_config(k=1),
    )
    assert np.all(result.sample_associations[:, 1, :] == 0)
    with pytest.raises(AssociationError, match="g2"):
        fit_association(
            x, sample_ids=samples, conditions=conditions, gene_names=["g1", "g2"],
            config=_config(k=1, association="correlation"),
        )


def test_correlation_mode_uses_conventional_per_sample_pearson():
    x = np.array(
        [
            [1, 2, 3], [2, 4, 2], [4, 3, 5],
            [2, 1, 4], [4, 3, 2], [5, 2, 6],
            [3, 3, 1], [4, 7, 2], [6, 4, 3],
        ], dtype=float
    )
    samples = ["C1"] * 3 + ["C2"] * 3 + ["D1"] * 3
    conditions = ["Healthy"] * 6 + ["Disease"] * 3
    result = fit_association(
        x, sample_ids=samples, conditions=conditions,
        gene_names=["g1", "g2", "g3"],
        config=_config(k=2, association="correlation"),
    )
    centered = x[:3] - x[:3].mean(axis=0)
    ss = np.sum(centered**2, axis=0)
    expected = centered.T @ centered / np.sqrt(np.outer(ss, ss))
    np.testing.assert_allclose(result.sample_associations[0], expected)
    np.testing.assert_allclose(np.diag(result.sample_associations[0]), 1.0)
    np.testing.assert_allclose(
        result.control_association,
        result.sample_associations[:2].mean(axis=0),
    )


def test_nonfinite_values_and_duplicate_gene_names_are_rejected():
    x, samples, conditions = _data()
    x[0, 0] = np.nan
    with pytest.raises(AssociationError, match="non-finite"):
        fit_association(
            x, sample_ids=samples, conditions=conditions, gene_names=["g1", "g2"],
            config=_config(k=1),
        )
    x, samples, conditions = _data()
    with pytest.raises(AssociationError, match="unique"):
        fit_association(
            x, sample_ids=samples, conditions=conditions, gene_names=["g", "g"],
            config=_config(k=1),
        )
