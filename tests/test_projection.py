import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from idc.config import IDCConfig
from idc.projection import (
    apply_cell_score_normalization,
    ProjectionError,
    ProjectionFailureStatus,
    extract_expression,
    fit_cell_score_normalization,
    fit_projection,
    transform_projection,
)


def _config(source="X", transform="identity", **projection):
    return IDCConfig.from_mapping(
        {
            "data": {
                "expression_source": source,
                "upstream_transform": transform,
                "upstream_target_sum": projection.pop("upstream_target_sum", None),
            },
            "projection": projection,
        }
    )


def _adata(matrix):
    data = ad.AnnData(
        X=matrix,
        var={"gene_id": ["g1", "g2", "g3"]},
    )
    data.var_names = ["g1", "g2", "g3"]
    return data


def test_global_control_reference_ddof_one_and_projection_equation():
    values = np.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [5.0, 8.0, 9.0],
            [7.0, 6.0, 11.0],
        ]
    )
    data = _adata(values)
    config = _config()
    theta = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    fit = fit_projection(
        data, gene_theta=theta, reference_mask=[True, True, False, False],
        reference_donor_ids=["d1", "d1", "d2", "d2"],
        config=config,
    )
    expected_mean = values[:2].mean(axis=0)
    expected_sd = values[:2].std(axis=0, ddof=1)
    np.testing.assert_allclose(fit.reference_mean, expected_mean)
    np.testing.assert_allclose(fit.reference_sd, expected_sd)
    assert fit.reference_sd_ddof == 1
    transformed = transform_projection(fit, data, config=config)
    expected = ((values - expected_mean) / expected_sd) @ theta
    np.testing.assert_allclose(transformed.cell_theta, expected)
    assert transformed.gene_names == ("g1", "g2", "g3")


def test_zero_reference_sd_gene_is_excluded_with_auditable_diagnostic():
    values = np.array([[1, 2, 3], [1, 4, 5], [2, 3, 4], [3, 2, 6]], dtype=float)
    fit = fit_projection(
        _adata(values), gene_theta=np.ones((3, 1)),
        reference_mask=[True, True, False, False],
        reference_donor_ids=["d1", "d1", "d2", "d2"], config=_config(),
    )
    assert fit.genes_before_reference == ("g1", "g2", "g3")
    assert fit.excluded_gene_ids == ("g1",)
    assert fit.excluded_gene_reasons == (("g1", "zero_control_reference_sd"),)
    assert fit.gene_names == ("g2", "g3")
    assert fit.gene_theta.shape == (2, 1)
    assert fit.reference_mean.shape == fit.reference_sd.shape == (2,)
    projected = transform_projection(fit, _adata(values), config=_config())
    assert projected.gene_names == ("g2", "g3")


def test_nonfinite_reference_sd_gene_is_excluded():
    values = np.array(
        [[1e308, 2, 3], [-1e308, 4, 5], [0, 3, 4], [1, 2, 6]], dtype=float
    )
    with np.errstate(over="ignore", invalid="ignore"):
        fit = fit_projection(
            _adata(values), gene_theta=np.ones((3, 1)),
            reference_mask=[True, True, False, False],
            reference_donor_ids=["d1", "d1", "d2", "d2"], config=_config(),
        )
    assert fit.excluded_gene_ids == ("g1",)
    assert fit.excluded_gene_reasons == (("g1", "nonfinite_control_reference_sd"),)
    assert fit.gene_names == ("g2", "g3")


def test_strict_reference_sd_action_is_configurable():
    values = np.array([[1, 2, 3], [1, 4, 5], [2, 3, 4], [3, 2, 6]], dtype=float)
    with pytest.raises(ProjectionError) as error:
        fit_projection(
            _adata(values), gene_theta=np.ones((3, 1)),
            reference_mask=[True, True, False, False],
            reference_donor_ids=["d1", "d1", "d2", "d2"],
            config=_config(zero_reference_sd_policy="fail"),
        )
    assert error.value.status is ProjectionFailureStatus.ZERO_REFERENCE_SD
    assert error.value.genes == ("g1",)


def test_reference_mean_and_ddof_one_sd_are_donor_balanced():
    # Donor A contributes 2 cells and donor B 4. Their means are equally
    # weighted, rather than allowing donor B's larger cell count to dominate.
    values = np.array(
        [
            [0.0, 1.0, 4.0], [2.0, 3.0, 8.0],
            [8.0, 5.0, 1.0], [10.0, 8.0, 3.0],
            [12.0, 6.0, 6.0], [14.0, 9.0, 7.0],
        ]
    )
    donors = ["A", "A", "B", "B", "B", "B"]
    fit = fit_projection(
        _adata(values), gene_theta=np.eye(3, 1),
        reference_mask=[True] * 6, reference_donor_ids=donors, config=_config(),
    )
    weights = np.array([1 / 4, 1 / 4, 1 / 8, 1 / 8, 1 / 8, 1 / 8])
    expected_mean = np.sum(weights[:, None] * values, axis=0)
    denom = 1 - np.sum(weights**2)
    expected_sd = np.sqrt(
        np.sum(weights[:, None] * (values - expected_mean) ** 2, axis=0) / denom
    )
    np.testing.assert_allclose(fit.reference_mean, expected_mean)
    np.testing.assert_allclose(fit.reference_sd, expected_sd)
    assert fit.reference_mean[0] == pytest.approx(6.0)
    assert fit.reference_cell_count == 6
    assert fit.reference_donor_ids == ("A", "B")
    assert fit.reference_cells_per_donor == (2, 4)


def test_retained_gene_order_is_original_order_after_multiple_exclusions():
    values = np.array(
        [[1, 2, 3], [1, 2, 5], [2, 3, 4], [3, 5, 6]], dtype=float
    )
    fit = fit_projection(
        _adata(values), gene_theta=np.ones((3, 1)),
        reference_mask=[True, True, False, False],
        reference_donor_ids=["d1", "d1", "d2", "d2"], config=_config(),
    )
    assert fit.excluded_gene_ids == ("g1", "g2")
    assert fit.gene_names == ("g3",)


def test_reference_support_and_nonfinite_expression_fail_explicitly():
    data = _adata(np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=float))
    with pytest.raises(ProjectionError) as error:
        fit_projection(
            data, gene_theta=np.ones((3, 1)), reference_mask=[True, False, False],
            reference_donor_ids=["d1", "d1", "d2"], config=_config(),
        )
    assert error.value.status is ProjectionFailureStatus.INSUFFICIENT_REFERENCE

    data.X[0, 0] = np.nan
    with pytest.raises(ProjectionError, match="non-finite"):
        fit_projection(
            data, gene_theta=np.ones((3, 1)), reference_mask=[True, True, False],
            reference_donor_ids=["d1", "d1", "d2"], config=_config(),
        )


def test_sparse_layer_source_and_gene_order_are_preserved():
    values = np.array([[1, 2, 4], [2, 5, 8], [3, 4, 10]], dtype=float)
    data = _adata(np.zeros_like(values))
    data.layers["counts"] = sparse.csr_matrix(values)
    config = _config(source="layer:counts")
    extracted, genes = extract_expression(data, config=config)
    np.testing.assert_array_equal(extracted, values)
    assert genes == ("g1", "g2", "g3")


def test_frozen_transform_rejects_configuration_or_gene_order_changes():
    values = np.array([[1, 2, 4], [2, 5, 8], [3, 4, 10]], dtype=float)
    data = _adata(values)
    config = _config()
    fit = fit_projection(
        data, gene_theta=np.eye(3, 2), reference_mask=[True, True, False],
        reference_donor_ids=["d1", "d1", "d2"], config=config,
    )
    with pytest.raises(ProjectionError) as error:
        transform_projection(fit, data, config=_config(transform="log1p"))
    assert error.value.status is ProjectionFailureStatus.CONFIG_MISMATCH

    data.var_names = ["g2", "g1", "g3"]
    with pytest.raises(ProjectionError, match="gene names/order"):
        transform_projection(fit, data, config=config)


def test_batch_reference_is_rejected_until_a_method_is_specified():
    with pytest.raises(ValueError, match="batch-aware reference"):
        _config(batch_key="Batch")


def test_configured_upstream_transforms_are_explicit():
    raw = np.array([[1.0, 3.0], [2.0, 2.0]])
    log_config = IDCConfig.from_mapping(
        {"data": {"expression_source": "X", "upstream_transform": "log1p"}}
    )
    transformed, _ = extract_expression(
        ad.AnnData(X=raw, var={"gene": ["a", "b"]}), config=log_config
    )
    np.testing.assert_allclose(transformed, np.log1p(raw))

    norm_config = IDCConfig.from_mapping(
        {
            "data": {
                "expression_source": "X",
                "upstream_transform": "normalize_total_log1p",
                "upstream_target_sum": 10.0,
            }
        }
    )
    normalized, _ = extract_expression(
        ad.AnnData(X=raw, var={"gene": ["a", "b"]}), config=norm_config
    )
    expected = np.log1p(raw * (10.0 / raw.sum(axis=1))[:, None])
    np.testing.assert_allclose(normalized, expected)


def test_cell_score_uses_cell_level_mean_sample_sd_and_freezes_for_transform():
    theta = np.array([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]])
    config = _config()
    state = fit_cell_score_normalization(theta, config=config)
    np.testing.assert_allclose(state.mean, [3.0, 2.0])
    np.testing.assert_allclose(state.sd, [2.0, 0.0])
    np.testing.assert_array_equal(state.degenerate_components, [False, True])
    scores = apply_cell_score_normalization(theta, state, config=config).cell_score
    np.testing.assert_allclose(scores, [[-1, 0], [0, 0], [1, 0]])

    # These held-out values are transformed by the training-fitted state;
    # neither mean nor SD is recomputed from them.
    held_out = apply_cell_score_normalization(np.array([[7.0, 99.0]]), state, config=config)
    np.testing.assert_allclose(held_out.cell_score, [[2.0, 0.0]])
    assert held_out.normalization is state


def test_one_cell_cell_score_marks_undefined_sample_sd_as_degenerate():
    state = fit_cell_score_normalization(np.array([[2.0, -3.0]]), config=_config())
    assert np.all(state.degenerate_components)
    scores = apply_cell_score_normalization(np.array([[5.0, 10.0]]), state, config=_config())
    np.testing.assert_array_equal(scores.cell_score, [[0.0, 0.0]])
