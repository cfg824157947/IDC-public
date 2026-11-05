from types import SimpleNamespace

import numpy as np
import pytest

from idc.config import IDCConfig
from idc.gene_score import GeneScoreError, fit_legacy_gene_direction, score_genes
from idc.projection import CellScoreNormalization
from idc.pu import PUFit, PUFitStatus


def _fit(theta, score_sd, pu_sd, beta, *, degenerate=None):
    theta = np.asarray(theta, dtype=float)
    k = theta.shape[1]
    if degenerate is None:
        degenerate = np.zeros(k, dtype=bool)
    idc_fit = SimpleNamespace(
        gene_theta=np.asarray(theta, dtype=float),
        gene_names=tuple(f"g{i}" for i in range(theta.shape[0])),
        cell_score_normalization=CellScoreNormalization(
            mean=np.zeros(k),
            sd=np.asarray(score_sd, dtype=float),
            degenerate_components=np.asarray(degenerate, dtype=bool),
            ddof=1,
            fitted_cell_count=10,
            normalization="cell_weighted",
        ),
    )
    pu_fit = SimpleNamespace(
        status=PUFitStatus.CONVERGED,
        coefficient=np.asarray(beta, dtype=float),
        scaling_sd=np.asarray(pu_sd, dtype=float),
    )
    return idc_fit, pu_fit


def test_three_new_scores_match_manual_formula_and_back_transform():
    theta = np.array([[1.0, -1.0], [0.5, 2.0]])
    idc_fit, pu_fit = _fit(theta, [2.0, 1.0], [2.0, 0.5], [8.0, 0.5])

    result = score_genes(idc_fit, pu_fit, config=IDCConfig())

    # u = beta / (sd_PU * sd_score) = (2, 1)
    np.testing.assert_allclose(result.cell_theta_direction, [2.0, 1.0])
    np.testing.assert_allclose(result.signed_gene_score, [1.0, 3.0])
    np.testing.assert_allclose(result.absolute_signed_gene_score, [1.0, 3.0])
    np.testing.assert_allclose(result.component_magnitude_gene_score, [3.0, 3.0])
    assert result.legacy_absolute_gene_score is None


def test_degenerate_embedding_component_maps_to_zero_and_is_recorded():
    theta = np.array([[3.0, 4.0]])
    idc_fit, pu_fit = _fit(
        theta, [0.0, 2.0], [1.0, 2.0], [99.0, 4.0], degenerate=[True, False]
    )

    result = score_genes(idc_fit, pu_fit, config=IDCConfig())

    np.testing.assert_allclose(result.cell_theta_direction, [0.0, 1.0])
    np.testing.assert_allclose(result.signed_gene_score, [4.0])
    assert result.degenerate_components == (0,)


def test_gene_order_and_selective_output_configuration_are_preserved():
    idc_fit, pu_fit = _fit([[1.0, 0.0], [0.0, -1.0]], [1, 1], [1, 1], [2, 3])
    config = IDCConfig.from_mapping(
        {"gene_score": {"outputs": ["signed", "component_magnitude"]}}
    )

    result = score_genes(idc_fit, pu_fit, config=config)

    assert result.gene_names == ("g0", "g1")
    assert result.signed_gene_score is not None
    assert result.absolute_signed_gene_score is None
    assert result.component_magnitude_gene_score is not None


def test_sign_flip_and_component_permutation_leave_gene_scores_invariant():
    theta = np.array([[1.0, -1.0], [0.25, 2.0]])
    idc_fit, pu_fit = _fit(theta, [2.0, 4.0], [0.5, 2.0], [3.0, 8.0])
    baseline = score_genes(idc_fit, pu_fit, config=IDCConfig())

    signs = np.array([-1.0, 1.0])
    sign_fit, sign_pu = _fit(
        theta * signs, np.array([2.0, 4.0]), [0.5, 2.0], np.array([3.0, 8.0]) * signs
    )
    sign_result = score_genes(sign_fit, sign_pu, config=IDCConfig())
    np.testing.assert_allclose(sign_result.signed_gene_score, baseline.signed_gene_score)
    np.testing.assert_allclose(
        sign_result.component_magnitude_gene_score,
        baseline.component_magnitude_gene_score,
    )

    order = np.array([1, 0])
    perm_fit, perm_pu = _fit(
        theta[:, order], np.array([2.0, 4.0])[order], np.array([0.5, 2.0])[order],
        np.array([3.0, 8.0])[order],
    )
    perm_result = score_genes(perm_fit, perm_pu, config=IDCConfig())
    np.testing.assert_allclose(perm_result.signed_gene_score, baseline.signed_gene_score)
    np.testing.assert_allclose(
        perm_result.component_magnitude_gene_score,
        baseline.component_magnitude_gene_score,
    )


def test_legacy_score_is_separately_enabled_and_uses_historical_direction():
    idc_fit, pu_fit = _fit([[1.0, -1.0]], [1, 1], [1, 1], [10, 10])
    config = IDCConfig.from_mapping(
        {
            "gene_score": {
                "outputs": ["component_magnitude", "legacy_absolute"],
                "legacy_reproduction": True,
            }
        }
    )

    result = score_genes(
        idc_fit, pu_fit, config=config, legacy_beta_historical=[2.0, 1.0]
    )

    np.testing.assert_allclose(result.component_magnitude_gene_score, [20.0])
    np.testing.assert_allclose(result.legacy_absolute_gene_score, [3.0])


def test_legacy_score_requires_historical_classifier_direction():
    idc_fit, pu_fit = _fit([[1.0, 1.0]], [1, 1], [1, 1], [1, 1])
    config = IDCConfig.from_mapping(
        {
            "gene_score": {
                "outputs": ["legacy_absolute"],
                "legacy_reproduction": True,
            }
        }
    )
    with pytest.raises(GeneScoreError, match="legacy_beta_historical"):
        score_genes(idc_fit, pu_fit, config=config)


def test_legacy_direction_reproduces_historical_second_classifier_transform():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.array([
        [-2.0, 0.2], [-1.0, 0.1], [-0.5, 1.1], [0.0, -0.3],
        [0.5, 0.4], [1.0, 0.8], [1.5, -0.2], [2.0, 0.6],
    ])
    p = np.array([0.1, 0.2, 0.4, 0.49, 0.51, 0.7, 0.8, 0.95])
    actual = fit_legacy_gene_direction(x, p, threshold=0.5, max_iter=1000)
    y = (p >= 0.5).astype(int)
    manual = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    manual.fit(x, y)
    scaler = manual.named_steps["standardscaler"]
    logistic = manual.named_steps["logisticregression"]
    np.testing.assert_allclose(actual.beta_historical, logistic.coef_[0] / scaler.scale_)
    np.testing.assert_allclose(actual.scaler_mean, scaler.mean_)
    np.testing.assert_allclose(actual.scaler_scale, scaler.scale_)
    assert actual.n_high_probability == 4
    assert actual.n_low_probability == 4


def test_failed_pu_fit_cannot_produce_gene_scores():
    idc_fit, pu_fit = _fit([[1.0]], [1], [1], [1])
    pu_fit = SimpleNamespace(
        status=PUFitStatus.FAILED, coefficient=None, scaling_sd=None
    )
    with pytest.raises(GeneScoreError, match="FAILED"):
        score_genes(idc_fit, pu_fit, config=IDCConfig())
