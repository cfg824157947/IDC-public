import numpy as np
import pytest

from idc import (
    compact_state_from_fits,
    fit_idc,
    fit_pu,
    load_compact_state,
    predict_compact_state,
    predict_pu,
    save_compact_state,
    transform_cells,
)
from helpers import tiny_adata, tiny_config


def test_compact_prediction_roundtrip_is_exact_and_omits_training_intermediates(tmp_path):
    data = tiny_adata()
    config = tiny_config()
    idc_fit = fit_idc(data, sample_key="SampleID", condition_key="Condition", config=config)
    scores = transform_cells(idc_fit, data, config=config)
    pu_fit = fit_pu(scores, sample_key="SampleID", condition_key="Condition", config=config)
    expected = predict_pu(pu_fit, scores).disease_like_probability
    state = compact_state_from_fits(idc_fit, pu_fit)

    prefix = tmp_path / "prediction_state"
    saved = save_compact_state(state, prefix)
    loaded = load_compact_state(prefix)
    actual, _, _ = predict_compact_state(loaded, data, config=config)

    np.testing.assert_array_equal(actual.disease_like_probability, expected)
    assert saved["total_bytes"] < 100_000
    assert loaded.projection_fit.gene_names == idc_fit.gene_names
    assert loaded.projection_fit.input_gene_names == idc_fit.projection_fit.input_gene_names
    with pytest.raises(FileExistsError):
        save_compact_state(loaded, prefix)


def test_compact_prediction_rejects_changed_config(tmp_path):
    data = tiny_adata()
    config = tiny_config()
    idc_fit = fit_idc(data, sample_key="SampleID", condition_key="Condition", config=config)
    scores = transform_cells(idc_fit, data, config=config)
    pu_fit = fit_pu(scores, sample_key="SampleID", condition_key="Condition", config=config)
    prefix = tmp_path / "prediction_state"
    save_compact_state(compact_state_from_fits(idc_fit, pu_fit), prefix)
    loaded = load_compact_state(prefix)
    changed = type(config).from_mapping({**config.to_dict(), "compute": {**config.to_dict()["compute"], "chunk_size": 17}})
    with pytest.raises(ValueError, match="config differs"):
        predict_compact_state(loaded, data, config=changed)
