import pytest

from idc.config import ConfigurationError, IDCConfig


def test_publication_defaults_are_explicit_and_resolved():
    config = IDCConfig()
    assert config.method.association == "covariance"
    assert config.method.n_components == 2
    assert config.pu.quantile_init == 0.5
    assert config.pu.membership_threshold == 0.5
    assert config.pu.categorical_threshold == 0.5
    assert config.pu.max_iter == 5
    assert config.pu.final_refit is True
    assert config.pu.logistic.max_iter == 2000
    assert config.projection.cell_score_ddof == 1
    assert config.projection.zero_reference_sd_policy == "exclude"
    assert config.compute.backend == "cpu"
    assert config.compute.dtype == "float64"


def test_yaml_round_trip_reproduces_resolved_configuration():
    config = IDCConfig.from_yaml(IDCConfig().to_yaml())
    assert config == IDCConfig()
    assert config.to_dict()["method"] == {
        "association": "covariance",
        "n_components": 2,
        "minimum_cells_per_sample": 2,
        "gene_filter": None,
    }


def test_compute_backend_configuration_round_trips():
    cpu = IDCConfig.from_mapping({"compute": {"backend": "cpu"}})
    assert IDCConfig.from_yaml(cpu.to_yaml()) == cpu
    gpu = IDCConfig.from_mapping(
        {"compute": {"backend": "gpu", "device": "cuda", "chunk_size": 2048}}
    )
    assert gpu.compute.backend == "gpu"
    assert gpu.compute.device == "cuda"
    assert gpu.compute.chunk_size == 2048
    assert IDCConfig.from_yaml(gpu.to_yaml()) == gpu


@pytest.mark.parametrize(
    "value, message",
    [
        ({"compute": {"backend": "tpu"}}, "compute.backend"),
        ({"compute": {"backend": "gpu"}}, "compute.device"),
        ({"compute": {"backend": "gpu", "device": "cpu"}}, "compute.device"),
        ({"compute": {"backend": "gpu", "device": "cuda:not-an-index"}}, "compute.device"),
        ({"compute": {"backend": "gpu", "device": "cuda:-1"}}, "compute.device"),
        ({"compute": {"dtype": "float32"}}, "float64"),
        ({"technical": {"dtype": "float32"}}, "not implemented or validated"),
        ({"compute": {"chunk_size": 0}}, "chunk_size"),
    ],
)
def test_invalid_compute_configuration_fails(value, message):
    with pytest.raises(ConfigurationError, match=message):
        IDCConfig.from_mapping(value)


@pytest.mark.parametrize("association", ["corr", "pearson", "", None])
def test_unknown_association_is_rejected(association):
    with pytest.raises(ConfigurationError, match="method.association"):
        IDCConfig.from_mapping({"method": {"association": association}})


@pytest.mark.parametrize(
    "mapping, message",
    [
        ({"method": {"n_components": 0}}, "n_components"),
        ({"pu": {"quantile_init": 1.0}}, "quantile_init"),
        ({"pu": {"membership_threshold": -0.1}}, "membership_threshold"),
        ({"pu": {"membership_update": "quantile"}}, "membership_quantile"),
        ({"pu": {"logistic": {"C": 0.0}}}, "logistic.C"),
        ({"pu": {"covariance": {"ridge_trace_scaled": float("nan")}}}, "ridge"),
        ({"projection": {"cell_score_ddof": -1}}, "cell_score_ddof"),
        ({"projection": {"zero_reference_sd_policy": "impute"}}, "zero_reference_sd_policy"),
        (
            {"projection": {"cell_score_normalization": "donor_balanced"}},
            "future scientific specification",
        ),
    ],
)
def test_invalid_ranges_and_combinations_are_rejected(mapping, message):
    with pytest.raises(ConfigurationError, match=message):
        IDCConfig.from_mapping(mapping)


def test_unknown_fields_are_rejected():
    with pytest.raises(ConfigurationError, match="unknown config.method fields"):
        IDCConfig.from_mapping({"method": {"assoc": "covariance"}})


def test_publication_values_and_alternatives_are_yaml_inputs():
    config = IDCConfig.from_mapping(
        {
            "method": {"association": "correlation", "n_components": 1},
            "pu": {"membership_update": "quantile", "membership_quantile": 0.5},
            "gene_score": {"outputs": ["signed", "component_magnitude"]},
        }
    )
    assert config.method.association == "correlation"
    assert config.method.n_components == 1
    assert config.pu.membership_quantile == 0.5
    assert config.gene_score.outputs == ("signed", "component_magnitude")
