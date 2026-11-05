import numpy as np
import pandas as pd
import anndata as ad

from idc.config import IDCConfig


def tiny_adata(seed=41):
    rng = np.random.default_rng(seed)
    rows = []
    obs = []
    for condition, prefix in (("Healthy", "H"), ("Disease", "D")):
        for donor_idx in range(4):
            donor = f"{prefix}{donor_idx}"
            n = 12
            groups = np.full(n, "Group1", dtype=object)
            if condition == "Disease":
                fraction = [0.25, 0.4, 0.6, 0.75][donor_idx]
                n_g2 = round(n * fraction)
                groups[-n_g2:] = "Group2"
            x = rng.normal(size=(n, 5))
            if condition == "Disease":
                x[groups == "Group2", 0] += 2.0
                x[groups == "Group2", 1] += 0.8
            rows.append(x)
            obs.extend(
                {
                    "SampleID": donor,
                    "Condition": condition,
                    "Group": groups[i],
                    "cell_type": "target",
                }
                for i in range(n)
            )
    matrix = np.vstack(rows)
    metadata = pd.DataFrame(obs, index=[f"cell{i}" for i in range(len(obs))])
    return ad.AnnData(
        X=matrix,
        obs=metadata,
        var=pd.DataFrame(index=[f"g{i}" for i in range(matrix.shape[1])]),
    )


def tiny_config():
    return IDCConfig.from_mapping(
        {
            "method": {"n_components": 2},
            "data": {
                "expression_source": "X",
                "upstream_transform": "identity",
                "control_label": "Healthy",
                "case_label": "Disease",
                "donor_key": "SampleID",
                "cell_type_key": "cell_type",
                "target_cell_type": "target",
            },
            "pu": {
                "membership_threshold": 0.0,
                "support": {
                    "minimum_cells_per_donor": 3,
                    "minimum_donors_per_class": 2,
                    "minimum_cells_per_class": 4,
                    "minimum_positives_per_case_donor": 1,
                },
            },
        }
    )
