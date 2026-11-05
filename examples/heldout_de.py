"""Small synthetic LODO/LGO example with the unchanged final DL test."""
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from quickstart import synthetic_data, configuration, fit
from idc import compact_state_from_fits, predict_compact_state
from idc.within_donor_nb import LeaveGeneOutScores
from idc.within_donor_dl import within_donor_test


def main():
    out = Path('example_output/heldout_de'); out.mkdir(parents=True, exist_ok=False)
    data, config = synthetic_data(), configuration()
    disease = data[data.obs.condition == 'Disease'].copy()
    scores = np.empty(disease.n_obs); lg = {}
    with threadpool_limits(limits=1):
        for donor in sorted(disease.obs.donor.unique()):
            train = data[data.obs.donor != donor].copy()
            held = disease[disease.obs.donor == donor].copy()
            representation, classifier, _ = fit(train, config)
            state = compact_state_from_fits(representation, classifier)
            prediction, _, _ = predict_compact_state(state, held, config=config)
            mask = (disease.obs.donor == donor).to_numpy()
            scores[mask] = prediction.disease_like_probability
            lg[donor] = LeaveGeneOutScores.from_compact_state(state, held.X,
                            held.var_names, disease.var_names)
            np.testing.assert_allclose(lg[donor].full, scores[mask], atol=1e-12)
        results, slopes, ranges, info = within_donor_test(disease.X,
            disease.obs.donor.to_numpy(), scores, disease.var_names, lg)
    tested = results.tested & results.DL_p.notna()
    assert tested.any() and (results.loc[tested, 'n_donors_fit'] >= 2).all()
    results.to_csv(out/'genes.tsv', sep='\t')
    slopes.to_csv(out/'donor_slopes_log2.tsv', sep='\t')
    ranges.to_csv(out/'score_ranges.tsv', sep='\t')
    print(f'Synthetic held-out DL check: {info["n_donors"]} eligible donors, {tested.sum()} tested genes; output {out}')


if __name__ == '__main__':
    main()
