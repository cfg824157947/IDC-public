"""Small synthetic full-data example; not a manuscript benchmark."""
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from threadpoolctl import threadpool_limits
from idc import IDCConfig, fit_idc, transform_cells, fit_pu, predict_pu, score_genes


def synthetic_data(seed=17):
    rng = np.random.default_rng(seed)
    blocks, metadata = [], []
    for condition, prefix in [('Healthy', 'H'), ('Disease', 'D')]:
        for d in range(4):
            n = 120
            affected = np.zeros(n, dtype=bool)
            if condition == 'Disease':
                affected[rng.choice(n, 60, replace=False)] = True
            mu = np.full((n, 30), 8.0)
            mu[affected, :8] *= 5.0
            mu *= rng.lognormal(0, .1, (1, 30))
            blocks.append(rng.poisson(mu))
            metadata.extend([{'donor': f'{prefix}{d}', 'condition': condition} for _ in range(n)])
    obs = pd.DataFrame(metadata, index=[f'cell{i}' for i in range(len(metadata))])
    return ad.AnnData(sparse.csr_matrix(np.vstack(blocks)), obs=obs,
                      var=pd.DataFrame(index=[f'gene{i}' for i in range(30)]))


def configuration():
    return IDCConfig.from_mapping({'data': {'expression_source': 'X',
        'upstream_transform': 'identity', 'control_label': 'Healthy',
        'case_label': 'Disease', 'donor_key': 'donor'}})


def fit(data, config):
    representation = fit_idc(data, sample_key='donor', condition_key='condition', config=config)
    embedding = transform_cells(representation, data, config=config)
    classifier = fit_pu(embedding, sample_key='donor', condition_key='condition', config=config)
    probabilities = predict_pu(classifier, embedding).disease_like_probability
    if probabilities is None:
        raise RuntimeError(f'PU fitting failed: {classifier.status}')
    return representation, classifier, probabilities


def main():
    out = Path('example_output/quickstart')
    out.mkdir(parents=True, exist_ok=False)
    data, config = synthetic_data(), configuration()
    with threadpool_limits(limits=1):
        representation, classifier, scores = fit(data, config)
        ranking = score_genes(representation, classifier, config=config)
    cells = data.obs.copy(); cells['score'] = scores
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    cells.to_csv(out/'cell_scores.tsv', sep='\t')
    cells.groupby(['donor', 'condition']).score.mean().to_csv(out/'donor_burden.tsv', sep='\t')
    pd.DataFrame({'gene': ranking.gene_names, 'gene_contribution_score':
        ranking.component_magnitude_gene_score}).sort_values('gene_contribution_score', ascending=False).to_csv(out/'gene_contribution.tsv', sep='\t', index=False)
    print(f'Completed synthetic full-data example: {len(scores)} cells; output {out}')


if __name__ == '__main__':
    main()
