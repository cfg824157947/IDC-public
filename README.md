# IDC — Identification of Disease-associated Cells

IDC identifies disease-associated cellular states in case–control single-cell data
using donor-balanced within-donor covariance contrasts and positive–unlabeled
(PU) learning. It provides cell scores, donor burden summaries, unsigned gene
contribution scores, and a separate within-donor differential-expression test.

## Install

Requires Python 3.10 or newer. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python examples/quickstart.py
python -m pytest
```

The distribution name is `idc-singlecell`; import it as `idc`. This is a local
source installation, not a claim of availability on PyPI. CPU execution is the
quickstart default. GPU execution additionally requires a suitable PyTorch/CUDA
installation; CUDA dependencies are not needed for the CPU example.

## Inputs and workflow

Use an AnnData object with cells as rows and genes as columns. Supply donor IDs
and case/control labels in `.obs`, and unique gene names in `.var_names`.
The example generates synthetic counts; no patient data or downloads are needed.

1. `fit_idc`: fit the two-dimensional covariance-contrast representation.
2. `transform_cells`: project cells through the fixed healthy reference and basis.
3. `fit_pu` / `predict_pu`: train the PU classifier and obtain scores in [0, 1].
4. Average cell scores within donors to summarize donor burden.
5. `score_genes(...).component_magnitude_gene_score`: unsigned disease-state
   gene contribution score (formerly HADS). It is not differential expression.
6. `idc.within_donor_dl.within_donor_test`: final manuscript within-donor NB/DL
   test. See [the input contract and held-out example](docs/within_donor_test.md).

PU scores are not calibrated disease probabilities. The historical internal field
`disease_like_probability` stores the score. A 0.5 cutoff has operational uses,
including eligibility checks; it is not a calibrated diagnostic threshold.

The quickstart is a full-data demonstration. For evaluation, split by donor and
fit preprocessing, IDC, and PU on training donors only. Predict held-out donors
without refitting; do not split cells from one donor across train and test.
The manuscript used five donor-disjoint simulation folds and real-data LODO.

## Scope and reproducibility

Core fitting/scoring modules are copied unchanged from the manuscript code.
The final DL function is extracted verbatim with portable imports; its saved
result convention and eligibility rules are retained. Older methods remain in
support modules for dependency compatibility and are not the recommended test.
See [reproduction notes](docs/reproduction.md) and `provenance/sources.json`.

No restricted datasets, learned donor models, manuscript files, credentials,
large results, or third-party comparator source code are included. Real datasets
must be obtained through their providers and applicable access agreements.

## Citation

Fengge Chang and Zhana Duren. *Contrasting within-donor gene covariation identifies
disease-associated cells and genes in case–control single-cell studies.*
Manuscript, 2026. No publication acceptance or DOI is asserted.

## License and release status

MIT License, copyright (c) 2026 Fengge Chang and IDC contributors. See [LICENSE](LICENSE).
Dependencies retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Version: v0.1.0. Intended repository: https://github.com/cfg824157947/IDC
This local release has not been pushed; the URL above is the author-selected destination.
