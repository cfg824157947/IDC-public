# Manuscript implementation and reproduction

The README uses the final presentation name Identification of Disease-associated
Cells. An older package docstring says Incremental Disease Characterization;
that historical string is preserved to keep the imported core byte-identical.

`provenance/sources.json` records original locations and hashes. Those paths are
provenance only; the installable core does not depend on cluster directories.
`reproduction/original/` retains the original final real-data DL runner and its
class selector for inspection, not as a portable command: they depend on the
private analysis workspace. Use `idc.within_donor_dl` for portable execution.

Scientific source families:
- IDC covariance/projection/PU/scoring: IDC_publish/src/idc.
- NB dispersion/LGO helpers: IDC/publication_snapshot/src/idc/within_donor_nb.py.
- Final donor-specific regression and DL pooling: fig45_gfv2_20261031_v1/scripts/dl_run.py.
- Real-data fitting gene exclusions: reproduction/gene_filter.py (MT-, cytosolic
  RPL/RPS exclusions, preserving nuclear respiratory-chain and MRPL/MRPS genes).

Fitting genes and DE testing genes are distinct. For the paper, fitting used
GENCODE v44 protein-coding genes with mean training-donor pseudobulk CPM > 1,
then applied the exclusion rules. This preparation is not automatically performed
by fit_idc; provide the intended fitting-gene matrix explicitly. DE uses raw counts
and its separate gene universe, not only the fitting genes.

T2D: RNA component of GSE200044, Wang et al. (2023). AD: Liu et al. (2025)
ROSMAP multi-region data, subject to provider access conditions. No patient data
are redistributed. Pathway files and comparator tools are not bundled.

This package provides the core method and a runnable synthetic example; it does
not claim one-command recreation of all manuscript figures. Full experiment
manifests, comparator dependencies, data downloads, and plotting campaigns are
separate from this software release candidate. Paper-specific fitting and test
settings must be matched when reproducing published numerical results.
