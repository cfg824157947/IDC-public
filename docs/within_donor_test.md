# Final within-donor differential-expression test

```python
from idc.within_donor_nb import LeaveGeneOutScores
from idc.within_donor_dl import within_donor_test

# C: CSR raw integer counts, disease cells x test genes
# donor: donor IDs aligned to C; score: original held-out cell scores
# genes: test-gene identifiers in C column order
# lg[d]: LeaveGeneOutScores object for donor d, in that donor's C row order
results, donor_slopes, score_ranges, info = within_donor_test(C, donor, score, genes, lg)
```

Construct each LGO object with `LeaveGeneOutScores.from_compact_state(state,
expression, expression_genes, test_genes)`, where state is the frozen model for
that held-out donor and expression is its raw-count projection input, including
all fitting genes. The helper supports identity upstream transformation only;
it raises for other transforms. Do not substitute log counts. Both the original
score and LGO objects must come from the same frozen model and match cell order.

Eligible disease donors have at least 20 cells strictly above and 20 strictly
below score 0.5. Six approximately equal rank bins are formed per donor and gene.
Median-of-ratios size factors are calculated across original-score donor x bin
profiles and reused by donor/bin ordinal for LGO bins, not re-estimated per gene.
The auxiliary pooled donor-intercept/common-slope model estimates shared gene
NB dispersion, with Cox–Reid adjustment and empirical-Bayes shrinkage.

Each donor slope uses that donor's six LGO pseudobulks with fixed dispersion and
size-factor offsets. Gene eligibility additionally requires at least two fitted
donors. DerSimonian–Laird random-effects pooling gives the final Wald P values;
BH is applied only to eligible finite-P genes. Use DL_p and DL_BH, not the
auxiliary v2_BH column. No count-splitting or old per-cell test is invoked here.

Outputs preserve the published convention: DL_log2 = pooled natural-log slope /
ln(2); delta_log2 = DL_log2 times the mean across donors of each donor's median
across jointly tested genes of the bin-6 minus bin-1 score separation. The shared
factor is returned as info['delta_factor']. It is not a gene-specific log fold
change between two externally defined diagnostic groups.

Run `python examples/heldout_de.py` for a small synthetic held-out workflow.
Synthetic outputs are software checks, not manuscript performance estimates.
