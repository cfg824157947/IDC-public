"""IDC-guided within-sample differential expression.

Compares disease-like and healthy-like cells *within* each donor, where the cell state comes from held-out (LODO) IDC+PU scores
(state = score > ``threshold``; cells exactly at the threshold are excluded). Two modes:

* ``"per_donor_nb"``: in every donor a cell-level negative binomial GLM (R glmGamPoi; ``~ state`` with per-cell size factors as a log
  offset, by default from total counts; quasi-likelihood F test; BH within donor), then a random-effects meta-analysis across donors
  (fixed-effect inverse-variance Cochran's Q, I^2 = max(0, (Q - df) / Q), DerSimonian-Laird tau^2, normal-theory random-effects z and P,
  BH across genes). Genes are *consistent* if the random-effects BH < ``alpha`` and the Q BH >= ``alpha``, *heterogeneous* if the Q BH
  < ``alpha``.
* ``"pseudobulk_deseq2"``: paired pseudobulk DESeq2 (``~ donor + state``) on summed counts per donor x state.
* ``"nb_python"``: Python-only per-donor NB GLM (:mod:`idc.within_donor_nb`; no R): deconvolution size factors, continuous score
  (default) or threshold design, quality covariates (default on: log10 total counts, log10 detected genes, % mito, z-scored within
  donor), shrunken dispersions, Wald tests, then a random-effects meta-analysis over the donors where each gene was tested (default >=
  max(3, ceil(D / 3)) donors; per donor a gene needs >= 3 detecting cells, at least one on each side of the threshold).

R (with glmGamPoi and DESeq2) is called through ``subprocess``; give ``rscript`` or set ``IDC_RSCRIPT``. This reproduces the
manuscript Figure 4F/5F analyses exactly when ``se_source="pvalue"``; the default ``se_source="model"`` uses the Wald-type SE of the
fitted model instead of the SE back-calculated from the P value.

Size factors (``size_factor``), computed per donor on the genes with nonzero total: ``"normed_sum"`` (total counts; manuscript
default), ``"poscounts"`` and ``"ratio"`` (DESeq2-style median of ratios, glmGamPoi; ``"ratio"`` needs genes nonzero in every cell and
fails on sparse single-cell data), ``"deconvolution"`` (pooling and deconvolution, scran ``calculateSumFactors`` or the identical
``scuttle::pooledSizeFactors``). Compositional caveat: with total-count size factors, a large shift of a few highly expressed genes
between states changes every cell's total and therefore appears as a shift of the opposite sign in all other genes. Ratio-type and
deconvolution size factors assume that most genes are not DE between the two states and are more robust to such shifts, but none of the
options can recover absolute changes: all log2FC are relative to the normalization, and the direction balance of the significant genes
should be read with this in mind.

Count splitting: ``split_counts`` thins counts into independent halves A and B (binomial, p = 0.5, fixed seed). With
``count_split=True`` the tests run on half B; the caller must then supply scores computed from half A (this function does not
recompute IDC scores).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from scipy import special, stats

R_DIR = Path(__file__).resolve().parent / "R"
NB_SCRIPT = R_DIR / "per_donor_nb.R"
DESEQ2_SCRIPT = R_DIR / "run_pseudobulk_deg.R"
DESEQ2_SETTINGS = {"test": "Wald", "fit_type": "parametric", "size_factor_type": "ratio", "beta_prior": False,
                   "adjusted_pvalue_alpha": 0.05, "independent_filtering": True, "cooks_distance_filtering": True}
CHI2_MEDIAN = 0.4549364231195724  # median of chi-square with 1 df
SIZE_FACTORS = ("normed_sum", "poscounts", "ratio", "deconvolution")


class WithinSampleDEError(RuntimeError):
    """Raised for invalid inputs or a failed R call."""


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted P values (NaN-preserving)."""
    p = np.asarray(p, float); out = np.full(p.shape, np.nan); ok = np.isfinite(p)
    q = p[ok]; n = len(q)
    if n:
        o = np.argsort(q); adj = np.minimum.accumulate((q[o] * n / np.arange(1, n + 1))[::-1])[::-1]
        r = np.empty(n); r[o] = np.minimum(adj, 1); out[ok] = r
    return out


def genomic_inflation(p: np.ndarray) -> float:
    """Genomic inflation factor lambda = median(chi2_1 quantile of P) / 0.4549 over finite P values."""
    p = np.asarray(p, float); p = p[np.isfinite(p)]
    return float(np.median(stats.chi2.isf(np.clip(p, 1e-300, 1.0), 1)) / CHI2_MEDIAN) if len(p) else float("nan")


def split_counts(counts: Any, seed: int = 20261025, p: float = 0.5) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Binomial thinning of integer counts into independent halves (A, B) with a fixed seed."""
    C = sp.csr_matrix(counts)
    if np.any(C.data < 0) or np.any(C.data != np.floor(C.data)):  # validate before any integer cast
        raise WithinSampleDEError("counts must be non-negative integers")
    C = C.astype(np.int64); rng = np.random.default_rng(seed); A = C.copy(); A.data = rng.binomial(C.data, p); A.eliminate_zeros()
    B = (C - A).tocsr(); B.eliminate_zeros()
    return A, B


def _rscript(rscript: str | None) -> str:
    exe = rscript or os.environ.get("IDC_RSCRIPT") or shutil.which("Rscript")
    if not exe or not Path(exe).exists():
        raise WithinSampleDEError("Rscript not found; pass rscript= or set IDC_RSCRIPT (R with glmGamPoi and DESeq2)")
    return exe


def r_versions(rscript: str | None = None) -> dict[str, str]:
    """R, glmGamPoi, DESeq2 and scran/scuttle (deconvolution size factors) versions of the R used for the tests."""
    code = ("cat(R.version.string, as.character(packageVersion('glmGamPoi')), as.character(packageVersion('DESeq2')), "
            "if (requireNamespace('scran', quietly = TRUE)) paste('scran', packageVersion('scran')) else paste('scuttle', packageVersion('scuttle')), sep='\\n')")
    out = subprocess.run([_rscript(rscript), "--vanilla", "-e", code], capture_output=True, text=True, check=True).stdout.splitlines()
    return {"R": out[0], "glmGamPoi": out[1], "DESeq2": out[2], "deconvolution": out[3]}


def per_donor_nb(counts: Any, genes: Sequence[str], is_test: np.ndarray, *, size_factor: str = "normed_sum", rscript: str | None = None,
                 workdir: str | Path | None = None) -> pd.DataFrame:
    """Run the per-donor glmGamPoi test for one donor (cells x genes counts; ``is_test`` = disease-like).

    The per-cell size factors used (geometric mean 1) are returned as a tuple in ``result.attrs["size_factors"]``
    (:func:`within_sample_de` moves them to ``WithinSampleDEResult.size_factors``).
    """
    if size_factor not in SIZE_FACTORS:
        raise WithinSampleDEError(f"size_factor must be one of {SIZE_FACTORS}")
    C = sp.csr_matrix(counts).astype(np.int64); is_test = np.asarray(is_test, bool)
    with tempfile.TemporaryDirectory(dir=workdir) as d:
        d = Path(d)
        scipy.io.mmwrite(str(d / "counts.mtx"), C.T.tocoo())
        (d / "genes.txt").write_text("\n".join(map(str, genes)) + "\n")
        (d / "states.txt").write_text("\n".join(np.where(is_test, "test", "ref")) + "\n")
        r = subprocess.run([_rscript(rscript), "--vanilla", str(NB_SCRIPT), *(str(d / f) for f in ("counts.mtx", "genes.txt", "states.txt", "out.tsv")),
                            size_factor], capture_output=True, text=True)
        if r.returncode != 0:
            raise WithinSampleDEError(f"glmGamPoi failed (size_factor={size_factor!r}): {r.stderr[-2000:]}")
        res = pd.read_csv(d / "out.tsv", sep="\t", index_col=0)
        res.attrs["size_factors"] = tuple(np.loadtxt(d / "out.tsv.size_factors", ndmin=1).tolist())
        return res


def pseudobulk_deseq2(pb_counts: pd.DataFrame, meta: pd.DataFrame, *, rscript: str | None = None,
                      workdir: str | Path | None = None) -> pd.DataFrame:
    """Paired pseudobulk DESeq2 (``~ donor + state``; disease-like vs healthy-like) through the repository adapter."""
    req = {"method": "DESeq2", "normalization": "median_of_ratios", "method_filtering": "nonzero_total",
           "contrast": {"factor": "state", "numerator": "disease_like", "denominator": "healthy_like"},
           "design_formula": "~ donor + state", "settings": DESEQ2_SETTINGS}
    with tempfile.TemporaryDirectory(dir=workdir) as d:
        d = Path(d); pb_counts.to_csv(d / "counts.tsv", sep="\t"); meta.to_csv(d / "metadata.tsv", sep="\t")
        (d / "request.json").write_text(json.dumps(req))
        r = subprocess.run([_rscript(rscript), "--vanilla", str(DESEQ2_SCRIPT), *(str(d / f) for f in ("counts.tsv", "metadata.tsv", "request.json", "out.tsv"))],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise WithinSampleDEError(f"DESeq2 failed: {r.stderr[-2000:]}")
        return pd.read_csv(d / "out.tsv", sep="\t", index_col=0)


def _trigamma_inverse(x: float) -> float:
    """Inverse of trigamma (Newton iteration as in limma::trigammaInverse)."""
    if x > 1e7:
        return 1 / np.sqrt(x)
    if x < 1e-6:
        return 1 / x
    y = 0.5 + 1 / x
    for _ in range(50):
        tri = float(special.polygamma(1, y)); dif = tri * (1 - tri / x) / float(special.polygamma(2, y)); y += dif
        if abs(-dif / y) < 1e-8:
            break
    return y


def fit_f_dist(x: np.ndarray, df1: np.ndarray) -> tuple[float, float]:
    """Moment estimation of a scaled F prior (d0, s0^2) for variance-like statistics x with df1 (Smyth 2004; limma fitFDist)."""
    x = np.asarray(x, float); df1 = np.asarray(df1, float); ok = np.isfinite(x) & (df1 > 0)
    x, df1 = x[ok], df1[ok]
    if len(x) < 3:
        return np.inf, float(np.mean(x)) if len(x) else 1.0
    x = np.maximum(x, 1e-5 * np.median(x[x > 0]) if np.any(x > 0) else 1e-8)
    e = np.log(x) - special.digamma(df1 / 2) + np.log(df1 / 2); em = e.mean()
    ev = np.sum((e - em) ** 2) / (len(e) - 1) - np.mean(special.polygamma(1, df1 / 2))
    if ev > 0:
        d0 = 2 * _trigamma_inverse(ev); return float(d0), float(np.exp(em + special.digamma(d0 / 2) - np.log(d0 / 2)))
    return np.inf, float(np.exp(em))


def meta_analysis(per_donor: Mapping[str, pd.DataFrame], *, se_source: str = "model", alpha: float = 0.05,
                  min_donors: int | None = None, method: str = "DL") -> pd.DataFrame:
    """Random-effects meta-analysis of per-donor log2FC across donors.

    ``min_donors=None`` (default; R modes, manuscript): genes must be tested in every donor. With ``min_donors=k`` each gene is combined
    over the donors where it was tested (valid log2FC, P and SE), and genes with fewer than k such donors are dropped; Q has
    (donors - 1) degrees of freedom per gene and ``n_donors`` records the count.

    ``se_source="model"`` uses the per-donor model SE; ``"pvalue"`` back-calculates SE = |log2FC| / Phi^-1(1 - P/2) (the manuscript
    Figure 4F/5F convention; genes with P >= 0.999 in any donor are then excluded because their SE is undefined).
    Q uses fixed-effect inverse-variance weights; tau^2 is the DerSimonian-Laird estimator; RE P is two-sided normal.
    """
    donors = list(per_donor)
    L = pd.DataFrame({d: r.log2FC for d, r in per_donor.items()}); P = pd.DataFrame({d: r.p_value for d, r in per_donor.items()})
    valid = pd.DataFrame({d: r.tested.astype(bool) for d, r in per_donor.items()}) & L.notna() & P.notna()
    if se_source == "pvalue":
        valid &= P < 0.999
        SE = (L.abs() / stats.norm.isf(P.clip(lower=1e-300) / 2))
    elif se_source == "model":
        SE = pd.DataFrame({d: r.SE for d, r in per_donor.items()}); valid &= SE.notna() & (SE > 0)
    else:
        raise WithinSampleDEError("se_source must be 'model' or 'pvalue'")
    if method not in ("DL", "moderated"):
        raise WithinSampleDEError("method must be 'DL' or 'moderated'")
    k = valid.sum(axis=1)
    ok = valid.all(axis=1) if min_donors is None else k >= max(int(min_donors), 2)
    g = L.index[ok]; valid = valid.loc[g]; L, SE = L.loc[g].where(valid), SE.loc[g].where(valid)
    w = 1 / SE ** 2; df = (k.loc[g] - 1).astype(float)
    fe = (w * L).sum(1) / w.sum(1); Q = (w * L.sub(fe, axis=0) ** 2).sum(1); Qp = stats.chi2.sf(Q, df)
    tau2 = np.maximum(0, (Q - df) / (w.sum(1) - (w ** 2).sum(1) / w.sum(1)))
    wr = 1 / (SE ** 2 + tau2.to_numpy()[:, None]); re = (wr * L).sum(1) / wr.sum(1); se = 1 / np.sqrt(wr.sum(1)); z = re / se
    rp = 2 * stats.norm.sf(np.abs(z))
    extra = {}
    if method == "moderated":
        # Donor-level moderated test (design note Stage 3, option 3): squeeze u = Q / df (limma fitFDist), map to tau^2 (DL map),
        # RE weights 1 / (s^2 + tau^2), residual scale moderated toward 1 with the same prior df, t with df0 + df.
        u = Q / df; d0, s02 = fit_f_dist(u.to_numpy(), df.to_numpy())
        ut = s02 if np.isinf(d0) else (d0 * s02 + df * u) / (d0 + df)
        c = w.sum(1) - (w ** 2).sum(1) / w.sum(1); tau2m = np.maximum(0, df * (ut - 1) / c)
        wm = 1 / (SE ** 2 + tau2m.to_numpy()[:, None]); re = (wm * L).sum(1) / wm.sum(1)
        r = (wm * L.sub(re, axis=0) ** 2).sum(1) / df; rt = 1.0 if np.isinf(d0) else (d0 + df * r) / (d0 + df)
        se = np.sqrt(rt / wm.sum(1)); z = re / se; dfm = np.inf if np.isinf(d0) else d0 + df
        rp = 2 * (stats.norm.sf(np.abs(z)) if np.isinf(d0) else stats.t.sf(np.abs(z), dfm))
        extra = {"tau2_moderated": tau2m, "moderated_df": dfm if np.isinf(d0) else dfm, "prior_df": d0, "prior_scale": s02}
    t = pd.DataFrame({"RE_log2FC": re, "RE_SE": se, "RE_z": z, "RE_p": rp, "RE_BH": bh(rp), "Q": Q, "Q_p": Qp, "Q_BH": bh(Qp),
                      "I2": np.where(Q > 0, np.maximum(0, (Q - df) / Q), 0.0), "tau2": tau2, "n_donors": k.loc[g].astype(int), **extra}, index=g)
    t["class"] = np.where(t.Q_BH < alpha, "heterogeneous", np.where(t.RE_BH < alpha, "consistent", "neither"))
    return t.join(L.add_prefix("log2FC_")).join(SE.add_prefix("SE_"))


@dataclass
class WithinSampleDEResult:
    """Results of :func:`within_sample_de`."""
    per_donor: dict[str, pd.DataFrame]
    meta: pd.DataFrame | None
    pseudobulk: pd.DataFrame | None
    donor_summary: pd.DataFrame
    skipped_donors: pd.DataFrame
    diagnostics: dict[str, Any]
    permutation_p: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    size_factors: dict[str, np.ndarray] = field(default_factory=dict)

    def per_donor_table(self) -> pd.DataFrame:
        """Long table: gene x donor with log2FC, SE (model), P, BH."""
        return pd.concat([r.assign(donor=d) for d, r in self.per_donor.items()]).rename_axis("gene").reset_index()

    def qq_plot(self, ax=None, include_permutations: bool = True):
        """QQ plot of the meta-analysis random-effects P (and per-donor P), with permutation-null curves and a 95% band."""
        import matplotlib.pyplot as plt
        ax = ax or plt.subplots(figsize=(3.4, 3.2))[1]

        def curve(p, **kw):
            p = np.sort(np.clip(np.asarray(p, float)[np.isfinite(p)], 1e-300, 1)); n = len(p); i = np.arange(1, n + 1)
            ax.plot(-np.log10(i / (n + 1)), -np.log10(p), **kw)
            return n
        series = {"RE P": self.meta.RE_p.to_numpy()} if self.meta is not None else {"DESeq2 P": self.pseudobulk.p_value.to_numpy()}
        for name, p in series.items():
            n = curve(p, color="#7B3294", lw=1.4, label=f"{name} (λ = {genomic_inflation(p):.2f})")
            i = np.arange(1, n + 1); e = -np.log10(i / (n + 1))
            ax.fill_between(e, -np.log10(stats.beta.ppf(.975, i, n - i + 1)), -np.log10(stats.beta.ppf(.025, i, n - i + 1)), color="#DDDDDD", lw=0, zorder=0)
        if include_permutations:
            for k, d in self.permutation_p.items():
                p = d.get("RE_p", d.get("DESeq2_p"))
                if p is not None: curve(p, color="#AAAAAA", lw=.9, label=f"{k} (λ = {genomic_inflation(p):.2f})")
        lim = ax.get_xlim()[1]; ax.plot([0, lim], [0, lim], "k--", lw=.6)
        ax.set_xlabel("Expected −log10 P"); ax.set_ylabel("Observed −log10 P"); ax.legend(frameon=False, fontsize=7)
        return ax


def within_sample_de(counts: Any, donor: Sequence[Any], score: Sequence[float], *, genes: Sequence[str],
                     condition: Sequence[Any] | None = None, case_label: Any | None = None, donors: Sequence[Any] | None = None,
                     mode: str = "per_donor_nb", threshold: float = 0.5, min_cells_per_state: int = 20,
                     count_split: bool | str | None = None, split_seed: int = 20261025, score_source: str | None = None, se_source: str = "model", size_factor: str | None = None, alpha: float = 0.05,
                     n_permutations: int | None = None, permutation_seed: int = 20261026, rscript: str | None = None,
                     workdir: str | Path | None = None, score_model: str = "continuous", quality: Any = "auto",
                     min_cells_detected: int = 3, min_cells_detected_per_state: int = 3,
                     min_donors_meta: int | None = None, per_donor_level: str = "cells", n_bins: int = 4,
                     leave_gene_out: Mapping[str, Any] | None = None, meta_method: str = "DL") -> WithinSampleDEResult:
    """IDC-guided within-sample differential expression (disease-like vs healthy-like cells within donors).

    Parameters
    ----------
    counts : cells x genes raw integer counts (array or scipy.sparse).
    donor : per-cell donor IDs.
    score : per-cell held-out (LODO) IDC+PU disease-like probability; state = score > ``threshold`` (cells equal to it are dropped).
    genes : gene identifiers (columns of ``counts``).
    condition, case_label : optional per-cell condition labels; if both are given only donors whose cells are ``case_label`` are used.
    donors : optional explicit donor subset (applied after the condition filter).
    mode : ``"per_donor_nb"`` (glmGamPoi per donor + random-effects meta-analysis), ``"pseudobulk_deseq2"`` (paired pseudobulk DESeq2),
        ``"both"`` (R modes), or ``"nb_python"`` (Python-only per-donor NB; see :mod:`idc.within_donor_nb`).
    threshold : score threshold (default 0.5; PU scores are not calibrated, so the threshold is exploratory).
    min_cells_per_state : donors with fewer cells in either state are skipped and recorded (default 20).
    count_split, split_seed, score_source : count splitting. Default: ``True`` for ``nb_python`` (confirmatory results), ``False`` for
        the R modes. ``True`` thins the supplied counts (Binomial p = 0.5, ``split_seed``; :func:`split_counts`) and tests on half B;
        ``"presplit"`` means the supplied counts already are half B; ``False`` tests on the supplied counts (ranking mode; the state
        definition and the test then share the same counts). For ``nb_python`` with ``True`` / ``"presplit"`` the caller must pass
        ``score_source="half_A"`` to declare that the held-out scores were computed on half A of the same split; otherwise the call
        raises, because tests on half B with full-data scores are not confirmatory.
    se_source : ``"model"`` (default; Wald-type SE from the glmGamPoi fit) or ``"pvalue"`` (back-calculated; reproduces the earlier
        manuscript values).
    size_factor : per-cell size factors. ``nb_python``: ``"deconvolution"`` (default; Python pooled deconvolution) or ``"normed_sum"``.
        ``per_donor_nb``: ``"normed_sum"`` (default; total counts, manuscript), ``"poscounts"``,
        ``"ratio"`` (median of ratios; fails on sparse data) or ``"deconvolution"`` (scran/scuttle pooling). See the module docstring for
        the compositional caveat.
    alpha : FDR level for the consistent/heterogeneous classification.
    n_permutations, permutation_seed : within-donor label permutations (group sizes kept) rerunning the same tests, for
        diagnostics (lambda and QQ null curves).
    rscript : path to Rscript with glmGamPoi and DESeq2 (or set ``IDC_RSCRIPT``); not used by ``nb_python``.
    score_model : ``nb_python`` only: ``"continuous"`` (x = held-out score; effect = log2 change from score 0 to 1) or ``"threshold"``
        (x = 1(score > threshold); cells at the threshold dropped). Donors need ``min_cells_per_state`` cells on each side in both.
    quality : ``nb_python`` only. ``"auto"`` (default) computes log10 total counts, log10 detected genes and % mito ("MT-" genes) from
        the supplied counts; better, pass a DataFrame (one row per input cell) from :func:`idc.within_donor_nb.quality_covariates` on
        all genes, since the supplied matrix is usually the IDC-filtered gene set. ``None`` fits without covariates. Covariates are
        z-scored within donor; constant or undefined ones (e.g. % mito without MT genes) are dropped and recorded.
    min_cells_detected, min_cells_detected_per_state : ``nb_python`` only. A gene is tested in a donor if it is nonzero in
        >= ``min_cells_detected`` cells with at least one on each side of the threshold (continuous), or in >=
        ``min_cells_detected_per_state`` cells of each state (threshold); a non-finite SE also makes it untested in that donor.
    min_donors_meta : ``nb_python`` only. Each gene is meta-analysed over the donors where it was tested, if there are at least this many
        (default max(3, ceil(D / 3)), D = donors tested); ``n_donors`` per gene is reported.
    n_permutations defaults to 1 for ``nb_python`` (diagnostics always reported) and 0 otherwise.
    per_donor_level, n_bins : ``nb_python`` only. ``"cells"`` (default; cell-level NB) or ``"bins"`` (score-binned pseudobulk per donor,
        ``n_bins`` equal-size bins by score; see :func:`idc.within_donor_nb.fit_donor_binned`).
    leave_gene_out : ``nb_python`` only. Mapping donor -> :class:`idc.within_donor_nb.LeaveGeneOutScores` (rows = that donor's cells in
        input order, columns = ``genes``). Each gene is then tested against scores with its own contribution removed from the IDC
        projection (double-dipping control without splitting); requires ``count_split=False``.
    meta_method : ``"DL"`` (default; DerSimonian-Laird random effects) or ``"moderated"`` (donor-level moderated test: empirical-Bayes
        shrinkage of the between-donor variance, t statistic with moderated df; RE_z then holds the t statistic).

    Returns
    -------
    WithinSampleDEResult with per-donor tables (log2FC, SE, P, BH), the meta-analysis table (RE_log2FC, RE_SE, RE_z, RE_p, RE_BH, Q, Q_p,
    Q_BH, I2, tau2 [DerSimonian-Laird], class), per-donor size factors (the tested cells of that donor, in input order), the pseudobulk DESeq2 table, a donor summary, skipped donors, diagnostics (lambda per
    donor, for RE P and Q P, and for permutations), and the settings / R versions used.
    """
    if mode not in {"per_donor_nb", "pseudobulk_deseq2", "both", "nb_python"}:
        raise WithinSampleDEError("mode must be 'per_donor_nb', 'pseudobulk_deseq2', 'both' or 'nb_python'")
    if n_permutations is None:
        n_permutations = 1 if mode == "nb_python" else 0
    if size_factor is None:
        size_factor = "deconvolution" if mode == "nb_python" else "normed_sum"
    if mode != "nb_python" and size_factor not in SIZE_FACTORS:
        raise WithinSampleDEError(f"size_factor must be one of {SIZE_FACTORS}")
    C = sp.csr_matrix(counts); genes = [str(g) for g in genes]
    donor = np.asarray(donor).astype(str); score = np.asarray(score, float)
    if C.shape != (len(donor), len(genes)) or len(score) != len(donor):
        raise WithinSampleDEError("counts must be cells x genes and match donor, score and genes")
    if not np.isfinite(score).all():
        raise WithinSampleDEError("scores must be finite")
    if np.any(C.data < 0) or np.any(C.data != np.floor(C.data)):
        raise WithinSampleDEError("counts must be non-negative integers")
    C = C.astype(np.int64)
    if count_split is None:
        count_split = mode == "nb_python"
    if count_split not in (True, False, "presplit"):
        raise WithinSampleDEError("count_split must be True, False or 'presplit'")
    if leave_gene_out is not None and (mode != "nb_python" or count_split is not False):
        raise WithinSampleDEError("leave_gene_out requires mode='nb_python' and count_split=False (it replaces splitting)")
    if mode == "nb_python" and count_split is not False and score_source != "half_A":
        raise WithinSampleDEError(
            "mode='nb_python' is confirmatory by default (count_split=True): compute the held-out scores on half A "
            "(idc.split_counts(counts, seed=split_seed)[0], or your own split of the full raw matrix) and pass score_source='half_A' "
            "(with count_split='presplit' if the counts given are already half B); use count_split=False for ranking without splitting")
    if count_split is True:
        _, C = split_counts(C, seed=split_seed)
    use = np.ones(len(donor), bool)
    if condition is not None and case_label is not None:
        use &= np.asarray(condition).astype(str) == str(case_label)
    if donors is not None:
        use &= np.isin(donor, np.asarray(donors).astype(str))
    if mode == "nb_python":
        from . import within_donor_nb
        res = within_donor_nb.run(C, donor, score, genes, use, score_model=score_model, quality=quality, size_factor=size_factor,
                                  threshold=threshold, min_cells_per_state=min_cells_per_state, min_cells_detected=min_cells_detected,
                                  min_cells_detected_per_state=min_cells_detected_per_state, min_donors_meta=min_donors_meta,
                                  alpha=alpha, n_permutations=n_permutations, permutation_seed=permutation_seed,
                                  per_donor_level=per_donor_level, n_bins=n_bins, leave_gene_out=leave_gene_out, meta_method=meta_method)
        res.settings.update(count_split=count_split, split_seed=split_seed if count_split is True else None, case_label=case_label,
                            score_source=score_source, analysis_mode=("leave_gene_out" if leave_gene_out is not None else
                                                                      "ranking" if count_split is False else "confirmatory"))
        return res
    use &= score != threshold
    rows, skipped, groups = [], [], {}
    for d in sorted(set(donor[use])):
        idx = np.flatnonzero(use & (donor == d)); t = score[idx] > threshold
        rows.append({"donor": d, "n_disease_like": int(t.sum()), "n_healthy_like": int((~t).sum())})
        if min(t.sum(), (~t).sum()) < min_cells_per_state:
            skipped.append({**rows[-1], "reason": f"fewer than {min_cells_per_state} cells in a state"}); continue
        groups[d] = (idx, t)
    if len(groups) < 2:
        raise WithinSampleDEError(f"fewer than two donors pass min_cells_per_state={min_cells_per_state}")
    summary = pd.DataFrame(rows); summary["tested"] = summary.donor.isin(groups)
    per, meta, pb = {}, None, None
    if mode in {"per_donor_nb", "both"}:
        per = {d: per_donor_nb(C[idx], genes, t, size_factor=size_factor, rscript=rscript, workdir=workdir) for d, (idx, t) in groups.items()}
        sfs = {d: np.asarray(r.attrs.pop("size_factors")) for d, r in per.items()}
        meta = meta_analysis(per, se_source=se_source, alpha=alpha)
    else:
        sfs = {}
    if mode in {"pseudobulk_deseq2", "both"}:
        pb = _pseudobulk(C, genes, groups, rscript=rscript, workdir=workdir)
    diag = {"lambda_per_donor": {d: genomic_inflation(r.p_value) for d, r in per.items()}}
    if meta is not None:
        diag.update(lambda_RE=genomic_inflation(meta.RE_p), lambda_Q=genomic_inflation(meta.Q_p),
                    n_consistent=int((meta["class"] == "consistent").sum()), n_heterogeneous=int((meta["class"] == "heterogeneous").sum()))
    if pb is not None:
        diag["lambda_DESeq2"] = genomic_inflation(pb.p_value)
    perm = {}
    rng = np.random.default_rng(permutation_seed)
    for k in range(n_permutations):
        pg = {d: (idx, rng.permutation(t)) for d, (idx, t) in groups.items()}; key = f"permutation_{k + 1}"; perm[key] = {}
        if mode in {"per_donor_nb", "both"}:
            pp = {d: per_donor_nb(C[idx], genes, t, size_factor=size_factor, rscript=rscript, workdir=workdir) for d, (idx, t) in pg.items()}
            for r in pp.values(): r.attrs.pop("size_factors", None)
            pm = meta_analysis(pp, se_source=se_source, alpha=alpha)
            perm[key].update(RE_p=pm.RE_p.to_numpy(), Q_p=pm.Q_p.to_numpy())
            diag.setdefault("lambda_permutation_per_donor", {})[key] = {d: genomic_inflation(r.p_value) for d, r in pp.items()}
            diag.setdefault("lambda_permutation_RE", {})[key] = genomic_inflation(pm.RE_p)
        if mode in {"pseudobulk_deseq2", "both"}:
            pq = _pseudobulk(C, genes, pg, rscript=rscript, workdir=workdir)
            perm[key]["DESeq2_p"] = pq.p_value.to_numpy(); diag.setdefault("lambda_permutation_DESeq2", {})[key] = genomic_inflation(pq.p_value)
    settings = {"mode": mode, "threshold": threshold, "min_cells_per_state": min_cells_per_state, "count_split": count_split,
                "split_seed": split_seed if count_split else None, "se_source": se_source, "size_factor": size_factor, "alpha": alpha, "n_permutations": n_permutations,
                "permutation_seed": permutation_seed if n_permutations else None, "case_label": case_label,
                "tau2_estimator": "DerSimonian-Laird", "Q_weights": "fixed-effect inverse variance",
                "per_donor_model": "glmGamPoi glm_gp(~ state, size_factors=size_factor, overdispersion=TRUE); test_de QL F test",
                "model_SE": "sqrt(phi_QL * [(X'WX)^-1]_state) / ln 2, W = mu / (1 + theta mu)"}
    try:
        settings["r_versions"] = r_versions(rscript)
    except Exception as exc:  # pragma: no cover - versions are informational
        settings["r_versions"] = repr(exc)
    return WithinSampleDEResult(per, meta, pb, summary, pd.DataFrame(skipped, columns=["donor", "n_disease_like", "n_healthy_like", "reason"]),
                                diag, perm, settings, sfs)


def _pseudobulk(C: sp.csr_matrix, genes: list[str], groups: Mapping[str, tuple[np.ndarray, np.ndarray]], *, rscript, workdir) -> pd.DataFrame:
    cols, meta = {}, []
    for d, (idx, t) in groups.items():
        for s, m in (("disease_like", t), ("healthy_like", ~t)):
            cols[f"{d}__{s}"] = np.asarray(C[idx[m]].sum(0)).ravel(); meta.append({"sample": f"{d}__{s}", "donor": d, "state": s})
    return pseudobulk_deseq2(pd.DataFrame(cols, index=genes).astype(np.int64), pd.DataFrame(meta).set_index("sample"), rscript=rscript, workdir=workdir)
