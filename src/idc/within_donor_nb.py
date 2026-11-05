"""Python-only within-donor negative binomial DE (numpy/scipy; no R).

Per donor: pooled-cell deconvolution size factors (scran style), a per-gene NB GLM on cells
``log mu_c = log s_c + b0 + b1 x_c [+ quality covariates]`` fitted by IRLS vectorized across genes, NB2 dispersions (Cox-Reid adjusted
gene-wise MLE, local-median mean-dispersion trend, empirical-Bayes shrinkage of log dispersion toward the trend as in DESeq2) and a Wald
test from the Fisher information. ``x_c`` is the held-out score (continuous mode; effect = b1 / ln 2 = log2 change from score 0 to 1) or
1(score > 0.5) (threshold mode). Donors are combined by :func:`idc.within_sample_de.meta_analysis`. Design: docs/DESIGN_within_donor_nb.md.
Entry point: ``idc.within_sample_de(..., mode="nb_python")``; confirmatory by default (count splitting: scores from half A, tests on
half B; ``count_split=False`` for ranking).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import optimize, special, stats
from scipy.sparse.linalg import lsqr

from .within_sample_de import WithinSampleDEError, bh

POOL_SIZES = tuple(range(21, 102, 5))
LOG_ALPHA_MIN, LOG_ALPHA_MAX = np.log(1e-6), np.log(1e3)
GRID = np.linspace(LOG_ALPHA_MIN, LOG_ALPHA_MAX, 13)
PRIOR_VAR_MIN = 0.25  # DESeq2 lower bound for the log-dispersion prior variance
SINGLE_CELL_WEIGHT = 1e-3  # sqrt(1e-6), scran weight of the per-cell equations


def quality_covariates(counts: Any, genes: Sequence[str], mito_prefix: str = "MT-") -> pd.DataFrame:
    """Per-cell log10 total counts, log10 detected genes and % mito from a cells x genes count matrix (use all genes, not the
    IDC-filtered set). % mito is NaN when no gene has the prefix (the column is then dropped by the fit)."""
    C = sp.csr_matrix(counts); tot = np.asarray(C.sum(1)).ravel().astype(float)
    det = np.diff(C.indptr) if C.has_canonical_format else np.asarray((C > 0).sum(1)).ravel()
    mt = np.char.startswith(np.asarray(genes).astype(str), mito_prefix)
    pct = 100 * np.asarray(C[:, mt].sum(1)).ravel() / np.maximum(tot, 1) if mt.any() else np.full(C.shape[0], np.nan)
    return pd.DataFrame({"log10_total_counts": np.log10(np.maximum(tot, 1)), "log10_detected_genes": np.log10(np.maximum(det, 1)), "pct_mito": pct})


def deconvolution_size_factors(counts: Any, sizes: Sequence[int] = POOL_SIZES, min_mean: float = 0.1) -> np.ndarray:
    """Pooled-cell deconvolution size factors (scran style, one cluster), strictly positive, geometric mean 1.

    Cells are placed on a ring ordered by library size (odd ranks ascending, even ranks descending); for every pool size and start cell,
    the pool's factor is the median over genes of pooled library-normalized expression / average library-normalized expression (genes with
    library-size-adjusted mean >= ``min_mean``). The pooled factors are the sums of theta_j = s_j / L_j over the pool; the system (plus
    one low-weight equation per cell) is solved by least squares with theta_j > 0 enforced, and s_j = theta_j L_j.
    """
    C = sp.csr_matrix(counts, dtype=float); n = C.shape[0]
    lib = np.asarray(C.sum(1)).ravel()
    if np.any(lib <= 0):
        raise WithinSampleDEError("deconvolution size factors need cells with positive library size")
    keep = np.asarray((sp.diags(lib.mean() / lib) @ C).mean(0)).ravel() >= min_mean
    if keep.sum() < 20:
        keep = np.asarray(C.sum(0)).ravel() > 0
    Z = (sp.diags(1 / lib) @ C[:, keep]).toarray(); ref = Z.mean(0)
    order = np.argsort(lib, kind="stable"); ring = np.concatenate([order[0::2], order[1::2][::-1]])
    cs = np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(np.vstack([Z[ring], Z[ring]]), axis=0)])
    sizes = [s for s in sizes if s < n] or [max(2, n // 2)]
    rows, cols, rhs, eq = [], [], [], 0
    for s in sizes:
        rhs.append(np.median((cs[s:s + n] - cs[:n]) / ref, axis=1))
        rows.append(np.repeat(np.arange(eq, eq + n), s)); cols.append(ring[(np.arange(n)[:, None] + np.arange(s)[None, :]) % n].ravel()); eq += n
    rows.append(np.arange(eq, eq + n)); cols.append(np.arange(n)); rhs.append(SINGLE_CELL_WEIGHT * np.median(Z / ref, axis=1))
    vals = np.concatenate([np.ones(sum(len(r) for r in rows[:-1])), np.full(n, SINGLE_CELL_WEIGHT)])
    A = sp.csr_matrix((vals, (np.concatenate(rows), np.concatenate(cols))), shape=(eq + n, n)); b = np.concatenate(rhs)
    theta = lsqr(A, b, atol=1e-12, btol=1e-12, iter_lim=20 * n)[0]
    if np.any(theta <= 0):
        lb = 1e-8 * np.median(np.abs(theta[theta > 0])) if np.any(theta > 0) else 1e-12
        theta = optimize.lsq_linear(A, b, bounds=(lb, np.inf), lsmr_tol="auto").x
    s = theta * lib
    return s / np.exp(np.log(s).mean())


def normed_sum_size_factors(counts: Any) -> np.ndarray:
    lib = np.asarray(sp.csr_matrix(counts).sum(1)).ravel().astype(float)
    return lib / np.exp(np.log(lib).mean())


# ---------------------------------------------------------------------------------------------------------------- NB GLM machinery
def _trigamma(x):
    """Trigamma for x > 0: recurrence up to x >= 6, then the asymptotic series (abs. error < 1e-11); scipy's polygamma is ~30x slower."""
    x = np.array(x, float); acc = np.zeros_like(x)
    for _ in range(6):
        m = x < 6
        if not m.any():
            break
        acc[m] += 1 / x[m] ** 2; x[m] += 1
    z = 1 / (x * x)
    return acc + 1 / x + z / 2 + z / x * (1 / 6 - z * (1 / 30 - z * (1 / 42 - z / 30)))


def _pairs(X):
    p = X.shape[1]; ij = [(i, j) for i in range(p) for j in range(i, p)]
    return np.column_stack([X[:, i] * X[:, j] for i, j in ij]), ij


class _Design:
    """Design with shared columns ``Xs`` (n x q) and an optional gene-specific column ``xg`` (n x G) at position ``pos``.

    The gene-specific column carries leave-gene-out scores or per-gene bin scores; without it the design is ``Xs`` itself."""

    def __init__(self, Xs, xg=None, pos=1, _pairs_cache=None):
        self.Xs, self.xg, self.pos = Xs, xg, pos
        self.p = Xs.shape[1] + (xg is not None)
        self.P, self.ij = _pairs_cache if _pairs_cache is not None else _pairs(Xs)
        self.sidx = [j if (xg is None or j < pos) else j + 1 for j in range(Xs.shape[1])]

    def sub(self, cols):
        return self if self.xg is None else _Design(self.Xs, self.xg[:, cols], self.pos, (self.P, self.ij))

    def lin(self, beta):
        eta = self.Xs @ beta[:, self.sidx].T
        return eta + self.xg * beta[:, self.pos] if self.xg is not None else eta

    def xtwx(self, W):
        S = self.P.T @ W; A = np.empty((W.shape[1], self.p, self.p)); si = self.sidx
        for k, (i, j) in enumerate(self.ij):
            A[:, si[i], si[j]] = A[:, si[j], si[i]] = S[k]
        if self.xg is not None:
            WX = W * self.xg; c = self.Xs.T @ WX
            for i in range(self.Xs.shape[1]):
                A[:, si[i], self.pos] = A[:, self.pos, si[i]] = c[i]
            A[:, self.pos, self.pos] = (WX * self.xg).sum(0)
        return A

    def xtwz(self, Wz):
        out = np.empty((Wz.shape[1], self.p)); out[:, self.sidx] = (self.Xs.T @ Wz).T
        if self.xg is not None:
            out[:, self.pos] = (self.xg * Wz).sum(0)
        return out


def _deviance(Y, mu, alpha):
    r = 1 / alpha
    return 2 * (special.xlogy(Y, Y) - special.xlogy(Y, mu) - (Y + r) * np.log1p((Y - mu) / (mu + r))).sum(0)


def _mu(design, off, beta):
    return np.exp(np.clip(off[:, None] + design.lin(beta), -50, 50))


def _irls(Y, design, off, alpha, beta=None, maxit=50, tol=1e-8, ridge=1e-8):
    """IRLS for an NB2 GLM with fixed per-gene dispersion; returns beta (G x p), converged flag, iterations."""
    n, G = Y.shape; p = design.p; I = np.eye(p) * ridge
    if beta is None:
        beta = np.zeros((G, p)); beta[:, 0] = np.log(np.maximum(Y.sum(0), 0.5) / np.exp(off).sum())
    beta = beta.copy(); dev = _deviance(Y, _mu(design, off, beta), alpha)
    conv = np.zeros(G, bool); iters = np.zeros(G, int)
    for _ in range(maxit):
        act = np.flatnonzero(~conv)
        if not len(act):
            break
        Ya, al, ba, Da = Y[:, act], alpha[act], beta[act], design.sub(act)
        mu = _mu(Da, off, ba); W = mu / (1 + al * mu); z = Da.lin(ba) + (Ya - mu) / mu
        bn = np.linalg.solve(Da.xtwx(W) + I, Da.xtwz(W * z)[..., None])[..., 0]
        dn = _deviance(Ya, _mu(Da, off, bn), al); step = bn - ba
        for _h in range(10):  # step halving for genes whose deviance increased
            bad = ~np.isfinite(dn) | (dn > dev[act] * (1 + 1e-10) + 1e-10)
            if not bad.any():
                break
            step[bad] /= 2; bn[bad] = ba[bad] + step[bad]; dn[bad] = _deviance(Ya[:, bad], _mu(Da.sub(bad), off, bn[bad]), al[bad])
        iters[act] += 1
        conv[act] = np.abs(dn - dev[act]) / (np.abs(dn) + 0.1) < tol
        beta[act], dev[act] = bn, dn
    return beta, conv, iters


class _CRObjective:
    """Cox-Reid adjusted NB2 profile log-likelihood in theta = log(alpha) with mu fixed (constant -lgamma(y + 1) omitted), its first
    and second derivatives, and a vectorized maximizer (coarse grid, then safeguarded Newton on the derivative within a bracket)."""

    def __init__(self, Y, mu, design):
        self.mu, self.design, self.G = mu, design, mu.shape[1]
        if Y is not None:
            r, c = np.nonzero(Y); self.c, self.y, self.mu_nz = c, Y[r, c], mu[r, c]

    def subset(self, keep):
        """Objective restricted to the genes in ``keep`` (boolean)."""
        o = _CRObjective(None, self.mu[:, keep], self.design.sub(keep))
        m = keep[self.c]; o.c = (np.cumsum(keep) - 1)[self.c[m]]; o.y, o.mu_nz = self.y[m], self.mu_nz[m]
        return o

    def __call__(self, theta, prior_mean=None, prior_var=None):
        a = np.exp(theta); r = 1 / a
        ll = -(r[None, :] * np.log1p(a[None, :] * self.mu)).sum(0)
        an = a[self.c]
        nz = special.gammaln(self.y + 1 / an) - special.gammaln(1 / an) + self.y * np.log(an * self.mu_nz / (1 + an * self.mu_nz))
        ll += np.bincount(self.c, nz, minlength=self.G)
        ll -= 0.5 * np.linalg.slogdet(self.design.xtwx(self.mu / (1 + a[None, :] * self.mu)))[1]
        if prior_mean is not None:
            ll -= (theta - prior_mean) ** 2 / (2 * prior_var)
        return ll

    def derivatives(self, theta, prior_mean=None, prior_var=None):
        a = np.exp(theta); r = 1 / a; t = a[None, :] * self.mu; u = 1 / (1 + t)
        Lr = (t * u - np.log1p(t)).sum(0); Lrr = a * ((t * u) ** 2).sum(0)  # y = 0 parts of d/dr and d2/dr2, all cells
        rn = r[self.c]; dn = rn + self.mu_nz; nnz = np.bincount(self.c, minlength=self.G)
        Lr += np.bincount(self.c, special.digamma(self.y + rn) - self.y / dn, minlength=self.G) - nnz * special.digamma(r)
        Lrr += np.bincount(self.c, _trigamma(self.y + rn) + self.y / dn ** 2, minlength=self.G) - nnz * _trigamma(r)
        g = -r * Lr; h = r ** 2 * Lrr + r * Lr
        W = self.mu * u; Wt = -W * t * u; Wtt = Wt * (1 - t) * u
        Ai = np.linalg.inv(self.design.xtwx(W)); At = self.design.xtwx(Wt)
        Att = self.design.xtwx(Wtt); B = Ai @ At
        g -= 0.5 * np.trace(B, axis1=1, axis2=2)
        h -= 0.5 * (np.trace(Ai @ Att, axis1=1, axis2=2) - np.einsum("gij,gji->g", B, B))
        if prior_mean is not None:
            g -= (theta - prior_mean) / prior_var; h -= 1 / prior_var
        return g, h

    def _newton(self, lo, hi, theta, n_iter=40, tol=1e-6, prior_mean=None, prior_var=None):
        """Safeguarded Newton on d/dtheta within [lo, hi] (bisection when the step leaves the bracket or h >= 0)."""
        for it in range(n_iter):
            g, h = self.derivatives(theta, prior_mean=prior_mean, prior_var=prior_var)
            lo = np.where(g > 0, theta, lo); hi = np.where(g > 0, hi, theta)
            step = theta - g / np.where(h < 0, h, -1.0)
            new = np.where((h < 0) & (step > lo) & (step < hi), step, (lo + hi) / 2)
            act = ~((np.abs(new - theta) < tol) | (hi - lo < tol)); theta = new
            if not act.any():
                break
            if act.mean() < 0.25:  # continue the stragglers on a column subset
                sub = self.subset(act); pm = None if prior_mean is None else prior_mean[act]
                theta = theta.copy()
                theta[act] = sub._newton(lo[act], hi[act], theta[act], n_iter - it - 1, tol, pm, prior_var)
                break
        return theta

    def maximize(self):
        """Gene-wise CR MLE of log(alpha)."""
        vals = np.stack([self(np.full(self.G, g)) for g in GRID]); k = np.argmax(vals, axis=0)
        lo = GRID[np.maximum(k - 1, 0)]; hi = GRID[np.minimum(k + 1, len(GRID) - 1)]
        return self._newton(lo, hi, GRID[k].astype(float))

    def maximize_map(self, theta_hat, prior_mean, prior_var):
        """MAP of log(alpha) under the log-normal prior; bracketed by the gene-wise estimate and the prior mean."""
        lo, hi = np.minimum(theta_hat, prior_mean), np.maximum(theta_hat, prior_mean)
        return self._newton(lo, hi, (lo + hi) / 2, prior_mean=prior_mean, prior_var=prior_var)


def _loc_median_trend(log_mean, log_alpha, use, frac=0.1):
    o = np.argsort(log_mean[use]); xm = log_mean[use][o]; ya = log_alpha[use][o]
    w = max(int(frac * len(o)), 5)
    sm = pd.Series(ya).rolling(w, center=True, min_periods=1).median().to_numpy()
    return np.interp(log_mean, xm, sm)


def vif(X: np.ndarray) -> np.ndarray:
    """Variance inflation factor of each non-intercept column of a design with an intercept in column 0."""
    out = []
    for j in range(1, X.shape[1]):
        others = np.delete(X, j, axis=1); fit = others @ np.linalg.lstsq(others, X[:, j], rcond=None)[0]
        ss = np.sum((X[:, j] - X[:, j].mean()) ** 2); r2 = 1 - np.sum((X[:, j] - fit) ** 2) / ss if ss > 0 else np.nan
        out.append(1 / max(1 - r2, 1e-12))
    return np.asarray(out)


def _nb_fit(get_y, design_for, disp_design_for, off, gi, G, norm, n_obs, p_disp, chunk_size, warm_start=True):
    """Shared per-gene NB machinery: moments -> IRLS -> gene-wise CR MLE (dispersion design) -> local-median trend -> EB prior -> MAP
    -> final IRLS (full design) -> Wald SE of column 1. Returns a dict of per-gene arrays (length G) and the prior variances."""
    a_gw = np.full(G, np.nan); beta_d = {}
    s = np.exp(off)
    for g0 in range(0, len(gi), chunk_size):
        g = gi[g0:g0 + chunk_size]; Y = get_y(g); Dd = disp_design_for(g)
        Yn = Y / s[:, None]; m = Yn.mean(0); v = Yn.var(0, ddof=1)
        a0 = np.clip((v - m * np.mean(1 / s)) / np.maximum(m, 1e-12) ** 2, 1e-4, 1e2)
        b, _, _ = _irls(Y, Dd, off, a0)
        a_gw[g] = np.exp(_CRObjective(Y, _mu(Dd, off, b), Dd).maximize()); beta_d[g0] = b
    tested = np.zeros(G, bool); tested[gi] = True
    lm = np.log(np.maximum(norm, 1e-12)); use = tested & (a_gw > np.exp(LOG_ALPHA_MIN) * 100)
    if use.sum() < 10:
        use = tested
    trend = np.exp(_loc_median_trend(lm, np.log(a_gw), use)); resid = np.log(a_gw[use]) - np.log(trend[use])
    mad_var = (stats.median_abs_deviation(resid, scale="normal")) ** 2
    prior_var = max(mad_var - (_trigamma((n_obs - p_disp) / 2) if n_obs > p_disp else 0.0), PRIOR_VAR_MIN)
    a_map = np.full(G, np.nan); out_b = np.full(G, np.nan); out_se = np.full(G, np.nan); conv = np.zeros(G, bool); iters = np.zeros(G, int)
    for g0 in range(0, len(gi), chunk_size):
        g = gi[g0:g0 + chunk_size]; Y = get_y(g); Dd = disp_design_for(g); Df = design_for(g)
        lmap = _CRObjective(Y, _mu(Dd, off, beta_d[g0]), Dd).maximize_map(np.log(a_gw[g]), np.log(trend[g]), prior_var)
        outlier = np.log(a_gw[g]) - np.log(trend[g]) > 2 * np.sqrt(prior_var)
        am = np.where(outlier, a_gw[g], np.exp(lmap)); a_map[g] = am
        b, cv, it = _irls(Y, Df, off, am, beta=beta_d[g0] if warm_start else None)
        mu = _mu(Df, off, b); cov = np.linalg.inv(Df.xtwx(mu / (1 + am * mu)))
        out_b[g], out_se[g], conv[g], iters[g] = b[:, 1], np.sqrt(cov[:, 1, 1]), cv, it
    return {"b": out_b, "se": out_se, "a_gw": a_gw, "trend": trend, "a_map": a_map, "conv": conv, "iters": iters,
            "prior_var": prior_var, "mad_var": mad_var}


def _finish(fit, tested, norm, ndet):
    bad_se = tested & ~(np.isfinite(fit["se"]) & (fit["se"] > 0) & np.isfinite(fit["b"])); tested = tested & ~bad_se
    b, se = np.where(tested, fit["b"], np.nan), np.where(tested, fit["se"], np.nan)
    z = b / se; pv = 2 * stats.norm.sf(np.abs(z))
    tab = pd.DataFrame({"log2FC": b / np.log(2), "SE": se / np.log(2), "z": z, "p_value": pv, "adjusted_p_value": bh(pv),
                        "tested": tested, "converged": fit["conv"], "iterations": fit["iters"], "alpha_genewise": fit["a_gw"],
                        "alpha_trend": np.where(tested, fit["trend"], np.nan), "alpha_map": fit["a_map"], "mean_normalized": norm, "n_detected": ndet})
    return tab, int(bad_se.sum())


def _standardize(Q, names):
    Q = np.asarray(Q, float); qn = list(names) if names is not None else [f"q{j}" for j in range(Q.shape[1])]
    Q = (Q - Q.mean(0)) / np.where(Q.std(0) > 0, Q.std(0), 1)
    dropped = [j for j in range(Q.shape[1]) if not np.isfinite(Q[:, j]).all() or Q[:, j].std() == 0]
    return np.delete(Q, dropped, axis=1), [c for j, c in enumerate(qn) if j not in dropped], [qn[j] for j in dropped]


def fit_donor(counts: Any, x: np.ndarray, *, quality: np.ndarray | None = None, quality_names: Sequence[str] | None = None,
              size_factors: np.ndarray | None = None, size_factor: str = "deconvolution", threshold_mode: bool = False,
              min_cells_detected: int = 3, min_cells_detected_per_state: int = 3, split_at: float = 0.5,
              x_gene=None, chunk_size: int = 2000) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Fit the per-gene NB GLM in one donor. Returns (per-gene table, size factors, info).

    ``x_gene`` (optional): callable mapping gene column indices to an n x len(cols) matrix of gene-specific scores (leave-gene-out);
    the score column of the design is then gene-specific, while the gene rule and VIF use ``x``.

    Gene rule: threshold mode, count > 0 in >= ``min_cells_detected_per_state`` cells of each state; continuous mode, count > 0 in
    >= ``min_cells_detected`` cells, at least one of them with x > ``split_at`` and one with x < ``split_at``. A gene whose final Wald SE
    is not finite (or not positive) is marked untested.

    Table columns: log2FC (b1 / ln 2), SE (Wald, log2 scale), z, p_value, adjusted_p_value (BH within donor), tested, converged,
    iterations, alpha_genewise, alpha_trend, alpha_map, mean_normalized, n_detected.
    """
    Ycsc = sp.csc_matrix(counts, dtype=float); n, G = Ycsc.shape; x = np.asarray(x, float)
    if size_factors is None:
        size_factors = deconvolution_size_factors(Ycsc) if size_factor == "deconvolution" else normed_sum_size_factors(Ycsc)
    off = np.log(size_factors)
    Q, kept_q, dropped_q = _standardize(np.empty((n, 0)) if quality is None else np.asarray(quality, float).reshape(n, -1), quality_names)
    X = np.column_stack([np.ones(n), x, Q]); p = X.shape[1]
    full = _Design(X); Xs = np.column_stack([np.ones(n), Q]); pc = _pairs(Xs)
    design_for = (lambda g: full) if x_gene is None else (lambda g: _Design(Xs, x_gene(g), 1, pc))
    D = (Ycsc > 0).astype(np.int8).tocsc(); ndet = np.diff(D.indptr)
    hi, lo = x > split_at, x < split_at
    det_hi, det_lo = np.asarray(D[hi].sum(0)).ravel(), np.asarray(D[lo].sum(0)).ravel()
    if threshold_mode:
        tested = (det_hi >= min_cells_detected_per_state) & (det_lo >= min_cells_detected_per_state)
    else:
        tested = (ndet >= min_cells_detected) & (det_hi >= 1) & (det_lo >= 1)
    n_rule = int(tested.sum())
    gi = np.flatnonzero(tested); norm = np.asarray((sp.diags(1 / size_factors) @ Ycsc).mean(0)).ravel()
    fit = _nb_fit(lambda g: Ycsc[:, g].toarray(), design_for, design_for, off, gi, G, norm, n, p, chunk_size)
    tab, n_bad = _finish(fit, tested, norm, ndet); prior_var = fit["prior_var"]
    info = {"n_cells": int(n), "n_tested": int(tab.tested.sum()), "n_not_converged": int((tab.tested & ~tab.converged).sum()), "prior_var": float(prior_var),
            "mad_var": float(fit["mad_var"]), "n_dispersion_outliers_kept_genewise": int(np.sum(tested & (np.log(fit["a_gw"]) - np.log(fit["trend"]) > 2 * np.sqrt(prior_var)))),
            "quality_covariates_dropped": dropped_q, "n_design_columns": int(p), "n_genes_pass_detection_rule": n_rule,
            "n_genes_nonfinite_SE": n_bad, "vif": dict(zip(["score", *kept_q], vif(X).round(4).tolist())), "level": "cells",
            "leave_gene_out": x_gene is not None}
    return tab, size_factors, info



def median_of_ratios(Y: np.ndarray) -> np.ndarray:
    """DESeq2 median-of-ratios size factors for a samples x genes matrix (genes positive in every sample; poscounts-style geometric
    means over positive values if fewer than 10 such genes), scaled to geometric mean 1."""
    pos = np.all(Y > 0, axis=0)
    if pos.sum() >= 10:
        lg = np.log(Y[:, pos]); s = np.exp(np.median(lg - lg.mean(0), axis=1))
    else:
        lg = np.where(Y > 0, np.log(np.where(Y > 0, Y, 1)), 0.0); gm = lg.sum(0) / Y.shape[0]
        keep = Y.sum(0) > 0
        s = np.array([np.exp(np.median((np.log(Y[i, keep & (Y[i] > 0)]) - gm[keep & (Y[i] > 0)]))) for i in range(Y.shape[0])])
    return s / np.exp(np.log(s).mean())


def _bins(order, K):
    """Bin index (0..K-1) for positions 0..n-1 in sorted order; equal sizes up to 1."""
    n = len(order); b = np.empty(n, int); b[order] = (np.arange(n) * K) // n
    return b


def fit_donor_binned(counts: Any, x: np.ndarray, *, n_bins: int = 4, quality: pd.DataFrame | np.ndarray | None = None,
                     quality_names: Sequence[str] | None = None, x_gene=None, min_total: int = 3, min_bins_nonzero: int = 2,
                     chunk_size: int = 4000) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Score-binned pseudobulk NB fit for one donor (design note Stage 3, option 2).

    Cells are sorted by score into ``n_bins`` equal bins; counts are summed per bin. Per gene, log mu_k = log s_k + b0 + b1 * mean score_k
    + gamma * quality_k (bin medians of the quality columns, z-scored across bins; undefined or constant ones dropped), with
    median-of-ratios size factors across bins. Dispersion: gene-wise CR MLE under [1, score] (df K - 2), local-median trend, EB shrinkage,
    MAP; Wald SE of b1 from the full model. Gene rule: total count >= ``min_total`` and nonzero in >= ``min_bins_nonzero`` bins.
    With ``x_gene`` (leave-gene-out), each gene's bins follow its own score; size factors and bin quality use the full-score bins.
    """
    Ycsc = sp.csc_matrix(counts, dtype=float); n, G = Ycsc.shape; x = np.asarray(x, float); K = int(n_bins)
    b = _bins(np.argsort(x, kind="stable"), K); M = sp.csr_matrix((np.ones(n), (b, np.arange(n))), shape=(K, n))
    Yb_all = np.asarray((M @ Ycsc).todense()); xb = (M @ x) / np.asarray(M.sum(1)).ravel()
    sf = median_of_ratios(Yb_all + 0.0); off = np.log(sf)
    qn = None
    if quality is not None:
        qa = np.asarray(quality, float).reshape(n, -1); qn = quality_names
        Qb = np.column_stack([[np.nanmedian(qa[b == k, j]) for k in range(K)] for j in range(qa.shape[1])]) if qa.shape[1] else np.empty((K, 0))
    else:
        Qb = np.empty((K, 0))
    Qb, kept_q, dropped_q = _standardize(Qb, qn)
    if Qb.shape[1] > max(K - 3, 0):  # keep the design estimable: at most K - 3 quality columns (first ones kept)
        dropped_q += kept_q[max(K - 3, 0):]; kept_q = kept_q[:max(K - 3, 0)]; Qb = Qb[:, :max(K - 3, 0)]
    Xs = np.column_stack([np.ones(K), Qb]); pc = _pairs(Xs); ones = np.ones((K, 1)); po = _pairs(ones)
    nz_bins = (Yb_all > 0).sum(0); tot = Yb_all.sum(0)
    tested = (tot >= min_total) & (nz_bins >= min_bins_nonzero); gi = np.flatnonzero(tested)
    norm = (Yb_all / sf[:, None]).mean(0)
    if x_gene is None:
        get_y = lambda g: Yb_all[:, g]
        xg_of = lambda g: np.repeat(xb[:, None], len(g), axis=1)
    else:
        cache = {}

        def per_gene(g):
            key = (int(g[0]), len(g))
            if key not in cache:
                S = x_gene(g); o = np.argsort(S, axis=0, kind="stable"); rank = np.empty_like(o); np.put_along_axis(rank, o, np.arange(n)[:, None], axis=0)
                bb = (rank * K) // n; Yg = Ycsc[:, g].toarray(); Yk = np.zeros((K, len(g))); Sk = np.zeros((K, len(g)))
                for k in range(K):
                    m = bb == k; Yk[k] = (Yg * m).sum(0); Sk[k] = (S * m).sum(0) / m.sum(0)
                cache.clear(); cache[key] = (Yk, Sk)
            return cache[key]
        get_y = lambda g: per_gene(g)[0]
        xg_of = lambda g: per_gene(g)[1]
    design_for = lambda g: _Design(Xs, xg_of(g), 1, pc)
    disp_for = lambda g: _Design(ones, xg_of(g), 1, po)
    fit = _nb_fit(get_y, design_for, disp_for, off, gi, G, norm, K, 2, chunk_size, warm_start=False)
    tab, n_bad = _finish(fit, tested, norm, np.diff((Ycsc > 0).astype(np.int8).tocsc().indptr))
    info = {"n_cells": int(n), "n_bins": K, "bin_cells": np.bincount(b, minlength=K).tolist(), "bin_mean_score": xb.round(4).tolist(),
            "size_factors_bins": sf.round(4).tolist(), "n_tested": int(tab.tested.sum()), "n_not_converged": int((tab.tested & ~tab.converged).sum()),
            "prior_var": float(fit["prior_var"]), "quality_covariates_dropped": dropped_q, "quality_covariates_used": kept_q,
            "n_design_columns": int(Xs.shape[1] + 1), "residual_df": int(K - Xs.shape[1] - 1), "n_genes_nonfinite_SE": n_bad,
            "level": f"bins{K}", "leave_gene_out": x_gene is not None}
    return tab, sf, info



def joint_binned_de(counts: Any, donor: Sequence[Any], score: np.ndarray, genes: Sequence[str], *, n_bins: int = 6, leave_gene_out: Any = None,
                    threshold: float = 0.5, min_cells_per_state: int = 20, min_total: int = 3, min_samples_nonzero: int = 2,
                    chunk_size: int = 2000) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Joint binned NB model across donors (candidate; design: manuscript plan within_donor_joint_binned_20261028_v1).

    Per donor and gene, cells are binned by score (leave-gene-out score of that gene if ``leave_gene_out`` maps donor ->
    :class:`LeaveGeneOutScores` aligned to that donor's cells in input order): ``n_bins`` equal-size bins, or for ``n_bins=2`` score >
    ``threshold`` vs below. Counts are summed per bin (D x K samples). One NB GLM per gene: log mu = log s + intercept + donor dummies +
    beta * x, with x = bin mean score (K > 2) or the disease-like indicator (K = 2); median-of-ratios size factors across all samples
    (full-score bins); gene-wise Cox-Reid dispersion, local-median trend, EB shrinkage, MAP; Wald test on beta; BH.
    Donors need >= ``min_cells_per_state`` cells on each side of ``threshold``. Gene rule: total >= ``min_total`` and nonzero in >=
    ``min_samples_nonzero`` samples. Returns (gene table indexed by ``genes``, info)."""
    C = sp.csr_matrix(counts); donor = np.asarray(donor).astype(str); score = np.asarray(score, float); genes = [str(g) for g in genes]; K = int(n_bins)
    groups = {}
    for d in sorted(set(donor)):
        idx = np.flatnonzero(donor == d); s = score[idx]
        if K == 2:
            idx = idx[s != threshold]; s = score[idx]
        if min((s > threshold).sum(), (s < threshold).sum()) >= min_cells_per_state:
            groups[d] = idx
    if len(groups) < 2:
        raise WithinSampleDEError("fewer than two donors pass min_cells_per_state")
    dn = list(groups); D = len(dn); S = D * K; G = len(genes)

    def bins_of(v):
        if K == 2:
            return (v > threshold).astype(int)
        o = np.argsort(v, axis=0, kind="stable"); r = np.empty_like(o); np.put_along_axis(r, o, np.arange(v.shape[0])[:, None] if v.ndim == 2 else np.arange(len(v)), axis=0)
        return (r * K) // v.shape[0]

    Yfull = np.zeros((S, G)); xfull = np.zeros(S)
    for j, d in enumerate(dn):
        idx = groups[d]; b = bins_of(score[idx]); Cd = C[idx]
        for k in range(K):
            m = b == k; Yfull[j * K + k] = np.asarray(Cd[m].sum(0)).ravel(); xfull[j * K + k] = float(k) if K == 2 else score[idx][m].mean()
    sf = median_of_ratios(Yfull); off = np.log(sf); norm = (Yfull / sf[:, None]).mean(0)
    dummies = np.zeros((S, max(D - 1, 0)))
    for j in range(1, D):
        dummies[j * K:(j + 1) * K, j - 1] = 1.0
    Xs = np.column_stack([np.ones(S), dummies]); pc = _pairs(Xs)
    tested = (Yfull.sum(0) >= min_total) & ((Yfull > 0).sum(0) >= min_samples_nonzero); gi = np.flatnonzero(tested)
    if leave_gene_out is None:
        get_y = lambda g: Yfull[:, g]; xg_of = lambda g: np.repeat(xfull[:, None], len(g), axis=1)
    else:
        lg = {d: leave_gene_out[d].take(np.searchsorted(np.flatnonzero(donor == d), groups[d])) for d in dn}; cache = {}

        def per_gene(g):
            key = (int(g[0]), len(g))
            if key not in cache:
                Yk = np.zeros((S, len(g))); Xk = np.zeros((S, len(g)))
                for j, d in enumerate(dn):
                    Sg = lg[d](g); bb = bins_of(Sg); Yg = C[groups[d]][:, g].toarray()
                    for k in range(K):
                        m = bb == k; Yk[j * K + k] = (Yg * m).sum(0)
                        Xk[j * K + k] = float(k) if K == 2 else (Sg * m).sum(0) / np.maximum(m.sum(0), 1)
                cache.clear(); cache[key] = (Yk, Xk)
            return cache[key]
        get_y = lambda g: per_gene(g)[0]; xg_of = lambda g: per_gene(g)[1]
    design_for = lambda g: _Design(Xs, xg_of(g), 1, pc)
    fit = _nb_fit(get_y, design_for, design_for, off, gi, G, norm, S, D + 1, chunk_size)
    tab, n_bad = _finish(fit, tested, norm, (Yfull > 0).sum(0)); tab.index = genes
    info = {"donors": dn, "n_donors": D, "n_bins": K, "n_samples": S, "size_factors": sf.round(4).tolist(), "n_tested": int(tab.tested.sum()),
            "n_not_converged": int((tab.tested & ~tab.converged).sum()), "prior_var": float(fit["prior_var"]), "n_genes_nonfinite_SE": n_bad,
            "covariate": "disease-like indicator" if K == 2 else "bin mean score", "leave_gene_out": leave_gene_out is not None}
    return tab, info


class LeaveGeneOutScores:
    """Leave-gene-out IDC scores for one donor's cells (design note Stage 3, option 1).

    eta: per-cell PU linear predictor; expression: projection input expression for the test genes (cells x genes, CSC); center, weight:
    per test gene reference mean m_g and linear-predictor weight v_g (0 for genes outside the projection). The score of cell c when
    testing gene g is expit(eta_c - (e_cg - m_g) v_g)."""

    def __init__(self, eta, expression, center, weight):
        self.eta = np.asarray(eta, float); self.expression = sp.csc_matrix(expression, dtype=float)
        self.center = np.asarray(center, float); self.weight = np.asarray(weight, float)

    @property
    def full(self):
        return special.expit(self.eta)

    def take(self, rows):
        return LeaveGeneOutScores(self.eta[rows], self.expression[rows], self.center, self.weight)

    def __call__(self, cols):
        E = self.expression[:, cols].toarray()
        return special.expit(self.eta[:, None] - (E - self.center[cols]) * self.weight[cols])

    @classmethod
    def from_compact_state(cls, state, expression, expression_genes: Sequence[str], test_genes: Sequence[str]):
        """Build from a frozen IDC compact prediction state and the projection input expression (cells x ``expression_genes``; the same
        values the projection used, e.g. raw counts with an identity upstream transform)."""
        pf = state.projection_fit; nm = state.cell_score_normalization
        if getattr(pf, "upstream_transform", "identity") not in ("identity",):
            raise WithinSampleDEError("leave-gene-out scores need the projection input expression; non-identity transforms not supported")
        names = pd.Index([str(g) for g in expression_genes]); ret = list(map(str, pf.gene_names))
        ri = names.get_indexer(ret)
        if (ri < 0).any():
            raise WithinSampleDEError(f"{int((ri < 0).sum())} projection genes missing from the expression matrix")
        E = sp.csr_matrix(expression, dtype=float)[:, ri]
        theta, mean, sd = np.asarray(pf.gene_theta, float), np.asarray(pf.reference_mean, float), np.asarray(pf.reference_sd, float)
        deg = np.asarray(nm.degenerate_components, bool); nsd = np.where(deg, 1.0, np.asarray(nm.sd, float))
        ct = np.asarray(E @ (theta / sd[:, None])) - (mean / sd) @ theta
        sc = np.where(deg, 0.0, (ct - np.asarray(nm.mean, float)) / nsd)
        beta = np.asarray(state.pu_coefficient, float).ravel(); smu, ssd = np.asarray(state.pu_scaling_mean, float), np.asarray(state.pu_scaling_sd, float)
        eta = ((sc - smu) / ssd) @ beta + float(state.pu_intercept)
        coef = np.where(deg, 0.0, beta / (nsd * ssd)); v_ret = (theta / sd[:, None]) @ coef
        ti = pd.Index(ret).get_indexer([str(g) for g in test_genes]); inp = ti >= 0
        Et = (E[:, np.maximum(ti, 0)] @ sp.diags(inp.astype(float))).tocsc()  # genes outside the projection: zero column
        return cls(eta, Et, np.where(inp, mean[np.maximum(ti, 0)], 0.0), np.where(inp, v_ret[np.maximum(ti, 0)], 0.0))


def run(C: sp.csr_matrix, donor: np.ndarray, score: np.ndarray, genes: list[str], use: np.ndarray, *, score_model: str = "continuous",
        quality: pd.DataFrame | str | None = "auto", size_factor: str = "deconvolution", threshold: float = 0.5, min_cells_per_state: int = 20,
        min_cells_detected: int = 3, min_cells_detected_per_state: int = 3, min_donors_meta: int | None = None, alpha: float = 0.05,
        n_permutations: int = 1, permutation_seed: int = 20261026, chunk_size: int = 2000, per_donor_level: str = "cells",
        n_bins: int = 4, leave_gene_out: Any = None, meta_method: str = "DL"):
    """Python NB mode of :func:`idc.within_sample_de` (inputs already validated there)."""
    from .within_sample_de import WithinSampleDEResult, genomic_inflation, meta_analysis
    if score_model not in {"continuous", "threshold"}:
        raise WithinSampleDEError("score_model must be 'continuous' or 'threshold'")
    if size_factor not in {"deconvolution", "normed_sum"}:
        raise WithinSampleDEError("mode='nb_python' supports size_factor 'deconvolution' or 'normed_sum'")
    thr = score_model == "threshold"
    if per_donor_level not in ("cells", "bins"):
        raise WithinSampleDEError("per_donor_level must be 'cells' or 'bins'")
    if per_donor_level == "bins" and thr:
        raise WithinSampleDEError("per_donor_level='bins' uses the continuous score; score_model must be 'continuous'")
    lib = np.asarray(C.sum(1)).ravel(); use = use & (lib > 0)
    qdf, q_source = None, "none"
    if isinstance(quality, str):
        if quality != "auto":
            raise WithinSampleDEError("quality must be 'auto', None or a per-cell DataFrame")
        quality = quality_covariates(C, genes); q_source = "auto: computed from the supplied genes"
    elif quality is not None:
        q_source = "supplied"
    if quality is not None:
        qdf = pd.DataFrame(quality).reset_index(drop=True)
        if len(qdf) != len(donor):
            raise WithinSampleDEError("quality must have one row per cell")
    rows, skipped, groups = [], [], {}
    for d in sorted(set(donor[use])):
        idx = np.flatnonzero(use & (donor == d)); s = score[idx]
        n_hi, n_lo = int((s > threshold).sum()), int((s < threshold).sum())
        rows.append({"donor": d, "n_cells": len(idx), "n_disease_like": n_hi, "n_healthy_like": n_lo})
        if min(n_hi, n_lo) < min_cells_per_state:
            skipped.append({**rows[-1], "reason": f"fewer than {min_cells_per_state} cells on a side of {threshold}"}); continue
        if thr:
            idx = idx[s != threshold]
        groups[d] = idx
    if len(groups) < 2:
        raise WithinSampleDEError(f"fewer than two donors pass min_cells_per_state={min_cells_per_state}")
    summary = pd.DataFrame(rows); summary["tested"] = summary.donor.isin(groups)
    xof = (lambda idx: (score[idx] > threshold).astype(float)) if thr else (lambda idx: score[idx].astype(float))
    qof = (lambda idx: None) if qdf is None else (lambda idx: qdf.iloc[idx].to_numpy(float))
    qnames = None if qdf is None else [str(c) for c in qdf.columns]
    md = min_donors_meta if min_donors_meta is not None else max(3, int(np.ceil(len(groups) / 3)))
    fit_kw = dict(quality_names=qnames, threshold_mode=thr, min_cells_detected=min_cells_detected,
                  min_cells_detected_per_state=min_cells_detected_per_state, split_at=threshold, chunk_size=chunk_size)
    lgo = {}
    if leave_gene_out is not None:
        for d, idx in groups.items():
            if d not in leave_gene_out:
                raise WithinSampleDEError(f"leave_gene_out has no scores for donor {d}")
            obj = leave_gene_out[d]; all_d = np.flatnonzero(donor == d)
            if len(obj.eta) != len(all_d) or obj.expression.shape[1] != len(genes):
                raise WithinSampleDEError(f"leave_gene_out[{d}] must have one row per cell of the donor (input order) and one column per gene")
            lgo[d] = obj.take(np.searchsorted(all_d, idx))
    bin_q = None if qdf is None else [c for c in ("log10_detected_genes", "pct_mito") if c in qdf.columns]

    def xgene(d, order=None):
        if d not in lgo:
            return None
        base = lgo[d]; f = (lambda cols: (base(cols) > threshold).astype(float)) if thr else base
        return f if order is None else (lambda cols: f(cols)[order])

    def fit(d, idx, x, order=None, sf=None):
        if per_donor_level == "bins":
            q = None if not bin_q else qdf.iloc[idx][bin_q].to_numpy(float)
            return fit_donor_binned(C[idx], x, n_bins=n_bins, quality=q, quality_names=bin_q, x_gene=xgene(d, order))
        kw = {"size_factors": sf} if sf is not None else {"size_factor": size_factor}
        return fit_donor(C[idx], x, quality=qof(idx), x_gene=xgene(d, order), **kw, **fit_kw)

    per, sfs, infos = {}, {}, {}
    for d, idx in groups.items():
        per[d], sfs[d], infos[d] = fit(d, idx, xof(idx))
        per[d].index = genes
    meta = meta_analysis(per, se_source="model", alpha=alpha, min_donors=md, method=meta_method)
    diag = {"lambda_per_donor": {d: genomic_inflation(r.p_value) for d, r in per.items()}, "lambda_RE": genomic_inflation(meta.RE_p),
            "lambda_Q": genomic_inflation(meta.Q_p), "n_consistent": int((meta["class"] == "consistent").sum()),
            "n_heterogeneous": int((meta["class"] == "heterogeneous").sum()), "per_donor_fit": infos,
            "vif_per_donor": {d: i.get("vif", {}) for d, i in infos.items()}, "genes_in_meta": int(len(meta)),
            "genes_by_n_donors": {int(k): int(v) for k, v in meta.n_donors.value_counts().sort_index().items()}}
    dq = qdf if qdf is not None else quality_covariates(C, genes)
    diag["score_quality_spearman"] = {d: {c: (float(stats.spearmanr(score[idx], dq[c].to_numpy()[idx]).statistic)
                                              if np.isfinite(dq[c].to_numpy()[idx]).all() and dq[c].to_numpy()[idx].std() > 0 else None)
                                          for c in dq.columns} for d, idx in groups.items()}
    diag["score_quality_source"] = "supplied quality" if qdf is not None else "computed from the supplied genes (mito = 'MT-' genes among them)"
    perm, rng = {}, np.random.default_rng(permutation_seed)
    for k in range(n_permutations):
        key = f"permutation_{k + 1}"; pp = {}
        for d, idx in groups.items():
            o = rng.permutation(len(idx)); xp = xof(idx)[o]
            pp[d] = fit(d, idx, xp, order=o, sf=None if per_donor_level == "bins" else sfs[d])[0]
            pp[d].index = genes
        pm = meta_analysis(pp, se_source="model", alpha=alpha, min_donors=md, method=meta_method)
        perm[key] = {"RE_p": pm.RE_p.to_numpy(), "Q_p": pm.Q_p.to_numpy(), "genes": pm.index.to_numpy(),  # gene set can differ from r.meta
                     "per_donor_p": {d: r.p_value.to_numpy() for d, r in pp.items()}}
        diag.setdefault("lambda_permutation_per_donor", {})[key] = {d: genomic_inflation(r.p_value) for d, r in pp.items()}
        diag.setdefault("lambda_permutation_RE", {})[key] = genomic_inflation(pm.RE_p)
    settings = {"mode": "nb_python", "score_model": score_model, "per_donor_level": per_donor_level,
                "n_bins": n_bins if per_donor_level == "bins" else None, "leave_gene_out": leave_gene_out is not None, "meta_method": meta_method, "threshold": threshold, "min_cells_per_state": min_cells_per_state,
                "min_cells_detected": min_cells_detected, "min_cells_detected_per_state": min_cells_detected_per_state,
                "gene_rule": ("count > 0 in >= min_cells_detected_per_state cells of each state" if thr else
                              "count > 0 in >= min_cells_detected cells, >= 1 on each side of the threshold") + "; non-finite SE -> untested",
                "min_donors_meta": md, "min_donors_meta_rule": "given" if min_donors_meta is not None else "max(3, ceil(D / 3)), D = donors tested",
                "size_factor": size_factor, "alpha": alpha, "n_permutations": n_permutations,
                "permutation_seed": permutation_seed, "quality_covariates": None if qdf is None else list(qdf.columns), "quality_source": q_source,
                "per_donor_model": "NB2 GLM log mu = log s + b0 + b1 x [+ z-scored quality]; IRLS; Wald (Fisher information)",
                "effect": "b1 / ln 2" + (" (log2 change from score 0 to 1)" if not thr else " (disease-like vs healthy-like)"),
                "dispersion": "Cox-Reid adjusted MLE; local median trend (10% window) of log alpha on log mean normalized count; "
                              "log-normal prior, var = max(MAD^2 - trigamma((n - p)/2), 0.25); MAP; genes > 2 SD above trend keep gene-wise",
                "size_factor_method": "pooled deconvolution (scran style, one cluster, pool sizes 21..101 by 5, min mean 0.1)" if size_factor == "deconvolution" else "total counts",
                "tau2_estimator": "DerSimonian-Laird", "Q_weights": "fixed-effect inverse variance", "se_source": "model"}
    return WithinSampleDEResult(per, meta, None, summary, pd.DataFrame(skipped, columns=["donor", "n_cells", "n_disease_like", "n_healthy_like", "reason"]),
                                diag, perm, settings, sfs)
