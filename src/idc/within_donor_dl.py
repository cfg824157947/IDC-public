"""Final manuscript DL test. Scientific function bodies copied unchanged from dl_run.py.
Use only disease-donor cells with frozen held-out scores and matching LGO objects.
The auxiliary joint-model P values are NOT the final DL test.
"""
import numpy as np
import pandas as pd
from scipy import stats
from . import within_donor_nb as wdn
K = 6
LN2 = np.log(2)

def bins_of(v):
    o = np.argsort(v, axis=0, kind="stable"); r = np.empty_like(o)
    np.put_along_axis(r, o, np.arange(v.shape[0])[:, None] if v.ndim == 2 else np.arange(len(v)), axis=0); return (r * K) // v.shape[0]

def core(C, donor, score, genes, lg):
    v2, info = wdn.joint_binned_de(C, donor, score, genes, n_bins=K, leave_gene_out=lg)
    dn = info["donors"]; D = len(dn); S = D * K; G = len(genes); groups = {d: np.flatnonzero(donor == d) for d in dn}; Yfull = np.zeros((S, G))
    for j, d in enumerate(dn):
        b = bins_of(score[groups[d]]); Cd = C[groups[d]]
        for k in range(K): Yfull[j * K + k] = np.asarray(Cd[b == k].sum(0)).ravel()
    off = np.log(wdn.median_of_ratios(Yfull)); ones = np.ones((K, 1)); alpha = v2.alpha_map.to_numpy(); tested = np.flatnonzero(v2.tested.to_numpy())
    B = np.full((G, D), np.nan); SE = np.full((G, D), np.nan); RNG = np.full((G, D), np.nan)
    for g0 in range(0, len(tested), 1000):
        g = tested[g0:g0 + 1000]; Yk = np.zeros((S, len(g))); Xk = np.zeros((S, len(g))); al = alpha[g]
        for j, d in enumerate(dn):
            Sg = lg[d](g); bb = bins_of(Sg); Yg = C[groups[d]][:, g].toarray()
            for k in range(K): m = bb == k; Yk[j * K + k] = (Yg * m).sum(0); Xk[j * K + k] = (Sg * m).sum(0) / np.maximum(m.sum(0), 1)
        for j in range(D):
            r = slice(j * K, (j + 1) * K); yy = Yk[r]; ok = (yy.sum(0) >= 3) & ((yy > 0).sum(0) >= 2); RNG[g, j] = Xk[r][K - 1] - Xk[r][0]
            Dd = wdn._Design(ones, Xk[r], 1); bdd, _, _ = wdn._irls(yy, Dd, off[r], al); mud = wdn._mu(Dd, off[r], bdd)
            cov = np.linalg.inv(Dd.xtwx(mud / (1 + al * mud))); bb_ = np.where(ok, bdd[:, 1], np.nan); se_ = np.where(ok, np.sqrt(cov[:, 1, 1]), np.nan); B[g, j] = bb_; SE[g, j] = se_
    n = np.isfinite(B).sum(1); W = 1 / SE ** 2; fe = np.nansum(W * B, 1) / np.nansum(W, 1); Q = np.nansum(W * (B - fe[:, None]) ** 2, 1)
    c = np.nansum(W, 1) - np.nansum(W ** 2, 1) / np.nansum(W, 1); tau2 = np.maximum((Q - (n - 1)) / np.where(c > 0, c, np.nan), 0)
    wr = 1 / (SE ** 2 + tau2[:, None]); est = np.nansum(wr * B, 1) / np.nansum(wr, 1); se = 1 / np.sqrt(np.nansum(wr, 1)); z = est / se; p = 2 * stats.norm.sf(np.abs(z))
    t = pd.DataFrame({"DL_log2": est / LN2, "DL_SE_log2": se / LN2, "DL_z": z, "DL_p": p, "n_donors_fit": n, "I2": np.where(Q > 0, np.maximum((Q - (n - 1)) / Q, 0), 0),
                      "v2_log2FC": v2.log2FC.to_numpy(), "v2_BH": v2.adjusted_p_value.to_numpy(), "tested": v2.tested.to_numpy() & (n >= 2)}, index=genes)
    ok = t.tested & t.DL_p.notna(); t["DL_BH"] = np.nan; t.loc[ok, "DL_BH"] = stats.false_discovery_control(t.loc[ok, "DL_p"])
    t.loc[~ok, ["DL_log2", "DL_z", "DL_p"]] = np.nan
    factor = float(np.nanmean(np.nanmedian(RNG[tested], axis=0))); t["delta_log2"] = t.DL_log2 * factor
    return t, pd.DataFrame(B / LN2, index=genes, columns=dn), pd.DataFrame(RNG, index=genes, columns=dn), {"donors": dn, "n_donors": D, "delta_factor": factor, "v2_info": info}

within_donor_test = core
