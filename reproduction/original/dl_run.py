#!/usr/bin/env python3
"""ANALYSIS_PLAN.md Part 2 step 3: DL within-donor test with per-donor slopes and bin ranges stored. --data t2d_beta|<region>_<class> (runs only if selected).
Core = v2_slope_heterogeneity_20261031_v1/scripts/robust.py::core (same LGO bins, size factors, v2 MAP dispersion, per-donor NB fits, DL meta),
copied here to also return the per-donor slopes and (bin 6 - bin 1) LGO score ranges."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[v] = "4"
import argparse, json, sys, time
from pathlib import Path
import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp
from scipy import stats
from threadpoolctl import threadpool_limits
PKG = Path(__file__).resolve().parents[1]; DG = PKG.parent; GF = DG / "genefilter_v2_rerun_20261031_v1"
sys.path.insert(0, str(GF / "scripts")); import downstream as ds  # noqa: E402
wdn = ds.wdn; K = 6; LN2 = np.log(2)
sys.path.insert(0, str(Path(__file__).resolve().parent)); import sel  # noqa: E402
DATA = {k: v[:3] for k, v in sel.ALL.items()}  # T2D Beta + any AD class; only selected ones are run


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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", choices=list(DATA), required=True); a = ap.parse_args(); t0 = time.time(); data, cls, cond = DATA[a.data]
    if a.data not in sel.datasets(): print(f"{a.data}: not selected; skipped"); return
    out = PKG / f"work/dl/{a.data}"; out.mkdir(parents=True, exist_ok=False)
    tab, h5, genes, _ = ds.inputs(data, cls); tab = tab[tab.condition == cond]; sep = "," if data == "t2d" else "\t"
    s = pd.concat([pd.read_csv(r.pred, sep=sep, usecols=["cell_id", "disease_like_probability"]).assign(donor_id=r.donor) for r in tab.itertuples()])
    raw = ad.read_h5ad(h5, backed="r"); rix = pd.Index(raw.obs_names.astype(str)).get_indexer(s.cell_id.astype(str)); assert (rix >= 0).all()
    o = np.argsort(rix); full = sp.csr_matrix(raw.X[rix[o]]); full.data = np.rint(full.data); s = s.iloc[o].reset_index(drop=True)
    vn = raw.var_names.astype(str).tolist(); gi = pd.Index(vn).get_indexer(genes); C = sp.csr_matrix(full[:, gi].astype(np.int64))
    donor = s.donor_id.astype(str).to_numpy(); score = s.disease_like_probability.to_numpy(float); st = tab.set_index("donor").state
    with threadpool_limits(limits=4):
        lg = {d: wdn.LeaveGeneOutScores.from_compact_state(ds.load_compact_state(st[d]), full[np.flatnonzero(donor == d)], vn, genes) for d in sorted(set(donor))}
        t, Bd, R, info = core(C, donor, score, genes, lg)
    t.to_csv(out / "genes.tsv.gz", sep="\t"); Bd.to_csv(out / "per_donor_slope_log2.tsv.gz", sep="\t"); R.to_csv(out / "per_donor_bin6_minus_bin1.tsv.gz", sep="\t")
    sg = t.DL_BH < .05; info.update({"data": a.data, "DL_up": int((sg & (t.DL_log2 > 0)).sum()), "DL_down": int((sg & (t.DL_log2 < 0)).sum()), "genes_tested": int(t.tested.sum()), "seconds": round(time.time() - t0, 1)})
    (out / "run_info.json").write_text(json.dumps(info, indent=1, default=str) + "\n"); print({k: v for k, v in info.items() if k != "v2_info"})


if __name__ == "__main__":
    main()
