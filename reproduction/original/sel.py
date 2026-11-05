"""Datasets of this run: T2D Beta + the AD classes selected under gene filter v2 (pre-specified: held-out exact two-sided P < 0.05;
genefilter_v2_rerun_20261031_v1/tables/fig5_class_selection.tsv), and the existing case-control comparator run of each AD class."""
from pathlib import Path
import pandas as pd
DG = Path(__file__).resolve().parents[2]; G2 = DG / "genefilter_v2_rerun_20261031_v1"
CLASSES = ["Ast", "Exc", "Inh", "Mic_Immune", "OPC", "Oli", "Vasc"]
key = lambda reg, cls: f"{reg}_{cls.lower()}"
label = lambda reg, cls: f"{reg.upper()} {cls.replace('_', '/')}"
ALL = {"t2d_beta": ("t2d", "beta", "T2D", "T2D Beta"), **{key(r, c): (r, c, "lateAD", f"AD {label(r, c)}") for r in ("ec", "pfc") for c in CLASSES}}


def selected():
    s = pd.read_csv(G2 / "tables/fig5_class_selection.tsv", sep="\t"); return [(r, c) for r, c in zip(s.dataset, s["class"])]


def datasets():
    return {"t2d_beta": ALL["t2d_beta"], **{key(r, c): ALL[key(r, c)] for r, c in selected()}}


def comparator(reg, cls):
    if reg == "pfc": return DG / f"ad_lodo_deg_campaign_20261023_v1/deg/{cls}"
    return DG / ("ad_ec_oli_pipeline_20261027_v1/campaign/deg/Oli" if cls == "Oli" else f"ad_ec_v2_within_donor_20261028_v1/campaign/deg/{cls}")
