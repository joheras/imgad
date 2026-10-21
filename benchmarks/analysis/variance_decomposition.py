"""Variance decomposition of AUROC into extractor, detector and preprocessing effects (review items R3/RR3).

For each dataset the unit of analysis is one cell = extractor x detector x preprocessing, with the test AUROC
averaged over seeds (results.csv of each benchmark). Only the feature pipelines enter: 13 distinct extractors
(the RETFound mirror is dropped), 12 trained detectors, 2 preprocessings. Text-prompt detectors and Anomalib
models are not crossed with the other factors and are left out.

Reported per dataset, on the AUROC scale and on the logit scale:
  * eta^2 (share of the total sum of squares) of each factor in a balanced factorial model: extractor, detector,
    preprocessing, extractor x preprocessing, detector x preprocessing, and the residual (extractor x detector
    and the three-way interaction, which have one cell each); and the same within each preprocessing;
  * the difference eta^2(extractor) - eta^2(detector) with a paired bootstrap interval and the bootstrap
    probability that it is positive;
  * each analysis with and without RETFound CFP, the weakest extractor.
Intervals come from a two-way bootstrap that resamples extractors and detectors with replacement.

Usage:
    python benchmarks/analysis/variance_decomposition.py [--boot 2000] [--out benchmarks/analysis/variance_decomposition.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "Private dataset": (ROOT / "benchmarks/ophtalmology/results.csv", "miopia"),
    "PALM": (ROOT / "benchmarks/palm/report/results.csv", "palm-test"),
}

# Benchmark identifiers -> names used in the paper
EXTRACTORS = {
    "retizero-clip": "RetiZero-CLIP",
    "vit_large_patch16_224[cls]": "RetiZero (image)",
    "google/medsiglip-448": "MedSigLIP-448",
    "vit_base_patch16_224[cls]": "VisionFM fundus",
    "vit_small_patch14_reg4_dinov2[mean]": "RETFound-Green",
    "resnet50": "FLAIR (image)",
    "facebook/dinov3-vitb16-pretrain-lvd1689m": "DINOv3 ViT-B/16",
    "vit_base_patch16_224[mean]": "UrFound",
    "google/siglip2-base-patch16-224": "SigLIP 2 B/16",
    "facebook/dinov2-base": "DINOv2-B",
    "vit_large_patch16_224[mean]": "RETFound CFP",
    "flair-clip": "FLAIR-CLIP",
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": "BiomedCLIP",
}
WEAKEST = "RETFound CFP"
PREPS = ["l2", "l2+pca:0.95"]
COMPONENTS = ["extractor", "detector", "preprocessing", "extractor x preprocessing",
              "detector x preprocessing", "residual (extractor x detector, 3-way)"]


def detector_name(d: str) -> str:
    return "ensemble" if d.startswith("ensemble") else d


def load(csv: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df = df[df["error"].isna() | (df["error"].astype(str) == "")]
    df = df[df["dataset"].astype(str).str.split("/").str[-1].eq(split) & df["extractor"].isin(EXTRACTORS)]
    df = df[~df["detector"].str.startswith("zero_shot")]
    df = df.assign(extractor=df["extractor"].map(EXTRACTORS), detector=df["detector"].map(detector_name))
    return df.groupby(["extractor", "detector", "preprocessing"], as_index=False)["auroc"].mean()


def to_array(d: pd.DataFrame, preps) -> tuple:
    exts, dets = sorted(d["extractor"].unique()), sorted(d["detector"].unique())
    y = (d.set_index(["extractor", "detector", "preprocessing"])["auroc"]
         .reindex(pd.MultiIndex.from_product([exts, dets, preps])).to_numpy()
         .reshape(len(exts), len(dets), len(preps)))
    if np.isnan(y).any():
        raise ValueError("unbalanced design: missing cells")
    return y, exts, dets


def eta_squared(y: np.ndarray) -> dict:
    """eta^2 for a full array y[extractor, detector, preprocessing]."""
    gm = y.mean()
    sst = ((y - gm) ** 2).sum()
    n_e, n_d, n_p = y.shape
    me, md, mp = y.mean((1, 2)), y.mean((0, 2)), y.mean((0, 1))
    out = {"extractor": n_d * n_p * ((me - gm) ** 2).sum(), "detector": n_e * n_p * ((md - gm) ** 2).sum()}
    if n_p > 1:
        out["preprocessing"] = n_e * n_d * ((mp - gm) ** 2).sum()
        out["extractor x preprocessing"] = n_d * ((y.mean(1) - gm) ** 2).sum() - out["extractor"] - out["preprocessing"]
        out["detector x preprocessing"] = n_e * ((y.mean(0) - gm) ** 2).sum() - out["detector"] - out["preprocessing"]
    out["residual (extractor x detector, 3-way)"] = sst - sum(out.values())
    return {k: v / sst for k, v in out.items()}


def bootstrap(y: np.ndarray, n: int, rng) -> pd.DataFrame:
    res = []
    for _ in range(n):
        e = rng.integers(0, y.shape[0], y.shape[0])
        k = rng.integers(0, y.shape[1], y.shape[1])
        res.append(eta_squared(y[np.ix_(e, k, np.arange(y.shape[2]))]))
    return pd.DataFrame(res)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def analyse(d: pd.DataFrame, n_boot: int, rng) -> list:
    """Rows: (subset, scale, scope, component, estimate, lo, hi, p_positive)."""
    rows = []
    for subset, sub in [("all 13 extractors", d), (f"without {WEAKEST}", d[d["extractor"] != WEAKEST])]:
        for scale, f in [("AUROC", lambda v: v), ("logit(AUROC)", logit)]:
            for scope, preps in [("both preprocessings", PREPS), ("L2 only", ["l2"]), ("L2 + PCA only", ["l2+pca:0.95"])]:
                y, _, _ = to_array(sub[sub["preprocessing"].isin(preps)], preps)
                y = f(y)
                est = eta_squared(y)
                boot = bootstrap(y, n_boot, rng)
                for comp in COMPONENTS:
                    if comp in est:
                        rows.append((subset, scale, scope, comp, est[comp],
                                     boot[comp].quantile(0.025), boot[comp].quantile(0.975), np.nan))
                diff = boot["extractor"] - boot["detector"]
                rows.append((subset, scale, scope, "eta2(extractor) - eta2(detector)",
                             est["extractor"] - est["detector"], diff.quantile(0.025), diff.quantile(0.975),
                             (diff > 0).mean()))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("variance_decomposition.md"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    md = ["# Variance decomposition of AUROC (review items R3 and RR3)", "",
          "Share of the total sum of squares (eta^2) over extractor x detector x preprocessing cells of test AUROC "
          f"averaged over seeds, from each benchmark's results.csv. 95% intervals: two-way bootstrap over extractors "
          f"and detectors ({a.boot} replicates, seed {a.seed}). P(>0): bootstrap probability that the extractor "
          "explains more than the detector.", ""]
    for name, (csv, split) in DATASETS.items():
        rows = analyse(load(csv, split), a.boot, rng)
        for subset in dict.fromkeys(r[0] for r in rows):
            for scale in ("AUROC", "logit(AUROC)"):
                md += [f"## {name}, {subset}, {scale} scale", "",
                       "| Scope | Component | eta^2 | 95% interval | P(>0) |", "| --- | --- | ---: | ---: | ---: |"]
                for s, sc, scope, comp, v, lo, hi, p in rows:
                    if s == subset and sc == scale:
                        pp = "" if np.isnan(p) else f"{p:.2f}"
                        md.append(f"| {scope} | {comp} | {v:.3f} | {lo:.3f} to {hi:.3f} | {pp} |")
                md.append("")
    text = "\n".join(md)
    a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
