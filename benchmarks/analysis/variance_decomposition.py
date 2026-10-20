"""Variance decomposition of AUROC into extractor, detector and preprocessing effects (review item R3).

For each dataset the unit of analysis is one cell = extractor x detector x preprocessing, with the AUROC
averaged over seeds (the private dataset only has seed means, in its report's extractor x detector table;
PALM uses its test split from results.csv, averaged the same way). Only the feature pipelines enter: 13
distinct extractors (the RETFound mirror is dropped), 12 trained detectors, 2 preprocessings. Text-prompt
detectors and Anomalib models are not crossed with the other factors and are left out.

Reported per dataset:
  * eta^2 (share of the total sum of squares) of each factor and interaction in a balanced factorial
    model: extractor, detector, preprocessing, extractor x preprocessing, detector x preprocessing, and the
    residual (extractor x detector and the three-way interaction, which have one cell each);
  * the same within each preprocessing (two-way: extractor, detector, residual = extractor x detector);
  * each analysis with and without RETFound CFP, the weakest extractor;
  * 95% intervals from a two-way bootstrap that resamples extractors and detectors with replacement.

Usage:
    python benchmarks/analysis/variance_decomposition.py [--boot 2000] [--out benchmarks/analysis/variance_decomposition.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PALM_CSV = ROOT / "benchmarks/palm/report/results.csv"
PRIVATE_REPORT = ROOT / "benchmarks/ophtalmology/report.md"

# Report/CSV identifiers -> names used in the paper
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
MIRROR = "hf-hub:bitfount/RETFound_MAE[mean]"
WEAKEST = "RETFound CFP"


def detector_name(d: str) -> str:
    return "ensemble" if d.startswith("ensemble") else d


def load_palm() -> pd.DataFrame:
    df = pd.read_csv(PALM_CSV)
    df = df[(df["error"].isna() | (df["error"].astype(str) == ""))]
    df = df[df["dataset"].str.endswith("palm-test") & df["extractor"].isin(EXTRACTORS)]
    df = df[~df["detector"].str.startswith("zero_shot")]
    df = df.assign(extractor=df["extractor"].map(EXTRACTORS), detector=df["detector"].map(detector_name))
    return df.groupby(["extractor", "detector", "preprocessing"], as_index=False)["auroc"].mean()


def load_private() -> pd.DataFrame:
    """Parse the 'Mean AUROC: extractor x detector' table of the private dataset's report."""
    lines = PRIVATE_REPORT.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("| Detector | vit_small"))
    header = [c.strip() for c in lines[start].strip("|").split("|")]
    # column labels look like "vit_base_patch16_224 · cls] l2" or "resnet50  · l2+pca:0.95"
    labels = {
        "vit_small_patch14_reg4_dinov2 · mean]": "vit_small_patch14_reg4_dinov2[mean]",
        "vit_base_patch16_224 · cls]": "vit_base_patch16_224[cls]",
        "vit_base_patch16_224 · mean]": "vit_base_patch16_224[mean]",
        "vit_large_patch16_224 · cls]": "vit_large_patch16_224[cls]",
        "vit_large_patch16_224 · mean]": "vit_large_patch16_224[mean]",
        "RETFound_MAE · mean]": MIRROR,
        "resnet50  ·": "resnet50",
        "flair-clip  ·": "flair-clip",
        "retizero-clip  ·": "retizero-clip",
        "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224  ·": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "dinov2-base  ·": "facebook/dinov2-base",
        "siglip2-base-patch16-224  ·": "google/siglip2-base-patch16-224",
        "medsiglip-448  ·": "google/medsiglip-448",
        "dinov3-vitb16-pretrain-lvd1689m  ·": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    }
    cols = []
    for h in header[1:]:
        prep = "l2+pca:0.95" if h.endswith("l2+pca:0.95") else "l2"
        stem = h[: -len(prep)].rstrip()
        key = next(v for k, v in labels.items() if stem == k.rstrip() or stem == k)
        cols.append((key, prep))
    rows = []
    for l in lines[start + 2:]:
        if not l.startswith("|"):
            break
        cells = [c.strip().strip("*") for c in l.strip("|").split("|")]
        det = cells[0]
        if det in ("mean", "") or det.startswith("zero_shot"):
            continue
        for (ext, prep), v in zip(cols, cells[1:]):
            if ext in EXTRACTORS and v not in ("–", ""):
                rows.append((EXTRACTORS[ext], detector_name(det), prep, float(v)))
    return pd.DataFrame(rows, columns=["extractor", "detector", "preprocessing", "auroc"])


def eta_squared(d: pd.DataFrame, by_prep: bool) -> dict:
    """Shares of the total sum of squares in a balanced factorial design (one value per cell)."""
    y = d["auroc"].to_numpy()
    gm = y.mean()
    sst = ((y - gm) ** 2).sum()

    def ss(cols):
        m = d.groupby(cols)["auroc"].transform("mean").to_numpy()
        return ((m - gm) ** 2).sum()

    out = {"extractor": ss(["extractor"]), "detector": ss(["detector"])}
    if by_prep:
        out["preprocessing"] = ss(["preprocessing"])
        out["extractor x preprocessing"] = ss(["extractor", "preprocessing"]) - out["extractor"] - out["preprocessing"]
        out["detector x preprocessing"] = ss(["detector", "preprocessing"]) - out["detector"] - out["preprocessing"]
    out["residual (extractor x detector, 3-way)"] = sst - sum(out.values())
    return {k: v / sst for k, v in out.items()}


def ranges(d: pd.DataFrame) -> tuple:
    """The paper's original range-based effects, computed within one preprocessing."""
    p = d.pivot_table(index="detector", columns="extractor", values="auroc")
    return (p.max() - p.min()).mean(), (p.max(axis=1) - p.min(axis=1)).mean()


def eta_squared_array(y: np.ndarray) -> dict:
    """eta^2 for a full array y[extractor, detector, preprocessing] (same components as eta_squared)."""
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


def bootstrap(d: pd.DataFrame, by_prep: bool, n: int, rng) -> pd.DataFrame:
    """Two-way bootstrap: resample extractors and detectors with replacement (duplicates are distinct levels)."""
    preps = sorted(d["preprocessing"].unique()) if by_prep else [d["preprocessing"].iloc[0]]
    exts, dets = sorted(d["extractor"].unique()), sorted(d["detector"].unique())
    y = (d.set_index(["extractor", "detector", "preprocessing"])["auroc"]
         .reindex(pd.MultiIndex.from_product([exts, dets, preps])).to_numpy().reshape(len(exts), len(dets), len(preps)))
    res = []
    for _ in range(n):
        e = rng.integers(0, len(exts), len(exts))
        k = rng.integers(0, len(dets), len(dets))
        res.append(eta_squared_array(y[np.ix_(e, k, np.arange(len(preps)))]))
    return pd.DataFrame(res)


def analyse(name: str, d: pd.DataFrame, n_boot: int, rng) -> list:
    blocks = []
    for label, sub in [("all extractors", d), (f"without {WEAKEST}", d[d["extractor"] != WEAKEST])]:
        n_e, n_d = sub["extractor"].nunique(), sub["detector"].nunique()
        rows = []
        for scope, part, by_prep in [("both preprocessings", sub, True),
                                     ("L2 only", sub[sub["preprocessing"] == "l2"], False),
                                     ("L2 + PCA only", sub[sub["preprocessing"] == "l2+pca:0.95"], False)]:
            est = eta_squared(part, by_prep)
            boot = bootstrap(part, by_prep, n_boot, rng) if n_boot else None
            for k, v in est.items():
                lo, hi = (boot[k].quantile(0.025), boot[k].quantile(0.975)) if boot is not None else (np.nan, np.nan)
                rows.append((scope, k, v, lo, hi))
            if not by_prep:
                de, ee = ranges(part)
                rows.append((scope, "range effect: detector / extractor (paper's original measure)", de, ee, np.nan))
        blocks.append((f"{name}, {label} ({n_e} extractors x {n_d} detectors)", rows))
    return blocks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap replicates (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("variance_decomposition.md"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    md = ["# Variance decomposition of AUROC (review item R3)", "",
          "Share of the total sum of squares (eta^2) over extractor x detector x preprocessing cells, AUROC averaged "
          f"over seeds. 95% intervals: two-way bootstrap over extractors and detectors ({a.boot} replicates).", ""]
    for name, d in [("Private dataset (test set)", load_private()), ("PALM (test set)", load_palm())]:
        for title, rows in analyse(name, d, a.boot, rng):
            md += [f"## {title}", "", "| Scope | Component | eta^2 / value | 95% interval |", "| --- | --- | ---: | ---: |"]
            for scope, comp, v, lo, hi in rows:
                if comp.startswith("range"):
                    md.append(f"| {scope} | range effect, detector / extractor | {v:.3f} / {lo:.3f} | |")
                else:
                    md.append(f"| {scope} | {comp} | {v:.3f} | {lo:.3f}–{hi:.3f} |")
            md.append("")
    text = "\n".join(md)
    a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
