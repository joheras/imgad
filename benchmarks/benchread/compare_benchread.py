"""Choose imgad configurations on BenchReAD's validation set and compare them with BenchReAD's published numbers.

Reads the ``results.csv`` files written by ``imgad benchmark --out`` for ``configs/benchread_benchmark.yaml`` (and
``configs/benchread_anomalib.yaml``) and prints a markdown report with:

  * the configuration with the best mean validation AUROC (overall and per family), and its AUROC on RIADD and JSIEC;
  * a configuration fixed in advance (by default the one chosen on PALM), which uses no BenchReAD anomaly at all;
  * the oracle per test set (best on that test set, optimistic) and the validation-test rank correlation;
  * a table of these rows next to BenchReAD's published AUROCs (Fig. 2 of the paper), with each method's supervision;
  * with ``--scores`` and ``--data``: F1, specificity and sensitivity at imgad's threshold (calibrated on normal
    images only) for every anomaly category, averaged as in BenchReAD's Table 1, next to the published values.

Usage:
    python benchmarks/benchread/compare_benchread.py benchmarks/benchread/results.csv \\
        [benchmarks/benchread/results_anomalib.csv] [--scores scores.joblib [scores_anomalib.joblib] \\
        --data /data/benchread_imgad] [--out comparison.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

VAL, RIADD, JSIEC = "benchread-val", "benchread-riadd", "benchread-jsiec"
TESTS = (RIADD, JSIEC)
TEST_NAMES = {RIADD: "RIADD", JSIEC: "JSIEC"}
PALM_SELECTED = "vit_large_patch16_224[cls] | l2 | lof"  # RetiZero image encoder + L2 + LOF, chosen on PALM

# BenchReAD, Fig. 2: AUROC (%) normal vs all anomalies with 95% CI; supervision level of each method
METHODS = ["SoftPatch", "EDC", "SimpleNet", "PatchCore", "DDAD-ASR", "DRA", "NFM-DRA"]
SUPERVISION = {"SoftPatch": "unsupervised", "EDC": "one-class", "SimpleNet": "one-class", "PatchCore": "one-class",
               "DDAD-ASR": "semi-supervised", "DRA": "supervised", "NFM-DRA": "supervised"}
PUBLISHED_AUROC = {
    RIADD: {"SoftPatch": (75.4, 73.5, 77.3), "EDC": (58.3, 55.6, 60.9), "SimpleNet": (68.7, 66.4, 70.7),
            "PatchCore": (77.0, 75.1, 78.8), "DDAD-ASR": (64.7, 62.1, 67.2), "DRA": (85.6, 84.1, 87.1),
            "NFM-DRA": (89.0, 87.7, 90.2)},
    JSIEC: {"SoftPatch": (90.9, 86.7, 94.7), "EDC": (80.0, 72.6, 86.4), "SimpleNet": (76.6, 67.7, 85.0),
            "PatchCore": (92.3, 88.1, 95.8), "DDAD-ASR": (68.2, 56.7, 77.8), "DRA": (89.2, 85.9, 92.1),
            "NFM-DRA": (96.7, 94.8, 98.2)},
}
SEEN = {RIADD: {"DR", "AMD", "ODC", "MYA", "DN"}, JSIEC: {"DR", "LOC", "PM"}}
TYPE_NAMES = {"BF_wo_PDR": "BF w/o PDR", "BF_w_PDR": "BF w/ PDR"}  # folder name -> BenchReAD's table name

# BenchReAD, Table 1: (F1, specificity, sensitivity) % of each method in METHODS, normal vs each category. "Average"
# is the mean of the category rows and the pooled "All abnormal" row (exact for RIADD; JSIEC prints no pooled row)
TABLE1 = {
    "benchread-riadd": {
        "All abnormal": [(82.7, 5.8, 97.7), (83.0, 0.0, 100.0), (83.0, 0.9, 99.8), (82.4, 13.6, 94.9), (37.4, 88.6, 24.1), (79.3, 90.3, 68.3), (70.0, 99.3, 54.0)],
        "DR": [(54.8, 5.8, 97.0), (54.5, 0.0, 100.0), (54.6, 0.9, 99.8), (56.1, 13.6, 95.3), (26.5, 88.6, 18.2), (90.0, 90.3, 95.0), (93.2, 99.3, 88.3)],
        "AMD": [(18.0, 5.8, 100.0), (17.1, 0.0, 100.0), (17.2, 0.9, 100.0), (19.3, 13.6, 100.0), (33.3, 88.6, 42.0), (64.6, 90.3, 92.8), (87.9, 99.3, 84.1)],
        "ODC": [(30.8, 5.8, 92.3), (31.7, 0.0, 100.0), (31.7, 0.9, 99.4), (29.7, 13.6, 82.6), (22.3, 88.6, 18.7), (50.8, 90.3, 48.4), (28.9, 99.3, 17.4)],
        "MYA": [(19.2, 5.8, 100.0), (18.3, 0.0, 100.0), (18.5, 0.9, 100.0), (20.6, 13.6, 100.0), (32.2, 88.6, 38.7), (68.5, 90.3, 97.3), (90.4, 99.3, 88.0)],
        "DN": [(31.5, 5.8, 96.1), (31.2, 0.0, 100.0), (31.3, 0.9, 99.3), (30.6, 13.6, 86.8), (27.9, 88.6, 24.3), (67.7, 90.3, 73.0), (60.4, 99.3, 44.7)],
        "MH": [(50.0, 5.8, 100.0), (48.5, 0.0, 100.0), (48.7, 0.9, 100.0), (51.9, 13.6, 99.4), (38.8, 88.6, 29.8), (42.7, 90.3, 32.7), (35.9, 99.3, 22.2)],
        "BRVO": [(20.7, 5.8, 100.0), (19.7, 0.0, 100.0), (19.8, 0.9, 100.0), (21.9, 13.6, 98.8), (14.1, 88.6, 14.6), (66.4, 90.3, 89.0), (85.5, 99.3, 79.3)],
        "TSLN": [(11.3, 5.8, 100.0), (10.7, 0.0, 100.0), (10.5, 0.9, 97.5), (12.2, 13.6, 100.0), (18.8, 88.6, 30.0), (33.3, 90.3, 52.5), (30.2, 99.3, 20.0)],
        "CSR": [(13.9, 5.8, 94.4), (13.9, 0.0, 100.0), (14.0, 0.9, 100.0), (13.8, 13.6, 87.0), (12.9, 88.6, 16.7), (32.4, 90.3, 42.6), (31.4, 99.3, 20.4)],
        "CRVO": [(11.8, 5.8, 100.0), (11.2, 0.0, 100.0), (11.2, 0.9, 100.0), (12.7, 13.6, 100.0), (5.0, 88.6, 7.1), (54.4, 90.3, 95.2), (85.4, 99.3, 83.3)],
        "ODP": [(12.0, 5.8, 97.7), (11.6, 0.0, 100.0), (11.7, 0.9, 100.0), (12.1, 13.6, 90.9), (6.5, 88.6, 9.1), (31.0, 90.3, 45.5), (25.0, 99.3, 15.9)],
        "ODE": [(20.5, 5.8, 100.0), (19.5, 0.0, 100.0), (19.6, 0.9, 100.0), (21.9, 13.6, 100.0), (24.6, 88.6, 27.2), (36.9, 90.3, 40.7), (45.0, 99.3, 30.9)],
        "RS": [(17.5, 5.8, 100.0), (16.7, 0.0, 100.0), (16.8, 0.9, 100.0), (18.8, 13.6, 100.0), (32.7, 88.6, 41.8), (60.3, 90.3, 85.1), (86.6, 99.3, 82.1)],
        "CRS": [(9.7, 5.8, 100.0), (9.2, 0.0, 100.0), (9.3, 0.9, 100.0), (10.5, 13.6, 100.0), (13.6, 88.6, 23.5), (44.1, 90.3, 82.4), (78.1, 99.3, 73.5)],
        "RPEC": [(5.9, 5.8, 87.0), (6.4, 0.0, 100.0), (6.5, 0.9, 100.0), (6.4, 13.6, 87.0), (7.8, 88.6, 17.4), (27.5, 90.3, 60.9), (44.4, 99.3, 34.8)],
        "Average": [(25.6, 5.8, 97.6), (25.2, 0.0, 100.0), (25.3, 0.9, 99.7), (26.3, 13.6, 95.2), (22.1, 88.6, 24.0), (53.1, 90.3, 68.8), (61.1, 99.3, 52.4)],
    },
    "benchread-jsiec": {
        "DR": [(81.9, 10.5, 91.5), (84.8, 0.0, 100.0), (86.1, 13.2, 99.1), (84.3, 21.1, 93.4), (70.8, 47.4, 65.1), (90.9, 94.7, 84.9), (90.7, 100.0, 83.0)],
        "LOC": [(74.6, 10.5, 100.0), (72.5, 0.0, 100.0), (75.2, 13.2, 100.0), (76.9, 21.1, 100.0), (83.3, 47.4, 100.0), (71.6, 94.7, 58.0), (46.2, 100.0, 30.0)],
        "PM": [(76.1, 10.5, 100.0), (74.0, 0.0, 100.0), (76.6, 13.2, 100.0), (78.3, 21.1, 100.0), (79.7, 47.4, 90.7), (98.2, 94.7, 100.0), (100.0, 100.0, 100.0)],
        "BRVO": [(72.1, 10.5, 100.0), (69.8, 0.0, 100.0), (72.7, 13.2, 100.0), (74.6, 21.1, 100.0), (76.9, 47.4, 90.9), (97.8, 94.7, 100.0), (100.0, 100.0, 100.0)],
        "CRVO": [(56.4, 10.5, 100.0), (53.7, 0.0, 100.0), (57.1, 13.2, 100.0), (59.5, 21.1, 100.0), (66.7, 47.4, 95.5), (95.7, 94.7, 100.0), (100.0, 100.0, 100.0)],
        "RRD": [(77.0, 10.5, 100.0), (75.0, 0.0, 100.0), (77.6, 13.2, 100.0), (79.2, 21.1, 100.0), (85.1, 47.4, 100.0), (68.9, 94.7, 54.4), (77.4, 100.0, 63.2)],
        "MAC": [(81.3, 10.5, 100.0), (79.6, 0.0, 100.0), (81.8, 13.2, 100.0), (83.1, 21.1, 100.0), (88.1, 47.4, 100.0), (98.0, 94.7, 98.6), (99.3, 100.0, 98.6)],
        "ERM": [(60.5, 10.5, 100.0), (57.8, 0.0, 100.0), (61.2, 13.2, 100.0), (63.4, 21.1, 100.0), (70.4, 47.4, 96.2), (90.2, 94.7, 88.5), (87.0, 100.0, 76.9)],
        "MH": [(57.5, 10.5, 100.0), (54.8, 0.0, 100.0), (58.2, 13.2, 100.0), (60.5, 21.1, 100.0), (65.6, 47.4, 91.3), (81.0, 94.7, 73.9), (75.7, 100.0, 60.9)],
        "RP": [(56.4, 10.5, 100.0), (53.7, 0.0, 100.0), (57.1, 13.2, 100.0), (59.5, 21.1, 100.0), (68.8, 47.4, 100.0), (95.7, 94.7, 100.0), (100.0, 100.0, 100.0)],
        "YWSF": [(63.0, 10.5, 100.0), (60.4, 0.0, 100.0), (63.7, 13.2, 100.0), (65.9, 21.1, 100.0), (71.1, 47.4, 93.1), (96.7, 94.7, 100.0), (100.0, 100.0, 100.0)],
        "LS": [(54.1, 10.5, 100.0), (51.3, 0.0, 100.0), (54.8, 13.2, 100.0), (57.1, 21.1, 100.0), (66.7, 47.4, 100.0), (95.2, 94.7, 100.0), (100.0, 100.0, 100.0)],
        "BF w/o PDR": [(87.0, 10.5, 100.0), (85.3, 0.0, 99.1), (87.4, 13.2, 100.0), (88.4, 21.1, 100.0), (86.0, 47.4, 88.6), (29.4, 94.7, 17.5), (44.9, 100.0, 28.9)],
        "BF w/ PDR": [(72.6, 10.5, 100.0), (70.3, 0.0, 100.0), (73.2, 13.2, 100.0), (75.0, 21.1, 100.0), (70.0, 47.4, 77.8), (65.7, 94.7, 51.1), (78.4, 100.0, 64.4)],
        "Average": [(71.2, 10.5, 99.3), (69.3, 0.0, 99.9), (72.0, 13.2, 99.9), (73.5, 21.1, 99.5), (76.1, 47.4, 91.9), (83.9, 94.7, 80.0), (85.6, 100.0, 78.6)],
    },
}


def family(row) -> str:
    if row["extractor"] == "anomalib":
        return "Anomalib model"
    if str(row["detector"]).startswith("zero_shot"):
        return "Text prompts"
    return "Extractor + detector"


def split_of(dataset: str) -> str:
    return str(dataset).split("/")[-1]


def load_results(paths: List[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "error" in df.columns:
        failed = df["error"].notna() & (df["error"].astype(str) != "")
        if failed.any():
            print(f"({int(failed.sum())} failed runs ignored)", file=sys.stderr)
        df = df[~failed]
    df = df[df["auroc"].notna()].copy()
    df["split"] = df["dataset"].map(split_of)
    df["family"] = df.apply(family, axis=1)
    return df[df["split"].isin([VAL, *TESTS])]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration: mean/std AUROC per split and the mean bootstrap CI of each test split."""
    g = df.groupby(["config", "family", "split"])
    agg = g["auroc"].agg(["mean", "std"])
    for c in ("auroc_ci_low", "auroc_ci_high", "f1@thr", "tpr@thr", "fpr@thr"):
        if c in df.columns:
            agg[c] = g[c].mean()
    wide = agg.reset_index().pivot_table(index=["config", "family"], columns="split")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    return wide.reset_index().dropna(subset=[f"mean_{s}" for s in (VAL, *TESTS)])


def pct(x: float) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{100 * x:.1f}"


def cell(text: str) -> str:
    """Configuration labels use '|' as separator, which would split a markdown table cell."""
    return str(text).replace("|", "\\|")


def auroc_cell(r: pd.Series, split: str) -> str:
    out = f"{pct(r[f'mean_{split}'])}"
    if f"auroc_ci_low_{split}" in r and np.isfinite(r[f"auroc_ci_low_{split}"]):
        out += f" ({pct(r[f'auroc_ci_low_{split}'])}–{pct(r[f'auroc_ci_high_{split}'])})"
    sd = r.get(f"std_{split}", np.nan)
    if np.isfinite(sd) and sd > 0.0005:
        out += f" ± {pct(sd)}"
    return out


def chosen_rows(s: pd.DataFrame, preselected: Optional[str]) -> List[tuple]:
    """(label, row) for the configurations compared with BenchReAD."""
    out = [("imgad, best on validation", s.loc[s[f"mean_{VAL}"].idxmax()])]
    for fam, sub in s.groupby("family"):
        out.append((f"imgad {fam.lower()}, best on validation", sub.loc[sub[f"mean_{VAL}"].idxmax()]))
    if preselected:
        hit = s[s["config"] == preselected]
        if len(hit):
            out.append(("imgad, fixed in advance (chosen on PALM)", hit.iloc[0]))
        else:
            print(f"(preselected configuration {preselected!r} not in the results)", file=sys.stderr)
    pc = s[s["config"].str.contains("patchcore(image_size=224", regex=False)]
    if len(pc):
        out.append(("imgad Anomalib PatchCore, BenchReAD settings", pc.iloc[0]))
    return out


# --------------------------------------------------------------------------------------------- per category
def test_types(data_root: Path, category: str) -> List[str]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from imgad.data import mvtec_ad

    return mvtec_ad(data_root, category).test_types


def threshold_metrics(scores: np.ndarray, labels: np.ndarray, types: np.ndarray, thr: float) -> Dict[str, tuple]:
    """{category: (F1, specificity, sensitivity)} at the threshold, normal vs each category and vs all anomalies."""
    pred = scores > thr
    normal = labels == 0
    spc = float((~pred[normal]).mean())
    out = {}
    groups = {TYPE_NAMES.get(t, t): types == t for t in sorted(set(types[labels == 1]))}
    groups["All abnormal"] = labels == 1
    for name, m in groups.items():
        tp, fn = int(pred[m].sum()), int((~pred[m]).sum())
        fp = int(pred[normal].sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
        out[name] = (f1, spc, tp / (tp + fn))
    out["Average"] = tuple(float(np.mean([v[k] for v in out.values()])) for k in range(3))
    return out


def per_category(score_rows: List[dict], data_root: Path, configs: List[str]) -> Dict[tuple, Dict[str, tuple]]:
    """{(config, split): {category: mean (F1, SPC, SEN) over seeds}}."""
    types = {sp: np.asarray(test_types(data_root, sp)) for sp in TESTS}
    acc: Dict[tuple, List[Dict[str, tuple]]] = {}
    for r in score_rows:
        sp = split_of(r.get("dataset", ""))
        if sp not in TESTS or r.get("config") not in configs or "scores" not in r or r.get("error"):
            continue
        t = types[sp]
        if len(t) != len(r["scores"]):
            raise SystemExit(f"--data {data_root}: {sp} has {len(t)} test images but the scores have "
                             f"{len(r['scores'])}; pass the folder the benchmark was run on.")
        acc.setdefault((r["config"], sp), []).append(
            threshold_metrics(np.asarray(r["scores"]), np.asarray(r["labels"]), t, float(r["threshold"])))
    return {k: {c: tuple(float(np.mean([m[c][i] for m in ms])) for i in range(3)) for c in ms[0]}
            for k, ms in acc.items()}


# --------------------------------------------------------------------------------------------- report
def report(s: pd.DataFrame, preselected: Optional[str], cats: Optional[Dict[tuple, Dict[str, tuple]]]) -> str:
    out = ["# BenchReAD (fundus, one-class): imgad against the published results", ""]
    best = s.loc[s[f"mean_{VAL}"].idxmax()]
    out += [f"Selected on BenchReAD's validation set (100 normal + 100 anomalous, EDDFS + BRSET): **{cell(best['config'])}**,"
            f" validation AUROC {pct(best[f'mean_{VAL}'])}.", ""]
    for sp in TESTS:
        orc = s.loc[s[f"mean_{sp}"].idxmax()]
        rank = int((s[f"mean_{sp}"] > best[f"mean_{sp}"]).sum()) + 1
        rho = s[f"mean_{VAL}"].corr(s[f"mean_{sp}"], method="spearman")
        out.append(f"- {TEST_NAMES[sp]}: AUROC {pct(best[f'mean_{sp}'])}, rank {rank} of {len(s)}; oracle "
                   f"{cell(orc['config'])} {pct(orc[f'mean_{sp}'])} (optimism {100 * (orc[f'mean_{sp}'] - best[f'mean_{sp}']):+.1f}); "
                   f"Spearman validation–test {rho:.2f}")
    out.append("")

    rows = chosen_rows(s, preselected)
    out += ["## AUROC (%), normal vs all anomalies", "",
            "BenchReAD rows: Fig. 2 of the paper (one run, 95% CI). imgad rows: mean over seeds, mean bootstrap 95% CI "
            "and ± standard deviation over seeds when it is not zero. Only the one-class rows share BenchReAD's training "
            "data; the others use anomalies or unlabelled images in training.", "",
            "| Method | Training data | RIADD | JSIEC |", "| --- | --- | --- | --- |"]
    for m in METHODS:
        a = [PUBLISHED_AUROC[sp][m] for sp in TESTS]
        out.append(f"| BenchReAD: {m} | {SUPERVISION[m]} | " + " | ".join(f"{v:.1f} ({lo:.1f}–{hi:.1f})" for v, lo, hi in a) + " |")
    for label, r in rows:
        out.append(f"| {label}: {cell(r['config'])} | one-class | {auroc_cell(r, RIADD)} | {auroc_cell(r, JSIEC)} |")
    out.append("")

    out += ["## Threshold-dependent metrics (%)", "",
            "BenchReAD: Table 1, average of F1 / specificity / sensitivity over the anomaly categories and the pooled "
            "row; its threshold rule is not described. imgad: threshold calibrated on held-out normal training images "
            "(target 5% false positives, no anomalies), same averaging.", ""]
    hdr = "| Method | " + " | ".join(f"{TEST_NAMES[sp]} F1 | SPC | SEN" for sp in TESTS) + " |"
    out += [hdr, "| --- |" + " ---: |" * 6]
    for i, m in enumerate(METHODS):
        out.append(f"| BenchReAD: {m} ({SUPERVISION[m]}) | "
                   + " | ".join(" | ".join(f"{v:.1f}" for v in TABLE1[sp]["Average"][i]) for sp in TESTS) + " |")
    for label, r in rows:
        if cats and all((r["config"], sp) in cats for sp in TESTS):
            vals = [cats[(r["config"], sp)]["Average"] for sp in TESTS]
            out.append(f"| {label} | " + " | ".join(" | ".join(pct(v) for v in t) for t in vals) + " |")
        elif f"f1@thr_{RIADD}" in r:  # pooled metrics only (no --scores): normal vs all anomalies
            out.append(f"| {label} (pooled, all anomalies) | "
                       + " | ".join(f"{pct(r[f'f1@thr_{sp}'])} | {pct(1 - r[f'fpr@thr_{sp}'])} | {pct(r[f'tpr@thr_{sp}'])}"
                                    for sp in TESTS) + " |")
    out.append("")

    if cats:
        sel = best["config"]
        for sp in TESTS:
            if (sel, sp) not in cats:
                continue
            c = cats[(sel, sp)]
            out += [f"## {TEST_NAMES[sp]} per category: F1 / sensitivity (%) of the configuration selected on validation", "",
                    f"Specificity at the threshold: imgad {pct(c['Average'][1])}; BenchReAD PatchCore "
                    f"{TABLE1[sp]['Average'][3][1]:.1f}, DRA {TABLE1[sp]['Average'][5][1]:.1f}, NFM-DRA "
                    f"{TABLE1[sp]['Average'][6][1]:.1f}.", "",
                    "| Category | Seen in training | imgad | BenchReAD PatchCore | BenchReAD DRA | BenchReAD NFM-DRA |",
                    "| --- | --- | --- | --- | --- | --- |"]
            for name in [k for k in TABLE1[sp] if k != "Average"]:
                ours = c.get(name)
                seen = "–" if name == "All abnormal" else ("seen" if name in SEEN[sp] else "unseen")
                pub = [TABLE1[sp][name][i] for i in (3, 5, 6)]
                out.append(f"| {name} | {seen} | " + (f"{pct(ours[0])} / {pct(ours[2])}" if ours else "–") + " | "
                           + " | ".join(f"{p[0]:.1f} / {p[2]:.1f}" for p in pub) + " |")
            out.append("")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", type=Path, nargs="+", help="results.csv files from imgad benchmark --out")
    ap.add_argument("--scores", type=Path, nargs="*", default=[], help="joblib files from imgad benchmark --scores-out")
    ap.add_argument("--data", type=Path, help="the --out folder of prepare_benchread.py (needed with --scores)")
    ap.add_argument("--preselected", default=PALM_SELECTED,
                    help="configuration fixed in advance (default: the one chosen on PALM); '' to skip")
    ap.add_argument("--out", type=Path, help="also write the report to this markdown file")
    a = ap.parse_args(argv)

    for p in a.results + a.scores:
        if not p.exists():
            print(f"ERROR: {p} does not exist.", file=sys.stderr)
            return 1
    s = summarise(load_results(a.results))
    if s.empty:
        print(f"No configuration has results on {VAL}, {RIADD} and {JSIEC}.", file=sys.stderr)
        return 1
    cats = None
    if a.scores:
        if a.data is None or not a.data.is_dir():
            print("ERROR: --scores needs --data, the folder written by prepare_benchread.py (to know each test "
                  "image's anomaly category).", file=sys.stderr)
            return 1
        import joblib

        score_rows = [r for p in a.scores for r in joblib.load(p)]
        configs = [r["config"] for _, r in chosen_rows(s, a.preselected)]
        cats = per_category(score_rows, a.data, configs)
    text = report(s, a.preselected, cats)
    print(text)
    if a.out:
        a.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
