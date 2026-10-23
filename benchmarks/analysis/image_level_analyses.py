"""Image-level analyses of the private dataset (review items RR1, RR6 and RR10).

Needs the per-image scores that ``imgad benchmark --scores-out`` saves, the YAML used for that run (to recover
which image each score belongs to) and a metadata CSV with one row per test image:

    image,se,age,camera,patient
    img_000123.jpg,-4.25,11,Topcon NW400,P0457
    ...

``image`` is the file name (or path) of the image; ``se`` the spherical equivalent in dioptres; ``age`` in years;
``camera`` the fundus camera model; ``patient`` an anonymous patient identifier (both eyes of a child share it).
Columns other than ``image`` and ``se`` are optional; the corresponding analyses are skipped without them.

Outputs (in --out):
  * image_scores.csv.gz   one row per configuration x seed x test image: dataset, config, seed, image, label, score
  * image_level.md        RR1: AUROC of the chosen configuration by refractive subgroup, age band and camera
                          (each abnormal subgroup against all normal test images; within a camera, both classes
                          from that camera), with bootstrap 95% intervals;
                          RR6: selection on one half of the test set (split by patient) and AUROC on the other half,
                          in both directions and over repeated random splits, against the oracle;
  * score_hist_<config>.png  RR10: score distribution of the chosen configuration with its calibrated threshold.

Run it from the same directory as the benchmark: relative data paths in the YAML are resolved from there, as
imgad does.

Usage:
    imgad benchmark --config <private YAML> --scores-out scores.joblib --out results.csv
    python benchmarks/analysis/image_level_analyses.py --config <private YAML> --scores scores.joblib \
        --metadata metadata.csv --out benchmarks/analysis/image_level
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CONFIG = "retizero-clip | l2 | knn(k=1)"


def subgroup(se: float) -> str:
    """Refractive categories used in the paper (abnormal = SE <= -3.00 D or SE >= +2.00 D)."""
    if se < -6.0:
        return "severe myopia"
    if se <= -3.0:
        return "moderate myopia"
    if se >= 5.0:
        return "severe hypermetropia"
    if se >= 2.0:
        return "moderate hypermetropia"
    return "normal"


def load_scores(config_path: Path, scores_path: Path) -> pd.DataFrame:
    from imgad.cli import _datasets_from_config

    cfg = yaml.safe_load(config_path.read_text())
    datasets = {ds.name: ds for ds in _datasets_from_config(cfg)}
    rows = joblib.load(scores_path)
    out = []
    for r in rows:
        if r.get("error") or "scores" not in r:
            continue
        ds = datasets[r["dataset"]]
        if len(ds.test_paths) != len(r["scores"]):
            raise ValueError(f"{r['config']}: {len(r['scores'])} scores but {len(ds.test_paths)} test images; "
                             "is --config the YAML of this run?")
        out.append(pd.DataFrame({"dataset": r["dataset"], "config": r["config"], "seed": r["seed"],
                                 "image": [Path(p).name for p in ds.test_paths], "path": list(ds.test_paths),
                                 "label": r["labels"], "score": r["scores"], "threshold": r.get("threshold", np.nan)}))
    return pd.concat(out, ignore_index=True)


def auc_ci(y, s, n_boot=1000, rng=None):
    y, s = np.asarray(y), np.asarray(s)
    if y.min() == y.max():
        return np.nan, np.nan, np.nan
    a = roc_auc_score(y, s)
    rng = rng or np.random.default_rng(0)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    bs = []
    for _ in range(n_boot):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        bs.append(roc_auc_score(y[i], s[i]))
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def rr1(d: pd.DataFrame, rng) -> list:
    """AUROC by subgroup for one configuration (scores averaged over seeds)."""
    s = d.groupby(["image", "label", "group", "age", "camera"], dropna=False)["score"].mean().reset_index()
    normal = s[s["label"] == 0]
    out = ["| Subgroup | Abnormal images | AUROC (95% CI) |", "| --- | ---: | ---: |"]
    for g in ["moderate myopia", "severe myopia", "moderate hypermetropia", "severe hypermetropia"]:
        ab = s[(s["label"] == 1) & (s["group"] == g)]
        if len(ab):
            sub = pd.concat([normal, ab])
            a, lo, hi = auc_ci(sub["label"], sub["score"], rng=rng)
            out.append(f"| {g} vs normal | {len(ab)} | {a:.3f} ({lo:.3f}-{hi:.3f}) |")
    for name, mask in [("all myopia", s["group"].str.contains("myopia")),
                       ("all hypermetropia", s["group"].str.contains("hypermetropia"))]:
        ab = s[(s["label"] == 1) & mask]
        if len(ab):
            sub = pd.concat([normal, ab])
            a, lo, hi = auc_ci(sub["label"], sub["score"], rng=rng)
            out.append(f"| {name} vs normal | {len(ab)} | {a:.3f} ({lo:.3f}-{hi:.3f}) |")
    if s["age"].notna().any():
        out += ["", "| Age band | Normal | Abnormal | AUROC (95% CI) |", "| --- | ---: | ---: | ---: |"]
        for lo_a, hi_a in [(5, 9), (10, 12), (13, 15)]:
            sub = s[(s["age"] >= lo_a) & (s["age"] <= hi_a)]
            a, lo, hi = auc_ci(sub["label"], sub["score"], rng=rng)
            out.append(f"| {lo_a}-{hi_a} years | {(sub['label'] == 0).sum()} | {(sub['label'] == 1).sum()} | {a:.3f} ({lo:.3f}-{hi:.3f}) |")
    if s["camera"].notna().any():
        out += ["", "| Camera | Normal | Abnormal | AUROC (95% CI) |", "| --- | ---: | ---: | ---: |"]
        for cam, sub in s.groupby("camera"):
            a, lo, hi = auc_ci(sub["label"], sub["score"], rng=rng)
            out.append(f"| {cam} | {(sub['label'] == 0).sum()} | {(sub['label'] == 1).sum()} | {a:.3f} ({lo:.3f}-{hi:.3f}) |")
    return out


def rr6(d: pd.DataFrame, rng, n_splits: int) -> list:
    """Select the configuration on one half of the test set (split by patient), report AUROC on the other."""
    s = d.groupby(["config", "image", "label", "patient"], dropna=False)["score"].mean().reset_index()
    units = s.drop_duplicates("image")[["image", "patient"]]
    units["unit"] = units["patient"].fillna(units["image"])
    patients = units["unit"].unique()
    unit_of = dict(zip(units["image"], units["unit"]))
    s["unit"] = s["image"].map(unit_of)
    res = []
    for k in range(n_splits):
        a_set = set(rng.permutation(patients)[: len(patients) // 2])
        half = s["unit"].isin(a_set)
        for sel, rep in [(half, ~half), (~half, half)]:
            auc_sel = s[sel].groupby("config").apply(lambda g: roc_auc_score(g["label"], g["score"]))
            auc_rep = s[rep].groupby("config").apply(lambda g: roc_auc_score(g["label"], g["score"]))
            best = auc_sel.idxmax()
            res.append({"chosen": best, "selection_auroc": auc_sel[best], "report_auroc": auc_rep[best],
                        "oracle_report_auroc": auc_rep.max()})
    r = pd.DataFrame(res)
    out = [f"Over {n_splits} random patient-level splits (both directions, {len(r)} selections):", "",
           "| Quantity | Median | 2.5th-97.5th percentile |", "| --- | ---: | ---: |"]
    for c, label in [("selection_auroc", "AUROC of the chosen configuration on the selection half"),
                     ("report_auroc", "AUROC of the chosen configuration on the other half (unbiased)"),
                     ("oracle_report_auroc", "Best AUROC on the other half (oracle)")]:
        out.append(f"| {label} | {r[c].median():.3f} | {r[c].quantile(.025):.3f}-{r[c].quantile(.975):.3f} |")
    opt = r["selection_auroc"] - r["report_auroc"]
    out += ["", f"Optimism of choosing on the same images: median {opt.median():+.3f} "
            f"({opt.quantile(.025):+.3f} to {opt.quantile(.975):+.3f}).", "",
            "Configurations chosen most often:", ""]
    for cfg, n in r["chosen"].value_counts().head(5).items():
        out.append(f"- {cfg}: {n} of {len(r)}")
    return out


def rr10(d: pd.DataFrame, cfg: str, out_dir: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = d.groupby(["image", "label"])["score"].mean().reset_index()
    thr = d["threshold"].mean()
    fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=300)
    bins = np.linspace(s["score"].min(), s["score"].max(), 40)
    ax.hist(s.loc[s["label"] == 0, "score"], bins=bins, alpha=0.6, label="normal", color="#4C78A8")
    ax.hist(s.loc[s["label"] == 1, "score"], bins=bins, alpha=0.6, label="abnormal", color="#E45756")
    if np.isfinite(thr):
        ax.axvline(thr, ls="--", color="k", lw=1, label="threshold")
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("test images")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    name = "score_hist_" + "".join(c if c.isalnum() else "_" for c in cfg).strip("_") + ".png"
    fig.savefig(out_dir / name)
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, type=Path, help="YAML used for the benchmark run")
    ap.add_argument("--scores", required=True, type=Path, help="file written by imgad benchmark --scores-out")
    ap.add_argument("--metadata", type=Path, help="CSV with image, se[, age, camera, patient]")
    ap.add_argument("--chosen", default=DEFAULT_CONFIG, help=f"configuration for RR1/RR10 (default: {DEFAULT_CONFIG})")
    ap.add_argument("--splits", type=int, default=50, help="random patient-level splits for RR6")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/analysis/image_level"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)

    d = load_scores(a.config, a.scores)
    d.drop(columns=["path"]).to_csv(a.out / "image_scores.csv.gz", index=False)
    md = ["# Image-level analyses of the private dataset (RR1, RR6, RR10)", "",
          f"{d['config'].nunique()} configurations, {d['image'].nunique()} test images.", ""]

    if a.metadata:
        meta = pd.read_csv(a.metadata)
        meta["image"] = meta["image"].map(lambda p: Path(str(p)).name)
        for c in ["age", "camera", "patient"]:
            if c not in meta:
                meta[c] = np.nan
        d = d.merge(meta[["image", "se", "age", "camera", "patient"]], on="image", how="left")
        missing = d.drop_duplicates("image")["se"].isna().sum()
        if missing:
            md += [f"Warning: {missing} test images have no metadata row.", ""]
        d["group"] = d["se"].map(lambda v: subgroup(v) if pd.notna(v) else np.nan)
        mism = d.drop_duplicates("image").query("label == 1 and group == 'normal'")
        if len(mism):
            md += [f"Warning: {len(mism)} abnormal images have an SE in the normal range; check the thresholds.", ""]
        c = d[d["config"] == a.chosen]
        if c.empty:
            raise SystemExit(f"configuration {a.chosen!r} not in the scores file")
        md += [f"## RR1. AUROC of {a.chosen} by subgroup", ""] + rr1(c, rng) + [""]
    else:
        d["patient"] = np.nan
        md += ["No --metadata given: RR1 skipped; RR6 splits by image instead of by patient.", ""]

    md += ["## RR6. Selection on one half, evaluation on the other", ""] + rr6(d, rng, a.splits) + [""]
    c = d[d["config"] == a.chosen]
    if not c.empty:
        md += ["## RR10. Score distribution", "", f"![{a.chosen}]({rr10(c, a.chosen, a.out)})", ""]
    (a.out / "image_level.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
