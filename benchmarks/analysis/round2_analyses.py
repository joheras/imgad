"""Re-analyses requested in the second review round (RR2, RR4, RR5, RR7, RR8), from the released results.csv files.

All values are means over seeds unless stated. The RETFound mirror (identical weights to RETFound CFP) is dropped
everywhere, leaving 13 distinct extractors. "Collapsed" Anomalib models are those with test AUROC < 0.65 on PALM,
the easier dataset (UniNet, DRAEM, SuperSimpleNet, CFlow); this rule is post hoc and reported as such.

  RR2  family comparison with an equal selection budget: one pre-specified detector for every extractor
       (L2 + kNN k=1, L2 + Mahalanobis, L2 + LOF) against each Anomalib model's single run; with and without
       collapsed models; Mann-Whitney U per dataset; top configuration of each family on PALM with bootstrap CIs.
  RR4  rank agreement between datasets: Spearman rho and Kendall tau, overall and within each family, with and
       without collapsed models, under two selection rules applied identically to both datasets (best on test;
       fixed L2 + kNN k=1), with bootstrap intervals over methods.
  RR5  false-positive rate at the calibrated threshold (target 5%) by family and split, with exact
       Clopper-Pearson intervals for the selected configurations and the per-seed range.
  RR7  operating characteristics of the selected configurations: sensitivity and specificity with exact CIs at the
       calibrated threshold, specificity at 95% sensitivity, and PPV/NPV and counts per 1,000 screened at
       1%, 3% and 10% prevalence.
  RR8  counts without the mirror.

Usage:
    python benchmarks/analysis/round2_analyses.py [--boot 2000] [--out benchmarks/analysis/round2_analyses.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, kendalltau, mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "benchmarks/ophtalmology/results.csv"
PALM = ROOT / "benchmarks/palm/report/results.csv"
MIRROR = "hf-hub:bitfount/RETFound_MAE[mean]"
N_NORMAL = {"private": 908, "palm-val": 189, "palm-test": 187}
N_ABNORMAL = {"private": 946, "palm-val": 211, "palm-test": 213}
SELECTED = {  # configurations reported in the paper
    "private": "retizero-clip | l2 | knn(k=1)",           # best on test (no validation split)
    "palm-test": "vit_large_patch16_224[cls] | l2 | lof",  # chosen on PALM validation
}
FIXED_RULES = ["knn(k=1)", "mahalanobis", "lof"]


def family(r) -> str:
    if r["extractor"] == "anomalib":
        return "Anomalib"
    if str(r["detector"]).startswith("zero_shot"):
        return "Text prompts"
    return "Extractor + detector"


def load() -> pd.DataFrame:
    frames = []
    for csv, fix in [(PRIVATE, None), (PALM, None)]:
        d = pd.read_csv(csv)
        d = d[d["error"].isna() | (d["error"].astype(str) == "")]
        d = d[d["extractor"] != MIRROR].copy()
        d["split"] = d["dataset"].astype(str).str.split("/").str[-1].replace({"miopia": "private"})
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["family"] = d.apply(family, axis=1)
    d["method"] = np.where(d["extractor"] == "anomalib", d["detector"].str.split("(").str[0], d["extractor"])
    return d


def means(d: pd.DataFrame) -> pd.DataFrame:
    cols = ["auroc", "auroc_ci_low", "auroc_ci_high", "tpr@thr", "fpr@thr", "fpr@95tpr", "aupr", "f1_max", "f1@thr"]
    return (d.groupby(["split", "family", "method", "extractor", "preprocessing", "detector", "config"])[cols]
            .mean().reset_index())


def cp(k: float, n: int, alpha=0.05) -> tuple:
    k = int(round(k))
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def per_method(m: pd.DataFrame, split: str, rule: str) -> pd.Series:
    """One AUROC per method on a split: 'best' = best configuration on that split, or a fixed detector with L2."""
    s = m[(m["split"] == split) & (m["family"] != "Text prompts")]
    if rule == "best":
        return s.groupby("method")["auroc"].max()
    an = s[s["family"] == "Anomalib"].set_index("method")["auroc"]
    fx = s[(s["family"] == "Extractor + detector") & (s["preprocessing"] == "l2") & (s["detector"] == rule)]
    return pd.concat([fx.set_index("method")["auroc"], an])


def rr2(m: pd.DataFrame, collapsed: set) -> list:
    out = ["## RR2. Family comparison with an equal selection budget", "",
           "Each extractor with one pre-specified detector and L2 preprocessing (no selection), against each Anomalib "
           "model's single run. Medians over methods; two-sided Mann-Whitney U.", "",
           "| Dataset | Extractor rule | Anomalib set | Median extractor | Median Anomalib | U | p |",
           "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for split in ["private", "palm-test"]:
        s = m[m["split"] == split]
        an = s[s["family"] == "Anomalib"].set_index("method")["auroc"]
        for rule in FIXED_RULES + ["best of 24 (paper)"]:
            ex = s[s["family"] == "Extractor + detector"]
            ex = ex.groupby("method")["auroc"].max() if rule.startswith("best") else \
                ex[(ex["preprocessing"] == "l2") & (ex["detector"] == rule)].set_index("method")["auroc"]
            for label, a in [("all 19", an), ("15 not collapsed", an[~an.index.isin(collapsed)])]:
                u, p = mannwhitneyu(ex, a, alternative="two-sided")
                out.append(f"| {split} | L2 + {rule} | {label} | {ex.median():.3f} | {a.median():.3f} | {u:.1f} | {p:.2g} |"
                           if not rule.startswith("best") else
                           f"| {split} | {rule} | {label} | {ex.median():.3f} | {a.median():.3f} | {u:.1f} | {p:.2g} |")
    # top of each family on PALM, chosen on validation
    v, t = m[m["split"] == "palm-val"], m[m["split"] == "palm-test"]
    out += ["", "Top configuration of each family on PALM, chosen on the validation set (test AUROC, 95% bootstrap CI "
            "averaged over seeds). DeLong needs per-image scores, which are not released; overlapping intervals are "
            "reported instead.", "", "| Family | Configuration | Val AUROC | Test AUROC (95% CI) |", "| --- | --- | ---: | ---: |"]
    for fam in ["Extractor + detector", "Anomalib", "Text prompts"]:
        c = v[v["family"] == fam].sort_values("auroc", ascending=False).iloc[0]
        r = t[t["config"] == c["config"]].iloc[0]
        out.append(f"| {fam} | {c['config'].replace('|', '/')} | {c['auroc']:.3f} | {r['auroc']:.3f} ({r['auroc_ci_low']:.3f}-{r['auroc_ci_high']:.3f}) |")
    return out + [""]


def boot_corr(x, y, n, rng):
    x, y = np.asarray(x), np.asarray(y)
    rs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.unique(x[i]).size > 2:
            rs.append(spearmanr(x[i], y[i])[0])
    return np.nanpercentile(rs, [2.5, 97.5])


def rr4(m: pd.DataFrame, collapsed: set, n_boot: int, rng) -> list:
    out = ["## RR4. Rank agreement between the two datasets", "",
           "Spearman rho (95% bootstrap interval over methods) and Kendall tau between the per-method AUROC on the "
           "private test set and on the PALM test set, with the same selection rule on both datasets.", "",
           "| Rule | Methods | n | rho (95% CI) | tau |", "| --- | --- | ---: | ---: | ---: |"]
    fam = m.drop_duplicates("method").set_index("method")["family"]
    for rule in ["best", "knn(k=1)"]:
        a, b = per_method(m, "private", rule), per_method(m, "palm-test", rule)
        j = pd.concat([a.rename("private"), b.rename("palm")], axis=1).dropna()
        j["family"] = fam.reindex(j.index)
        label = "best configuration on test" if rule == "best" else "L2 + kNN (k=1) for extractors"
        for name, sub in [("all", j), ("all, without collapsed", j[~j.index.isin(collapsed)]),
                          ("extractors only", j[j["family"] == "Extractor + detector"]),
                          ("Anomalib only", j[j["family"] == "Anomalib"]),
                          ("Anomalib, without collapsed", j[(j["family"] == "Anomalib") & ~j.index.isin(collapsed)])]:
            r, tau = spearmanr(sub["private"], sub["palm"])[0], kendalltau(sub["private"], sub["palm"])[0]
            lo, hi = boot_corr(sub["private"], sub["palm"], n_boot, rng)
            out.append(f"| {label} | {name} | {len(sub)} | {r:.2f} ({lo:.2f} to {hi:.2f}) | {tau:.2f} |")
    # family label alone
    j = pd.concat([per_method(m, "private", "best").rename("private"), per_method(m, "palm-test", "best").rename("palm")], axis=1).dropna()
    lab = (fam.reindex(j.index) == "Extractor + detector").astype(int)
    out += ["", f"For reference, the family label alone has rho = {spearmanr(lab, j['private'])[0]:.2f} with the private "
            f"AUROC and {spearmanr(lab, j['palm'])[0]:.2f} with the PALM AUROC.", ""]
    return out


def rr5(m: pd.DataFrame, raw: pd.DataFrame) -> list:
    out = ["## RR5. False-positive rate at the calibrated threshold (target 5%)", "",
           "Distribution over configurations of the seed-averaged false-positive rate on normal images at the threshold "
           "set from out-of-sample normal training scores.", "",
           "| Split | Family | Configurations | Median FPR | IQR | Max | % > 5% | % > 10% |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split in ["private", "palm-val", "palm-test"]:
        for fam in ["Extractor + detector", "Text prompts", "Anomalib"]:
            f = m[(m["split"] == split) & (m["family"] == fam)]["fpr@thr"]
            if len(f):
                out.append(f"| {split} | {fam} | {len(f)} | {f.median():.3f} | {f.quantile(.25):.3f}-{f.quantile(.75):.3f} | "
                           f"{f.max():.3f} | {100 * (f > .05).mean():.0f} | {100 * (f > .10).mean():.0f} |")
    out += ["", "Selected configurations: FPR averaged over seeds, with an exact 95% Clopper-Pearson interval for one test set (the seeds share the test images), and per-seed range.", "",
            "| Split | Configuration | FPR | 95% CI | Per-seed range |", "| --- | --- | ---: | ---: | ---: |"]
    for split, cfg in [("private", SELECTED["private"]), ("palm-val", SELECTED["palm-test"]), ("palm-test", SELECTED["palm-test"])]:
        r = raw[(raw["split"] == split) & (raw["config"] == cfg)]
        n = N_NORMAL[split]
        k = (r["fpr@thr"] * n).mean()  # seeds share the test images: interval on one set of n images
        lo, hi = cp(k, n)
        out.append(f"| {split} | {cfg.replace('|', '/')} | {k / n:.3f} | {lo:.3f}-{hi:.3f} | "
                   f"{r['fpr@thr'].min():.3f}-{r['fpr@thr'].max():.3f} |")
    return out + [""]


def rr7(raw: pd.DataFrame) -> list:
    out = ["## RR7. Operating characteristics and predictive values of the selected configurations", "",
           "Sensitivity and specificity averaged over the 3 seeds, with exact 95% intervals for one test set. Specificity at 95% sensitivity "
           "from the ROC curve. PPV and NPV at the calibrated threshold for several prevalences, and outcomes per "
           "1,000 people screened.", "",
           "| Dataset | Sensitivity (95% CI) | Specificity (95% CI) | Specificity at 95% sensitivity |",
           "| --- | ---: | ---: | ---: |"]
    pp = []
    for split in ["private", "palm-test"]:
        r = raw[(raw["split"] == split) & (raw["config"] == SELECTED[split])]
        na, nn = N_ABNORMAL[split], N_NORMAL[split]  # seeds share the test images
        tp = (r["tpr@thr"] * na).mean()
        fp = (r["fpr@thr"] * nn).mean()
        se, sp = tp / na, 1 - fp / nn
        slo, shi = cp(tp, na)
        flo, fhi = cp(fp, nn)
        sp95 = 1 - r["fpr@95tpr"].mean()
        out.append(f"| {split} | {se:.3f} ({slo:.3f}-{shi:.3f}) | {sp:.3f} ({1 - fhi:.3f}-{1 - flo:.3f}) | {sp95:.3f} |")
        for prev in [0.01, 0.03, 0.10]:
            ppv = se * prev / (se * prev + (1 - sp) * (1 - prev))
            # bounds using the worst / best specificity in its CI
            ppv_lo = se * prev / (se * prev + fhi * (1 - prev))
            ppv_hi = se * prev / (se * prev + flo * (1 - prev)) if flo > 0 else 1.0
            npv = sp * (1 - prev) / (sp * (1 - prev) + (1 - se) * prev)
            per1000 = (1000 * prev * se, 1000 * (1 - prev) * (1 - sp), 1000 * prev * (1 - se))
            pp.append(f"| {split} | {100 * prev:.0f}% | {ppv:.2f} ({ppv_lo:.2f}-{ppv_hi:.2f}) | {npv:.3f} | "
                      f"{per1000[0]:.1f} | {per1000[1]:.1f} | {per1000[2]:.1f} |")
    out += ["", "| Dataset | Prevalence | PPV (range over the specificity CI) | NPV | True positives / 1,000 | False positives / 1,000 | Missed / 1,000 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"] + pp
    return out + [""]


def rr8(m: pd.DataFrame, collapsed: set) -> list:
    out = ["## RR8. Counts without the RETFound mirror", ""]
    for split in ["private", "palm-test"]:
        s = m[m["split"] == split]
        ex = s[s["family"] == "Extractor + detector"]
        an = s[s["family"] == "Anomalib"]
        best_ex = ex.groupby("method")["auroc"].max()
        above = (best_ex > an["auroc"].max()).sum()
        p = ex.pivot_table(index=["method", "detector"], columns="preprocessing", values="auroc").dropna()
        worse = (p["l2+pca:0.95"] < p["l2"]).sum()
        n_cfg = s["config"].nunique()
        out.append(f"- {split}: {n_cfg} configurations; {best_ex.size} distinct extractors, of which {above} score above "
                   f"every Anomalib model; PCA lowers AUROC in {worse} of {len(p)} extractor-detector pairs "
                   f"(mean change {(p['l2+pca:0.95'] - p['l2']).mean():+.3f}).")
    det = m[(m["split"] == "private") & (m["family"] == "Extractor + detector")].groupby(["detector", "preprocessing"])["auroc"].mean().unstack()
    out += ["", "Mean AUROC per detector over the 13 distinct extractors, private dataset (Table 6, private columns):", "",
            "| Detector | L2 | L2 + PCA |", "| --- | ---: | ---: |"]
    for dname, r in det.sort_values("l2", ascending=False).iterrows():
        out.append(f"| {dname.split('(members')[0]} | {r['l2']:.3f} | {r['l2+pca:0.95']:.3f} |")
    return out + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("round2_analyses.md"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    raw = load()
    m = means(raw)
    pt = m[(m["split"] == "palm-test") & (m["family"] == "Anomalib")]
    collapsed = set(pt[pt["auroc"] < 0.65]["method"])
    md = ["# Second-round re-analyses (RR2, RR4, RR5, RR7, RR8)", "",
          f"Collapsed Anomalib models (PALM test AUROC < 0.65): {', '.join(sorted(collapsed))}.", ""]
    md += rr2(m, collapsed) + rr4(m, collapsed, a.boot, rng) + rr5(m, raw) + rr7(raw) + rr8(m, collapsed)
    text = "\n".join(md)
    a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
