"""Choose configurations on PALM's validation set and report them on its test set.

Reads the ``results.csv`` written by ``imgad benchmark --out`` for ``configs/palm_benchmark.yaml`` and prints a
markdown report with:

  * the configuration with the best mean validation AUROC, and its test AUROC (the unbiased estimate);
  * the oracle: the best test AUROC over all configurations (optimistic, chosen on the test set);
  * the same selection inside each family (feature pipelines, Anomalib models, text prompts);
  * the rank correlation between validation and test AUROC over all configurations.

Usage:
    python benchmarks/palm/select_on_validation.py benchmarks/palm/results.csv [--top 10] [--out selection.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

VAL, TEST = "palm-val", "palm-test"


def family(row) -> str:
    if row["extractor"] == "anomalib":
        return "Anomalib model"
    if str(row["detector"]).startswith("zero_shot"):
        return "Text prompts"
    return "Extractor + detector"


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of AUROC over seeds per (split, configuration), one column per split."""
    df = df[df["auroc"].notna()].copy()
    df["split"] = df["dataset"].astype(str).str.split("/").str[-1]
    df = df[df["split"].isin([VAL, TEST])]
    df["family"] = df.apply(family, axis=1)
    g = df.groupby(["config", "family", "split"])["auroc"].agg(["mean", "std", "count"]).reset_index()
    wide = g.pivot_table(index=["config", "family"], columns="split", values=["mean", "std", "count"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    return wide.reset_index().dropna(subset=[f"mean_{VAL}", f"mean_{TEST}"])


def pick(s: pd.DataFrame) -> dict:
    sel = s.loc[s[f"mean_{VAL}"].idxmax()]
    orc = s.loc[s[f"mean_{TEST}"].idxmax()]
    return {"selected": sel["config"], "val": sel[f"mean_{VAL}"], "test": sel[f"mean_{TEST}"],
            "test_std": sel.get(f"std_{TEST}", float("nan")), "rank": int((s[f"mean_{TEST}"] > sel[f"mean_{TEST}"]).sum()) + 1,
            "oracle": orc["config"], "oracle_test": orc[f"mean_{TEST}"], "n": len(s)}


def cell(text: str) -> str:
    """Configuration labels use '|' as separator, which would split a markdown table cell."""
    return str(text).replace("|", "\\|")


def report(s: pd.DataFrame, top: int) -> str:
    out = ["# PALM: selection on validation, evaluation on test", ""]
    p = pick(s)
    out += [f"Selected on validation: **{p['selected']}**", "",
            f"- validation AUROC {p['val']:.3f}; **test AUROC {p['test']:.3f}** (± {p['test_std']:.3f} over seeds), "
            f"rank {p['rank']} of {p['n']} on test",
            f"- oracle (best on test, optimistic): {p['oracle']}, test AUROC {p['oracle_test']:.3f}",
            f"- optimism of choosing on test: {p['oracle_test'] - p['test']:+.3f}", ""]
    rho = s[f"mean_{VAL}"].corr(s[f"mean_{TEST}"], method="spearman")
    out += [f"Spearman correlation between validation and test AUROC over {len(s)} configurations: {rho:.3f}", ""]

    out += ["## Per family", "", "| Family | Selected on validation | Val AUROC | Test AUROC | Oracle test AUROC |",
            "| --- | --- | ---: | ---: | ---: |"]
    for fam, sub in s.groupby("family"):
        q = pick(sub)
        out.append(f"| {fam} | {cell(q['selected'])} | {q['val']:.3f} | {q['test']:.3f} | {q['oracle_test']:.3f} |")

    out += ["", f"## Top {top} on validation", "", "| # | Configuration | Family | Val AUROC | Test AUROC |",
            "| ---: | --- | --- | ---: | ---: |"]
    for i, (_, r) in enumerate(s.sort_values(f"mean_{VAL}", ascending=False).head(top).iterrows(), 1):
        out.append(f"| {i} | {cell(r['config'])} | {r['family']} | {r[f'mean_{VAL}']:.3f} | {r[f'mean_{TEST}']:.3f} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", type=Path, help="results.csv from imgad benchmark --out")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", type=Path, help="also write the report to this markdown file")
    a = ap.parse_args(argv)

    df = pd.read_csv(a.results)
    if "error" in df.columns:
        failed = df["error"].notna() & (df["error"].astype(str) != "")
        if failed.any():
            print(f"({int(failed.sum())} failed runs ignored)", file=sys.stderr)
        df = df[~failed]
    s = summarise(df)
    if s.empty:
        print(f"No configuration has results on both {VAL} and {TEST}.", file=sys.stderr)
        return 1
    text = report(s, a.top)
    print(text)
    if a.out:
        a.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
