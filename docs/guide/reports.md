# Benchmark reports

`imgad benchmark --report DIR` writes a markdown report with figures:

```bash
imgad benchmark --config configs/mvtec_benchmark.yaml --report report/ --seeds 0 1 2 --bootstrap 1000
```

```
report/
  report.md       # the report
  figures/*.png   # every figure referenced in the report
  results.csv     # one row per dataset × configuration × seed
  config.yaml     # configuration used
```

The report contains:

| Section | Content |
|---|---|
| Summary | best configuration, runner-up and significance, best extractors/detectors, hardest dataset |
| Data analysis | images per split, defect types, resolution, masks; example images with ground-truth masks; warnings (small or imbalanced test sets, few training images) |
| Methods | extractors (with dimension), preprocessing, detectors and Anomalib models with a short description; threshold protocol, metrics, repetitions and statistical tests |
| Results | ranking table and chart, extractor × detector heatmap, per-dataset results, ROC curves, score histograms with the threshold, per-defect-type AUROC, operating point, pixel-level metrics and example anomaly maps, spread across seeds, Friedman/Nemenyi/Wilcoxon (or DeLong) with a critical-difference diagram, computational cost |
| Discussion | extractor vs. detector effect, consistency across datasets, significance, hard defect types, Anomalib vs. global features, threshold calibration, localisation |
| Limitations and reproducibility | caveats derived from the setup, exact command, configuration and environment versions |

Figures need matplotlib (`pip install "imgad[report]"`); without it the report contains tables only.
The summary and discussion are generated from the numbers with simple rules: treat them as a first
draft to check and complete. To rebuild a report without rerunning the benchmark:

```bash
imgad benchmark --config cfg.yaml --scores-out scores.joblib
imgad report --scores scores.joblib --config cfg.yaml --out report/ --title "My benchmark"
```

From Python: `imgad.report.generate_report(rows, datasets, "report/", config=cfg)` with
`rows = benchmark(..., keep_scores=True)`.
