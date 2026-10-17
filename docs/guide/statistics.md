# Statistical analysis

Repetitions, confidence intervals and statistical tests.

## Repetitions

 `benchmark(..., seeds=[0, 1, 2, 3, 4], train_subsample=0.8)` runs every configuration
once per seed (column `seed`). The seed sets the detectors' `random_state` (unless the spec fixes it),
the calibration folds, the Anomalib seed and the **training subset** (`train_subsample`: a fraction if
float, a number of images if int). Without subsampling, deterministic detectors (kNN, Mahalanobis…)
give the same result for every seed. Features are extracted once and reused across seeds.

## Confidence intervals

 There are two sources of uncertainty, measured separately:

| Source | How | Where |
|---|---|---|
| Test sample | stratified bootstrap of the test images (`bootstrap=1000`) | `auroc_ci_low/high`, `aupr_ci_low/high` in each row |
| Training | repetitions with seeds + `train_subsample` | `stats.summarize_seeds(rows)`: mean, standard deviation and t-based CI of the mean |

## Statistical tests

`imgad.stats`:

| Question | Test | Function |
|---|---|---|
| Does A have a higher AUROC than B **on this dataset**? | DeLong (correlated AUROCs, same images) | `delong_test(y, sa, sb)`, `delong_vs_baseline(rows, base)` |
| Same for AUPR, F1-max or another metric | paired bootstrap | `paired_bootstrap_test(y, sa, sb, metric="aupr")` |
| Are there differences between configurations **across datasets**? | Friedman (+ Iman-Davenport correction) | `compare_configs(rows)` |
| Which pairs differ? | Nemenyi (critical difference of ranks) and Wilcoxon signed-rank + Holm | `compare_configs(rows)["pairs"]`, `plot_cd_diagram` |

To compare across datasets, seeds are first averaged within each dataset, and each dataset (e.g. each
MVTec category) is a block, following Demšar (2006). Wilcoxon-Holm usually has more power with few
configurations, while Nemenyi is the classical post-hoc test after Friedman and the one drawn in the CD
diagram. DeLong needs per-image scores (`keep_scores=True`).

```python
from imgad import benchmark, stats

rows = benchmark(extractors, detectors, datasets, seeds=range(5), train_subsample=0.8,
                 bootstrap=1000, keep_scores=True, anomalib_models=["patchcore"])
stats.summarize_seeds(rows)                          # mean ± std per (dataset, config)
res = stats.compare_configs(rows, metric="auroc")    # also aupr, pixel_auroc, aupro, fpr@95tpr…
print(stats.report_markdown(res))                    # mean ranks, Friedman, CD, pairs with Holm
stats.plot_cd_diagram(res, "cd.png")
stats.delong_vs_baseline(rows, baseline=res["configs"][0])
```

From the CLI:

```bash
imgad benchmark --config configs/mvtec_benchmark.yaml --seeds 0 1 2 3 4 --train-subsample 0.8 \
                --bootstrap 1000 --out results.csv --scores-out scores.joblib --stats
imgad stats --results results.csv --metric auroc --cd-plot cd.png --out stats.md
imgad stats --scores scores.joblib --baseline "openai/clip-vit-base-patch16 | l2 | knn(k=1)"   # + DeLong
```
