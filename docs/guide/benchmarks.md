# Benchmarks

`evaluate_scores` returns AUROC, AUPR, F1-max and FPR@95%TPR (and F1/TPR/FPR at the threshold if
predictions are given). `per_type_auroc` breaks the AUROC down by defect type.

`evaluate_maps(masks, maps)` gives the localisation metrics: pixel-AUROC, pixel-AUPR, pixel-F1-max and
AUPRO (FPR ≤ 0.3, the standard MVTec AD metric). Masks are resized to each map's resolution.

`benchmark(extractors, detectors, datasets, preprocessings, anomalib_models=[...])` evaluates the full
grid, extracting features once per (extractor, dataset), plus each Anomalib model per dataset (with
pixel metrics when masks are available). In YAML, use the `anomalib:` key (see
`configs/mvtec_benchmark.yaml`).

## When a run fails

A run that fails is logged and recorded, and the benchmark goes on with the others. This covers a model
that cannot be downloaded or loaded, an extraction error, a detector that cannot fit, or an Anomalib error.

- A failed run becomes a row with an `error` column (the message) and a `stage` (`load`, `extract`,
  `detector` or `anomalib`), and no metrics.
- The end of the log lists every failure.
- The CSV keeps the failed rows. The statistical analysis, the markdown tables and the report ignore them,
  and the report lists them in a *Failed runs* section.
- `successful_rows(rows)` keeps only the runs that finished.
- `on_error="raise"` (CLI `--stop-on-error`) stops at the first failure instead.

## Few-shot protocol

`k_shots=[0, 1, 2, 4, 8, 16]` (YAML `k_shots:`, CLI `--k-shots 0 1 4 16`) runs every configuration with `k`
normal training images drawn per seed, instead of the whole training set.

- The rows get a `k_shot` column, and the configuration name ends in `| k-shot`.
- `k = 0` is zero-shot: it runs only the [zero-shot detector](zero-shot.md).
- Detectors that cannot fit on `k` images are recorded as failed runs. For example, kNN needs two images.
- The report adds a *Few-shot learning curves* section: a table and a plot of the metric against `k`.

Use several seeds: with a handful of images, the choice of images matters as much as the method.


## Progress while the benchmark runs

The benchmark reports what it is doing at every moment: the plan (datasets and number of runs), each
phase (loading a model, extracting features with the number of cached images, training an Anomalib
model, calibrating its thresholds, scoring the test set, building the report) and one line per
finished run with a global counter, its metrics, the elapsed time and an ETA:

```
15:19:27 Benchmark plan: 2 dataset(s), 2 seed(s), 36 runs in total
15:19:27   feature pipelines: 2 extractor(s) × 1 preprocessing(s) × 4 detector(s) × 2 dataset(s) × 2 seed(s) = 32 runs
15:19:27 Extracting features [extractor 1/2, dataset 1/2]: clip-vit-base-patch16 on mvtec/bottle (292 images, 150 cached)...
15:19:31 [ 2/36   6%] mvtec/bottle · seed 0 · ... | l2 | mahalanobis: AUROC=0.9937 [0.980, 1.000] AUPR=0.9981 (0.04s) · elapsed 4s, ETA ~1m12s
15:19:36   training patchcore on 167 normal images (device cuda)...
```

The ETA is computed per kind of work (model loading, feature extraction, detector runs, Anomalib runs),
so a slow extraction does not distort the estimate for the cheap detector runs; kinds not timed yet
are flagged. If a step fails, the log says which one (`FAILED: fit mvtec/screw · seed 2 · ...`).

| Option | Console output |
|---|---|
| (default) | plan, phases and one line per run (INFO) |
| `-v`, `--verbose` | every step with its duration: preprocessing, fit, score, cache hits, figures (DEBUG) |
| `-q`, `--quiet` | warnings and errors only, no progress bars |
| `--log-file run.log` | additionally writes the full DEBUG log with timestamps to a file |

From Python, messages go to the standard `logging` module under the `imgad` logger:
`benchmark(..., verbose=True)` prints them when logging is not configured, and
`imgad.progress.setup_logging(logging.DEBUG, log_file="run.log")` selects the level and adds a file.

## Parallel benchmarks (multiple models on the GPU)

By default the benchmark runs sequentially. With `-j/--jobs` and/or `--devices` it runs in parallel:

```bash
imgad devices --config configs/mvtec_benchmark.yaml   # GPUs, free memory, memory estimate per model
imgad benchmark --config configs/mvtec_benchmark.yaml -j 8 --devices auto
```

- **Accelerator jobs.** Each feature extractor is loaded once in a worker process on a GPU and
  extracts the features of every dataset; each Anomalib run (model × dataset × seed) is also a job.
  Several jobs share a GPU when they fit in its memory.
- **Availability checks.** GPUs and their free memory are read with `nvidia-smi` (or NVML), without
  creating CUDA contexts in the main process. A job starts only when its estimated memory fits both in
  the budget (free memory at start × 0.9, minus what running jobs reserved) and in the memory that is
  free right now (other processes may be using the GPU). The first estimate comes from the checkpoint
  size on the Hub (or `--gpu-mem-gb`, default 4 GiB); after a model has run, its measured peak is used.
  A job that still runs out of memory is retried alone on an idle device.
- **Pipelining.** As soon as the features of an extractor are ready, its detector runs (dataset × seed ×
  preprocessing × detector) start on a pool of `-j` CPU processes while the GPU moves on.
- **Same results.** Rows are identical to the sequential run (only the timing columns differ) and come
  out in the same order.

| Option | YAML (`parallel:`) | Meaning |
|---|---|---|
| `-j N`, `--jobs N` | `jobs` | CPU worker processes for detector runs (`-1` = all cores) |
| `--devices auto` | `devices` | `auto` (all GPUs, else CPU), `cpu`, `cuda:0,cuda:1`… |
| `--gpu-jobs auto` | `gpu_jobs` | maximum concurrent jobs per GPU (`auto`: as many as fit, up to 4) |
| `--gpu-mem-gb 4` | `gpu_mem_gb` | memory assumed for jobs of unknown size |

From Python: `benchmark(..., n_jobs=8, devices="auto")`. On a laptop GPU (8 GB) with three ViT
extractors (DINOv3-B, DINOv2-S, DINOv2-B), five detectors, PatchCore and three seeds on two datasets,
the parallel run took 32 s instead of 59 s; the gain grows with larger datasets and more GPUs.
Parallel workers are started with `spawn`, so scripts that call `benchmark(..., n_jobs>1)` must use an
`if __name__ == "__main__":` guard.

## Example results

[`benchmarks/mvtec_4categories/`](https://github.com/joheras/imgad/tree/main/benchmarks/mvtec_4categories) contains a benchmark of 5 extractors
× 11 detectors on 4 MVTec AD categories (report, results CSV and configuration).
