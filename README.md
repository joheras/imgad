# imgad — Image anomaly detection with foundation models, classical detectors and Anomalib

[![CI](https://github.com/joheras/imgad/actions/workflows/ci.yml/badge.svg)](https://github.com/joheras/imgad/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/imgad.svg)](https://pypi.org/project/imgad/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/joheras/imgad/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-joheras.github.io%2Fimgad-blue.svg)](https://joheras.github.io/imgad/)

A modular framework for image-level anomaly detection in the **one-class** setting (training uses
normal images only):

```
images ──► FeatureExtractor ──► preprocessing ──► classical detector ──► anomaly score
          (CLIP, SigLIP, DINOv2…   (l2, PCA,         (kNN, Mahalanobis,     (+ calibrated threshold)
           any HF model)            standard…)        LOF, OCSVM, IF…)
```

It also wraps end-to-end [Anomalib](https://github.com/open-edge-platform/anomalib) models (PatchCore,
PaDiM, EfficientAD…) with pixel-level anomaly maps, and includes a benchmarking module with repeated
runs, confidence intervals, statistical tests and automatic markdown reports.

## Installation

```bash
pip install imgad                    # core
pip install "imgad[anomalib]"        # + Anomalib models (PatchCore, PaDiM, EfficientAD…)
pip install "imgad[report]"          # + figures in benchmark reports (matplotlib)
pip install "imgad[medical]"         # + DICOM, NIfTI and the medical model zoo (pydicom, nibabel, timm, open_clip)
pip install "imgad[all]"             # + open_clip, pyod, pandas, anomalib, matplotlib, pytest
```

From source: `git clone https://github.com/joheras/imgad && pip install -e "imgad[dev]"`.
Python ≥ 3.10.

## Quick start

```python
from imgad import AnomalyDetectionPipeline, HFFeatureExtractor, build_detector, data
from imgad.evaluation import evaluate_scores

ds = data.mvtec_ad("/path/to/mvtec_anomaly_detection", "bottle")

pipe = AnomalyDetectionPipeline(
    extractor=HFFeatureExtractor("openai/clip-vit-base-patch16"),
    detector=build_detector("knn", k=1),
    preprocessing=["l2"],
    cache_dir=".feature_cache",          # caches features per image
)
pipe.fit(ds.train_paths)                 # normal images only
scores = pipe.score(ds.test_paths)       # higher = more anomalous
preds  = pipe.predict(ds.test_paths)     # 0/1 with the calibrated threshold
print(evaluate_scores(ds.test_labels, scores, preds))

pipe.save("bottle.joblib")               # saves detector + config (not the model weights)
pipe = AnomalyDetectionPipeline.load("bottle.joblib")
```

## Inference with a trained model

The `.joblib` file stores the fitted detector, the preprocessing, the threshold, the extractor
configuration and metadata (number of training images, date, threshold origin). Hugging Face weights
are not stored: they are reloaded from the Hub (or the local HF cache) on `load`.

```python
from imgad import AnomalyPredictor

predictor = AnomalyPredictor.load("bottle.joblib")   # device auto (cuda/mps/cpu); device="cpu" to force it
r = predictor.predict_one("new.png")                 # path, PIL.Image or numpy array
r.is_anomaly, r.score, r.normal_percentile

results = predictor.predict("folder/")               # folder, list of paths/images, or a mix
predictor.save_results(results, "preds.csv")         # .csv or .json
for r in predictor.predict_iter("huge_folder/", chunk_size=256): ...   # constant memory
```

Each `Prediction` contains:

| Field | Meaning |
|---|---|
| `score` | raw score (detector-dependent scale; higher = more anomalous) |
| `is_anomaly` | `score > threshold` |
| `threshold` | threshold used |
| `normal_percentile` | fraction of normal training images with a lower score (0–1). Comparable across detectors: ~0.5 = typical normal image, 1.0 = rarer than any normal image seen |

**Tune the threshold** on a labelled validation set (no retraining):

```python
pipe = AnomalyDetectionPipeline.load("bottle.joblib")
pipe.calibrate_threshold(val_paths, val_labels, method="f1")   # f1 | youden | fpr | tpr
pipe.save("bottle_cal.joblib")
# Normal images only: method="fpr", target_fpr=0.01  -> ≤1% false alarms
```

Or on the fly: `AnomalyPredictor.load("bottle.joblib", threshold=0.02)`.

Shortcuts: the extractor can be an HF id (`"facebook/dinov2-base"`) and the detector a name
(`"mahalanobis"`) or a dict (`{"name": "knn", "k": 5}`).

## Feature extractors

`HFFeatureExtractor(model_name, pooling="auto", layer=None, normalize=True, ...)` works with any
vision model on the Hub that loads with `AutoModel`:

| Family | Examples | Default pooling |
|---|---|---|
| CLIP-style dual encoders | `openai/clip-*`, `google/siglip-*`, `google/siglip2-*`, `facebook/metaclip-*`, `kakaobrain/align-base`, `laion/CLIP-*` (HF format) | `projection` → `get_image_features` |
| Vision backbones | `facebook/dinov2-*`, `google/vit-*`, `microsoft/beit-*`, `facebook/convnext-*`, `microsoft/resnet-*` | `pooler` |

`pooling` options: `auto`, `projection`, `pooler`, `cls`, `mean` (mean of patch tokens), `cls+mean`.
`layer` selects intermediate layers (`layer=-4`, or `layer=[6, 9, 12]` to concatenate several), which
often capture local defects better than the more semantic last layer.

For **open_clip** checkpoints (`hf-hub:laion/...`, EVA-CLIP, DataComp…):
`OpenCLIPFeatureExtractor("hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K")`.

### Gated and private models on the Hugging Face Hub

Some models (e.g. DINOv3) are **gated**: accept their conditions on the model page and authenticate
with an [access token](https://huggingface.co/settings/tokens). imgad uses, in this order: an explicit
token (`token=` / `--hf-token-file`), the `HF_TOKEN` environment variable, or the token stored by
`imgad hf-login` (same as `hf auth login`). Tokens are **never** saved in `.joblib` files, configs,
feature-cache keys, reports or logs.

```bash
imgad hf-login                                            # paste the token (input hidden)
imgad hf-check facebook/dinov3-vitb16-pretrain-lvd1689m   # access status and what to do
imgad hf-check --config configs/mvtec_benchmark.yaml      # every Hub model of a benchmark
imgad benchmark --config cfg.yaml --hf-token-file ~/.hf_token   # or fit/predict/calibrate
```

```python
HFFeatureExtractor("facebook/dinov3-vitb16-pretrain-lvd1689m", token=my_token)
AnomalyPredictor.load("model.joblib", token=my_token)      # the token is not stored in the file
benchmark(..., token=my_token)                            # checks access to every Hub model first
```

- A benchmark checks access to all its Hub models **before** doing any work and lists, for each model
  that cannot be loaded, what to do (accept the conditions, log in, replace a rejected token…).
- If you have no access but the model is already in the local Hugging Face cache, it is loaded from the
  cache (with a warning).
- There is deliberately no `--hf-token VALUE` option: a token on the command line ends up in the shell
  history and the process list.

## Detectors

All detectors share the same API (`fit`, `score`, `predict`) and the convention **higher score = more
anomalous**.

| Name | Method | Notes |
|---|---|---|
| `knn` | Distance to the k nearest normal samples | Very strong baseline with CLIP/DINO; `metric="cosine"` |
| `mahalanobis` / `gaussian` | Mahalanobis distance (Ledoit-Wolf, OAS, MCD) | Very good with foundation-model features |
| `centroid` | Distance to the mean embedding | Minimal cost |
| `kmeans` | Distance to the nearest centroid | Multimodal normality |
| `lof` | Local Outlier Factor (novelty mode) | |
| `gmm` | Negative log-likelihood under a Gaussian mixture | |
| `kde` | Negative KDE log-density | Better after `pca:<n>` to few dimensions |
| `ocsvm` | One-Class SVM | |
| `iforest` | Isolation Forest | |
| `pca` | PCA reconstruction error | |
| `ensemble` | Mean of normalised detectors (z-score or rank) | |
| `sklearn` | Wraps any sklearn estimator | `SklearnDetector(estimator=EllipticEnvelope())` |
| `pyod` | Any PyOD model | `{"name": "pyod", "model": "ECOD"}` |

**Threshold**: after `fit`, `threshold_` is set to the `threshold_quantile` (0.95 by default) of
*out-of-sample* scores of the normal data: leave-one-out for kNN/LOF and 5-fold cross-validation for
the rest. This avoids the optimistic threshold of detectors that overfit the training set (KDE, OCSVM,
PCA…). It can be disabled with `det.calibration = "train"`.

## Anomalib models (with anomaly maps)

`AnomalibPipeline` wraps any **one-class** [Anomalib](https://github.com/open-edge-platform/anomalib)
model (`patchcore`, `padim`, `efficient_ad`, `fastflow`, `stfpm`, `reverse_distillation`, `cflow`,
`draem`…; `imgad list` shows them) with the same API as the feature pipelines, and adds **pixel-level
anomaly maps**:

```python
from imgad import AnomalibPipeline, evaluate_maps

pc = AnomalibPipeline("patchcore", backbone="wide_resnet50_2", layers=["layer2", "layer3"], image_size=256)
pc.fit(ds.train_paths)                              # normal images only
scores, maps = pc.score_and_maps(ds.test_paths)     # maps: list of arrays (model resolution)
evaluate_maps(ds.test_masks, maps)                  # pixel_auroc, pixel_aupr, pixel_f1_max, aupro
pc.predict_masks(maps)                              # binary masks (map > pixel_threshold_)
pc.save("bottle_pc.joblib")                         # loaded by AnomalyPredictor.load / imgad predict as usual
```

- Anomalib provides the models and the training loop; its post-processing (normalisation and threshold
  learnt on a labelled validation set) is disabled. Scores and maps are the **raw** model outputs
  (higher = more anomalous).
- **Threshold**: as for the detectors, the `threshold_quantile` (0.95) of the scores of *unseen* normal
  images: `calibration_split=0.2` holds out that 20% of the training images (not used for training).
  `calibration_split=0` uses training scores (optimistic, especially for PatchCore). The pixel threshold
  (`pixel_threshold_`) is the same quantile of the maximum of each held-out normal map; with validation
  masks it can be tuned with `calibrate_pixel_threshold(images, masks)` (pixel F1).
  `calibrate_threshold` works as in the other pipelines.
- Images are resized to `image_size` **without cropping**, so the maps cover the whole image.
  `score_and_maps(..., size="original")` returns them at each image's own size.
- The `.joblib` stores the configuration **and the weights** (e.g. the PatchCore memory bank): nothing
  needs to be downloaded to load it.
- Model kwargs are those of the Anomalib constructor (`backbone`, `layers`, `coreset_sampling_ratio`,
  `pre_trained`…). For models trained over several epochs: `max_epochs` or
  `trainer_kwargs={"max_steps": ...}`. EfficientAD trains with batch size 1 and downloads imagenette.
- One-class models only; zero/few-shot models (WinCLIP, AnomalyDINO…) are not supported.

Inference with maps:

```python
predictor = AnomalyPredictor.load("bottle_pc.joblib")
res = predictor.predict("folder/", return_maps=True)   # r.anomaly_map (original size), r.anomalous_area
predictor.save_maps(res, "maps/")                      # <folder>_<name>_heatmap.png + _mask.png
predictor.save_results(res, "preds.csv")               # includes the heatmap and mask paths
```

Heatmaps use a fixed scale per model (blue = typical normal, middle of the scale = pixel threshold,
red = far above it), so normal images look blue instead of being rescaled.

## Preprocessing

List of steps applied before the detector (fitted on the normal data):
`"l2"`, `"standard"`, `"center"`, `"pca:0.95"` (explained variance) or `"pca:64"` (components),
`"pca_whiten:<n>"`, or any sklearn transformer.

## Data

```python
data.mvtec_ad(root, "screw")                     # MVTec AD / MVTec layout (converted VisA, etc.)
data.from_folders(train_normal, test_normal, [test_anom_1, test_anom_2])
data.one_vs_rest(root, normal_class="cat")       # semantic AD from a classification dataset
```

Each returns an `AnomalyDataset` with `train_paths`, `test_paths`, `test_labels`, `test_types` and, for
MVTec, `test_masks` (path of the ground-truth mask in `ground_truth/<type>/<name>_mask.png`, or `None`
for normal images).

## Medical and high-bit-depth images (16-bit, DICOM, NIfTI)

```bash
pip install "imgad[medical]"     # pydicom + nibabel
```

Images with more than 8 bits (16-bit PNG/TIFF, DICOM, NIfTI, float arrays) are mapped to the 8 bits the
models expect with an **intensity rule**; 8-bit images are never changed.

| Rule | Mapping |
|---|---|
| `auto` (default) | 8-bit unchanged; DICOM with a window -> that window; otherwise per-image percentiles 0.5–99.5 |
| `window:lo,hi` | fixed range, the same for every image (e.g. `window:-1000,400` for CT in HU, `window:0,4095` for a 12-bit detector). **Recommended for anomaly detection**: per-image rules hide global intensity anomalies |
| `percentile:1,99`, `minmax` | per-image percentiles / min–max |
| `dicom` | the WindowCenter/WindowWidth stored in each DICOM file |

```python
HFFeatureExtractor("microsoft/rad-dino").set_intensity("window:0,4095")
build_extractor({"type": "hf", "model_name": "microsoft/rad-dino", "intensity": "dicom"})
benchmark(..., intensity="window:-1000,400")       # every extractor / Anomalib model without its own rule
refs = data.volume_slices("scans/", axis="z")       # NIfTI / multi-frame DICOM -> "scan.nii.gz::z=42" refs
```

CLI: `--intensity` in `fit` and `benchmark` (or `intensity:` in the YAML). DICOM files (`.dcm`) are read with
their rescale slope/intercept, MONOCHROME1 is inverted, and only pixel data is read: patient metadata is
never stored or logged. Scores of volume slices are per slice; aggregate them per volume (e.g. the maximum).
The rule is part of the feature-cache key and of saved pipelines.

**Medical model zoo.** `imgad models --modality histopathology` recommends feature extractors per modality
(RAD-DINO, MedSigLIP, BiomedCLIP, UNI/UNI2-h, Virchow2, H-optimus-0, Prov-GigaPath, Phikon-v2; for the eye
RETFound, RETFound-Green, VisionFM, RetiZero, FLAIR and UrFound; + general baselines), with access and
license; use them as `build_extractor("zoo:rad-dino")` or `- zoo:uni` in a benchmark YAML (`--yaml` prints the
block). Pathology and retinal models load through the `timm` extractor, which also loads raw checkpoints
(`checkpoint="hf-hub:…/file.pth"`, a URL or `gdrive:<id>/<name>`) without executing code. See the
[documentation](https://joheras.github.io/imgad/medical/) for medical foundation models, benchmarks and
evaluation practices, and the [ophthalmology page](https://joheras.github.io/imgad/ophthalmology/) for
fundus and OCT.

## Evaluation and benchmarks

`evaluate_scores` returns AUROC, AUPR, F1-max and FPR@95%TPR (and F1/TPR/FPR at the threshold if
predictions are given). `per_type_auroc` breaks the AUROC down by defect type.

`evaluate_maps(masks, maps)` gives the localisation metrics: pixel-AUROC, pixel-AUPR, pixel-F1-max and
AUPRO (FPR ≤ 0.3, the standard MVTec AD metric). Masks are resized to each map's resolution.

`benchmark(extractors, detectors, datasets, preprocessings, anomalib_models=[...])` evaluates the full
grid, extracting features once per (extractor, dataset), plus each Anomalib model per dataset (with
pixel metrics when masks are available). In YAML, use the `anomalib:` key (see
`configs/mvtec_benchmark.yaml`).

### Progress while the benchmark runs

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

### Parallel benchmarks (multiple models on the GPU)

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

## Benchmark reports

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

## Statistical analysis: repetitions, confidence intervals and tests

**Repetitions.** `benchmark(..., seeds=[0, 1, 2, 3, 4], train_subsample=0.8)` runs every configuration
once per seed (column `seed`). The seed sets the detectors' `random_state` (unless the spec fixes it),
the calibration folds, the Anomalib seed and the **training subset** (`train_subsample`: a fraction if
float, a number of images if int). Without subsampling, deterministic detectors (kNN, Mahalanobis…)
give the same result for every seed. Features are extracted once and reused across seeds.

**Confidence intervals.** There are two sources of uncertainty, measured separately:

| Source | How | Where |
|---|---|---|
| Test sample | stratified bootstrap of the test images (`bootstrap=1000`) | `auroc_ci_low/high`, `aupr_ci_low/high` in each row |
| Training | repetitions with seeds + `train_subsample` | `stats.summarize_seeds(rows)`: mean, standard deviation and t-based CI of the mean |

**Statistical tests** (`imgad.stats`):

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

## CLI

```bash
imgad list
imgad fit --model openai/clip-vit-base-patch16 --detector knn -p k=1 --prep l2 \
          --train data/bottle/train/good --out bottle.joblib

# inference: images and/or folders; output to stdout, .csv or .json
imgad predict --model bottle.joblib new.png data/bottle/test --out preds.csv
imgad predict --model bottle.joblib folder/ --only-anomalies --threshold 0.02 --device cpu

# re-tune the threshold on labelled validation data (f1 | youden | fpr | tpr)
imgad calibrate --model bottle.joblib --normal val/good --anomalous val/defects --method f1 --out bottle_cal.joblib
imgad info --model bottle_cal.joblib         # configuration, threshold and metadata

imgad benchmark --config configs/mvtec_benchmark.yaml --out results.csv --report report/ --log-file run.log
imgad benchmark --config configs/mvtec_benchmark.yaml -j 8 --devices auto   # parallel, several models per GPU
imgad devices                                # GPUs, free memory and parallel plan

# gated Hugging Face models (see "Gated and private models")
imgad hf-login
imgad hf-check facebook/dinov3-vitb16-pretrain-lvd1689m

# Anomalib models: -p sets the model kwargs; predict saves heatmaps and masks
imgad fit --anomalib patchcore -p backbone=wide_resnet50_2 "layers=[layer2,layer3]" \
          --image-size 256 --train data/bottle/train/good --out bottle_pc.joblib
imgad predict --model bottle_pc.joblib data/bottle/test --maps-dir maps/ --out preds.csv
```

## Extending

Registering your own detector or extractor makes it available by name in Python, YAML and the CLI (see
`examples/extending.py`):

```python
from imgad import BaseDetector, register_detector

@register_detector("my_detector")
class MyDetector(BaseDetector):
    def _fit(self, X): ...
    def _score(self, X): ...   # higher = more anomalous
```

## Example results

[`benchmarks/mvtec_4categories/`](https://github.com/joheras/imgad/tree/main/benchmarks/mvtec_4categories) contains a benchmark of 5 extractors
× 11 detectors on 4 MVTec AD categories (report, results CSV and configuration).

## Project layout

```
src/imgad/
  features/      base.py (interface, registry, cache) · huggingface.py · open_clip.py
  detectors/     base.py (interface, registry, calibration) · classical.py
  anomalib_backend.py (AnomalibPipeline) · maps.py (masks, heatmaps)
  preprocessing.py · pipeline.py · inference.py · data.py · evaluation.py · stats.py · report.py
  progress.py (logging, ETA) · hub.py (gated models, tokens)
  parallel.py · devices.py (parallel benchmarks, GPU memory checks)
  imageio.py (16-bit, DICOM, NIfTI, intensity rules) · cli.py
configs/         example benchmark YAML files
examples/        quickstart.py · inference.py · compare_models.py · extending.py · anomalib_models.py
benchmarks/      example benchmark results
tests/           pytest (offline: a tiny CLIP generated locally; untrained ResNet-18 for Anomalib)
```

## Possible next steps

- **Zero-/few-shot** detection with text prompts (WinCLIP, AnomalyCLIP) using the text encoder.
- Native PatchCore on patch features of foundation models (CLIP/DINOv2), to compare with Anomalib's (CNN).
- Test-time augmentation (TTA) and multi-scale aggregation.
- Few-shot sweep (`train_subsample` = 1, 2, 4, 8… images) reusing the seeds and the tests in `imgad.stats`.

## License

Distributed under the [Apache 2.0](https://github.com/joheras/imgad/blob/main/LICENSE) license.

## Acknowledgements

This work is partially supported by Agencia de Desarrollo Económico de La Rioja [ADER 2022-I-IDI-00015].
