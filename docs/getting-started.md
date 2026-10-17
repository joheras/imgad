# Getting started

## Installation

imgad needs Python ≥ 3.10 and PyTorch.

=== "pip"

    ```bash
    pip install imgad
    pip install "imgad[anomalib,medical,report]"   # optional extras
    ```

=== "From source"

    ```bash
    git clone https://github.com/joheras/imgad
    cd imgad
    pip install -e ".[dev]"
    pytest                                          # offline test suite
    ```

| Extra | Adds |
|---|---|
| `anomalib` | Anomalib models (PatchCore, PaDiM, EfficientAD, FastFlow…) with anomaly maps |
| `medical` | DICOM (`pydicom`) and NIfTI (`nibabel`) reading |
| `report` | figures in benchmark reports (`matplotlib`) |
| `open_clip`, `pyod`, `pandas` | open_clip checkpoints, PyOD detectors, DataFrames |
| `all` | everything above + `pytest` |

## Your data

imgad works with lists of image paths. For the common layouts there are loaders that return an
`AnomalyDataset` (train paths, test paths, test labels, defect types and, if available, masks):

```python
from imgad import data

ds = data.mvtec_ad("/data/mvtec", "screw")          # root/<category>/{train/good, test/<type>, ground_truth/<type>}
ds = data.from_folders("train/normal", "test/normal", ["test/defect_a", "test/defect_b"])
print(ds.summary())
```

## Train, evaluate and save

```python
from imgad import AnomalyDetectionPipeline
from imgad.evaluation import evaluate_scores

pipe = AnomalyDetectionPipeline(
    extractor="facebook/dinov2-base",       # any HF id, or HFFeatureExtractor(...)
    detector={"name": "knn", "k": 1},       # or "mahalanobis", "lof", ...
    preprocessing=["l2"],
    cache_dir=".feature_cache",             # features are cached per image
)
pipe.fit(ds.train_paths)
scores = pipe.score(ds.test_paths)
preds = pipe.predict(ds.test_paths)         # 0/1 with the automatic threshold
print(evaluate_scores(ds.test_labels, scores, preds))
pipe.save("screw.joblib")
```

The automatic threshold is set from normal images only: it is the 95% quantile of their
out-of-sample scores, so you can expect about 5% false alarms. To change it, see
[Inference and thresholds](guide/inference.md).

## Use it on new images

```python
from imgad import AnomalyPredictor

predictor = AnomalyPredictor.load("screw.joblib")
r = predictor.predict_one("new.png")
print(r.is_anomaly, r.score, r.normal_percentile)
predictor.save_results(predictor.predict("folder/"), "predictions.csv")
```

## Compare many models at once

Put the grid in a YAML file:

```yaml title="benchmark.yaml"
data: {type: mvtec, root: /data/mvtec, categories: [bottle, screw, carpet]}
cache_dir: .feature_cache
extractors:
  - openai/clip-vit-base-patch16
  - google/siglip-base-patch16-224
  - facebook/dinov2-base
preprocessing: [[l2]]
detectors: [{name: knn, k: 1}, mahalanobis, iforest]
seeds: [0, 1, 2]
train_subsample: 0.8
bootstrap: 1000
```

```bash
imgad benchmark --config benchmark.yaml --report report/ -j 8 --devices auto
```

This writes `report/report.md` (data analysis, methods, results with figures, statistical tests and
discussion). For the details, see [Benchmarks](guide/benchmarks.md) and
[Benchmark reports](guide/reports.md).
