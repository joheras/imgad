# Inference and thresholds

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
