# imgad

**Image anomaly detection with foundation models, classical detectors and Anomalib, with statistical
benchmarking and support for medical images.**

imgad detects anomalous images in the **one-class** setting: it learns only from normal images and flags
whatever does not look like them. Defects on a production line, lesions in a scan or out-of-distribution
inputs are typical uses.

```
images ──► FeatureExtractor ──► preprocessing ──► classical detector ──► anomaly score
          (CLIP, SigLIP, DINOv2,   (l2, PCA,         (kNN, Mahalanobis,     (+ calibrated threshold)
           RAD-DINO, any HF model)  standard…)        LOF, OCSVM, IF…)
```

<div class="grid cards" markdown>

-   :material-cube-outline: **Any foundation model**

    Any vision model on the Hugging Face Hub (CLIP, SigLIP, DINOv2/v3, RAD-DINO…) or open_clip
    checkpoint becomes a feature extractor, including gated and private models.
    [Feature pipelines](guide/pipelines.md)

-   :material-map-marker-radius: **Anomalib models with maps**

    PatchCore, PaDiM, EfficientAD… under the same API, with pixel-level anomaly maps, heatmaps and
    localisation metrics (pixel-AUROC, AUPRO).
    [Anomalib models](guide/anomalib.md)

-   :material-chart-box-outline: **Rigorous benchmarks**

    Run grids of extractors × detectors × datasets with repeated seeds, bootstrap confidence intervals,
    DeLong, Friedman/Nemenyi and Wilcoxon-Holm tests, and markdown reports with figures. They can run
    in parallel, with several models sharing a GPU.
    [Benchmarks](guide/benchmarks.md)

-   :material-hospital-box-outline: **Medical imaging**

    Reads 16-bit images, DICOM and NIfTI, with configurable intensity windows. Pages on medical
    foundation models, medical benchmarks and clinical evaluation practice.
    [Medical imaging](medical.md)

</div>

## Install

```bash
pip install imgad                  # core
pip install "imgad[anomalib]"      # + Anomalib models (PatchCore, PaDiM…)
pip install "imgad[medical]"       # + DICOM and NIfTI
pip install "imgad[all]"           # everything
```

## In 10 lines

```python
from imgad import AnomalyDetectionPipeline, data
from imgad.evaluation import evaluate_scores

ds = data.mvtec_ad("/path/to/mvtec_anomaly_detection", "bottle")
pipe = AnomalyDetectionPipeline("google/siglip-base-patch16-224", "mahalanobis", preprocessing=["l2"])
pipe.fit(ds.train_paths)                          # normal images only
scores = pipe.score(ds.test_paths)                # higher = more anomalous
print(evaluate_scores(ds.test_labels, scores, pipe.predict(ds.test_paths)))
pipe.save("bottle.joblib")
```

```bash
imgad predict --model bottle.joblib new_images/ --out predictions.csv
```

Next: [Getting started](getting-started.md).
