# Feature pipelines

A feature pipeline turns each image into an embedding with a frozen pretrained model, preprocesses it and scores it with a one-class detector fitted on normal images only:

```python
from imgad import AnomalyDetectionPipeline, HFFeatureExtractor, build_detector

pipe = AnomalyDetectionPipeline(HFFeatureExtractor("facebook/dinov2-base", pooling="cls+mean"),
                                build_detector("mahalanobis"), preprocessing=["l2"])
```

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


**timm models** (including Hub checkpoints such as the pathology models UNI or Virchow2):
`TimmFeatureExtractor("hf-hub:org/name", pooling="auto" | "cls" | "mean" | "cls+mean", timm_kwargs={...})`, or
`{"type": "timm", ...}` in specs. Classes in `timm_kwargs` are given as import paths (`"torch.nn.SiLU"`) so that
configurations stay serialisable. Ready-made medical recipes: `build_extractor("zoo:uni")`, see the
[model zoo](../medical.md#2-choose-a-feature-extractor).

Gated or private models (DINOv3, UNI, CONCH…) need a token: see [Gated Hugging Face models](gated-models.md).

## Preprocessing

List of steps applied before the detector (fitted on the normal data):
`"l2"`, `"standard"`, `"center"`, `"pca:0.95"` (explained variance) or `"pca:64"` (components),
`"pca_whiten:<n>"`, or any sklearn transformer.

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
