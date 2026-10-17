# Data and image formats

## Dataset loaders

```python
data.mvtec_ad(root, "screw")                     # MVTec AD / MVTec layout (converted VisA, etc.)
data.from_folders(train_normal, test_normal, [test_anom_1, test_anom_2])
data.one_vs_rest(root, normal_class="cat")       # semantic AD from a classification dataset
```

Each returns an `AnomalyDataset` with `train_paths`, `test_paths`, `test_labels`, `test_types` and, for
MVTec, `test_masks` (path of the ground-truth mask in `ground_truth/<type>/<name>_mask.png`, or `None`
for normal images).

## High-bit-depth and medical images

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
The rule is part of the feature-cache key and of saved pipelines. See [Medical imaging](../medical.md) for medical foundation models, benchmarks and
evaluation practices.
