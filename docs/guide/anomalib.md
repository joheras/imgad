# Anomalib models and anomaly maps

```bash
pip install "imgad[anomalib]"
```

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
