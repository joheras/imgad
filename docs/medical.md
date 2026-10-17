# Medical imaging

One-class anomaly detection fits medicine well. Normal studies are plentiful while pathologies are rare,
varied and expensive to annotate, so a model that learns only from normal anatomy can flag what deviates
from it: for screening triage, quality control or out-of-distribution checks before another model runs.

Medical images are different from photographs, though, and this page explains how to adapt imgad to them.

!!! warning "Research use only"
    imgad is a research tool, not a medical device. Several of the models below are released for research
    use only (see their licenses). Any clinical use needs validation on your own data and the applicable
    regulatory approval.

## 1. Read the images correctly

```bash
pip install "imgad[medical]"      # pydicom + nibabel
```

Medical images usually have **more than 8 bits**: 12–16-bit X-ray detectors, CT in Hounsfield units, MRI
intensities. Foundation models expect 8-bit RGB, so imgad maps every image with an **intensity rule**.
8-bit images are never changed.

| Rule | When to use it |
|---|---|
| `window:lo,hi` | **Default choice for anomaly detection.** The same mapping for every image, so a globally brighter or darker study stays different. CT: a radiological window in HU; X-ray: the detector range, e.g. `window:0,4095` |
| `dicom` | Use the window stored in each DICOM file (as radiologists view it). Different files may use different windows |
| `percentile:1,99` / `minmax` | Per-image normalisation. Robust to scanner differences, but hides global intensity anomalies |
| `auto` | 8-bit unchanged; DICOM window if present; else percentiles 0.5–99.5 |

Typical CT windows (center/width → `window:lo,hi`):

| Window | Center / width (HU) | Rule |
|---|---|---|
| Brain | 40 / 80 | `window:0,80` |
| Soft tissue / abdomen | 40 / 400 | `window:-160,240` |
| Lung | −600 / 1500 | `window:-1350,150` |
| Bone | 400 / 1800 | `window:-500,1300` |

```python
from imgad import HFFeatureExtractor, benchmark, build_extractor

ext = HFFeatureExtractor("microsoft/rad-dino").set_intensity("window:0,4095")
ext = build_extractor({"type": "hf", "model_name": "microsoft/rad-dino", "intensity": "dicom"})
rows = benchmark(..., intensity="window:-160,240")   # every model that does not set its own rule
```

From the CLI: `imgad fit ... --intensity window:0,4095` or `intensity:` at the top of the benchmark YAML.
The rule is part of the feature-cache key and is stored with the trained model, so inference uses the same
mapping.

**DICOM** (`.dcm`) files can be used anywhere a path is accepted, including folders and MVTec-style layouts.
imgad applies the rescale slope/intercept (CT → HU) and inverts MONOCHROME1 so that bright means dense.
It reads **only the pixel data**: patient name, IDs and dates are never stored, logged or written to results.

### Volumes (CT, MRI)

NIfTI volumes and multi-frame DICOM are analysed slice by slice. `data.volume_slices` returns references
such as `"scan.nii.gz::z=42"`, which every imgad function accepts as image paths:

```python
from imgad import AnomalyDetectionPipeline, data

train = data.volume_slices("normal_scans/", axis="z", step=2)       # skips empty slices
test = data.volume_slices("new_scans/", axis="z")
pipe = AnomalyDetectionPipeline({"type": "hf", "model_name": "facebook/dinov2-base",
                                 "intensity": "window:0,80"}, "mahalanobis", preprocessing=["l2"])
pipe.fit(train)
scores = pipe.score(test)
```

Volumes are reoriented to RAS, so the x/y/z axes are comparable across files. Each slice gets its own
score, so aggregate the slice scores to decide per volume, e.g. with the maximum:

```python
from collections import defaultdict
from imgad.imageio import file_of

per_volume = defaultdict(list)
for ref, s in zip(test, scores):
    per_volume[file_of(ref)].append(s)
volume_score = {v: max(s) for v, s in per_volume.items()}
```

## 2. Choose a feature extractor

General-purpose models (DINOv2/v3, CLIP, SigLIP) are strong baselines. Domain-specific models often do
better, and imgad can benchmark both under the same protocol:

| Model | Domain | Access | Use in imgad |
|---|---|---|---|
| [RAD-DINO](https://huggingface.co/microsoft/rad-dino) (Nature Machine Intelligence, 2025) | Chest X-ray | Public, research use | `"microsoft/rad-dino"` (DINOv2 backbone, 768-d) |
| [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | Biomedical figures (PMC-15M) | Public, research use | `{"type": "open_clip", "model_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"}` (`pip install "imgad[open_clip]"`) |
| UNI / UNI2-h, CONCH ([Mahmood Lab](https://github.com/mahmoodlab)) · Virchow2 (Paige) | Histopathology | **Gated**; non-commercial licenses (e.g. CONCH: CC-BY-NC-ND 4.0) | Request access, then `imgad hf-login` (see [Gated models](guide/gated-models.md)). These checkpoints load through timm/their own code, not `AutoModel`: register a small extractor ([Extending](guide/pipelines.md#extending)) |
| DINOv2 / DINOv3, SigLIP, CLIP | General | Public (DINOv3 gated) | baselines |

Check the access and license of a set of models before running a long benchmark:

```bash
imgad hf-check microsoft/rad-dino hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 MahmoodLab/UNI
```

!!! tip "Lesions are local"
    A single global embedding per image dilutes small findings, as it does with small industrial defects.
    Intermediate layers (`layer=[6, 9, 12]`, `pooling="mean"`) and patch-level models (PatchCore through
    the [Anomalib backend](guide/anomalib.md)) localise better and give anomaly maps.

## 3. Use the medical benchmarks

Two public benchmarks with an anomaly-detection protocol:

- **[BMAD](https://openaccess.thecvf.com/content/CVPR2024W/VAND/html/Bao_BMAD_Benchmarks_for_Medical_Anomaly_Detection_CVPRW_2024_paper.html)**
  (CVPR Workshops 2024). Six datasets: brain MRI (BraTS2021), liver CT (BTCV + LiTS), retinal OCT
  (RESC, OCT2017), chest X-ray (RSNA) and histopathology (Camelyon16). Each has `train/good` and
  `valid/` / `test/` folders with `good` and `Ungood` subfolders; brain MRI adds `img/` and
  `anomaly_mask/` subfolders.
  [Code and data](https://github.com/DorisBao/BMAD) (CC BY-NC-SA).
- **[MedIAnomaly](https://arxiv.org/abs/2404.04518)** (Medical Image Analysis, 2025). Seven datasets
  across chest X-ray, brain MRI, retinal fundus, dermatoscopy and histopathology, with 30 methods
  compared. [Code and data](https://github.com/caiyu6666/MedIAnomaly).

The folder layouts plug into `data.from_folders`. Point at the image folders, not at the mask folders:

```python
from imgad import data

brain = data.from_folders("BMAD/Brain/train/good/img", "BMAD/Brain/test/good/img",
                          ["BMAD/Brain/test/Ungood/img"], name="bmad/brain")
camelyon = data.from_folders("BMAD/Camelyon16/train/good", "BMAD/Camelyon16/test/good",
                             ["BMAD/Camelyon16/test/Ungood"], name="bmad/camelyon16")
```

Use the `valid/` split to choose the detector, the intensity rule and the threshold, and keep `test/` for
the final numbers.

## 4. Evaluate like a clinical study

Most of what makes a medical evaluation credible is already in imgad, provided it is used carefully:

- **Split by patient, not by image.** If one patient has several images or slices, put all of them on the
  same side of the split. Otherwise near-duplicates leak from training to test and inflate every metric.
  Build `train_paths` / `test_paths` from patient lists, and with volumes, split the volumes before calling
  `volume_slices`.
- **Report operating points, not only AUROC.** Clinicians care about sensitivity at a fixed specificity
  (or the other way round). `evaluate_scores` gives FPR@95%TPR. `calibrate_threshold(method="fpr",
  target_fpr=0.05)` fixes the false-alarm rate on normal validation images, and `method="tpr"` fixes the
  sensitivity.
- **Quantify uncertainty.** Medical test sets are small: use `bootstrap=1000` for confidence intervals,
  several `seeds` with `train_subsample` for training variability, and DeLong to compare two models on the
  same images ([Statistical analysis](guide/statistics.md)).
- **Check robustness across sites and scanners.** Domain shift between hospitals is the main failure mode.
  Treat each site (or scanner, or subgroup) as a separate dataset in the benchmark: the per-dataset results
  and the Friedman/Wilcoxon tests show whether a model's ranking holds across sites.
- **Calibrate on the target site.** The automatic threshold is estimated on the training distribution.
  Recalibrate it on a few normal studies from each new site (`imgad calibrate --method fpr`).

## 5. Protect patient data

- Only pixel data is read from DICOM/NIfTI, but **file paths end up in results, reports and logs**. If your
  file names contain patient names or IDs, pseudonymise them before running imgad.
- Benchmark reports include example images. Do not publish reports built on non-public data.
- Hugging Face tokens are never stored by imgad ([Gated models](guide/gated-models.md)).

## 6. Not yet in imgad

These are natural next steps; contributions are welcome.

- Patient-aware split helpers and a "sensitivity at fixed specificity" metric in reports.
- Loaders for BMAD and MedIAnomaly, and a timm-based extractor for pathology models such as UNI.
- Patch-level kNN on foundation-model features (anomaly maps without Anomalib).
- Zero/few-shot detection with vision-language models adapted to medicine, e.g.
  [MVFA](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_Adapting_Visual-Language_Models_for_Generalizable_Anomaly_Detection_in_Medical_Images_CVPR_2024_paper.html)
  (CVPR 2024).
- Showing model licenses in `imgad hf-check` and in reports.
