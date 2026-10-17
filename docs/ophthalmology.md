# Ophthalmology

Eye imaging is a natural fit for one-class anomaly detection. Screening programmes (diabetic retinopathy,
glaucoma, AMD) see mostly normal eyes, and the long tail of rarer retinal diseases is hard to cover with
labelled data. RetiZero's authors, for example, collected text for more than 400 fundus diseases. A
model trained on normal retinas can flag any deviation, including conditions absent from every training
set. That is exactly where supervised classifiers fail.

This page covers the two most common modalities, **colour fundus photographs (CFP)** and **retinal OCT
B-scans**. It explains the foundation models available in imgad for them, what the literature does for
retinal anomaly detection, and how to run a credible benchmark.

!!! warning "Research use only"
    Most retinal foundation models are released for non-commercial research only (see the licenses below).
    imgad is not a medical device, and any clinical use needs validation on your own data and regulatory
    approval.

## 1. Images

- **Fundus photographs** are ordinary 8-bit RGB images (JPEG/PNG) and need no intensity rule. The
  circular field of view sits on a black background. Models resize the whole image, so crop each image
  to its field of view first when your devices frame the retina differently. Otherwise the border
  becomes a "device signature" that a detector can learn.
- **OCT B-scans** are greyscale. Exported PNG/JPEG B-scans load directly. DICOM OCT volumes (multi-frame)
  load frame by frame with slice references (`scan.dcm::frame=60`, see [Medical imaging](medical.md#volumes-ct-mri)).
  Proprietary vendor formats (`.e2e`, `.fda`, `.img`) need converting first, for example with
  [OCT-Converter](https://github.com/marksgraham/OCT-Converter).
- Wide-field, FFA, slit-lamp or external-eye photographs are different domains. Use models trained on them
  (VisionFM releases encoders for these modalities) or benchmark the general baselines.

## 2. Foundation models

```bash
pip install "imgad[medical]"
imgad models -m fundus -v           # also: -m oct, --public-only, --commercial, --yaml
```

| Model | Images | Architecture · embedding | Access · source · download | License / terms |
|---|---|---|---|---|
| `zoo:retfound-cfp` | fundus | MAE ViT-L/16 · mean patch token | gated · HF Hub · 3.9 GB | CC-BY-NC-4.0 |
| `zoo:retfound-oct` | OCT | MAE ViT-L/16 · mean patch token | gated · HF Hub · 3.9 GB | CC-BY-NC-4.0 |
| `zoo:retfound-dinov2` | fundus | DINOv2 ViT-L/14 · class token | gated · HF Hub · 1.6 GB | CC-BY-NC-4.0 |
| `zoo:retfound-green` | fundus | ViT-S/14 + 4 registers · mean patch token (392 px) | public · GitHub release · 87 MB | custom non-commercial licence |
| `zoo:flair` | fundus | ResNet-50 (vision–language) · avg pool (512 px) | public · HF Hub · 0.5 GB | Apache-2.0 |
| `zoo:urfound` | fundus, OCT | MAE + text ViT-B/16 · mean patch token | public · HF Hub · 1.8 GB | MIT |
| `zoo:retizero` | fundus | ViT-L/16 + LoRA (vision–language) · class token | public · Google Drive · 1.7 GB | no license file in the repository |
| `zoo:visionfm-fundus` | fundus | iBOT ViT-B/16 · class token | public · Google Drive · 1.5 GB | CC-BY-NC-4.0 |
| `zoo:visionfm-oct` | OCT | iBOT ViT-B/16 · class token | public · Google Drive · 1.5 GB | CC-BY-NC-4.0 |
| `zoo:retfound-cfp-mirror` | fundus | copy of RETFound CFP | public · HF Hub · 1.2 GB | CC-BY-NC-4.0 |
| `zoo:retfound-oct-mirror` | OCT | copy of RETFound OCT | public · HF Hub · 1.2 GB | CC-BY-NC-4.0 |

All of them are used like any other extractor, e.g. `build_extractor("zoo:visionfm-fundus")` or
`- zoo:flair` in a benchmark YAML. On top of these, `zoo:medsiglip` (gated) and `zoo:biomedclip` cover
ophthalmology among other modalities, and `dinov2-base` / `siglip2-base` are the general baselines.

How they differ:

- **RETFound** ([Zhou et al., Nature 2023](https://www.nature.com/articles/s41586-023-06555-x)) is the
  reference model. It is a ViT-L trained with masked autoencoding on about 1.6M unlabelled retinal images
  from Moorfields Eye Hospital, with separate CFP and OCT weights. Its authors later released DINOv2
  versions trained on the AlzEye and Shanghai cohorts, and a DINOv3 version, which imgad does not load
  yet. The official weights are gated: request access on each Hugging Face page and log in with
  `imgad hf-login`. The `-mirror` cards are public third-party copies in timm format. They are convenient
  to try the model, but they were not verified against the official files here.
- **RETFound-Green** ([Engelmann & Bernabeu, 2024](https://arxiv.org/abs/2405.00117)) is a ViT-S trained
  on 75k public images with a token-reconstruction objective. Its download is 14× smaller than RETFound's and it
  computes embeddings faster, which makes it a strong first candidate. Its licence excludes projects with industry involvement.
- **VisionFM** ([Qiu et al., NEJM AI 2024](https://ai.nejm.org/doi/full/10.1056/AIoa2300221)) trains one
  iBOT ViT-B per ophthalmic modality: fundus, OCT, FFA, slit lamp, external eye, ultrasound, UBM and MRI.
  imgad has the fundus and OCT encoders. The others load with the same recipe by changing the Google Drive
  id (listed in the VisionFM README) and the normalisation constants (its `utils.get_stats`).
- **Vision–language models.** These were trained with expert text and are designed for zero-shot
  classification. Here they are used as image encoders.
    - **FLAIR** ([Silva-Rodríguez et al., MedIA 2025](https://arxiv.org/abs/2308.07898)): a ResNet-50
      trained on 38 public datasets (288k images).
    - **RetiZero** ([Wang et al., Nature Communications 2025](https://doi.org/10.1038/s41467-025-60577-9)):
      RETFound with LoRA, trained on 341,896 images covering more than 400 diseases. imgad merges the LoRA
      weights at load time.
    - **UrFound** ([Yu et al., MICCAI 2024](https://arxiv.org/abs/2408.05618)): a single ViT-B for both
      fundus and OCT, trained on ~180k images.

Loading details. Several of these models are not Hub models: their weights are raw PyTorch checkpoints
on Hugging Face, GitHub or Google Drive. imgad loads them into the matching timm architecture
(`checkpoint`, `checkpoint_key`, `checkpoint_prefix`) with each model's own preprocessing (`data_config`).
It also reports any weight that was not found in the file:

```python
from imgad import build_extractor

ext = build_extractor({
    "type": "timm", "model_name": "vit_base_patch16_224", "pooling": "cls",
    "checkpoint": "gdrive:<file id>/visionfm_ffa.pth",   # or hf-hub:org/repo/file.pth, https://…, a local path
    "checkpoint_key": "teacher", "checkpoint_prefix": "backbone.",
    "data_config": {"mean": [0.2020, 0.2021, 0.2020], "std": [0.1741, 0.1740, 0.1740], "crop_pct": 1.0},
})
```

Checkpoints are read with `torch.load(weights_only=True)`, so a checkpoint file cannot execute code.
Downloads are cached under the torch hub directory (`~/.cache/torch/hub/checkpoints`). Google Drive
sometimes answers with a quota page instead of the file. imgad detects this and asks you to download
the file manually and pass the local path.

!!! note "Verified loading"
    Every public card in the table was loaded and checked: all weights were found, and the embedding
    dimension matches. The LoRA merge of RetiZero was checked against the adapter computation. The gated
    RETFound cards follow the official loading code, but they could not be run without access.

## 3. Anomaly detection in retinal images

What the literature does, and how it maps onto imgad:

| Family | Representative work | In imgad |
|---|---|---|
| Reconstruction / generative: train on healthy scans, score the reconstruction error | AnoGAN and f-AnoGAN on OCT ([Schlegl et al., MedIA 2019](https://doi.org/10.1016/j.media.2019.01.010)); disease-marker candidates in OCT ([Seeböck et al., 2018](https://arxiv.org/abs/1810.13404)); [Sparse-GAN](https://arxiv.org/abs/1911.12527) (ISBI 2020); [P-Net](https://arxiv.org/abs/2008.03632) (ECCV 2020), which reconstructs from the vessel / layer structure | Anomalib reconstruction models (e.g. `draem`) through the [Anomalib backend](guide/anomalib.md) |
| Knowledge distillation: a student mimics a teacher on normal data | OCT volume screening, AUROC 0.94 ([Aresta et al., TVST 2025](https://tvst.arvojournals.org/article.aspx?articleid=2802739)) | Anomalib `stfpm`, `reverse_distillation`, `efficient_ad` |
| Feature embedding + memory bank or density | PatchCore ranks second overall in [BenchReAD](https://arxiv.org/abs/2507.10492) and is the most robust to *unseen* anomalies | Foundation-model features + `knn` / `mahalanobis` / `ocsvm` (this page's models), or Anomalib `patchcore` for anomaly maps |
| Zero/few-shot vision–language | [MVFA](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_Adapting_Visual-Language_Models_for_Generalizable_Anomaly_Detection_in_Medical_Images_CVPR_2024_paper.html) (CVPR 2024) on OCT17 and RESC; FLAIR and RetiZero zero-shot disease recognition | Not yet: the image encoders are available, text prompting is not |
| Supervised / semi-supervised, with labelled anomalies | DRA and NFM-DRA, the best methods on BenchReAD, but weaker on unseen anomaly types | Out of scope (imgad is one-class) |

Practical consequences:

- **Global embeddings first.** Many retinal findings are diffuse or large (cupping, haemorrhages, fluid),
  and a kNN on foundation-model embeddings is a strong, cheap baseline. For small lesions (microaneurysms,
  drusen), compare intermediate layers and PatchCore, and look at the anomaly maps.
- **Benchmark several encoders.** No single retinal model wins everywhere, and general models (DINOv2/v3)
  are competitive on some tasks. Run the comparison with `seeds` and `bootstrap` and test the differences
  ([Statistical analysis](guide/statistics.md)) before choosing one.
- **Report results per anomaly type.** An image-level AUROC averages easy and hard diseases. The per-type
  AUROC (`per_type_auroc`, also in the benchmark report) shows which conditions a model misses.
  Keep some disease types out of every design decision, to measure the *unseen anomaly* case honestly.

## 4. Datasets and benchmarks

| Benchmark | Modality | Anomaly-detection protocol |
|---|---|---|
| [BMAD](https://github.com/DorisBao/BMAD) (CVPRW 2024) | OCT | RESC (pixel masks, edema) and OCT2017, `train/good` + `valid` / `test` with `good` / `Ungood` |
| [MVFA splits](https://github.com/MediaBrain-SJTU/MVFA-AD) (CVPR 2024) | OCT | OCT17 (image labels) and RESC (masks), for zero/few-shot |
| [BenchReAD](https://github.com/DopamineLcy/BenchReAD) (MICCAI 2025) | fundus, OCT | Fundus: train on EDDFS + BRSET, test on RIADD and JSIEC, with seen and unseen anomaly types. OCT: train on OCT2017, test on OCT2017, OCTDL and OCTID |
| [MedIAnomaly](https://github.com/caiyu6666/MedIAnomaly) (MedIA 2025) | fundus | one retinal fundus dataset among seven medical ones |

!!! warning "OCT2017 is saturated"
    BenchReAD reports near-perfect AUROC for CNV on OCT2017 for several methods. Use it for development, but
    report results on harder or external sets (OCTDL, OCTID, RESC) too.

The OCT2017 folders (`train|test/{NORMAL,CNV,DME,DRUSEN}`) map onto `from_folders`. Use its official test
split, which has different patients from the training split. Do not use `one_vs_rest`: it would re-split
by image, and B-scans of the same patient would end up on both sides.

```python
from imgad import data

oct2017 = data.from_folders("OCT2017/train/NORMAL", "OCT2017/test/NORMAL",
                            ["OCT2017/test/CNV", "OCT2017/test/DME", "OCT2017/test/DRUSEN"], name="oct2017")
```

The BMAD copies of RESC and OCT2017 use its `train/good`, `valid|test/{good,Ungood}` layout, as in the
[BMAD example](medical.md#3-use-the-medical-benchmarks). Point at the image folders, not at the mask
folders.

A benchmark of retinal encoders on fundus images:

```yaml
data:
  type: folders
  name: fundus-screening
  train_normal: /data/fundus/train/normal
  test_normal: /data/fundus/test/normal
  test_anomalous: [/data/fundus/test/dr, /data/fundus/test/glaucoma, /data/fundus/test/amd]

extractors:                      # `imgad models -m fundus --public-only --yaml` prints such a block
  - zoo:retfound-green
  - zoo:visionfm-fundus
  - zoo:flair
  - zoo:retizero
  - zoo:urfound
  - zoo:dinov2-base              # general baseline
preprocessing: [[l2]]
detectors: [{name: knn, k: 1}, mahalanobis]
seeds: [0, 1, 2]
train_subsample: 0.8
bootstrap: 1000
```

## 5. Evaluation pitfalls specific to the eye

- **Split by patient, and keep both eyes together.** Fellow eyes are strongly correlated, and so are
  repeated visits and neighbouring B-scans.
- **Image quality is a confounder.** Ungradable, blurred or poorly illuminated photographs look
  "anomalous" to every detector. Decide whether they count as anomalies, and exclude them or report
  them separately (JSIEC even has blurred-fundus categories).
- **Devices and populations shift.** Camera models, field of view, pupil dilation and ethnicity change the
  image distribution. Treat each site or device as a separate dataset, and recalibrate the threshold on
  normal images of the target site.
- **The operating point matters.** Screening needs high sensitivity at an acceptable referral rate. Report
  the sensitivity at a fixed specificity, not only AUROC ([Medical imaging](medical.md#4-evaluate-like-a-clinical-study)).
