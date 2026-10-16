# Anomaly detection benchmark: foundation models + classical detectors

*`imgad` framework · MVTec AD (4 categories) · 16 October 2026*

## Summary

**Question:** with a pretrained feature extractor and a classical detector trained on normal images only, which combination detects image-level anomalies best?

- **Best overall combination:** SigLIP B/16 + kNN+Mahal. ensemble, with a mean AUROC of **0.974** over the 4 categories.
- **Best extractor** (with its best detector): SigLIP B/16 0.974, DINOv2-B · layers 6/9/12 0.966, DINOv2-B 0.959, CLIP ViT-B/16 0.933, CLIP ViT-B/16 · layers 6/9/12 0.918.
- **Best detector** (mean over extractors): Mahalanobis 0.946, kNN+Mahal. ensemble 0.945, PCA (90% var.) 0.944, kNN (k=1) 0.926; the worst: OCSVM 0.828, Centroid 0.836.
- **Hardest category:** screw (best AUROC 0.927, SigLIP B/16 + kNN+Mahal. ensemble).
- Choosing the best combination per category would give a mean AUROC of 0.978 (optimistic bound: the selection is made on the test set).


**Main conclusions**

1. **SigLIP B/16 is the best extractor** in all 4 categories (tied on *bottle*, where almost everything scores 1.000), and also the best on average over all detectors (0.936 vs. 0.911 for DINOv2-B and 0.885 for CLIP).
2. **Gaussian detectors are the best with any extractor.** Mahalanobis, PCA and the kNN+Mahalanobis ensemble take the top three places with 4 of the 5 extractors; with SigLIP, kNN (k=1) overtakes PCA by 0.001. Isolation Forest, One-Class SVM and the distance to the centroid are the bottom three with all 5 extractors.
3. **The detector matters almost as much as the extractor.** With CLIP, the mean AUROC ranges from 0.830 (OCSVM) to 0.933 (PCA) depending on the detector. With Mahalanobis, changing the extractor moves it between 0.916 and 0.968.
4. **Intermediate layers with global pooling do not help.** With DINOv2 they barely change the result (+0.001 on average) and with CLIP they make it worse (−0.016), although they improve *hazelnut*.
5. **Default recommendation:** SigLIP + Mahalanobis (0.968 on average), simpler than the ensemble and only 0.006 behind.

## Experimental setup

| | |
|---|---|
| Data | MVTec AD (copy `foersben/mvtec-ad` on the HF Hub): bottle, hazelnut, screw, carpet |
| Training | normal images only (`train/good`): bottle 209, hazelnut 391, screw 320, carpet 280 |
| Test | normal / anomalous: bottle 20/63, hazelnut 40/70, screw 41/119, carpet 28/89 |
| Extractors | CLIP ViT-B/16; CLIP ViT-B/16 · layers 6/9/12; SigLIP B/16; DINOv2-B; DINOv2-B · layers 6/9/12 |
| Preprocessing | L2 normalisation |
| Detectors | kNN (k=1), kNN (k=5), Mahalanobis, Centroid, k-means (8), LOF, GMM (4), OCSVM, Isolation Forest, PCA (90% var.), kNN+Mahal. ensemble |
| Main metric | image-level AUROC (also AUPR, F1-max, FPR@95%TPR) |
| Hardware | CPU (2 cores), no GPU |

Pooling: CLIP and SigLIP use the projected embedding (`get_image_features`); DINOv2-B concatenates the CLS token and the mean of the patch tokens; the "layers 6/9/12" variants concatenate the mean of the patch tokens of those ViT layers (dimension 3×768). Decision threshold: 95% quantile of out-of-sample normal scores.

## Mean AUROC (4 categories): extractor × detector

| Detector | CLIP ViT-B/16 | CLIP ViT-B/16 · layers 6/9/12 | SigLIP B/16 | DINOv2-B | DINOv2-B · layers 6/9/12 |
|---|---:|---:|---:|---:|---:|
| Mahalanobis | 0.924 | 0.916 | 0.968 | 0.959 | 0.966 |
| kNN+Mahal. ensemble | 0.921 | 0.911 | **0.974** | 0.959 | 0.960 |
| PCA (90% var.) | 0.933 | 0.918 | 0.964 | 0.956 | 0.951 |
| kNN (k=1) | 0.902 | 0.877 | 0.965 | 0.949 | 0.938 |
| kNN (k=5) | 0.898 | 0.864 | 0.955 | 0.935 | 0.929 |
| LOF | 0.897 | 0.818 | 0.950 | 0.911 | 0.923 |
| k-means (8) | 0.873 | 0.813 | 0.946 | 0.901 | 0.903 |
| GMM (4) | 0.869 | 0.803 | 0.926 | 0.896 | 0.897 |
| Isolation Forest | 0.846 | 0.777 | 0.904 | 0.864 | 0.808 |
| Centroid | 0.844 | 0.787 | 0.867 | 0.844 | 0.837 |
| OCSVM | 0.830 | 0.770 | 0.881 | 0.843 | 0.818 |
| *Extractor mean* | *0.885* | *0.841* | *0.936* | *0.911* | *0.903* |

## Results per category

AUROC of each extractor with its best detector on that category (detector in brackets).

| Extractor | bottle | hazelnut | screw | carpet |
|---|---:|---:|---:|---:|
| CLIP ViT-B/16 | 1.000 (Mahalanobis) | 0.949 (PCA (90% var.)) | 0.814 (PCA (90% var.)) | 0.976 (Mahalanobis) |
| CLIP ViT-B/16 · layers 6/9/12 | 1.000 (Mahalanobis) | 0.971 (PCA (90% var.)) | 0.775 (Mahalanobis) | 0.929 (Mahalanobis) |
| SigLIP B/16 | 1.000 (kNN (k=1)) | 0.998 (PCA (90% var.)) | 0.927 (kNN+Mahal. ensemble) | 0.986 (Mahalanobis) |
| DINOv2-B | 1.000 (kNN (k=1)) | 0.975 (Mahalanobis) | 0.900 (kNN (k=1)) | 0.984 (Mahalanobis) |
| DINOv2-B · layers 6/9/12 | 0.998 (Mahalanobis) | 0.979 (Mahalanobis) | 0.905 (Mahalanobis) | 0.980 (Mahalanobis) |

## Analysis per defect type

Best overall combination (SigLIP B/16 + kNN+Mahalanobis ensemble). AUROC of each defect type against all normal test images of its category.

| Category | Defect type | No. test | AUROC |
|---|---|---:|---:|
| bottle | broken_large | 20 | 1.000 |
| bottle | broken_small | 22 | 1.000 |
| bottle | contamination | 21 | 1.000 |
| hazelnut | crack | 18 | 1.000 |
| hazelnut | cut | 17 | 0.991 |
| hazelnut | hole | 18 | 1.000 |
| hazelnut | print | 17 | 0.957 |
| screw | manipulated_front | 24 | 0.986 |
| screw | scratch_head | 24 | 0.933 |
| screw | scratch_neck | 25 | 0.969 |
| screw | thread_side | 23 | 0.874 |
| screw | thread_top | 23 | 0.870 |
| carpet | color | 19 | 0.998 |
| carpet | cut | 17 | 0.996 |
| carpet | hole | 17 | 0.996 |
| carpet | metal_contamination | 17 | 0.966 |
| carpet | thread | 19 | 0.955 |

Failures concentrate on **small, local** defects: the threads of *screw* (`thread_side` 0.874 and `thread_top` 0.870), `thread` and `metal_contamination` on *carpet*, and `print` on *hazelnut*. Large defects or defects that change the global appearance (breaks, holes, colour changes) are detected almost perfectly. This is expected with a single embedding per image: a defect covering a tiny fraction of the image barely shifts the global vector. On *screw*, in addition, the screws appear in arbitrary orientations, so much of the variation of the embedding is due to pose rather than to defects.

## Effect of using intermediate layers

AUROC difference (intermediate layers − last layer), with the best detector of each variant.

| Backbone | bottle | hazelnut | screw | carpet | Mean |
|---|---:|---:|---:|---:|---:|
| CLIP ViT-B/16 | +0.000 | +0.023 | -0.039 | -0.047 | -0.016 |
| DINOv2-B | -0.002 | +0.004 | +0.005 | -0.004 | +0.001 |

## Operating point (automatic threshold)

For the best overall combination (SigLIP B/16 + kNN+Mahal. ensemble) with the automatically calibrated threshold (95% quantile of out-of-sample normal scores, without using any anomaly):

| Category | AUROC | AUPR | F1-max | FPR@95%TPR | TPR@thr | FPR@thr | F1@thr |
|---|---:|---:|---:|---:|---:|---:|---:|
| bottle | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| hazelnut | 0.987 | 0.993 | 0.971 | 0.050 | 0.886 | 0.050 | 0.925 |
| screw | 0.927 | 0.973 | 0.924 | 0.390 | 0.639 | 0.024 | 0.776 |
| carpet | 0.982 | 0.995 | 0.972 | 0.071 | 1.000 | 0.464 | 0.932 |

## Discussion

**Why Gaussian detectors win.** With L2-normalised foundation-model embeddings, the normal images of a category form a compact and roughly unimodal cloud. Mahalanobis (with Ledoit-Wolf covariance) and PCA reconstruction explicitly model the directions along which normal data vary. Thus a small deviation along a direction in which normal data barely vary counts much more than the same deviation along a high-variance direction. kNN and the distance to the centroid treat all directions equally. Isolation Forest makes random axis-aligned splits, which suit dense 512–2304-dimensional vectors poorly because the information is spread across dimensions. The One-Class SVM with an RBF kernel and `gamma="scale"` is sensitive to scale and was not tuned.

**Intermediate layers.** The literature shows that intermediate layers help with local defects, but that result comes from methods that compare individual **patches** (e.g. PatchCore). Here the patch tokens of each layer are averaged into a single vector, so local information is diluted anyway. In addition, the dimension rises to 2304 with only 209–391 training images, which hurts the detectors that estimate densities or covariances. With CLIP, intermediate layers improve *hazelnut* (+0.023) but worsen *carpet* (−0.047) and *screw* (−0.039).

**Automatic threshold.** For the best combination, the threshold set at the 95% quantile of the normal training scores (without using any anomaly) gives a test FPR of 0–5% on *bottle*, *hazelnut* and *screw*, as expected. On *carpet* the FPR rises to 46% (13 of 28 normal test images): the scores of the normal test images are higher than those of the training images. The ranking is still good (AUROC 0.982), so this is a calibration problem, not a separability problem. In production the threshold should be recalibrated on a small validation set (`imgad calibrate --method fpr`).

**Context.** Patch-level methods (e.g. PatchCore, Roth et al., CVPR 2022) reach image-level AUROCs around 0.99 on MVTec AD. The global embedding + classical detector approach is simpler, trains in seconds and makes switching backbones trivial, but its ceiling is set by small defects. The mean of 0.974 obtained here (on 4 categories) is a solid baseline.

## Feature dimension

| Extractor | Dimension |
|---|---:|
| CLIP ViT-B/16 | 512 |
| CLIP ViT-B/16 · layers 6/9/12 | 2304 |
| SigLIP B/16 | 768 |
| DINOv2-B | 1536 |
| DINOv2-B · layers 6/9/12 | 2304 |

On the 2-core CPU used, extraction with a ViT-B/16 at 224 px ran at about 2.5–4 images/s (including decoding 900–1024 px PNGs); with a GPU it is in the order of hundreds of images/s. Fitting and scoring any detector takes less than a few seconds per category.

## Limitations

- **Only 4 of the 15 categories** of MVTec AD. *bottle* turns out to be trivial and *screw* is one of the hardest; the conclusions may change with the full set.
- **Small test sets** (20–41 normal images per category): AUROC differences of ±0.01–0.02 between combinations are not significant.
- **No hyperparameter tuning:** all detectors use default values. Picking the best combination by looking at the test set is optimistic; a rigorous comparison would select on a validation set.
- **Image-level metrics only**: localisation (pixel-AUROC) is not evaluated.
- **Base models only (ViT-B)** for lack of a GPU; *large* models could change the ranking.

## Proposed next steps

1. Complete the 15 categories of MVTec AD and add VisA, with a GPU.
2. Try larger backbones: SigLIP-L / SigLIP2, DINOv2-L, CLIP ViT-L/14.
3. Implement a **patch-level** extractor with a memory bank (PatchCore-style) to tackle local defects and obtain anomaly maps.
4. Separate validation from test to select the detector and hyperparameters without bias.
5. Zero-shot detection with text prompts (WinCLIP-style), using the text encoder of CLIP/SigLIP.

## Reproducibility

Files in this folder:

- `results.csv`: the 220 rows (5 extractors × 11 detectors × 4 categories) with AUROC, AUPR, F1-max, FPR@95%TPR, metrics at the threshold and dimension.
- `config.yaml`: exact benchmark configuration (change `root` to the local path of MVTec AD).

```bash
# data: copy of MVTec AD on the Hub (original layout)
huggingface-cli download foersben/mvtec-ad --repo-type dataset --local-dir mvtec \
    --include "bottle/*" "hazelnut/*" "screw/*" "carpet/*"

imgad benchmark --config benchmarks/mvtec_4categories/config.yaml --out results.csv
```

Environment: `imgad` 0.1.0, torch 2.14, transformers 5.20, scikit-learn 1.9, Python 3.13; 2-core CPU without GPU.

## Appendix: AUROC per category, extractor and detector

#### bottle

| Detector | CLIP ViT-B/16 | CLIP ViT-B/16 · layers 6/9/12 | SigLIP B/16 | DINOv2-B | DINOv2-B · layers 6/9/12 |
|---|---:|---:|---:|---:|---:|
| kNN (k=1) | 0.994 | 0.995 | 1.000 | 1.000 | 0.989 |
| kNN (k=5) | 0.992 | 0.992 | 1.000 | 0.998 | 0.990 |
| Mahalanobis | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 |
| Centroid | 0.993 | 0.983 | 0.992 | 0.991 | 0.976 |
| k-means (8) | 0.985 | 0.979 | 1.000 | 0.998 | 0.984 |
| LOF | 0.992 | 0.994 | 1.000 | 0.997 | 0.988 |
| GMM (4) | 0.983 | 0.994 | 1.000 | 0.990 | 0.980 |
| OCSVM | 0.985 | 0.981 | 0.986 | 0.994 | 0.973 |
| Isolation Forest | 0.991 | 0.993 | 0.997 | 0.990 | 0.986 |
| PCA (90% var.) | 1.000 | 1.000 | 1.000 | 1.000 | 0.995 |
| kNN+Mahal. ensemble | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 |

#### hazelnut

| Detector | CLIP ViT-B/16 | CLIP ViT-B/16 · layers 6/9/12 | SigLIP B/16 | DINOv2-B | DINOv2-B · layers 6/9/12 |
|---|---:|---:|---:|---:|---:|
| kNN (k=1) | 0.884 | 0.891 | 0.966 | 0.917 | 0.907 |
| kNN (k=5) | 0.898 | 0.866 | 0.966 | 0.919 | 0.909 |
| Mahalanobis | 0.942 | 0.961 | 0.996 | 0.975 | 0.979 |
| Centroid | 0.871 | 0.790 | 0.878 | 0.885 | 0.846 |
| k-means (8) | 0.890 | 0.849 | 0.959 | 0.921 | 0.919 |
| LOF | 0.905 | 0.860 | 0.976 | 0.918 | 0.932 |
| GMM (4) | 0.893 | 0.841 | 0.949 | 0.924 | 0.920 |
| OCSVM | 0.878 | 0.792 | 0.930 | 0.914 | 0.879 |
| Isolation Forest | 0.862 | 0.863 | 0.919 | 0.902 | 0.790 |
| PCA (90% var.) | 0.949 | 0.971 | 0.998 | 0.965 | 0.974 |
| kNN+Mahal. ensemble | 0.927 | 0.947 | 0.987 | 0.959 | 0.963 |

#### screw

| Detector | CLIP ViT-B/16 | CLIP ViT-B/16 · layers 6/9/12 | SigLIP B/16 | DINOv2-B | DINOv2-B · layers 6/9/12 |
|---|---:|---:|---:|---:|---:|
| kNN (k=1) | 0.754 | 0.700 | 0.925 | 0.900 | 0.881 |
| kNN (k=5) | 0.729 | 0.679 | 0.896 | 0.846 | 0.847 |
| Mahalanobis | 0.776 | 0.775 | 0.889 | 0.875 | 0.905 |
| Centroid | 0.541 | 0.495 | 0.679 | 0.529 | 0.575 |
| k-means (8) | 0.647 | 0.526 | 0.869 | 0.716 | 0.757 |
| LOF | 0.715 | 0.512 | 0.864 | 0.755 | 0.809 |
| GMM (4) | 0.629 | 0.479 | 0.806 | 0.696 | 0.727 |
| OCSVM | 0.484 | 0.448 | 0.670 | 0.504 | 0.484 |
| Isolation Forest | 0.570 | 0.458 | 0.753 | 0.597 | 0.637 |
| PCA (90% var.) | 0.814 | 0.772 | 0.882 | 0.881 | 0.865 |
| kNN+Mahal. ensemble | 0.780 | 0.769 | 0.927 | 0.895 | 0.904 |

#### carpet

| Detector | CLIP ViT-B/16 | CLIP ViT-B/16 · layers 6/9/12 | SigLIP B/16 | DINOv2-B | DINOv2-B · layers 6/9/12 |
|---|---:|---:|---:|---:|---:|
| kNN (k=1) | 0.974 | 0.921 | 0.971 | 0.979 | 0.973 |
| kNN (k=5) | 0.973 | 0.917 | 0.959 | 0.978 | 0.971 |
| Mahalanobis | 0.976 | 0.929 | 0.986 | 0.984 | 0.980 |
| Centroid | 0.972 | 0.882 | 0.920 | 0.970 | 0.951 |
| k-means (8) | 0.970 | 0.899 | 0.955 | 0.970 | 0.953 |
| LOF | 0.976 | 0.906 | 0.961 | 0.974 | 0.965 |
| GMM (4) | 0.970 | 0.898 | 0.948 | 0.973 | 0.959 |
| OCSVM | 0.974 | 0.857 | 0.938 | 0.961 | 0.936 |
| Isolation Forest | 0.962 | 0.796 | 0.948 | 0.964 | 0.819 |
| PCA (90% var.) | 0.970 | 0.928 | 0.975 | 0.976 | 0.971 |
| kNN+Mahal. ensemble | 0.976 | 0.929 | 0.982 | 0.982 | 0.978 |
