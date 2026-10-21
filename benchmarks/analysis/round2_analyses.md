# Second-round re-analyses (RR2, RR4, RR5, RR7, RR8)

Collapsed Anomalib models (PALM test AUROC < 0.65): cflow, draem, supersimplenet, uninet.

## RR2. Family comparison with an equal selection budget

Each extractor with one pre-specified detector and L2 preprocessing (no selection), against each Anomalib model's single run. Medians over methods; two-sided Mann-Whitney U.

| Dataset | Extractor rule | Anomalib set | Median extractor | Median Anomalib | U | p |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| private | L2 + knn(k=1) | all 19 | 0.596 | 0.527 | 223.0 | 0.00015 |
| private | L2 + knn(k=1) | 15 not collapsed | 0.596 | 0.529 | 173.0 | 0.00055 |
| private | L2 + mahalanobis | all 19 | 0.571 | 0.527 | 216.0 | 0.00042 |
| private | L2 + mahalanobis | 15 not collapsed | 0.571 | 0.529 | 166.0 | 0.0017 |
| private | L2 + lof | all 19 | 0.588 | 0.527 | 237.0 | 1.5e-05 |
| private | L2 + lof | 15 not collapsed | 0.588 | 0.529 | 185.0 | 6.1e-05 |
| private | best of 24 (paper) | all 19 | 0.596 | 0.527 | 241.0 | 7.1e-06 |
| private | best of 24 (paper) | 15 not collapsed | 0.596 | 0.529 | 189.0 | 2.8e-05 |
| palm-test | L2 + knn(k=1) | all 19 | 0.982 | 0.944 | 200.0 | 0.0035 |
| palm-test | L2 + knn(k=1) | 15 not collapsed | 0.982 | 0.960 | 148.0 | 0.021 |
| palm-test | L2 + mahalanobis | all 19 | 0.989 | 0.944 | 218.5 | 0.00029 |
| palm-test | L2 + mahalanobis | 15 not collapsed | 0.989 | 0.960 | 166.5 | 0.0016 |
| palm-test | L2 + lof | all 19 | 0.985 | 0.944 | 209.0 | 0.0011 |
| palm-test | L2 + lof | 15 not collapsed | 0.985 | 0.960 | 157.0 | 0.0066 |
| palm-test | best of 24 (paper) | all 19 | 0.990 | 0.944 | 224.5 | 0.00012 |
| palm-test | best of 24 (paper) | 15 not collapsed | 0.990 | 0.960 | 172.5 | 0.0006 |

Top configuration of each family on PALM, chosen on the validation set (test AUROC, 95% bootstrap CI averaged over seeds). DeLong needs per-image scores, which are not released; overlapping intervals are reported instead.

| Family | Configuration | Val AUROC | Test AUROC (95% CI) |
| --- | --- | ---: | ---: |
| Extractor + detector | vit_large_patch16_224[cls] / l2 / lof | 0.995 | 0.989 (0.982-0.995) |
| Anomalib | anomalib / - / dfm(image_size=256) | 0.985 | 0.989 (0.981-0.995) |
| Text prompts | google/medsiglip-448 / l2 / zero_shot(modality=fundus) | 0.987 | 0.988 (0.977-0.997) |

## RR4. Rank agreement between the two datasets

Spearman rho (95% bootstrap interval over methods) and Kendall tau between the per-method AUROC on the private test set and on the PALM test set, with the same selection rule on both datasets.

| Rule | Methods | n | rho (95% CI) | tau |
| --- | --- | ---: | ---: | ---: |
| best configuration on test | all | 32 | 0.83 (0.67 to 0.91) | 0.65 |
| best configuration on test | all, without collapsed | 28 | 0.79 (0.60 to 0.89) | 0.60 |
| best configuration on test | extractors only | 13 | 0.71 (0.16 to 0.94) | 0.54 |
| best configuration on test | Anomalib only | 19 | 0.68 (0.28 to 0.88) | 0.50 |
| best configuration on test | Anomalib, without collapsed | 15 | 0.58 (0.10 to 0.85) | 0.39 |
| L2 + kNN (k=1) for extractors | all | 32 | 0.68 (0.43 to 0.84) | 0.50 |
| L2 + kNN (k=1) for extractors | all, without collapsed | 28 | 0.58 (0.29 to 0.80) | 0.42 |
| L2 + kNN (k=1) for extractors | extractors only | 13 | 0.54 (-0.04 to 0.88) | 0.41 |
| L2 + kNN (k=1) for extractors | Anomalib only | 19 | 0.68 (0.28 to 0.88) | 0.50 |
| L2 + kNN (k=1) for extractors | Anomalib, without collapsed | 15 | 0.58 (0.10 to 0.84) | 0.39 |

For reference, the family label alone has rho = 0.81 with the private AUROC and 0.70 with the PALM AUROC.

## RR5. False-positive rate at the calibrated threshold (target 5%)

Distribution over configurations of the seed-averaged false-positive rate on normal images at the threshold set from out-of-sample normal training scores.

| Split | Family | Configurations | Median FPR | IQR | Max | % > 5% | % > 10% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| private | Extractor + detector | 312 | 0.054 | 0.041-0.060 | 0.077 | 62 | 0 |
| private | Text prompts | 10 | 0.054 | 0.050-0.058 | 0.063 | 60 | 0 |
| private | Anomalib | 19 | 0.060 | 0.057-0.066 | 0.097 | 95 | 0 |
| palm-val | Extractor + detector | 312 | 0.026 | 0.004-0.044 | 0.115 | 19 | 1 |
| palm-val | Text prompts | 15 | 0.048 | 0.040-0.056 | 0.102 | 33 | 7 |
| palm-val | Anomalib | 19 | 0.092 | 0.065-0.105 | 0.295 | 84 | 26 |
| palm-test | Extractor + detector | 312 | 0.011 | 0.000-0.024 | 0.093 | 3 | 0 |
| palm-test | Text prompts | 15 | 0.023 | 0.010-0.034 | 0.077 | 13 | 0 |
| palm-test | Anomalib | 19 | 0.046 | 0.032-0.060 | 0.230 | 37 | 5 |

Selected configurations: FPR averaged over seeds, with an exact 95% Clopper-Pearson interval for one test set (the seeds share the test images), and per-seed range.

| Split | Configuration | FPR | 95% CI | Per-seed range |
| --- | --- | ---: | ---: | ---: |
| private | retizero-clip / l2 / knn(k=1) | 0.065 | 0.050-0.083 | 0.064-0.066 |
| palm-val | vit_large_patch16_224[cls] / l2 / lof | 0.000 | 0.000-0.019 | 0.000-0.000 |
| palm-test | vit_large_patch16_224[cls] / l2 / lof | 0.002 | 0.000-0.020 | 0.000-0.005 |

## RR7. Operating characteristics and predictive values of the selected configurations

Sensitivity and specificity averaged over the 3 seeds, with exact 95% intervals for one test set. Specificity at 95% sensitivity from the ROC curve. PPV and NPV at the calibrated threshold for several prevalences, and outcomes per 1,000 people screened.

| Dataset | Sensitivity (95% CI) | Specificity (95% CI) | Specificity at 95% sensitivity |
| --- | ---: | ---: | ---: |
| private | 0.144 (0.123-0.169) | 0.935 (0.917-0.950) | 0.164 |
| palm-test | 0.739 (0.673-0.795) | 0.998 (0.980-1.000) | 0.943 |

| Dataset | Prevalence | PPV (range over the specificity CI) | NPV | True positives / 1,000 | False positives / 1,000 | Missed / 1,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| private | 1% | 0.02 (0.02-0.03) | 0.991 | 1.4 | 64.3 | 8.6 |
| private | 3% | 0.06 (0.05-0.08) | 0.972 | 4.3 | 63.0 | 25.7 |
| private | 10% | 0.20 (0.16-0.24) | 0.908 | 14.4 | 58.5 | 85.6 |
| palm-test | 1% | 0.81 (0.28-1.00) | 0.997 | 7.4 | 1.8 | 2.6 |
| palm-test | 3% | 0.93 (0.54-1.00) | 0.992 | 22.2 | 1.7 | 7.8 |
| palm-test | 10% | 0.98 (0.81-1.00) | 0.972 | 73.9 | 1.6 | 26.1 |

## RR8. Counts without the RETFound mirror

- private: 341 configurations; 13 distinct extractors, of which 8 score above every Anomalib model; PCA lowers AUROC in 118 of 156 extractor-detector pairs (mean change -0.019).
- palm-test: 346 configurations; 13 distinct extractors, of which 4 score above every Anomalib model; PCA lowers AUROC in 138 of 156 extractor-detector pairs (mean change -0.099).

Mean AUROC per detector over the 13 distinct extractors, private dataset (Table 6, private columns):

| Detector | L2 | L2 + PCA |
| --- | ---: | ---: |
| lof | 0.599 | 0.550 |
| knn(k=1) | 0.596 | 0.557 |
| ensemble | 0.590 | 0.579 |
| knn(k=5) | 0.580 | 0.541 |
| mahalanobis | 0.577 | 0.565 |
| pca | 0.569 | 0.561 |
| gmm | 0.560 | 0.563 |
| ocsvm | 0.559 | 0.554 |
| iforest | 0.557 | 0.562 |
| kmeans(n_clusters=8) | 0.556 | 0.553 |
| centroid | 0.552 | 0.489 |
| kde | 0.552 | 0.548 |
