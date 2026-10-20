# Variance decomposition of AUROC (review item R3)

Share of the total sum of squares (eta^2) over extractor x detector x preprocessing cells, AUROC averaged over seeds. 95% intervals: two-way bootstrap over extractors and detectors (2000 replicates).

## Private dataset (test set), all extractors (13 extractors x 12 detectors)

| Scope | Component | eta^2 / value | 95% interval |
| --- | --- | ---: | ---: |
| both preprocessings | extractor | 0.492 | 0.198–0.786 |
| both preprocessings | detector | 0.132 | 0.022–0.314 |
| both preprocessings | preprocessing | 0.047 | 0.007–0.156 |
| both preprocessings | extractor x preprocessing | 0.022 | 0.005–0.074 |
| both preprocessings | detector x preprocessing | 0.065 | 0.013–0.161 |
| both preprocessings | residual (extractor x detector, 3-way) | 0.242 | 0.110–0.328 |
| L2 only | extractor | 0.676 | 0.342–0.880 |
| L2 only | detector | 0.170 | 0.047–0.428 |
| L2 only | residual (extractor x detector, 3-way) | 0.155 | 0.053–0.266 |
| L2 only | range effect, detector / extractor | 0.063 / 0.135 | |
| L2 + PCA only | extractor | 0.417 | 0.175–0.802 |
| L2 + PCA only | detector | 0.240 | 0.028–0.526 |
| L2 + PCA only | residual (extractor x detector, 3-way) | 0.343 | 0.148–0.498 |
| L2 + PCA only | range effect, detector / extractor | 0.109 / 0.127 | |

## Private dataset (test set), without RETFound CFP (12 extractors x 12 detectors)

| Scope | Component | eta^2 / value | 95% interval |
| --- | --- | ---: | ---: |
| both preprocessings | extractor | 0.467 | 0.163–0.789 |
| both preprocessings | detector | 0.137 | 0.019–0.348 |
| both preprocessings | preprocessing | 0.048 | 0.005–0.162 |
| both preprocessings | extractor x preprocessing | 0.024 | 0.005–0.077 |
| both preprocessings | detector x preprocessing | 0.076 | 0.017–0.186 |
| both preprocessings | residual (extractor x detector, 3-way) | 0.247 | 0.107–0.344 |
| L2 only | extractor | 0.669 | 0.295–0.885 |
| L2 only | detector | 0.170 | 0.040–0.479 |
| L2 only | residual (extractor x detector, 3-way) | 0.161 | 0.053–0.281 |
| L2 only | range effect, detector / extractor | 0.062 / 0.135 | |
| L2 + PCA only | extractor | 0.382 | 0.148–0.782 |
| L2 + PCA only | detector | 0.272 | 0.038–0.591 |
| L2 + PCA only | residual (extractor x detector, 3-way) | 0.346 | 0.145–0.495 |
| L2 + PCA only | range effect, detector / extractor | 0.113 / 0.127 | |

## PALM (test set), all extractors (13 extractors x 12 detectors)

| Scope | Component | eta^2 / value | 95% interval |
| --- | --- | ---: | ---: |
| both preprocessings | extractor | 0.124 | 0.012–0.427 |
| both preprocessings | detector | 0.204 | 0.070–0.360 |
| both preprocessings | preprocessing | 0.164 | 0.071–0.318 |
| both preprocessings | extractor x preprocessing | 0.028 | 0.008–0.115 |
| both preprocessings | detector x preprocessing | 0.176 | 0.059–0.339 |
| both preprocessings | residual (extractor x detector, 3-way) | 0.303 | 0.076–0.508 |
| L2 only | extractor | 0.780 | 0.155–0.910 |
| L2 only | detector | 0.048 | 0.018–0.491 |
| L2 only | residual (extractor x detector, 3-way) | 0.172 | 0.066–0.477 |
| L2 only | range effect, detector / extractor | 0.042 / 0.140 | |
| L2 + PCA only | extractor | 0.143 | 0.027–0.452 |
| L2 + PCA only | detector | 0.482 | 0.141–0.842 |
| L2 + PCA only | residual (extractor x detector, 3-way) | 0.375 | 0.086–0.600 |
| L2 + PCA only | range effect, detector / extractor | 0.425 / 0.346 | |

## PALM (test set), without RETFound CFP (12 extractors x 12 detectors)

| Scope | Component | eta^2 / value | 95% interval |
| --- | --- | ---: | ---: |
| both preprocessings | extractor | 0.030 | 0.007–0.114 |
| both preprocessings | detector | 0.224 | 0.082–0.390 |
| both preprocessings | preprocessing | 0.168 | 0.073–0.338 |
| both preprocessings | extractor x preprocessing | 0.031 | 0.007–0.128 |
| both preprocessings | detector x preprocessing | 0.203 | 0.061–0.359 |
| both preprocessings | residual (extractor x detector, 3-way) | 0.343 | 0.082–0.544 |
| L2 only | extractor | 0.279 | 0.115–0.626 |
| L2 only | detector | 0.329 | 0.129–0.561 |
| L2 only | residual (extractor x detector, 3-way) | 0.393 | 0.164–0.517 |
| L2 only | range effect, detector / extractor | 0.032 / 0.031 | |
| L2 + PCA only | extractor | 0.072 | 0.018–0.272 |
| L2 + PCA only | detector | 0.515 | 0.175–0.877 |
| L2 + PCA only | residual (extractor x detector, 3-way) | 0.412 | 0.096–0.624 |
| L2 + PCA only | range effect, detector / extractor | 0.429 / 0.278 | |
