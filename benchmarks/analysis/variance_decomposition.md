# Variance decomposition of AUROC (review items R3 and RR3)

Share of the total sum of squares (eta^2) over extractor x detector x preprocessing cells of test AUROC averaged over seeds, from each benchmark's results.csv. 95% intervals: two-way bootstrap over extractors and detectors (2000 replicates, seed 0). P(>0): bootstrap probability that the extractor explains more than the detector.

## Private dataset, all 13 extractors, AUROC scale

| Scope | Component | eta^2 | 95% interval | P(>0) |
| --- | --- | ---: | ---: | ---: |
| both preprocessings | extractor | 0.491 | 0.198 to 0.785 |  |
| both preprocessings | detector | 0.132 | 0.022 to 0.313 |  |
| both preprocessings | preprocessing | 0.047 | 0.007 to 0.156 |  |
| both preprocessings | extractor x preprocessing | 0.022 | 0.005 to 0.074 |  |
| both preprocessings | detector x preprocessing | 0.065 | 0.013 to 0.161 |  |
| both preprocessings | residual (extractor x detector, 3-way) | 0.242 | 0.110 to 0.328 |  |
| both preprocessings | eta2(extractor) - eta2(detector) | 0.359 | -0.085 to 0.754 | 0.93 |
| L2 only | extractor | 0.676 | 0.343 to 0.880 |  |
| L2 only | detector | 0.170 | 0.046 to 0.426 |  |
| L2 only | residual (extractor x detector, 3-way) | 0.155 | 0.054 to 0.266 |  |
| L2 only | eta2(extractor) - eta2(detector) | 0.506 | -0.071 to 0.823 | 0.96 |
| L2 + PCA only | extractor | 0.417 | 0.174 to 0.802 |  |
| L2 + PCA only | detector | 0.239 | 0.028 to 0.527 |  |
| L2 + PCA only | residual (extractor x detector, 3-way) | 0.344 | 0.149 to 0.499 |  |
| L2 + PCA only | eta2(extractor) - eta2(detector) | 0.177 | -0.344 to 0.762 | 0.73 |

## Private dataset, all 13 extractors, logit(AUROC) scale

| Scope | Component | eta^2 | 95% interval | P(>0) |
| --- | --- | ---: | ---: | ---: |
| both preprocessings | extractor | 0.492 | 0.207 to 0.784 |  |
| both preprocessings | detector | 0.132 | 0.022 to 0.312 |  |
| both preprocessings | preprocessing | 0.048 | 0.006 to 0.155 |  |
| both preprocessings | extractor x preprocessing | 0.022 | 0.005 to 0.073 |  |
| both preprocessings | detector x preprocessing | 0.065 | 0.014 to 0.159 |  |
| both preprocessings | residual (extractor x detector, 3-way) | 0.242 | 0.115 to 0.328 |  |
| both preprocessings | eta2(extractor) - eta2(detector) | 0.360 | -0.073 to 0.750 | 0.95 |
| L2 only | extractor | 0.672 | 0.355 to 0.880 |  |
| L2 only | detector | 0.172 | 0.046 to 0.432 |  |
| L2 only | residual (extractor x detector, 3-way) | 0.156 | 0.054 to 0.273 |  |
| L2 only | eta2(extractor) - eta2(detector) | 0.500 | -0.061 to 0.825 | 0.96 |
| L2 + PCA only | extractor | 0.419 | 0.167 to 0.806 |  |
| L2 + PCA only | detector | 0.239 | 0.025 to 0.547 |  |
| L2 + PCA only | residual (extractor x detector, 3-way) | 0.343 | 0.146 to 0.485 |  |
| L2 + PCA only | eta2(extractor) - eta2(detector) | 0.180 | -0.357 to 0.771 | 0.70 |

## Private dataset, without RETFound CFP, AUROC scale

| Scope | Component | eta^2 | 95% interval | P(>0) |
| --- | --- | ---: | ---: | ---: |
| both preprocessings | extractor | 0.467 | 0.176 to 0.785 |  |
| both preprocessings | detector | 0.137 | 0.021 to 0.340 |  |
| both preprocessings | preprocessing | 0.048 | 0.004 to 0.166 |  |
| both preprocessings | extractor x preprocessing | 0.024 | 0.005 to 0.072 |  |
| both preprocessings | detector x preprocessing | 0.076 | 0.017 to 0.185 |  |
| both preprocessings | residual (extractor x detector, 3-way) | 0.248 | 0.109 to 0.337 |  |
| both preprocessings | eta2(extractor) - eta2(detector) | 0.330 | -0.133 to 0.746 | 0.90 |
| L2 only | extractor | 0.669 | 0.304 to 0.884 |  |
| L2 only | detector | 0.169 | 0.040 to 0.456 |  |
| L2 only | residual (extractor x detector, 3-way) | 0.161 | 0.053 to 0.297 |  |
| L2 only | eta2(extractor) - eta2(detector) | 0.500 | -0.140 to 0.838 | 0.93 |
| L2 + PCA only | extractor | 0.381 | 0.147 to 0.785 |  |
| L2 + PCA only | detector | 0.272 | 0.030 to 0.595 |  |
| L2 + PCA only | residual (extractor x detector, 3-way) | 0.347 | 0.143 to 0.516 |  |
| L2 + PCA only | eta2(extractor) - eta2(detector) | 0.110 | -0.419 to 0.748 | 0.64 |

## Private dataset, without RETFound CFP, logit(AUROC) scale

| Scope | Component | eta^2 | 95% interval | P(>0) |
| --- | --- | ---: | ---: | ---: |
| both preprocessings | extractor | 0.467 | 0.173 to 0.790 |  |
| both preprocessings | detector | 0.137 | 0.018 to 0.349 |  |
| both preprocessings | preprocessing | 0.048 | 0.005 to 0.163 |  |
| both preprocessings | extractor x preprocessing | 0.024 | 0.005 to 0.075 |  |
| both preprocessings | detector x preprocessing | 0.076 | 0.018 to 0.177 |  |
| both preprocessings | residual (extractor x detector, 3-way) | 0.247 | 0.108 to 0.345 |  |
| both preprocessings | eta2(extractor) - eta2(detector) | 0.330 | -0.153 to 0.757 | 0.90 |
| L2 only | extractor | 0.665 | 0.310 to 0.881 |  |
| L2 only | detector | 0.172 | 0.042 to 0.464 |  |
| L2 only | residual (extractor x detector, 3-way) | 0.163 | 0.053 to 0.278 |  |
| L2 only | eta2(extractor) - eta2(detector) | 0.493 | -0.152 to 0.831 | 0.94 |
| L2 + PCA only | extractor | 0.383 | 0.147 to 0.791 |  |
| L2 + PCA only | detector | 0.270 | 0.030 to 0.584 |  |
| L2 + PCA only | residual (extractor x detector, 3-way) | 0.346 | 0.141 to 0.495 |  |
| L2 + PCA only | eta2(extractor) - eta2(detector) | 0.113 | -0.416 to 0.757 | 0.65 |

## PALM, all 13 extractors, AUROC scale

| Scope | Component | eta^2 | 95% interval | P(>0) |
| --- | --- | ---: | ---: | ---: |
| both preprocessings | extractor | 0.124 | 0.013 to 0.404 |  |
| both preprocessings | detector | 0.204 | 0.072 to 0.360 |  |
| both preprocessings | preprocessing | 0.164 | 0.072 to 0.316 |  |
| both preprocessings | extractor x preprocessing | 0.028 | 0.008 to 0.107 |  |
| both preprocessings | detector x preprocessing | 0.176 | 0.059 to 0.335 |  |
| both preprocessings | residual (extractor x detector, 3-way) | 0.303 | 0.073 to 0.514 |  |
| both preprocessings | eta2(extractor) - eta2(detector) | -0.080 | -0.337 to 0.287 | 0.29 |
| L2 only | extractor | 0.780 | 0.153 to 0.907 |  |
| L2 only | detector | 0.048 | 0.018 to 0.502 |  |
| L2 only | residual (extractor x detector, 3-way) | 0.172 | 0.069 to 0.464 |  |
| L2 only | eta2(extractor) - eta2(detector) | 0.732 | -0.327 to 0.885 | 0.82 |
| L2 + PCA only | extractor | 0.143 | 0.025 to 0.450 |  |
| L2 + PCA only | detector | 0.482 | 0.149 to 0.842 |  |
| L2 + PCA only | residual (extractor x detector, 3-way) | 0.375 | 0.087 to 0.606 |  |
| L2 + PCA only | eta2(extractor) - eta2(detector) | -0.339 | -0.811 to 0.218 | 0.14 |

## PALM, all 13 extractors, logit(AUROC) scale

| Scope | Component | eta^2 | 95% interval | P(>0) |
| --- | --- | ---: | ---: | ---: |
| both preprocessings | extractor | 0.216 | 0.026 to 0.503 |  |
| both preprocessings | detector | 0.201 | 0.084 to 0.348 |  |
| both preprocessings | preprocessing | 0.244 | 0.104 to 0.462 |  |
| both preprocessings | extractor x preprocessing | 0.035 | 0.006 to 0.120 |  |
| both preprocessings | detector x preprocessing | 0.142 | 0.055 to 0.266 |  |
| both preprocessings | residual (extractor x detector, 3-way) | 0.163 | 0.061 to 0.262 |  |
| both preprocessings | eta2(extractor) - eta2(detector) | 0.015 | -0.280 to 0.395 | 0.50 |
| L2 only | extractor | 0.718 | 0.263 to 0.895 |  |
| L2 only | detector | 0.129 | 0.039 to 0.437 |  |
| L2 only | residual (extractor x detector, 3-way) | 0.153 | 0.048 to 0.402 |  |
| L2 only | eta2(extractor) - eta2(detector) | 0.589 | -0.132 to 0.853 | 0.94 |
| L2 + PCA only | extractor | 0.196 | 0.039 to 0.512 |  |
| L2 + PCA only | detector | 0.567 | 0.254 to 0.857 |  |
| L2 + PCA only | residual (extractor x detector, 3-way) | 0.237 | 0.077 to 0.406 |  |
| L2 + PCA only | eta2(extractor) - eta2(detector) | -0.371 | -0.808 to 0.237 | 0.10 |

## PALM, without RETFound CFP, AUROC scale

| Scope | Component | eta^2 | 95% interval | P(>0) |
| --- | --- | ---: | ---: | ---: |
| both preprocessings | extractor | 0.030 | 0.006 to 0.120 |  |
| both preprocessings | detector | 0.224 | 0.080 to 0.391 |  |
| both preprocessings | preprocessing | 0.168 | 0.071 to 0.341 |  |
| both preprocessings | extractor x preprocessing | 0.031 | 0.007 to 0.132 |  |
| both preprocessings | detector x preprocessing | 0.203 | 0.059 to 0.360 |  |
| both preprocessings | residual (extractor x detector, 3-way) | 0.343 | 0.077 to 0.552 |  |
| both preprocessings | eta2(extractor) - eta2(detector) | -0.194 | -0.380 to 0.029 | 0.05 |
| L2 only | extractor | 0.279 | 0.113 to 0.636 |  |
| L2 only | detector | 0.329 | 0.118 to 0.553 |  |
| L2 only | residual (extractor x detector, 3-way) | 0.393 | 0.171 to 0.513 |  |
| L2 only | eta2(extractor) - eta2(detector) | -0.050 | -0.413 to 0.499 | 0.49 |
| L2 + PCA only | extractor | 0.072 | 0.016 to 0.261 |  |
| L2 + PCA only | detector | 0.515 | 0.171 to 0.877 |  |
| L2 + PCA only | residual (extractor x detector, 3-way) | 0.412 | 0.097 to 0.618 |  |
| L2 + PCA only | eta2(extractor) - eta2(detector) | -0.443 | -0.854 to 0.076 | 0.05 |

## PALM, without RETFound CFP, logit(AUROC) scale

| Scope | Component | eta^2 | 95% interval | P(>0) |
| --- | --- | ---: | ---: | ---: |
| both preprocessings | extractor | 0.075 | 0.013 to 0.222 |  |
| both preprocessings | detector | 0.244 | 0.113 to 0.387 |  |
| both preprocessings | preprocessing | 0.288 | 0.129 to 0.510 |  |
| both preprocessings | extractor x preprocessing | 0.038 | 0.006 to 0.138 |  |
| both preprocessings | detector x preprocessing | 0.179 | 0.076 to 0.290 |  |
| both preprocessings | residual (extractor x detector, 3-way) | 0.176 | 0.054 to 0.282 |  |
| both preprocessings | eta2(extractor) - eta2(detector) | -0.170 | -0.327 to 0.046 | 0.05 |
| L2 only | extractor | 0.453 | 0.175 to 0.737 |  |
| L2 only | detector | 0.266 | 0.097 to 0.530 |  |
| L2 only | residual (extractor x detector, 3-way) | 0.281 | 0.088 to 0.466 |  |
| L2 only | eta2(extractor) - eta2(detector) | 0.187 | -0.308 to 0.614 | 0.80 |
| L2 + PCA only | extractor | 0.105 | 0.025 to 0.340 |  |
| L2 + PCA only | detector | 0.654 | 0.304 to 0.884 |  |
| L2 + PCA only | residual (extractor x detector, 3-way) | 0.241 | 0.066 to 0.442 |  |
| L2 + PCA only | eta2(extractor) - eta2(detector) | -0.549 | -0.849 to 0.012 | 0.03 |
