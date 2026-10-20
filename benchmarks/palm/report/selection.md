# PALM: selection on validation, evaluation on test

Selected on validation: **vit_large_patch16_224[cls] | l2 | lof**

- validation AUROC 0.995; **test AUROC 0.989** (± 0.001 over seeds), rank 50 of 370 on test
- oracle (best on test, optimistic): retizero-clip | l2+pca:0.95 | mahalanobis, test AUROC 0.996
- optimism of choosing on test: +0.007

Spearman correlation between validation and test AUROC over 370 configurations: 0.956

## Per family

| Family | Selected on validation | Val AUROC | Test AUROC | Oracle test AUROC |
| --- | --- | ---: | ---: | ---: |
| Anomalib model | anomalib \| - \| dfm(image_size=256) | 0.985 | 0.989 | 0.992 |
| Extractor + detector | vit_large_patch16_224[cls] \| l2 \| lof | 0.995 | 0.989 | 0.996 |
| Text prompts | google/medsiglip-448 \| l2 \| zero_shot(modality=fundus) | 0.987 | 0.988 | 0.988 |

## Top 10 on validation

| # | Configuration | Family | Val AUROC | Test AUROC |
| ---: | --- | --- | ---: | ---: |
| 1 | vit_large_patch16_224[cls] \| l2 \| lof | Extractor + detector | 0.995 | 0.989 |
| 2 | facebook/dinov3-vitb16-pretrain-lvd1689m \| l2 \| lof | Extractor + detector | 0.992 | 0.993 |
| 3 | facebook/dinov3-vitb16-pretrain-lvd1689m \| l2+pca:0.95 \| kmeans(n_clusters=8) | Extractor + detector | 0.992 | 0.991 |
| 4 | facebook/dinov3-vitb16-pretrain-lvd1689m \| l2 \| kmeans(n_clusters=8) | Extractor + detector | 0.992 | 0.993 |
| 5 | facebook/dinov3-vitb16-pretrain-lvd1689m \| l2 \| gmm | Extractor + detector | 0.991 | 0.993 |
| 6 | retizero-clip \| l2 \| lof | Extractor + detector | 0.991 | 0.993 |
| 7 | google/medsiglip-448 \| l2 \| knn(k=5) | Extractor + detector | 0.990 | 0.989 |
| 8 | facebook/dinov3-vitb16-pretrain-lvd1689m \| l2 \| knn(k=5) | Extractor + detector | 0.990 | 0.992 |
| 9 | facebook/dinov2-base \| l2 \| lof | Extractor + detector | 0.990 | 0.985 |
| 10 | retizero-clip \| l2 \| gmm | Extractor + detector | 0.990 | 0.993 |
