# PALM benchmark (pathologic myopia)

Second dataset for the ophthalmology benchmark: the same grid as `benchmarks/ophtalmology/`, run on
[PALM](https://doi.org/10.6084/m9.figshare.c.6224616.v1) (Fang et al., *Sci. Data* 11, 99, 2024). PALM
has 1,200 colour fundus photographs split by patient into 400 training, 400 validation and 400 test
images, labelled pathologic myopia (PM) or not. Its licence allows research and educational use only.

One-class setting: models are fitted on the **187 non-PM training images** (80% per seed). PM images are
the anomalies. The configuration is **chosen on the validation set** and **reported on the test set**,
so the test AUROC is not optimistic.

| Category | Train (normal) | Test normal | Test anomalous | Use |
|---|---:|---:|---:|---|
| `palm-val` | 187 | 189 | 211 | choose the configuration |
| `palm-test` | 187 | 187 | 213 | report the chosen configuration |
| `palm-lesion` (optional) | 187 | 187 | PM test images with lesion masks | pixel metrics of Anomalib maps |

## Steps

```bash
# 1. Download PALM from Figshare and extract it (e.g. to /data/PALM), then lay it out for imgad:
python benchmarks/palm/prepare_palm.py --palm /data/PALM --out /data/palm_imgad --lesion-masks

# 2. Set data.root in configs/palm_benchmark.yaml to /data/palm_imgad and run the benchmark:
imgad benchmark --config configs/palm_benchmark.yaml --out benchmarks/palm/results.csv \
  --scores-out benchmarks/palm/scores.joblib --report benchmarks/palm/report --stats \
  -j 8 --devices auto --hf-token-file token

# 3. Selection on validation, evaluation on test:
python benchmarks/palm/select_on_validation.py benchmarks/palm/results.csv --out benchmarks/palm/selection.md

# 4. DeLong against the selected configuration on the test set, with Holm correction (see imgad stats --help):
imgad stats --scores benchmarks/palm/scores.joblib --baseline "<selected configuration>"
```

`prepare_palm.py` reads both the Figshare release (`Classification Labels.xlsx` per set, lesion masks under
`Lesion Masks/Atrophy` and `Lesion Masks/Detachment`) and the 2019 challenge release (training images
named `P*`, `N*`, `H*`). It stops if a split does not have 400 labelled images; `--force` writes what it
found. Images are symlinked; `--copy` copies them. `--crop-fov` writes copies cropped to the field of view
(masks cropped the same way), for the cropping ablation: lay it out into another folder and run the same
YAML with `data.root` pointing there.

## Results

The run of 2026-10-20 is in [`report/`](report/): the imgad report (`report.md`, figures), `results.csv`
(one row per dataset × configuration × seed), the configuration used, and [`selection.md`](report/selection.md)
from `select_on_validation.py`.

- Chosen on validation: RetiZero image encoder (`vit_large_patch16_224[cls]`) + L2 + LOF, validation AUROC
  0.995, **test AUROC 0.989** (95% CI 0.982–0.995). The best configuration on test would give 0.996, so
  choosing on the test set inflates the result by 0.007. Validation and test AUROC have a Spearman
  correlation of 0.956 over the 370 configurations.
- With the threshold calibrated on normal images, 0.2% false positives and 74% sensitivity on test.
- Best Anomalib models: RAD 0.992 and DFM 0.989 on test. Median over methods: 0.989 for extractors with their
  best detector, 0.944 for Anomalib models.
- The report's Friedman/Nemenyi section treats `palm-val` and `palm-test` as two independent datasets.
  They share the training images, so read it with care.

## What the YAML adds to the myopia benchmark

- Two text-prompt variants naming myopic findings (atrophy, lacquer cracks, myopic CNV, staphyloma): one
  text-only (true zero-shot, `mode: text`) and one combined with the 1-NN visual score (`mode: both`).
- Commented-out Anomalib runs at 448–512 px, for the resolution ablation.

