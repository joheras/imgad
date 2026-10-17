# Command line

All commands: `imgad <command> -h` shows every option.

```bash
imgad list
imgad fit --model openai/clip-vit-base-patch16 --detector knn -p k=1 --prep l2 \
          --train data/bottle/train/good --out bottle.joblib

# inference: images and/or folders; output to stdout, .csv or .json
imgad predict --model bottle.joblib new.png data/bottle/test --out preds.csv
imgad predict --model bottle.joblib folder/ --only-anomalies --threshold 0.02 --device cpu

# re-tune the threshold on labelled validation data (f1 | youden | fpr | tpr)
imgad calibrate --model bottle.joblib --normal val/good --anomalous val/defects --method f1 --out bottle_cal.joblib
imgad info --model bottle_cal.joblib         # configuration, threshold and metadata

imgad benchmark --config configs/mvtec_benchmark.yaml --out results.csv --report report/ --log-file run.log
imgad benchmark --config configs/mvtec_benchmark.yaml -j 8 --devices auto   # parallel, several models per GPU
imgad devices                                # GPUs, free memory and parallel plan

# gated Hugging Face models (see [Gated Hugging Face models](gated-models.md))
imgad hf-login
imgad hf-check facebook/dinov3-vitb16-pretrain-lvd1689m

# Anomalib models: -p sets the model kwargs; predict saves heatmaps and masks
imgad fit --anomalib patchcore -p backbone=wide_resnet50_2 "layers=[layer2,layer3]" \
          --image-size 256 --train data/bottle/train/good --out bottle_pc.joblib
imgad predict --model bottle_pc.joblib data/bottle/test --maps-dir maps/ --out preds.csv
```

## Commands

| Command | What it does |
|---|---|
| `imgad list` | available extractors, detectors, preprocessing steps and Anomalib models |
| `imgad fit` | train a feature pipeline (`--model`) or an Anomalib model (`--anomalib`) on normal images |
| `imgad predict` | score images/folders; `--maps-dir` saves heatmaps and masks (Anomalib) |
| `imgad calibrate` | re-tune the threshold on labelled validation images |
| `imgad info` | configuration, threshold and metadata of a saved model |
| `imgad benchmark` | run a YAML grid; `--report`, `--stats`, `-j/--devices` (parallel), `--intensity`, `--k-shots` (few-shot protocol), `--stop-on-error` |
| `imgad report` | rebuild a report from `benchmark --scores-out` |
| `imgad stats` | statistical comparison of saved results |
| `imgad devices` | GPUs, free memory and memory estimate per model |
| `imgad models` | recommended medical feature extractors per modality (`-m`), `--check`, `--yaml`, `--zero-shot` (vision–language models) ([Medical imaging](../medical.md#2-choose-a-feature-extractor)) |
| `imgad hf-login` / `hf-logout` / `hf-check` | tokens and access to gated Hugging Face models |
