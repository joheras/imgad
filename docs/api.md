# API reference

Generated from the docstrings.

## Pipelines

::: imgad.pipeline.AnomalyDetectionPipeline

::: imgad.anomalib_backend.AnomalibPipeline

::: imgad.pipeline.load_pipeline

## Inference

::: imgad.inference.AnomalyPredictor

::: imgad.inference.Prediction

## Feature extractors

::: imgad.features.base.FeatureExtractor

::: imgad.features.huggingface.HFFeatureExtractor

::: imgad.features.open_clip.OpenCLIPFeatureExtractor

::: imgad.features.timm_models.TimmFeatureExtractor

::: imgad.features.base.build_extractor

## Medical model zoo

::: imgad.zoo
    options:
      members: [recommend, get, list_cards, benchmark_yaml, ModelCard, MODALITIES]

## Detectors

::: imgad.detectors.base.BaseDetector

::: imgad.detectors.base.build_detector

## Data and image reading

::: imgad.data
    options:
      members: [AnomalyDataset, mvtec_ad, from_folders, one_vs_rest, volume_slices, list_images]

::: imgad.imageio
    options:
      members: [read_image, to_uint8, parse_intensity, split_ref, file_of]

## Evaluation and benchmarks

::: imgad.evaluation
    options:
      members: [benchmark, evaluate_scores, evaluate_maps, aupro, per_type_auroc]

::: imgad.stats
    options:
      members: [bootstrap_ci, delong_test, paired_bootstrap_test, summarize_seeds, compare_configs, delong_vs_baseline, plot_cd_diagram, report_markdown]

::: imgad.report.generate_report

## Hugging Face access

::: imgad.hub
    options:
      members: [check_access, check_models, AccessStatus, GatedModelError, login, read_token_file]

## Devices and progress

::: imgad.devices
    options:
      members: [list_devices, resolve_devices, ResourceManager, estimate_extractor_bytes]

::: imgad.progress.setup_logging
