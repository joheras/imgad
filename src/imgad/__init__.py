"""imgad: image anomaly detection with foundation-model features + classical detectors.

Typical usage::

    from imgad import HFFeatureExtractor, build_detector, AnomalyDetectionPipeline

    extractor = HFFeatureExtractor("openai/clip-vit-base-patch32")
    pipe = AnomalyDetectionPipeline(extractor, build_detector("knn", k=1), preprocessing=["l2"])
    pipe.fit(normal_image_paths)
    scores = pipe.score(test_image_paths)
    pipe.save("model.joblib")

    # inference later / elsewhere
    from imgad import AnomalyPredictor
    predictor = AnomalyPredictor.load("model.joblib")
    predictor.predict_one("new.png")   # -> Prediction(score, is_anomaly, normal_percentile, ...)

    # end-to-end anomalib models with pixel-level anomaly maps (pip install "imgad[anomalib]")
    from imgad import AnomalibPipeline
    pipe = AnomalibPipeline("patchcore", backbone="wide_resnet50_2").fit(normal_image_paths)
    scores, maps = pipe.score_and_maps(test_image_paths)
"""

from .features import (
    FeatureExtractor,
    HFFeatureExtractor,
    OpenCLIPFeatureExtractor,
    TimmFeatureExtractor,
    build_extractor,
    register_extractor,
    list_extractors,
)
from .detectors import (
    BaseDetector,
    build_detector,
    register_detector,
    list_detectors,
)
from .preprocessing import build_preprocessing
from .pipeline import AnomalyDetectionPipeline, load_pipeline
from .anomalib_backend import AnomalibPipeline, list_anomalib_models
from .inference import AnomalyPredictor, Prediction
from .evaluation import evaluate_maps, evaluate_scores, benchmark
from . import data, hub, stats, zoo
from .hub import GatedModelError

__version__ = "0.4.0"

__all__ = [
    "FeatureExtractor",
    "HFFeatureExtractor",
    "OpenCLIPFeatureExtractor",
    "TimmFeatureExtractor",
    "build_extractor",
    "register_extractor",
    "list_extractors",
    "BaseDetector",
    "build_detector",
    "register_detector",
    "list_detectors",
    "build_preprocessing",
    "AnomalyDetectionPipeline",
    "AnomalibPipeline",
    "list_anomalib_models",
    "load_pipeline",
    "AnomalyPredictor",
    "Prediction",
    "evaluate_scores",
    "evaluate_maps",
    "benchmark",
    "data",
    "hub",
    "zoo",
    "GatedModelError",
    "stats",
]
