"""Quickstart: CLIP features + kNN on one MVTec AD category.

    python examples/quickstart.py --root /ruta/a/mvtec_anomaly_detection --category bottle
"""

import argparse

from imgad import AnomalyDetectionPipeline, HFFeatureExtractor, build_detector, data
from imgad.evaluation import evaluate_scores, per_type_auroc

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--category", default="bottle")
ap.add_argument("--model", default="openai/clip-vit-base-patch16")
args = ap.parse_args()

ds = data.mvtec_ad(args.root, args.category)
print(ds.summary())

pipe = AnomalyDetectionPipeline(
    extractor=HFFeatureExtractor(args.model, batch_size=32),
    detector=build_detector("knn", k=1, metric="cosine"),
    preprocessing=["l2"],
    cache_dir=".feature_cache",
)
pipe.fit(ds.train_paths)

scores = pipe.score(ds.test_paths)
preds = pipe.predict(ds.test_paths)  # umbral = cuantil 95% de scores normales (out-of-fold)

print(evaluate_scores(ds.test_labels, scores, preds))
print(per_type_auroc(ds.test_labels, scores, ds.test_types))

pipe.save(f"{args.category}.joblib")
# Más tarde: AnomalyDetectionPipeline.load(f"{args.category}.joblib").score(["nueva.png"])
