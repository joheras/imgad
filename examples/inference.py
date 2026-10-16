"""Inference with an already-trained model.

    # 1) train and save (once)
    imgad fit --model openai/clip-vit-base-patch16 --detector knn -p k=1 \
              --train /path/to/mvtec/bottle/train/good --out bottle.joblib

    # 2) inference
    python examples/inference.py --model bottle.joblib /path/to/new_images
"""

import argparse

from imgad import AnomalyDetectionPipeline, AnomalyPredictor

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("inputs", nargs="+", help="images and/or folders")
ap.add_argument("--val-normal", default=None, help="optional: folder of normal validation images to re-tune the threshold")
ap.add_argument("--val-anomalous", default=None, help="optional: folder of anomalous validation images")
args = ap.parse_args()

# --- (optional) recalibrate the threshold on a labelled validation set -----------------
if args.val_normal:
    from imgad.data import list_images

    pipe = AnomalyDetectionPipeline.load(args.model)
    normal = list_images(args.val_normal)
    anomalous = list_images(args.val_anomalous) if args.val_anomalous else []
    method = "f1" if anomalous else "fpr"  # with normals only: keep FPR <= 5%
    print(pipe.calibrate_threshold(normal + anomalous, [0] * len(normal) + [1] * len(anomalous), method=method))
    predictor = AnomalyPredictor(pipe)
else:
    predictor = AnomalyPredictor.load(args.model)

print(f"threshold = {predictor.threshold:.6g}")

# --- inference ---------------------------------------------------------------------------
results = predictor.predict(args.inputs)
for r in sorted(results, key=lambda r: -r.score):
    flag = "ANOMALY" if r.is_anomaly else "ok"
    print(f"{flag:8s} score={r.score:.5f} percentile={r.normal_percentile:.3f}  {r.source}")

predictor.save_results(results, "predictions.csv")

# Single image (path, PIL.Image or numpy array):
#   r = predictor.predict_one("img.png");  r.is_anomaly, r.score, r.normal_percentile
# Huge folders with constant memory:
#   for r in predictor.predict_iter("/path/to/huge_folder", chunk_size=256): ...
