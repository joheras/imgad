"""Anomalib models (PatchCore, PaDiM, ...) vs. foundation-model features, with pixel-level maps.

    pip install "imgad[anomalib]"
    python examples/anomalib_models.py --root /path/to/mvtec_anomaly_detection --category bottle
"""

import argparse

from imgad import AnomalibPipeline, AnomalyDetectionPipeline, AnomalyPredictor, data, evaluate_maps
from imgad.evaluation import evaluate_scores

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--category", default="bottle")
ap.add_argument("--maps-dir", default="maps_patchcore")
args = ap.parse_args()

ds = data.mvtec_ad(args.root, args.category)
print(ds.summary(), "| masks:", ds.has_masks)

# --- 1) PatchCore (anomalib): image scores + anomaly maps -------------------------------------
pc = AnomalibPipeline("patchcore", backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                      coreset_sampling_ratio=0.1, image_size=256)
pc.fit(ds.train_paths)                                    # 20% of the normals are held out for the threshold
scores, maps = pc.score_and_maps(ds.test_paths)           # maps at model resolution (256x256)
print("PatchCore  image:", evaluate_scores(ds.test_labels, scores, scores > pc.threshold_))
print("PatchCore  pixel:", evaluate_maps(ds.test_masks, maps))
pc.save(f"{args.category}_patchcore.joblib")              # includes weights + memory bank

# --- 2) same API as a feature pipeline, for comparison ------------------------------------------
clip = AnomalyDetectionPipeline("openai/clip-vit-base-patch16", {"name": "knn", "k": 1}, preprocessing=["l2"])
clip.fit(ds.train_paths)
s = clip.score(ds.test_paths)
print("CLIP + kNN image:", evaluate_scores(ds.test_labels, s, s > clip.threshold_))

# --- 3) inference with heatmaps ---------------------------------------------------------------
predictor = AnomalyPredictor.load(f"{args.category}_patchcore.joblib")
results = predictor.predict(ds.test_paths[-5:], return_maps=True)   # maps at original image size
predictor.save_maps(results, args.maps_dir)                         # <name>_heatmap.png + <name>_mask.png
for r in results:
    print(f"{r.score:8.3f}  area={r.anomalous_area:.3%}  {'ANOMALY' if r.is_anomaly else 'ok':7s}  {r.heatmap}")
