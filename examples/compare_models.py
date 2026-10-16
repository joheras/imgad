"""Compare several HF backbones x classical detectors on MVTec AD and save a CSV.

    python examples/compare_models.py --root /ruta/a/mvtec_anomaly_detection --categories bottle screw
    python examples/compare_models.py --root /ruta/a/mvtec_anomaly_detection --seeds 0 1 2 --cd-plot cd.png
"""

import argparse
import csv

from imgad import data
from imgad import stats
from imgad.evaluation import benchmark, to_markdown

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--categories", nargs="*", default=None)
ap.add_argument("--out", default="results.csv")
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--train-subsample", type=float, default=None, help="e.g. 0.8 so that seeds change the training set")
ap.add_argument("--cd-plot", default=None)
args = ap.parse_args()

cats = args.categories or data.mvtec_categories(args.root)
datasets = [data.mvtec_ad(args.root, c) for c in cats]

extractors = [
    "openai/clip-vit-base-patch16",
    "google/siglip-base-patch16-224",
    {"type": "hf", "model_name": "facebook/dinov2-base", "pooling": "cls+mean"},
]
detectors = [
    {"name": "knn", "k": 1},
    "mahalanobis",
    "lof",
    "iforest",
    "ocsvm",
    {"name": "pca", "n_components": 0.9},
]

rows = benchmark(extractors, detectors, datasets, preprocessings=[["l2"]], cache_dir=".feature_cache",
                 seeds=args.seeds, train_subsample=args.train_subsample, bootstrap=1000, keep_scores=True)

with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k not in ("scores", "labels")], extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

print(to_markdown(rows))

# Media por (extractor, detector) sobre categorías
try:
    import pandas as pd

    df = pd.DataFrame(rows)
    print(df.pivot_table(index="detector", columns="extractor", values="auroc", aggfunc="mean").round(4))
except ImportError:
    pass

# Tests estadísticos entre configuraciones (cada categoría es un bloque)
if len(datasets) >= 2:
    res = stats.compare_configs(rows, metric="auroc")
    print(stats.report_markdown(res, max_pairs=15))
    if args.cd_plot:
        stats.plot_cd_diagram(res, args.cd_plot)
    # ¿la mejor configuración supera a las demás en cada categoría? (DeLong, Holm por categoría)
    for t in stats.delong_vs_baseline(rows, baseline=res["configs"][0]):
        if t["significant"]:
            print(f"[{t['dataset']} seed={t['seed']}] {t['config']}: ΔAUROC={t['diff']:+.4f} p={t['p_holm']:.3g}")
