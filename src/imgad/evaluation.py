"""Metrics and a benchmarking helper that reuses extracted features across detectors."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve

from .detectors import BaseDetector, build_detector
from .features import FeatureExtractor, build_extractor
from .preprocessing import build_preprocessing


def evaluate_scores(y_true, scores, y_pred: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Image-level metrics. ``y_true``: 1 = anomaly. ``scores``: higher = more anomalous."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    out: Dict[str, float] = {}
    if len(np.unique(y_true)) < 2:
        raise ValueError("evaluate_scores needs both normal and anomalous samples")
    out["auroc"] = roc_auc_score(y_true, scores)
    out["aupr"] = average_precision_score(y_true, scores)
    p, r, _ = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    out["f1_max"] = float(f1.max())
    fpr, tpr, _ = roc_curve(y_true, scores)
    out["fpr@95tpr"] = float(fpr[np.searchsorted(tpr, 0.95, side="left")])
    if y_pred is not None:
        y_pred = np.asarray(y_pred).astype(int)
        out["f1@thr"] = f1_score(y_true, y_pred)
        out["tpr@thr"] = float(y_pred[y_true == 1].mean())
        out["fpr@thr"] = float(y_pred[y_true == 0].mean())
    return {k: float(v) for k, v in out.items()}


def evaluate_maps(masks: Sequence, maps: Sequence[np.ndarray], max_fpr: float = 0.3) -> Dict[str, float]:
    """Pixel-level metrics. ``masks``: ground-truth masks (paths, arrays or ``None`` for defect-free
    images); ``maps``: anomaly maps (higher = more anomalous). Masks are resized (nearest) to each
    map's resolution, so evaluating at model resolution keeps memory low.

    Returns pixel AUROC, pixel AUPR, pixel F1-max and AUPRO (normalised area under the
    per-region-overlap curve up to ``max_fpr``, the standard MVTec AD localisation metric)."""
    from .maps import load_mask

    if len(masks) != len(maps):
        raise ValueError("masks and maps must have the same length")
    gts = [load_mask(g, np.shape(m)) for g, m in zip(masks, maps)]
    y = np.concatenate([g.ravel() for g in gts])
    s = np.concatenate([np.ravel(m) for m in maps]).astype(np.float64)
    if not y.any() or y.all():
        raise ValueError("evaluate_maps needs both normal and anomalous pixels")
    p, r, _ = precision_recall_curve(y, s)
    return {
        "pixel_auroc": float(roc_auc_score(y, s)),
        "pixel_aupr": float(average_precision_score(y, s)),
        "pixel_f1_max": float((2 * p * r / np.clip(p + r, 1e-12, None)).max()),
        "aupro": aupro(gts, maps, max_fpr),
    }


def aupro(masks: Sequence[np.ndarray], maps: Sequence[np.ndarray], max_fpr: float = 0.3, n_thresholds: int = 500) -> float:
    """Area under the PRO curve (mean overlap per connected defect region vs. pixel FPR) for
    FPR in [0, ``max_fpr``], normalised to [0, 1]. ``masks`` are boolean arrays shaped like ``maps``."""
    from scipy import ndimage

    normal, regions = [], []
    for g, m in zip(masks, maps):
        m = np.asarray(m, dtype=np.float64)
        normal.append(m[~g])
        lab, n = ndimage.label(g)
        regions += [np.sort(m[lab == i]) for i in range(1, n + 1)]
    if not regions:
        return float("nan")
    normal = np.sort(np.concatenate(normal))
    # thresholds spanning FPR in [0, max_fpr] (decreasing threshold -> increasing FPR)
    thr = np.quantile(normal, 1 - np.linspace(0, max_fpr, n_thresholds))
    fpr = 1 - np.searchsorted(normal, thr, side="right") / len(normal)
    pro = np.mean([1 - np.searchsorted(r, thr, side="right") / len(r) for r in regions], axis=0)
    keep = fpr <= max_fpr
    fpr, pro = np.r_[0.0, fpr[keep], max_fpr], np.r_[pro[keep][0] if keep.any() else 0.0, pro[keep], np.interp(max_fpr, fpr, pro)]
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(pro, fpr) / max_fpr)


def per_type_auroc(y_true, scores, types: Sequence[str], normal_name: str = "good") -> Dict[str, float]:
    """AUROC of each anomaly type vs. all normal test samples."""
    y_true, scores, types = np.asarray(y_true), np.asarray(scores), np.asarray(types)
    norm = y_true == 0
    res = {}
    for t in sorted(set(types[y_true == 1])):
        m = norm | (types == t)
        res[t] = float(roc_auc_score(y_true[m], scores[m]))
    return res


def benchmark(
    extractors: Iterable[Union[FeatureExtractor, str, Dict[str, Any]]] = (),
    detectors: Iterable[Union[BaseDetector, str, Dict[str, Any]]] = (),
    datasets: Iterable = (),
    preprocessings: Iterable[Optional[Sequence]] = (None,),
    cache_dir: Optional[str] = None,
    verbose: bool = True,
    anomalib_models: Iterable[Union[str, Dict[str, Any]]] = (),
    seeds: Sequence[int] = (0,),
    train_subsample: Optional[Union[float, int]] = None,
    bootstrap: int = 0,
    ci_alpha: float = 0.05,
    keep_scores: bool = False,
) -> List[Dict[str, Any]]:
    """Grid evaluation: extractor x preprocessing x detector x dataset, plus each anomalib model x
    dataset (``anomalib_models``: names or ``{"name": "patchcore", "backbone": ...}`` specs).

    Features are extracted once per (extractor, dataset) and reused for all detectors and seeds.
    Anomalib rows also get pixel-level metrics (see :func:`evaluate_maps`) when the dataset has masks.

    Repetitions and uncertainty:
      * ``seeds``: every configuration is run once per seed (one row each, column ``seed``). The seed
        sets the detectors' ``random_state`` (unless fixed in the spec), the calibration folds, the
        anomalib seed and the training subset.
      * ``train_subsample``: fraction (float) or number (int) of normal training images drawn per
        seed. Without it, deterministic detectors (kNN, Mahalanobis...) give the same result for
        every seed; with it, the spread over seeds measures sensitivity to the training set.
      * ``bootstrap``: if > 0, adds ``<metric>_ci_low/high`` for AUROC and AUPR from that many
        stratified bootstrap resamples of the test set (``ci_alpha`` = 0.05 -> 95% CI).
      * ``keep_scores``: store per-image ``scores`` and ``labels`` in each row (needed for paired tests
        such as :func:`imgad.stats.delong_vs_baseline`).

    Returns a list of result rows (``pandas.DataFrame(rows)`` for a table); analyse them with
    :mod:`imgad.stats`.
    """
    import copy

    rows: List[Dict[str, Any]] = []
    datasets = list(datasets)
    detectors = list(detectors)
    preprocessings = list(preprocessings)
    seeds = list(seeds) or [0]
    opts = dict(bootstrap=bootstrap, ci_alpha=ci_alpha, keep_scores=keep_scores)
    for ext in extractors:
        ext = ext if isinstance(ext, FeatureExtractor) else build_extractor(ext)
        ext_name = getattr(ext, "model_name", type(ext).__name__)
        if getattr(ext, "pooling", None) not in (None, "auto"):
            ext_name += f"[{ext.pooling}" + (f"@{ext.layer}" if getattr(ext, "layer", None) is not None else "") + "]"
        for ds in datasets:
            t0 = time.time()
            Xtr_all = ext.extract(ds.train_paths, cache_dir=cache_dir, show_progress=verbose)
            Xte = ext.extract(ds.test_paths, cache_dir=cache_dir, show_progress=verbose)
            t_feat = time.time() - t0
            for seed in seeds:
                Xtr = Xtr_all[_subsample(len(Xtr_all), train_subsample, seed)]
                for pp in preprocessings:
                    prep = build_preprocessing(pp)
                    Ztr = prep.fit_transform(Xtr) if prep is not None else Xtr
                    Zte = prep.transform(Xte) if prep is not None else Xte
                    for det_spec in detectors:
                        det = (
                            copy.deepcopy(det_spec)
                            if isinstance(det_spec, BaseDetector)
                            else build_detector(det_spec)
                        )
                        _seed_detector(det, det_spec, seed)
                        t1 = time.time()
                        det.fit(Ztr)
                        s = det.score(Zte)
                        row = {
                            "dataset": ds.name,
                            "extractor": ext_name,
                            "preprocessing": "+".join(map(str, pp)) if pp else "-",
                            "detector": _det_name(det_spec, det),
                            "seed": seed,
                            "n_train": len(Ztr),
                        }
                        _add_metrics(row, ds.test_labels, s, det.predict(Zte), seed=seed, **opts)
                        row.update(
                            feat_dim=Ztr.shape[1], feat_time_s=round(t_feat, 2), det_time_s=round(time.time() - t1, 3)
                        )
                        rows.append(row)
                        if verbose:
                            _log(row)
    for spec in anomalib_models:
        for ds in datasets:
            for seed in seeds:
                rows.append(_benchmark_anomalib(spec, ds, verbose, seed, train_subsample, **opts))
    return rows


def _subsample(n: int, size: Optional[Union[float, int]], seed: int) -> np.ndarray:
    if size is None:
        return np.arange(n)
    k = int(round(size * n)) if isinstance(size, float) else int(size)
    k = max(1, min(n, k))
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def _seed_detector(det: BaseDetector, spec, seed: int) -> None:
    fixed = isinstance(spec, dict) and "random_state" in spec
    if hasattr(det, "random_state") and not fixed:
        det.random_state = seed
    det.random_state_calibration = seed


def _add_metrics(row, y, s, pred, seed: int, bootstrap: int, ci_alpha: float, keep_scores: bool) -> None:
    row["config"] = f"{row['extractor']} | {row['preprocessing']} | {row['detector']}"
    row.update(evaluate_scores(y, s, pred))
    if bootstrap:
        from .stats import bootstrap_ci

        for m, ci in bootstrap_ci(y, s, ("auroc", "aupr"), n_boot=bootstrap, alpha=ci_alpha, seed=seed).items():
            row[f"{m}_ci_low"], row[f"{m}_ci_high"] = ci["low"], ci["high"]
    if keep_scores:
        row["scores"], row["labels"] = np.asarray(s, dtype=float), np.asarray(y, dtype=int)


def _log(row) -> None:
    ci = f" [{row['auroc_ci_low']:.4f}, {row['auroc_ci_high']:.4f}]" if "auroc_ci_low" in row else ""
    pix = f" pixel-AUROC={row['pixel_auroc']:.4f} AUPRO={row['aupro']:.4f}" if "pixel_auroc" in row else ""
    print(f"[{row['dataset']}] seed={row['seed']} {row['config']}: AUROC={row['auroc']:.4f}{ci} AUPR={row['aupr']:.4f}{pix}")


def _benchmark_anomalib(spec, ds, verbose: bool, seed: int = 0, train_subsample=None, **opts) -> Dict[str, Any]:
    from .anomalib_backend import build_anomalib_pipeline

    pipe = build_anomalib_pipeline(spec, seed=seed)
    train = [ds.train_paths[i] for i in _subsample(len(ds.train_paths), train_subsample, seed)]
    t0 = time.time()
    pipe.fit(train, show_progress=verbose)
    t1 = time.time()
    if ds.has_masks:
        s, maps = pipe.score_and_maps(ds.test_paths, show_progress=verbose)
    else:
        s, maps = pipe.score(ds.test_paths, show_progress=verbose), None
    row = {"dataset": ds.name, "extractor": "anomalib", "preprocessing": "-", "detector": _det_name(spec, pipe),
           "seed": seed, "n_train": len(train)}
    _add_metrics(row, ds.test_labels, s, (s > pipe.threshold_).astype(int), seed=seed, **opts)
    if maps is not None:
        row.update(evaluate_maps(ds.test_masks, maps))
    row.update(fit_time_s=round(t1 - t0, 2), test_time_s=round(time.time() - t1, 2))
    if verbose:
        _log(row)
    return row


def _det_name(spec, det) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        extra = {k: v for k, v in spec.items() if k != "name"}
        return spec["name"] + (f"({', '.join(f'{k}={v}' for k, v in extra.items())})" if extra else "")
    return repr(det)


def to_markdown(rows: List[Dict[str, Any]], metrics=("auroc", "aupr", "f1_max", "fpr@95tpr")) -> str:
    """Markdown table; pixel metrics are appended when some row has them."""
    metrics = list(metrics) + [k for k in ("pixel_auroc", "aupro") if any(k in r for r in rows) and k not in metrics]
    seed_col = ["seed"] if len({r.get("seed", 0) for r in rows}) > 1 else []
    cols = ["dataset", "extractor", "preprocessing", "detector", *seed_col, *metrics]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = (r.get(c, "-") for c in cols)
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in cells) + " |")
    return "\n".join(lines)
