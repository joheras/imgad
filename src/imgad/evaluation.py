"""Metrics and a benchmarking helper that reuses extracted features across detectors."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve

from .detectors import BaseDetector, build_detector
from .detectors.zero_shot import needs_text
from .features import FeatureExtractor, build_extractor
from .preprocessing import build_preprocessing
from .progress import BenchmarkProgress, ensure_console, fmt_duration, logger, step


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
    token=None,
    check_hub_access: bool = True,
    intensity: Optional[str] = None,
    n_jobs: int = 1,
    devices: Union[None, str, Sequence[str]] = None,
    gpu_jobs: Union[int, str] = "auto",
    gpu_mem_gb: Optional[float] = None,
    k_shots: Optional[Sequence[int]] = None,
    on_error: str = "continue",
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
      * Progress (plan, current phase, one line per finished run with counter, elapsed time and ETA)
        is logged under the ``imgad`` logger; ``verbose=True`` shows it on the console unless logging is
        already configured (see :func:`imgad.progress.setup_logging`) and enables progress bars.
      * ``keep_scores``: store per-image ``scores`` and ``labels`` in each row (needed for paired tests
        such as :func:`imgad.stats.delong_vs_baseline`).
      * ``k_shots``: few-shot protocol, e.g. ``[0, 1, 2, 4, 8, 16]``. Every configuration is run with ``k``
        normal training images drawn per seed (column ``k_shot``; the configuration name ends in
        ``| k-shot``), instead of the whole training set or ``train_subsample``. ``k = 0`` is zero-shot and runs
        only detectors that need no training images (``zero_shot``); detectors that cannot fit on ``k``
        images are recorded as failed runs.
      * ``intensity``: rule mapping images with more than 8 bits (16-bit, DICOM, NIfTI) to 8 bits for every
        extractor and Anomalib model that does not set its own (see :mod:`imgad.imageio`).

    Gated / private Hub models: ``token`` is passed to the extractors (otherwise ``HF_TOKEN`` or the
    stored login are used; see :mod:`imgad.hub`). With ``check_hub_access`` (default) access to every
    Hub model is checked before any work starts, and a :class:`imgad.hub.GatedModelError` lists what
    to do for each model that cannot be loaded.

    Parallel execution (see :mod:`imgad.parallel`; results are identical to the sequential run):
      * ``n_jobs``: CPU worker processes for the detector runs (-1 = all cores). ``1`` together with
        ``devices=None`` runs everything sequentially in this process.
      * ``devices``: where feature extraction and Anomalib models run: ``"auto"`` (every GPU, or the
        CPU), ``"cpu"``, ``"cuda:0,cuda:1"`` or a list. Each extractor is loaded once in a worker
        process and its detector runs start on the CPU pool as soon as its features are ready.
      * ``gpu_jobs``: maximum concurrent jobs per GPU (``"auto"``: as many as fit in memory, up to 4).
      * ``gpu_mem_gb``: memory assumed for a job whose needs are unknown (default: estimated from the
        checkpoint size for Hub models, 4 GiB otherwise); after the first run of a model its measured
        peak is used. A job starts only when it fits in the free memory of a device.

    Zero-shot detectors (:mod:`imgad.detectors.zero_shot`) run only with extractors that encode text
    (vision–language models) and without preprocessing that changes the embeddings (``None`` or ``l2``);
    other combinations are skipped. The prompts are encoded once per extractor.

    Failures: with ``on_error="continue"`` (default) a run that fails (loading or extracting with a model,
    fitting a detector, an Anomalib model...) is logged, recorded as a row with an ``error`` column and no
    metrics, and the benchmark goes on with the rest; ``"raise"`` stops at the first failure.
    :func:`successful_rows` drops failed rows; :mod:`imgad.stats` and the report ignore them.

    Returns a list of result rows (``pandas.DataFrame(rows)`` for a table); analyse them with
    :mod:`imgad.stats`.
    """
    if on_error not in ("continue", "raise"):
        raise ValueError(f"on_error must be 'continue' or 'raise', got {on_error!r}")
    ensure_console(verbose)
    rows: List[Dict[str, Any]] = []
    extractors = [_with_intensity(e, intensity) for e in extractors]
    anomalib_models = [_with_intensity(m, intensity, anomalib=True) for m in anomalib_models]
    datasets = list(datasets)
    detectors = list(detectors)
    preprocessings = list(preprocessings)
    seeds = list(seeds) or [0]
    shots = [int(k) for k in k_shots] if k_shots else [None]
    if k_shots and train_subsample is not None:
        logger.warning("k_shots is set: train_subsample is ignored (each run uses k training images)")
        train_subsample = None
    opts = dict(bootstrap=bootstrap, ci_alpha=ci_alpha, keep_scores=keep_scores)
    n_feat = len(extractors) * len(datasets) * len(seeds) * len(preprocessings) * len(detectors) * len(shots)
    n_anom = len(anomalib_models) * len(datasets) * len(seeds) * len(shots)
    prog = BenchmarkProgress({"load": len(extractors), "extract": len(extractors) * len(datasets),
                              "detector": n_feat, "anomalib": n_anom})
    _log_plan(logger, extractors, preprocessings, detectors, anomalib_models, datasets, seeds, n_feat, n_anom,
              train_subsample, bootstrap)
    if k_shots:
        logger.info(f"  few-shot protocol: k = {', '.join(map(str, shots))} normal training image(s) per seed")
    if check_hub_access:
        _check_hub_access(extractors, token)
    if n_jobs != 1 or devices is not None:
        from .parallel import run_parallel

        rows = run_parallel(extractors, detectors, datasets, preprocessings, anomalib_models, seeds, train_subsample,
                            opts, cache_dir, n_jobs, devices, gpu_jobs, gpu_mem_gb, prog, token=token, shots=shots,
                            on_error=on_error)
        _log_end(logger, prog, rows)
        return rows

    runs_per_ds = len(seeds) * len(shots) * len(preprocessings) * len(detectors)
    for i_ext, ext in enumerate(extractors, 1):
        spec_name = _spec_name(ext)
        try:
            with step(f"Extractor {i_ext}/{len(extractors)}: building {spec_name}", logging.INFO) as st:
                ext = ext if isinstance(ext, FeatureExtractor) else build_extractor(ext, token=token)
                if hasattr(ext, "_load"):
                    logger.info(f"  loading model weights ({getattr(ext, 'model_name', '')} on {getattr(ext, 'device', '?')})")
                    ext._load()
                text = text_bank(ext, detectors)
        except Exception as e:
            _on_failure(e, on_error, f"loading {spec_name}")
            for ds in datasets:
                rows.append(failed_row(ds.name, spec_name, "-", "-", None, None, "load", e))
            prog.skip("load")
            prog.skip("extract", len(datasets))
            prog.skip("detector", runs_per_ds * len(datasets))
            continue
        prog.add("load", st.seconds)
        ext_name = extractor_label(ext)
        for i_ds, ds in enumerate(datasets, 1):
            n_img = len(ds.train_paths) + len(ds.test_paths)
            cached = ext.cached_count(list(ds.train_paths) + list(ds.test_paths), cache_dir)
            msg = (f"Extracting features [extractor {i_ext}/{len(extractors)}, dataset {i_ds}/{len(datasets)}]: "
                   f"{ext_name} on {ds.name} ({n_img} images" + (f", {cached} cached" if cached else "") + ")")
            try:
                with step(msg, logging.INFO) as st:
                    Xtr_all = ext.extract(ds.train_paths, cache_dir=cache_dir, show_progress=verbose)
                    Xte = ext.extract(ds.test_paths, cache_dir=cache_dir, show_progress=verbose)
            except Exception as e:
                _on_failure(e, on_error, f"extracting {ext_name} on {ds.name}")
                rows.append(failed_row(ds.name, ext_name, "-", "-", None, None, "extract", e))
                prog.skip("extract")
                prog.skip("detector", runs_per_ds)
                continue
            t_feat = st.seconds
            prog.add("extract", t_feat)
            logger.info(f"  features: {Xtr_all.shape[1]}-d, {fmt_duration(t_feat)} · {prog.timing()}")
            for seed in seeds:
                for k in shots:
                    for pp in preprocessings:
                        for det_spec in detectors:
                            row, dt = safe_detector_run(Xtr_all, Xte, ds.test_labels, ds.name, ext_name, t_feat,
                                                        det_spec, seed, pp, train_subsample, opts, k, text, on_error)
                            _record(rows, prog, "detector", row, dt)
    for i_m, spec in enumerate(anomalib_models, 1):
        for i_ds, ds in enumerate(datasets, 1):
            for seed in seeds:
                for k in shots:
                    t1 = time.time()
                    if k == 0:
                        _record(rows, prog, "anomalib", None, 0.0)
                        continue
                    logger.info(f"Anomalib model {i_m}/{len(anomalib_models)} [dataset {i_ds}/{len(datasets)}, seed {seed}"
                                + (f", {k}-shot" if k is not None else "") + f"]: {_anomalib_name(spec)} on {ds.name}")
                    try:
                        row = _benchmark_anomalib(spec, ds, verbose, seed, train_subsample, k_shot=k, **opts)
                    except Exception as e:
                        _on_failure(e, on_error, f"Anomalib {_anomalib_name(spec)} on {ds.name} (seed {seed})")
                        row = failed_row(ds.name, "anomalib", "-", _anomalib_name(spec), seed, k, "anomalib", e)
                    _record(rows, prog, "anomalib", row, time.time() - t1)
    _log_end(logger, prog, rows)
    return rows


# ---------------------------------------------------------------------------- failures, skips, few-shot
def successful_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows of runs that finished (without an ``error``)."""
    return [r for r in rows if not r.get("error")]


def failed_row(ds_name: str, extractor: str, pp: str, detector: str, seed, k_shot, stage: str, err) -> Dict[str, Any]:
    row: Dict[str, Any] = {"dataset": ds_name, "extractor": extractor, "preprocessing": pp, "detector": detector,
                           "seed": seed if seed is not None else "-"}
    if k_shot is not None:
        row["k_shot"] = k_shot
    row.update(config=config_label(row), stage=stage,
               error=err if isinstance(err, str) else f"{type(err).__name__}: {err}")
    return row


def config_label(row: Dict[str, Any]) -> str:
    """``extractor | preprocessing | detector`` (``| k-shot`` in the few-shot protocol)."""
    label = f"{row['extractor']} | {row['preprocessing']} | {row['detector']}"
    return label + (f" | {row['k_shot']}-shot" if row.get("k_shot") is not None else "")


def _on_failure(err: Exception, on_error: str, what: str) -> None:
    if on_error == "raise":
        raise err
    logger.error(f"FAILED: {what} ({type(err).__name__}: {err}); continuing with the other runs")


def _record(rows, prog, kind: str, row, dt: float) -> None:
    if row is None:  # skipped combination
        prog.skip(kind)
        return
    rows.append(row)
    prog.add(kind, dt)
    _log_run(logger, prog, row, dt)


def text_bank(ext, detectors) -> Optional[Dict[str, Any]]:
    """Prompt embeddings needed by the zero-shot detectors in ``detectors``, encoded with ``ext`` once
    (``None`` if there are none, or if ``ext`` cannot encode text: those runs are then skipped)."""
    prompts = prompts_of(detectors)
    if not prompts:
        return None
    if not (hasattr(ext, "has_text_encoder") and ext.has_text_encoder()):
        logger.info(f"  {extractor_label(ext)} has no text encoder: zero-shot detectors are skipped with it")
        return None
    with step(f"  encoding {len(prompts)} text prompt(s)"):
        emb = ext.encode_text(prompts)
    return {"emb": dict(zip(prompts, emb)), "scale": float(ext.logit_scale)}


def prompts_of(detectors) -> List[str]:
    out: Dict[str, None] = {}
    for d in detectors:
        if needs_text(d):
            det = d if isinstance(d, BaseDetector) else build_detector(d)
            out.update(dict.fromkeys(det.prompts()))
    return list(out)


def _skip_reason(det_spec, pp, k_shot, text) -> Optional[str]:
    if needs_text(det_spec):
        if text is None:
            return "the extractor has no text encoder"
        if pp and [str(p) for p in pp] != ["l2"]:
            return "preprocessing changes the image-text embedding space"
        return None
    if k_shot == 0:
        return "needs training images"
    return None


def safe_detector_run(Xtr_all, Xte, labels, ds_name: str, ext_name: str, t_feat: float, det_spec, seed: int, pp,
                      train_subsample, opts, k_shot=None, text=None, on_error: str = "continue"):
    """:func:`_detector_run` that returns ``(None, 0)`` for a skipped combination and a failed row (see
    :func:`failed_row`) instead of raising when ``on_error="continue"``."""
    if _skip_reason(det_spec, pp, k_shot, text):
        return None, 0.0
    t0 = time.time()
    try:
        return _detector_run(Xtr_all, Xte, labels, ds_name, ext_name, t_feat, det_spec, seed, pp, train_subsample,
                             opts, k_shot=k_shot, text=text)
    except Exception as e:
        pp_name = "+".join(map(str, pp)) if pp else "-"
        det_name = _det_name(det_spec, det_spec)
        _on_failure(e, on_error, f"{ds_name} · seed {seed} · {ext_name} | {pp_name} | {det_name}"
                    + (f" | {k_shot}-shot" if k_shot is not None else ""))
        return failed_row(ds_name, ext_name, pp_name, det_name, seed, k_shot, "detector", e), time.time() - t0


def _detector_run(Xtr_all, Xte, labels, ds_name: str, ext_name: str, t_feat: float, det_spec, seed: int, pp,
                  train_subsample, opts, k_shot: Optional[int] = None,
                  text: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], float]:
    """One benchmark run of a feature pipeline: training subset for ``seed`` (``k_shot`` images in the few-shot
    protocol) -> preprocessing -> detector -> metrics. Shared by the sequential and the parallel benchmark so
    both give identical rows. ``text``: prompt embeddings for zero-shot detectors (:func:`text_bank`)."""
    import copy

    size = k_shot if k_shot is not None else train_subsample
    Xtr = Xtr_all[_subsample(len(Xtr_all), size, seed)]
    pp_name = "+".join(map(str, pp)) if pp else "-"
    t1 = time.time()
    with step(f"preprocessing {pp_name} (seed {seed}, {len(Xtr)} training images)"):
        prep = build_preprocessing(pp)
        if prep is not None and len(Xtr):
            Ztr, Zte = prep.fit_transform(Xtr), prep.transform(Xte)
        else:  # no preprocessing, or zero-shot (nothing to fit on; only 'l2' gets here, the detector applies it)
            Ztr, Zte = Xtr, Xte
    det = copy.deepcopy(det_spec) if isinstance(det_spec, BaseDetector) else build_detector(det_spec)
    _seed_detector(det, det_spec, seed)
    if needs_text(det):
        det.set_text(text["emb"], text["scale"])
    row = {"dataset": ds_name, "extractor": ext_name, "preprocessing": pp_name, "detector": _det_name(det_spec, det),
           "seed": seed, "n_train": len(Ztr)}
    if k_shot is not None:
        row["k_shot"] = k_shot
    what = f"{ds_name} · seed {seed} · {ext_name} | {pp_name} | {row['detector']}" + (
        f" | {k_shot}-shot" if k_shot is not None else "")
    with step(f"fit {what}"):
        det.fit(Ztr)
    with step(f"score + metrics {what}"):
        s = det.score(Zte)
        _add_metrics(row, labels, s, det.predict(Zte), seed=seed, **opts)
    dt = time.time() - t1
    row.update(threshold=float(det.threshold_), threshold_quantile=det.threshold_quantile, feat_dim=Ztr.shape[1],
               feat_time_s=round(t_feat, 2), det_time_s=round(dt, 3))
    return row, dt


def extractor_label(ext: FeatureExtractor) -> str:
    """Name of an extractor in result rows, e.g. ``facebook/dinov2-base[mean@[6, 9, 12]]``."""
    name = getattr(ext, "model_name", type(ext).__name__)
    if getattr(ext, "pooling", None) not in (None, "auto"):
        name += f"[{ext.pooling}" + (f"@{ext.layer}" if getattr(ext, "layer", None) is not None else "") + "]"
    return name


def _with_intensity(spec, intensity, anomalib: bool = False):
    """Apply a benchmark-wide intensity rule to a spec that does not set its own."""
    if intensity is None:
        return spec
    if isinstance(spec, str) and spec.startswith("zoo:") and not anomalib:
        from .zoo import get

        spec = get(spec).extractor_spec()
    if isinstance(spec, str):
        return {"name": spec, "intensity": intensity} if anomalib else {"type": "hf", "model_name": spec,
                                                                         "intensity": intensity}
    if isinstance(spec, dict):
        return spec if "intensity" in spec else {**spec, "intensity": intensity}
    if isinstance(spec, FeatureExtractor) and spec.intensity == "auto":
        import copy

        return copy.copy(spec).set_intensity(intensity)
    return spec


def _hub_name(spec) -> Optional[str]:
    """Hub id of an extractor spec (str, dict or instance), if it loads from the Hub."""
    if isinstance(spec, str) and spec.startswith("zoo:"):
        from .zoo import get

        return get(spec).hub_id
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict) and str(spec.get("checkpoint", "")).startswith(("hf-hub:", "hf_hub:")):
        return "hf-hub:" + "/".join(spec["checkpoint"].split(":", 1)[1].split("/")[:2])
    if isinstance(spec, dict):
        name = spec.get("model_name") if spec.get("type", "hf") in ("hf", "open_clip", "timm") else None
        return name.replace("hf_hub:", "hf-hub:") if isinstance(name, str) else name
    ckpt = getattr(spec, "checkpoint", None)
    if isinstance(ckpt, str) and ckpt.startswith(("hf-hub:", "hf_hub:")):
        return "hf-hub:" + "/".join(ckpt.split(":", 1)[1].split("/")[:2])
    name = getattr(spec, "model_name", None)
    return name.replace("hf_hub:", "hf-hub:") if isinstance(name, str) else name


def _check_hub_access(extractors, token) -> None:
    from .hub import GatedModelError, check_models

    names = [n for n in map(_hub_name, extractors) if n]
    if not names:
        return
    with step(f"Checking access to {len(names)} Hugging Face model(s)", logging.INFO):
        statuses = check_models(names, token)
    for st in statuses:
        level = logging.INFO if st.ok else logging.ERROR
        logger.log(level, f"  {st.repo_id}: {st.state}" + (f" (user {st.user})" if st.user else ""))
    bad = [st for st in statuses if not st.ok]
    if bad:
        raise GatedModelError("Some Hugging Face models cannot be accessed:\n\n" + "\n\n".join(st.explain() for st in bad))


def _spec_name(spec) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return str(spec.get("model_name") or spec.get("name") or f"{spec.get('type', 'hf')} extractor")
    return getattr(spec, "model_name", type(spec).__name__)


def _log_plan(log, extractors, preprocessings, detectors, anomalib_models, datasets, seeds, n_feat, n_anom,
              train_subsample, bootstrap) -> None:
    log.info(f"Benchmark plan: {len(datasets)} dataset(s), {len(seeds)} seed(s), {n_feat + n_anom} runs in total")
    for ds in datasets:
        log.info(f"  {ds.summary()}")
    if n_feat:
        log.info(f"  feature pipelines: {len(extractors)} extractor(s) × {len(preprocessings)} preprocessing(s) × "
                 f"{len(detectors)} detector(s) × {len(datasets)} dataset(s) × {len(seeds)} seed(s) = {n_feat} runs")
    if n_anom:
        log.info(f"  anomalib: {len(anomalib_models)} model(s) × {len(datasets)} dataset(s) × {len(seeds)} seed(s) = {n_anom} runs")
    extra = []
    if train_subsample is not None:
        extra.append(f"train_subsample={train_subsample}")
    if bootstrap:
        extra.append(f"bootstrap={bootstrap}")
    if extra:
        log.info("  options: " + ", ".join(extra))


def _log_run(log, prog, row, seconds: float) -> None:
    if row.get("error"):
        log.error(f"{prog.counter()} {row['dataset']} · seed {row['seed']} · {row['config']}: FAILED ({row['error']}) "
                  f"· {prog.timing()}")
        return
    ci = f" [{row['auroc_ci_low']:.3f}, {row['auroc_ci_high']:.3f}]" if "auroc_ci_low" in row else ""
    pix = f" pixel-AUROC={row['pixel_auroc']:.4f} AUPRO={row['aupro']:.4f}" if "pixel_auroc" in row else ""
    took = f"{seconds:.2f}s" if seconds < 60 else fmt_duration(seconds)
    log.info(f"{prog.counter()} {row['dataset']} · seed {row['seed']} · {row['config']}: "
             f"AUROC={row['auroc']:.4f}{ci} AUPR={row['aupr']:.4f}{pix} ({took}) · {prog.timing()}")


def _log_end(log, prog, rows) -> None:
    failed = [r for r in rows if r.get("error")]
    rows = successful_rows(rows)
    if failed:
        log.warning(f"{len(failed)} run(s) failed (rows with an 'error' column):")
        for r in failed[:20]:
            log.warning(f"  {r['dataset']} · {r['config']} [{r.get('stage', '?')}]: {r['error']}")
        if len(failed) > 20:
            log.warning(f"  … and {len(failed) - 20} more")
    if not rows:
        log.info("Benchmark finished: no successful runs.")
        return
    by_cfg: Dict[str, List[float]] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(r["auroc"])
    best = max(by_cfg, key=lambda c: np.mean(by_cfg[c]))
    log.info(f"Benchmark finished: {len(rows)} runs in {fmt_duration(prog.elapsed)}. "
             f"Best mean AUROC {np.mean(by_cfg[best]):.4f}: {best}")


def _subsample(n: int, size: Optional[Union[float, int]], seed: int) -> np.ndarray:
    if size is None:
        return np.arange(n)
    if isinstance(size, (int, np.integer)) and size == 0:  # zero-shot
        return np.arange(0)
    k = int(round(size * n)) if isinstance(size, float) else int(size)
    k = max(1, min(n, k))
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def _seed_detector(det: BaseDetector, spec, seed: int) -> None:
    fixed = isinstance(spec, dict) and "random_state" in spec
    if hasattr(det, "random_state") and not fixed:
        det.random_state = seed
    det.random_state_calibration = seed


def _add_metrics(row, y, s, pred, seed: int, bootstrap: int, ci_alpha: float, keep_scores: bool) -> None:
    row["config"] = config_label(row)
    row.update(evaluate_scores(y, s, pred))
    if bootstrap:
        from .stats import bootstrap_ci

        for m, ci in bootstrap_ci(y, s, ("auroc", "aupr"), n_boot=bootstrap, alpha=ci_alpha, seed=seed).items():
            row[f"{m}_ci_low"], row[f"{m}_ci_high"] = ci["low"], ci["high"]
    if keep_scores:
        row["scores"], row["labels"] = np.asarray(s, dtype=float), np.asarray(y, dtype=int)


def _benchmark_anomalib(spec, ds, verbose: bool, seed: int = 0, train_subsample=None, k_shot: Optional[int] = None,
                        **opts) -> Dict[str, Any]:
    from .anomalib_backend import build_anomalib_pipeline

    pipe = build_anomalib_pipeline(spec, seed=seed)
    size = k_shot if k_shot is not None else train_subsample
    train = [ds.train_paths[i] for i in _subsample(len(ds.train_paths), size, seed)]
    t0 = time.time()
    pipe.fit(train, show_progress=verbose)
    t1 = time.time()
    what = "scoring test images" + (" + anomaly maps" if ds.has_masks else "")
    with step(f"  {what} ({len(ds.test_paths)} images)", logging.INFO):
        if ds.has_masks:
            s, maps = pipe.score_and_maps(ds.test_paths, show_progress=verbose)
        else:
            s, maps = pipe.score(ds.test_paths, show_progress=verbose), None
    row = {"dataset": ds.name, "extractor": "anomalib", "preprocessing": "-", "detector": _anomalib_name(spec),
           "seed": seed, "n_train": len(train)}
    if k_shot is not None:
        row["k_shot"] = k_shot
    _add_metrics(row, ds.test_labels, s, (s > pipe.threshold_).astype(int), seed=seed, **opts)
    row.update(threshold=float(pipe.threshold_), threshold_quantile=pipe.threshold_quantile)
    if maps is not None:
        with step("  pixel-level metrics (pixel-AUROC, AUPRO)"):
            row.update(evaluate_maps(ds.test_masks, maps))
        if opts.get("keep_scores"):  # a few maps for the report's example figure
            row.update(example_maps=_example_maps(ds, s, maps), pixel_threshold=float(pipe.pixel_threshold_),
                       map_reference=float(pipe.map_reference_))
    row.update(fit_time_s=round(t1 - t0, 2), test_time_s=round(time.time() - t1, 2))
    return row


def _example_maps(ds, scores, maps, n_anomalous: int = 3) -> Dict[int, np.ndarray]:
    """Index -> map (<= 128 px) for one anomalous image per defect type (highest score) and the median
    normal image."""
    from .maps import resize_map

    y, types = np.asarray(ds.test_labels), np.asarray(ds.test_types)
    picked = []
    for t in sorted(set(types[y == 1]))[:n_anomalous]:
        idx = np.flatnonzero((y == 1) & (types == t))
        picked.append(int(idx[np.argmax(scores[idx])]))
    normal = np.flatnonzero(y == 0)
    if len(normal):
        picked.append(int(normal[np.argsort(scores[normal])[len(normal) // 2]]))
    out = {}
    for i in picked:
        h, w = maps[i].shape
        f = min(1.0, 128 / max(h, w))
        out[i] = resize_map(maps[i], (max(1, round(h * f)), max(1, round(w * f))))
    return out


_ANOMALIB_NOISE = ("device", "pre_trained", "batch_size", "train_batch_size", "num_workers", "seed")


def _anomalib_name(spec) -> str:
    """Anomalib spec name without run-time settings (device, batch size...) that do not define the model."""
    if isinstance(spec, dict):
        spec = {k: v for k, v in spec.items() if k not in _ANOMALIB_NOISE}
    return _det_name(spec, spec)


def _det_name(spec, det) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        extra = {k: v for k, v in spec.items() if k != "name"}
        return spec["name"] + (f"({', '.join(f'{k}={v}' for k, v in extra.items())})" if extra else "")
    return repr(det)


def to_markdown(rows: List[Dict[str, Any]], metrics=("auroc", "aupr", "f1_max", "fpr@95tpr")) -> str:
    """Markdown table; pixel metrics are appended when some row has them."""
    rows = successful_rows(rows)
    metrics = list(metrics) + [k for k in ("pixel_auroc", "aupro") if any(k in r for r in rows) and k not in metrics]
    seed_col = ["seed"] if len({r.get("seed", 0) for r in rows}) > 1 else []
    shot_col = ["k_shot"] if any("k_shot" in r for r in rows) else []
    cols = ["dataset", "extractor", "preprocessing", "detector", *shot_col, *seed_col, *metrics]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = (r.get(c, "-") for c in cols)
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in cells) + " |")
    return "\n".join(lines)
