"""Command line interface.

    imgad list
    imgad fit   --model openai/clip-vit-base-patch32 --detector knn -p k=1 --prep l2 \
                --train data/bottle/train/good --out bottle.joblib
    imgad fit   --anomalib patchcore -p backbone=wide_resnet50_2 --train data/bottle/train/good --out pc.joblib
    imgad predict --model bottle.joblib new.png folder/ --out preds.csv
    imgad predict --model pc.joblib folder/ --maps-dir maps/ --out preds.csv
    imgad calibrate --model bottle.joblib --normal val/good --anomalous val/defects --method f1
    imgad info --model bottle.joblib
    imgad benchmark --config configs/mvtec_benchmark.yaml --out results.csv --seeds 0 1 2 --bootstrap 1000 --stats
    imgad stats --results results.csv --metric auroc --cd-plot cd.png --out stats.md
    imgad hf-login                                  # token for gated Hugging Face models
    imgad hf-check facebook/dinov3-vitb16-pretrain-lvd1689m
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Any, Dict, List

import numpy as np
import yaml


def _parse_value(v: str) -> Any:
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def _kv(pairs: List[str]) -> Dict[str, Any]:
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        out[k] = _parse_value(v)
    return out


def _write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    scalar = (str, int, float, bool, type(None), np.number)
    rows = [{k: v for k, v in r.items() if isinstance(v, scalar)} for r in rows]  # drop per-image arrays / maps
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _token(args):
    """Token from --hf-token-file (never from a plain command-line value, which would leak into the shell
    history and the process list). None -> HF_TOKEN or the stored login."""
    path = getattr(args, "hf_token_file", None)
    if not path:
        return None
    from imgad.hub import read_token_file

    return read_token_file(path)


def _add_token_arg(p) -> None:
    p.add_argument("--hf-token-file", default=None, metavar="PATH",
                   help="file with a Hugging Face token for gated/private models (default: HF_TOKEN or `imgad hf-login`)")


def cmd_list(args):
    from imgad import list_anomalib_models, list_detectors, list_extractors

    print("extractors:", ", ".join(list_extractors()))
    print("detectors: ", ", ".join(list_detectors()))
    print("preprocessing: l2, standard, center, pca:<n>, pca_whiten:<n>")
    am = list_anomalib_models()
    print("anomalib:  ", ", ".join(am) if am else '(not installed: pip install -e ".[anomalib]")')


def cmd_fit(args):
    from imgad.data import list_images

    if args.anomalib:
        return _fit_anomalib(args, list_images(args.train))
    if not args.model:
        raise SystemExit("fit: give --model (HF id) or --anomalib (anomalib model name)")
    from imgad import AnomalyDetectionPipeline

    ext = {"type": args.extractor_type, "model_name": args.model, "batch_size": args.batch_size, **_kv(args.ext_param)}
    if args.intensity:
        ext["intensity"] = args.intensity
    token = _token(args)
    if token is not None:
        ext["token"] = token  # used to download the model; never saved
    if args.device:
        ext["device"] = args.device
    det = {"name": args.detector, **_kv(args.param)}
    pipe = AnomalyDetectionPipeline(ext, det, preprocessing=args.prep, cache_dir=args.cache_dir)
    paths = list_images(args.train)
    print(f"Fitting on {len(paths)} normal images\n{pipe}")
    pipe.fit(paths)
    pipe.save(args.out)
    print(f"Saved to {args.out} (threshold={pipe.threshold_:.6f})")


def _fit_anomalib(args, paths):
    from imgad import AnomalibPipeline

    kw = {"batch_size": args.batch_size, "device": args.device, **_kv(args.param)}
    if args.intensity:
        kw["intensity"] = args.intensity
    if args.image_size:
        kw["image_size"] = args.image_size
    if args.max_epochs:
        kw["max_epochs"] = args.max_epochs
    pipe = AnomalibPipeline(args.anomalib, **kw)
    print(f"Fitting on {len(paths)} normal images\n{pipe}")
    pipe.fit(paths)
    pipe.save(args.out)
    print(f"Saved to {args.out} (threshold={pipe.threshold_:.6f}, pixel_threshold={pipe.pixel_threshold_:.6f})")


def cmd_predict(args):
    from imgad import AnomalyPredictor

    pred = AnomalyPredictor.load(args.model, device=args.device, threshold=args.threshold, batch_size=args.batch_size,
                                 token=_token(args))
    if args.maps_dir and not pred.supports_maps:
        raise SystemExit("--maps-dir needs a model with anomaly maps (trained with --anomalib)")
    if args.maps_dir:  # stream in chunks: maps at full resolution are large
        results = []
        for r in pred.predict_iter(args.inputs, chunk_size=64, recursive=not args.no_recursive, return_maps=True):
            if r.is_anomaly or not args.only_anomalies:
                pred.save_maps([r], args.maps_dir)
            r.anomaly_map = None
            results.append(r)
    else:
        results = pred.predict(args.inputs, recursive=not args.no_recursive, show_progress=not args.quiet)
    n_anom = sum(r.is_anomaly for r in results)
    shown = [r for r in results if r.is_anomaly] if args.only_anomalies else results
    if args.out:
        pred.save_results(shown, args.out)
        print(f"{len(results)} images, {n_anom} anomalous (threshold={pred.threshold:.6g}) -> {args.out}")
        if args.maps_dir:
            print(f"heatmaps and masks -> {args.maps_dir}")
    else:
        print("score\tpercentile\tanomaly\tsource")
        for r in shown:
            print(f"{r.score:.6f}\t{r.normal_percentile:.4f}\t{int(r.is_anomaly)}\t{r.source}")
        print(f"# {len(results)} images, {n_anom} anomalous (threshold={pred.threshold:.6g})", file=sys.stderr)


def cmd_calibrate(args):
    from imgad import load_pipeline
    from imgad.data import list_images

    pipe = load_pipeline(args.model, device=args.device, batch_size=args.batch_size, token=_token(args))
    normal = [p for d in args.normal for p in list_images(d)]
    anomalous = [p for d in (args.anomalous or []) for p in list_images(d)]
    paths, labels = normal + anomalous, [0] * len(normal) + [1] * len(anomalous)
    print(f"Calibrating on {len(normal)} normal + {len(anomalous)} anomalous images (method={args.method})")
    old = pipe.threshold_
    res = pipe.calibrate_threshold(paths, labels, method=args.method, target_fpr=args.target_fpr, target_tpr=args.target_tpr)
    out = args.out or args.model
    pipe.save(out)
    print(f"threshold {old:.6g} -> {res['threshold']:.6g}")
    print(json.dumps(res, indent=2))
    print(f"Saved to {out}")


def cmd_info(args):
    from imgad import load_pipeline

    print(json.dumps(load_pipeline(args.model, device="cpu").info(), indent=2, default=str))


def _datasets_from_config(cfg: Dict[str, Any]):
    from imgad import data

    dcfg = cfg["data"]
    kind = dcfg.get("type", "mvtec")
    if kind == "mvtec":
        cats = dcfg.get("categories") or data.mvtec_categories(dcfg["root"])
        return [data.mvtec_ad(dcfg["root"], c) for c in cats]
    if kind == "folders":
        return [data.from_folders(dcfg["train_normal"], dcfg["test_normal"], dcfg["test_anomalous"], dcfg.get("name", "folders"))]
    if kind == "one_vs_rest":
        classes = dcfg["normal_classes"]
        return [
            data.one_vs_rest(dcfg["root"], c, dcfg.get("train_fraction", 0.8), dcfg.get("max_anomalies"), dcfg.get("seed", 0))
            for c in classes
        ]
    raise ValueError(f"Unknown data.type {kind!r}")


def cmd_benchmark(args):
    from imgad.evaluation import benchmark, to_markdown

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    _setup_progress(args)
    datasets = _datasets_from_config(cfg)
    seeds = args.seeds if args.seeds is not None else cfg.get("seeds", [0])
    bootstrap = args.bootstrap if args.bootstrap is not None else cfg.get("bootstrap", 0)
    subsample = args.train_subsample if args.train_subsample is not None else cfg.get("train_subsample")
    rows = benchmark(
        extractors=cfg.get("extractors") or [],
        detectors=cfg.get("detectors") or [],
        anomalib_models=cfg.get("anomalib") or [],
        datasets=datasets,
        preprocessings=cfg.get("preprocessing", [None]),
        cache_dir=cfg.get("cache_dir"),
        seeds=seeds,
        train_subsample=subsample,
        bootstrap=bootstrap,
        ci_alpha=cfg.get("ci_alpha", 0.05),
        keep_scores=bool(args.scores_out or args.report),
        verbose=not args.quiet,
        intensity=args.intensity or cfg.get("intensity"),
        token=_token(args),
        **_parallel_kwargs(args, cfg),
    )
    if args.out:
        _write_csv(rows, args.out)
        print(f"Wrote {len(rows)} rows to {args.out}")
    if args.scores_out:
        import joblib

        joblib.dump(rows, args.scores_out)
        print(f"Per-image scores -> {args.scores_out} (imgad stats --scores ...)")
    print(to_markdown(rows))
    if args.report:
        from imgad.report import generate_report

        path = generate_report(rows, datasets, args.report, config=cfg, title=args.report_title, metric=args.metric,
                               alpha=cfg.get("ci_alpha", 0.05), command="imgad " + " ".join(args.argv))
        print(f"Report -> {path}")
    if args.stats:
        _print_stats(rows, args.metric, cfg.get("ci_alpha", 0.05), None, None)


def _print_stats(rows, metric: str, alpha: float, cd_plot, out, baseline=None) -> None:
    from imgad import stats

    lines = []
    if len({r.get("seed", 0) for r in rows}) > 1:
        summ = stats.summarize_seeds(rows, metrics=(metric,), alpha=alpha)
        lines += [f"### Repetitions ({metric}: mean ± std, {1 - alpha:.0%} CI of the mean)", "",
                  "| dataset | configuration | seeds | mean ± std | CI |", "|---|---|---|---|---|"]
        for r in summ:
            if f"{metric}_std" in r:
                lines.append(f"| {r['dataset']} | {stats.md_escape(r['config'])} | {r['n_seeds']} | {r[f'{metric}_mean']:.4f} ± "
                             f"{r[f'{metric}_std']:.4f} | [{r[f'{metric}_ci_low']:.4f}, {r[f'{metric}_ci_high']:.4f}] |")
        lines.append("")
    try:
        res = stats.compare_configs(rows, metric=metric, alpha=alpha)
        lines.append(stats.report_markdown(res))
        if cd_plot and "cd" in res:
            stats.plot_cd_diagram(res, cd_plot)
            lines.append(f"\nCD diagram -> {cd_plot}")
    except ValueError as e:
        res = None
        lines.append(f"(comparison across datasets skipped: {e})")
    if baseline is not None or (rows and "scores" in rows[0]):
        base = baseline or (res["configs"][0] if res else rows[0]["config"])
        tests = stats.delong_vs_baseline(rows, base, alpha)
        lines += ["", f"### DeLong vs. {stats.md_escape(base)} (per dataset and seed, Holm within each block)", "",
                  "| dataset | seed | configuration | ΔAUROC | CI | p (Holm) |", "|---|---|---|---|---|---|"]
        lines += [f"| {t['dataset']} | {t['seed']} | {stats.md_escape(t['config'])} | {t['diff']:+.4f} | [{t['diff_low']:+.4f}, "
                  f"{t['diff_high']:+.4f}] | {t['p_holm']:.3g}{' *' if t['significant'] else ''} |" for t in tests]
    text = "\n".join(lines)
    print(text)
    if out:
        with open(out, "w") as f:
            f.write(text + "\n")
        print(f"Report -> {out}")


def _setup_progress(args) -> None:
    """Console verbosity: -q = warnings only (no progress bars), default = phases and runs, -v = every step.
    --log-file always receives the full DEBUG log."""
    import logging

    from imgad.progress import setup_logging

    level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level, log_file=args.log_file)


def _add_progress_args(p) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("-q", "--quiet", action="store_true", help="only warnings and errors (no progress bars)")
    g.add_argument("-v", "--verbose", action="store_true", help="log every step (fit, score, cache, timings)")
    p.add_argument("--log-file", default=None, help="also write the full (DEBUG) log with timestamps to this file")


def _parallel_kwargs(args, cfg) -> dict:
    """Parallel options: CLI flags override the ``parallel:`` section of the YAML."""
    p = dict(cfg.get("parallel") or {})
    out = {}
    jobs = args.jobs if args.jobs is not None else p.get("jobs")
    if jobs is not None:
        out["n_jobs"] = int(jobs)
    devices = args.devices if args.devices is not None else p.get("devices")
    if devices is not None:
        out["devices"] = devices
    gpu_jobs = args.gpu_jobs if args.gpu_jobs is not None else p.get("gpu_jobs")
    if gpu_jobs is not None:
        out["gpu_jobs"] = gpu_jobs if gpu_jobs == "auto" else int(gpu_jobs)
    gpu_mem = args.gpu_mem_gb if args.gpu_mem_gb is not None else p.get("gpu_mem_gb")
    if gpu_mem is not None:
        out["gpu_mem_gb"] = float(gpu_mem)
    return out


def cmd_devices(args):
    from imgad.devices import GiB, ResourceManager, estimate_extractor_bytes, list_devices, resolve_devices

    devs = list_devices()
    print("Devices:")
    for d in devs:
        print(f"  {d}")
    try:
        use = resolve_devices(args.devices)
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    rm = ResourceManager(use, args.gpu_jobs if args.gpu_jobs == "auto" else int(args.gpu_jobs))
    print(f"With --devices {args.devices} the benchmark would run accelerator jobs on:")
    for line in rm.describe():
        print(f"  {line}")
    if args.config:
        from imgad.evaluation import _hub_name

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        default = int((args.gpu_mem_gb or 4) * GiB)
        print("Estimated GPU memory per feature extractor (checkpoint size + activations + CUDA context):")
        for spec in cfg.get("extractors") or []:
            name = _hub_name(spec) or str(spec)
            conf = spec if isinstance(spec, dict) else {"model_name": spec}
            est = estimate_extractor_bytes(conf, default)
            print(f"  {name}: ~{est / GiB:.1f} GiB" + ("  (default, size unknown)" if est == default else ""))


def cmd_report(args):
    import joblib

    from imgad.report import generate_report

    _setup_progress(args)
    rows = joblib.load(args.scores)
    cfg, datasets = None, []
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        datasets = _datasets_from_config(cfg)
    path = generate_report(rows, datasets, args.out, config=cfg, title=args.title, metric=args.metric, alpha=args.alpha)
    print(f"Report -> {path}")


def cmd_stats(args):
    if bool(args.results) == bool(args.scores):
        raise SystemExit("stats: give either --results (CSV) or --scores (joblib from benchmark --scores-out)")
    if args.results:
        from imgad.stats import load_results

        rows = load_results(args.results)
    else:
        import joblib

        rows = joblib.load(args.scores)
    _print_stats(rows, args.metric, args.alpha, args.cd_plot, args.out, args.baseline)


def cmd_hf_login(args):
    from imgad.hub import login

    token = _token(args)
    user = login(token)
    print(f"Logged in to Hugging Face as {user} (token stored like `hf auth login`).")


def cmd_hf_logout(args):
    from imgad.hub import logout

    logout()
    print("Logged out from Hugging Face (stored token removed).")


def cmd_hf_check(args):
    from imgad.hub import check_models, token_source, whoami

    names = list(args.models)
    if args.config:
        from imgad.evaluation import _hub_name

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        names += [n for n in map(_hub_name, cfg.get("extractors") or []) if n]
    if not names:
        raise SystemExit("hf-check: give model ids and/or --config")
    token = _token(args)
    user = whoami(token)
    print(f"Token: {token_source(token)}" + (f" (user {user})" if user else ""))
    statuses = check_models(names, token)
    if not statuses:
        print("No Hugging Face Hub models to check.")
        return
    for st in statuses:
        print(f"[{'ok' if st.ok else 'NO'}] {st.explain()}")
    if not all(st.ok for st in statuses):
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="imgad", description="Image anomaly detection with foundation-model features")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List available extractors / detectors").set_defaults(func=cmd_list)

    f = sub.add_parser("fit", help="Fit a pipeline on a folder of normal images")
    f.add_argument("--model", help="HF model id (or open_clip name with --extractor-type open_clip)")
    f.add_argument("--anomalib", default=None, help="anomalib model instead of extractor+detector (patchcore, padim, ...); -p sets its kwargs")
    f.add_argument("--image-size", type=int, default=None, help="(anomalib) input size, default 256")
    f.add_argument("--max-epochs", type=int, default=None, help="(anomalib) epochs for trainable models")
    f.add_argument("--extractor-type", default="hf")
    f.add_argument("-e", "--ext-param", nargs="*", default=[], help="extractor kwargs, e.g. pooling=mean layer=-2")
    f.add_argument("--detector", default="knn")
    f.add_argument("-p", "--param", nargs="*", default=[], help="detector (or anomalib model) kwargs, e.g. k=5 metric=cosine")
    f.add_argument("--prep", nargs="*", default=["l2"], help="preprocessing steps")
    f.add_argument("--train", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--batch-size", type=int, default=32)
    f.add_argument("--device", default=None)
    f.add_argument("--cache-dir", default=None)
    f.add_argument("--intensity", default=None,
                   help="images with >8 bits (16-bit, DICOM, NIfTI): auto | minmax | dicom | percentile:1,99 | window:lo,hi")
    _add_token_arg(f)
    f.set_defaults(func=cmd_fit)

    for name in ("predict", "score"):  # "score" kept as alias
        s = sub.add_parser(name, help="Run inference with a trained model on images and/or folders")
        s.add_argument("inputs", nargs="*", help="image files and/or folders")
        s.add_argument("--model", "--pipeline", dest="model", required=True, help="trained .joblib")
        s.add_argument("--images", action="append", default=[], help="(legacy) folder or image; same as positional")
        s.add_argument("--out", default=None, help=".csv or .json output (default: print to stdout)")
        s.add_argument("--threshold", type=float, default=None, help="override the saved threshold")
        s.add_argument("--only-anomalies", action="store_true")
        s.add_argument("--maps-dir", default=None, help="(anomalib models) write heatmap overlays + defect masks here")
        s.add_argument("--no-recursive", action="store_true")
        s.add_argument("--batch-size", type=int, default=None)
        s.add_argument("--device", default=None)
        s.add_argument("-q", "--quiet", action="store_true")
        _add_token_arg(s)
        s.set_defaults(func=cmd_predict)

    c = sub.add_parser("calibrate", help="Re-tune the decision threshold of a trained model on labelled validation images")
    c.add_argument("--model", required=True)
    c.add_argument("--normal", nargs="+", required=True, help="folders with normal validation images")
    c.add_argument("--anomalous", nargs="*", default=[], help="folders with anomalous validation images")
    c.add_argument("--method", default="f1", choices=["f1", "youden", "fpr", "tpr"])
    c.add_argument("--target-fpr", type=float, default=0.05)
    c.add_argument("--target-tpr", type=float, default=0.95)
    c.add_argument("--out", default=None, help="output .joblib (default: overwrite --model)")
    c.add_argument("--batch-size", type=int, default=None)
    c.add_argument("--device", default=None)
    _add_token_arg(c)
    c.set_defaults(func=cmd_calibrate)

    i = sub.add_parser("info", help="Show the configuration and metadata of a trained model")
    i.add_argument("--model", required=True)
    i.set_defaults(func=cmd_info)

    b = sub.add_parser("benchmark", help="Run an extractor x detector grid (and anomalib models) from a YAML config")
    b.add_argument("--config", required=True)
    b.add_argument("--out", default=None)
    b.add_argument("--seeds", type=int, nargs="+", default=None, help="repetitions (overrides 'seeds' in the YAML)")
    b.add_argument("--train-subsample", type=_parse_value, default=None,
                   help="per-seed subset of normal training images: fraction (0.8) or count (50)")
    b.add_argument("--bootstrap", type=int, default=None, help="bootstrap resamples for AUROC/AUPR CIs (0 = off)")
    b.add_argument("--scores-out", default=None, help="save rows + per-image scores (.joblib) for paired tests")
    b.add_argument("--stats", action="store_true", help="print seed summary and statistical comparison")
    b.add_argument("--metric", default="auroc")
    b.add_argument("--report", default=None, metavar="DIR",
                   help="write a markdown report (data analysis, methods, results with figures, discussion) to DIR")
    b.add_argument("--report-title", default=None)
    b.add_argument("--intensity", default=None,
                   help="rule for images with >8 bits for every model (overrides `intensity:` in the YAML)")
    _add_progress_args(b)
    g = b.add_argument_group("parallel execution (default: sequential)")
    g.add_argument("-j", "--jobs", type=int, default=None, help="CPU worker processes for detector runs (-1 = all cores)")
    g.add_argument("--devices", default=None,
                   help="where feature extractors / anomalib models run: auto (all GPUs), cpu, cuda:0,cuda:1 ...")
    g.add_argument("--gpu-jobs", default=None, help="max concurrent jobs per GPU (default auto: as many as fit in memory)")
    g.add_argument("--gpu-mem-gb", type=float, default=None,
                   help="GPU memory assumed for a job of unknown size (default: from the checkpoint size, else 4)")
    _add_token_arg(b)
    b.set_defaults(func=cmd_benchmark)

    rp = sub.add_parser("report", help="Build the markdown report from a benchmark --scores-out file")
    rp.add_argument("--scores", required=True, help="file written by benchmark --scores-out")
    rp.add_argument("--config", default=None, help="benchmark YAML (enables the dataset analysis and example images)")
    rp.add_argument("--out", default="report", help="output folder")
    rp.add_argument("--title", default=None)
    rp.add_argument("--metric", default="auroc")
    rp.add_argument("--alpha", type=float, default=0.05)
    _add_progress_args(rp)
    rp.set_defaults(func=cmd_report)

    st = sub.add_parser("stats", help="Statistical comparison of benchmark results (ranks, Friedman, Nemenyi, Wilcoxon, DeLong)")
    st.add_argument("--results", default=None, help="benchmark CSV")
    st.add_argument("--scores", default=None, help="benchmark --scores-out file (enables per-dataset DeLong tests)")
    st.add_argument("--metric", default="auroc", help="auroc | aupr | f1_max | fpr@95tpr | pixel_auroc | aupro ...")
    st.add_argument("--alpha", type=float, default=0.05)
    st.add_argument("--baseline", default=None, help="config for DeLong tests (default: best average rank)")
    st.add_argument("--cd-plot", default=None, help="save a critical-difference diagram (needs matplotlib)")
    st.add_argument("--out", default=None, help="write the report as markdown")
    st.set_defaults(func=cmd_stats)
    dv = sub.add_parser("devices", help="Show GPUs/CPU, free memory and how a parallel benchmark would use them")
    dv.add_argument("--devices", default="auto")
    dv.add_argument("--gpu-jobs", default="auto")
    dv.add_argument("--gpu-mem-gb", type=float, default=None)
    dv.add_argument("--config", default=None, help="benchmark YAML: estimate the GPU memory of its extractors")
    dv.set_defaults(func=cmd_devices)

    hl = sub.add_parser("hf-login", help="Store a Hugging Face token for gated/private models (input hidden)")
    _add_token_arg(hl)
    hl.set_defaults(func=cmd_hf_login)
    sub.add_parser("hf-logout", help="Remove the stored Hugging Face token").set_defaults(func=cmd_hf_logout)
    hc = sub.add_parser("hf-check", help="Check access to Hugging Face models (gated, private, token...)")
    hc.add_argument("models", nargs="*", help="model ids, e.g. facebook/dinov3-vitb16-pretrain-lvd1689m")
    hc.add_argument("--config", default=None, help="benchmark YAML: check every Hub extractor in it")
    _add_token_arg(hc)
    hc.set_defaults(func=cmd_hf_check)
    return ap


def main(argv=None):
    ap = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = ap.parse_args(argv)
    args.argv = argv
    if getattr(args, "func", None) is cmd_predict:
        args.inputs = list(args.inputs) + list(args.images)
        if not args.inputs:
            ap.error("predict: give at least one image or folder")
    from imgad.hub import GatedModelError

    try:
        args.func(args)
    except GatedModelError as e:  # access problems: print the instructions, not a traceback
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
