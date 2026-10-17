"""Parallel benchmark: accelerator jobs (feature extraction, Anomalib models) on GPUs with memory admission,
and detector runs on a pool of CPU processes, pipelined.

Used by :func:`imgad.evaluation.benchmark` when ``n_jobs != 1`` or ``devices`` is given::

    benchmark(extractors, detectors, datasets, n_jobs=8, devices="auto")

Scheduling:

* **Accelerator jobs** run in ``spawn`` worker processes (CUDA cannot be forked). A feature extraction
  job loads one extractor on its device, extracts the features of every dataset (filling the feature
  cache) and frees the model; an Anomalib job trains and tests one model on one dataset and seed.
* A job starts only when :class:`imgad.devices.ResourceManager` finds a device with a free slot and
  enough memory for its estimate (checkpoint size for Hub models, then the peak measured in previous
  jobs of the same model). If it still runs out of memory, it is retried alone on an idle device.
* As soon as the features of an extractor are ready, its detector runs (dataset × seed × preprocessing
  × detector) are submitted to the CPU pool, while the GPU moves on to the next model.
* Rows are returned in the same order as the sequential benchmark and are identical to it (apart from
  the timing columns).
"""

from __future__ import annotations

import gc
import inspect
import itertools
import logging
import multiprocessing as mp
import os
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .devices import CUDA_CONTEXT_BYTES, GiB, ResourceManager, estimate_extractor_bytes, resolve_devices
from .progress import fmt_duration, logger

DEFAULT_JOB_BYTES = 4 * GiB


# ============================================================================ worker side
def _init_worker(n_threads: int) -> None:
    """Limit math threads per worker so that parallel processes do not oversubscribe the CPU, and silence
    progress bars (several workers drawing bars on the same terminal is unreadable; the parent reports)."""
    os.environ["TQDM_DISABLE"] = "1"
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(n_threads)
    except Exception:  # pragma: no cover
        pass
    try:
        import torch

        torch.set_num_threads(n_threads)
    except Exception:  # pragma: no cover
        pass
    logging.getLogger("imgad").setLevel(logging.WARNING)  # the parent process does the reporting


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # pragma: no cover
        pass
    return "out of memory" in str(exc).lower()


class _cuda_peak:
    """Measures the peak memory reserved by PyTorch on ``device`` during the block (0 off CUDA) and
    releases the cached memory at the end."""

    def __init__(self, device: str):
        self.device, self.peak = device, 0

    def __enter__(self):
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.set_device(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def __exit__(self, *exc):
        gc.collect()
        if self.device.startswith("cuda"):
            import torch

            self.peak = int(torch.cuda.max_memory_reserved(self.device))
            torch.cuda.empty_cache()
        return False


def _extract_job(ext_cls, ext_cfg: Dict[str, Any], device: str, datasets, cache_dir, token=None,
                 intensity: str = "auto") -> Dict[str, Any]:
    """Load the extractor on ``device`` and extract train/test features for every dataset. ``token`` (gated
    Hub models) travels only through the process pipe; it is never part of the stored config."""
    t0 = time.time()
    cfg = dict(ext_cfg)
    params = inspect.signature(ext_cls.__init__).parameters
    if "device" in params:
        cfg["device"] = device
    if token is not None and "token" in params:
        cfg["token"] = token
    try:
        with _cuda_peak(device) as peak:
            ext = ext_cls(**cfg)
            ext.set_intensity(intensity)
            if hasattr(ext, "_load"):
                ext._load()
            t_load = time.time() - t0
            feats = {}
            for ds in datasets:
                t1 = time.time()
                Xtr = ext.extract(ds.train_paths, cache_dir=cache_dir, show_progress=False)
                Xte = ext.extract(ds.test_paths, cache_dir=cache_dir, show_progress=False)
                feats[ds.name] = (Xtr, Xte, time.time() - t1)
            del ext
    except Exception as e:
        if _is_oom(e):
            return {"oom": True, "error": f"{type(e).__name__}: {e}"}
        raise
    return {"features": feats, "peak_bytes": peak.peak, "load_s": t_load, "seconds": time.time() - t0}


def _detector_job(Xtr_all, Xte, labels, ds_name, ext_name, t_feat, det_spec, seed, pp, train_subsample, opts):
    from .evaluation import _detector_run

    return _detector_run(Xtr_all, Xte, labels, ds_name, ext_name, t_feat, det_spec, seed, pp, train_subsample, opts)


def _anomalib_job(spec, ds, seed: int, train_subsample, opts, device: str) -> Dict[str, Any]:
    from .evaluation import _benchmark_anomalib

    spec = {"name": spec} if isinstance(spec, str) else dict(spec)
    spec["device"] = device
    t0 = time.time()
    try:
        with _cuda_peak(device) as peak:
            row = _benchmark_anomalib(spec, ds, False, seed, train_subsample, **opts)
    except Exception as e:
        if _is_oom(e):
            return {"oom": True, "error": f"{type(e).__name__}: {e}"}
        raise
    return {"row": row, "peak_bytes": peak.peak, "seconds": time.time() - t0}


# ============================================================================ parent side
@dataclass
class _Job:
    id: int
    kind: str                       # "extract" | "anomalib"
    key: Tuple                      # model identity, for learned memory estimates
    label: str
    est: int
    payload: Dict[str, Any] = field(default_factory=dict)
    exclusive: bool = False
    device: Optional[str] = None
    started: float = 0.0


def _cpu_workers(n_jobs: int) -> int:
    n = os.cpu_count() or 1
    if n_jobs is None or n_jobs == 0:
        return 1
    return max(1, n + 1 + n_jobs) if n_jobs < 0 else int(n_jobs)


def run_parallel(extractors, detectors, datasets, preprocessings, anomalib_models, seeds, train_subsample, opts,
                 cache_dir, n_jobs, devices, gpu_jobs, gpu_mem_gb, prog, token=None) -> List[Dict[str, Any]]:
    from .anomalib_backend import build_anomalib_pipeline  # noqa: F401  (fail early if misconfigured)
    from .evaluation import _anomalib_name, _log_run, extractor_label
    from .features import FeatureExtractor, build_extractor

    devs = resolve_devices("auto" if devices is None else devices)
    rm = ResourceManager(devs, gpu_jobs)
    n_cpu = _cpu_workers(n_jobs)
    default_est = int(gpu_mem_gb * GiB) if gpu_mem_gb else DEFAULT_JOB_BYTES
    acc_threads = max(1, (os.cpu_count() or 1) // max(1, rm.total_capacity() + n_cpu)) if any(
        d.kind == "cpu" for d in devs) else 2
    logger.info(f"Parallel execution: {n_cpu} CPU worker(s) for detector runs; accelerator jobs on:")
    for line in rm.describe():
        logger.info(f"  {line}")
    prog.workers = {"extract": rm.total_capacity(), "anomalib": rm.total_capacity(), "detector": n_cpu}

    # ---------------------------------------------------------------- jobs
    ids = itertools.count()
    pending: deque = deque()
    for i_ext, spec in enumerate(extractors):
        ext = spec if isinstance(spec, FeatureExtractor) else build_extractor(spec, token=token)  # lazy: no weights
        cfg = ext.get_config()
        est = estimate_extractor_bytes(cfg, default_est) if not gpu_mem_gb else default_est
        pending.append(_Job(next(ids), "extract", ("extract", ext.identifier), f"features of {extractor_label(ext)}", est,
                            {"cls": type(ext), "cfg": cfg, "intensity": ext.intensity, "name": extractor_label(ext),
                             "i_ext": i_ext}))
    for i_m, spec in enumerate(anomalib_models):
        for i_ds, ds in enumerate(datasets):
            for i_s, seed in enumerate(seeds):
                name = _anomalib_name(spec)
                pending.append(_Job(next(ids), "anomalib", ("anomalib", name), f"{name} on {ds.name} (seed {seed})",
                                    default_est, {"spec": spec, "ds": ds, "seed": seed, "order": (1, i_m, i_ds, i_s)}))
    learned: Dict[Tuple, int] = {}

    ctx = mp.get_context("spawn")
    acc_pool = ProcessPoolExecutor(rm.total_capacity(), mp_context=ctx, initializer=_init_worker, initargs=(acc_threads,))
    cpu_pool = ProcessPoolExecutor(n_cpu, mp_context=ctx, initializer=_init_worker, initargs=(1,))
    running: Dict[Future, Any] = {}
    rows: List[Tuple[Tuple, Dict[str, Any]]] = []

    def launch() -> None:
        started = True
        while started:
            started = False
            for job in list(pending):
                est = learned.get(job.key, job.est)
                exclusive = job.exclusive or not rm.fits_somewhere(est)
                dev = rm.try_acquire(job.id, est, exclusive=exclusive)
                if dev is None:
                    continue
                pending.remove(job)
                job.device, job.started = dev, time.time()
                if job.kind == "extract":
                    fut = acc_pool.submit(_extract_job, job.payload["cls"], job.payload["cfg"], dev, datasets, cache_dir,
                                          token, job.payload["intensity"])
                else:
                    p = job.payload
                    fut = acc_pool.submit(_anomalib_job, p["spec"], p["ds"], p["seed"], train_subsample, opts, dev)
                running[fut] = job
                mem = ""
                if dev.startswith("cuda"):
                    mem = f"needs ~{est / GiB:.1f} GiB" + (", alone on the device" if exclusive else "")
                    avail = rm.available(dev)
                    if avail is not None and not exclusive:
                        mem += f", {(avail + est) / GiB:.1f} GiB were available"
                    mem = f" ({mem})"
                logger.info(f"Started {job.label} on {dev}{mem} · {rm.n_running()} accelerator job(s) running, "
                            f"{len(pending)} waiting")
                started = True
                break

    def submit_detectors(job: _Job, feats: Dict[str, Any]) -> int:
        n = 0
        for i_ds, ds in enumerate(datasets):
            Xtr, Xte, t_feat = feats[ds.name]
            for i_s, seed in enumerate(seeds):
                for i_pp, pp in enumerate(preprocessings):
                    for i_det, det_spec in enumerate(detectors):
                        fut = cpu_pool.submit(_detector_job, Xtr, Xte, ds.test_labels, ds.name, job.payload["name"],
                                              t_feat, det_spec, seed, pp, train_subsample, opts)
                        running[fut] = ("detector", (0, job.payload["i_ext"], i_ds, i_s, i_pp, i_det))
                        n += 1
        return n

    try:
        launch()
        while running:
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in done:
                job = running.pop(fut)
                if isinstance(job, tuple):  # detector run
                    try:
                        row, dt = fut.result()
                    except Exception as e:
                        logger.error(f"FAILED: detector run {job[1]} ({type(e).__name__}: {e})")
                        raise
                    rows.append((job[1], row))
                    prog.add("detector", dt)
                    _log_run(logger, prog, row, dt)
                    continue
                rm.release(job.id)
                try:
                    res = fut.result()
                except Exception as e:
                    logger.error(f"FAILED: {job.label} on {job.device} ({type(e).__name__}: {e})")
                    raise
                if res.get("oom"):
                    if job.exclusive:
                        raise MemoryError(f"{job.label} does not fit on {job.device} even alone: {res['error']}")
                    logger.warning(f"Out of memory: {job.label} on {job.device}; retrying it alone on an idle device")
                    job.exclusive = True
                    pending.appendleft(job)
                    continue
                if res.get("peak_bytes"):
                    learned[job.key] = int((res["peak_bytes"] + CUDA_CONTEXT_BYTES) * 1.15)
                peak = f", peak {res['peak_bytes'] / GiB:.2f} GiB" if res.get("peak_bytes") else ""
                took = fmt_duration(res["seconds"])
                if job.kind == "extract":
                    feats = res["features"]
                    prog.add("load", res["load_s"])
                    for _ in feats:
                        prog.add("extract", (res["seconds"] - res["load_s"]) / max(1, len(feats)))
                    dim = next(iter(feats.values()))[0].shape[1] if feats else 0
                    n = submit_detectors(job, feats)
                    logger.info(f"Finished {job.label} on {job.device}: {len(feats)} dataset(s), {dim}-d, {took}{peak} "
                                f"-> {n} detector run(s) queued · {prog.timing()}")
                else:
                    row = res["row"]
                    rows.append((job.payload["order"], row))
                    prog.add("anomalib", res["seconds"])
                    logger.info(f"Finished {job.label} on {job.device}: {took}{peak}")
                    _log_run(logger, prog, row, res["seconds"])
            launch()
        if pending:  # pragma: no cover - only if no device can ever run a job
            raise RuntimeError(f"{len(pending)} job(s) could not be scheduled on {[d.name for d in devs]}")
    except BaseException:
        for f in running:
            f.cancel()
        acc_pool.shutdown(wait=False, cancel_futures=True)
        cpu_pool.shutdown(wait=False, cancel_futures=True)
        raise
    acc_pool.shutdown()
    cpu_pool.shutdown()
    return [r for _, r in sorted(rows, key=lambda kv: kv[0])]
