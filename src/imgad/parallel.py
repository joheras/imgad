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
* If a worker process dies (segfault, killed for lack of host memory...), the whole CPU pool breaks and
  every run in it fails with it. At most ``2 × n_jobs`` detector runs are handed to the pool at a time, so
  only those are affected; each is retried alone in a separate one-process pool, and only a run that kills
  its process even alone is recorded as failed.
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
from concurrent.futures.process import BrokenProcessPool
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
                 intensity: str = "auto", detectors=()) -> Dict[str, Any]:
    """Load the extractor on ``device`` and extract train/test features for every dataset (and the prompt
    embeddings of the zero-shot ``detectors``). ``token`` (gated Hub models) travels only through the process
    pipe; it is never part of the stored config."""
    from .evaluation import text_bank

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
            text = text_bank(ext, detectors)
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
    return {"features": feats, "text": text, "peak_bytes": peak.peak, "load_s": t_load, "seconds": time.time() - t0}


def _detector_job(Xtr_all, Xte, labels, ds_name, ext_name, t_feat, det_spec, seed, pp, train_subsample, opts,
                  k_shot=None, text=None, on_error="continue"):
    from .evaluation import safe_detector_run

    return safe_detector_run(Xtr_all, Xte, labels, ds_name, ext_name, t_feat, det_spec, seed, pp, train_subsample,
                             opts, k_shot, text, on_error)


def _anomalib_job(spec, ds, seed: int, train_subsample, opts, device: str, k_shot=None) -> Dict[str, Any]:
    from .evaluation import _benchmark_anomalib

    spec = {"name": spec} if isinstance(spec, str) else dict(spec)
    spec["device"] = device
    t0 = time.time()
    try:
        with _cuda_peak(device) as peak:
            row = _benchmark_anomalib(spec, ds, False, seed, train_subsample, k_shot=k_shot, **opts)
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
    crashes: int = 0                # times its worker process died (it is retried once, alone)


def _cpu_workers(n_jobs: int) -> int:
    n = os.cpu_count() or 1
    if n_jobs is None or n_jobs == 0:
        return 1
    return max(1, n + 1 + n_jobs) if n_jobs < 0 else int(n_jobs)


def run_parallel(extractors, detectors, datasets, preprocessings, anomalib_models, seeds, train_subsample, opts,
                 cache_dir, n_jobs, devices, gpu_jobs, gpu_mem_gb, prog, token=None, shots=(None,),
                 on_error: str = "continue") -> List[Dict[str, Any]]:
    from .anomalib_backend import build_anomalib_pipeline  # noqa: F401  (fail early if misconfigured)
    from .evaluation import _anomalib_name, _log_run, _on_failure, _skip_reason, extractor_label, failed_row
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
                for i_k, k in enumerate(shots):
                    if k == 0:  # zero-shot: Anomalib models need training images
                        prog.skip("anomalib")
                        continue
                    name = _anomalib_name(spec)
                    shot = f", {k}-shot" if k is not None else ""
                    pending.append(_Job(next(ids), "anomalib", ("anomalib", name), f"{name} on {ds.name} (seed {seed}{shot})",
                                        default_est, {"spec": spec, "ds": ds, "seed": seed, "k": k,
                                                      "order": (1, i_m, i_ds, i_s, i_k)}))
    learned: Dict[Tuple, int] = {}

    ctx = mp.get_context("spawn")
    sizes = {"acc": (rm.total_capacity(), acc_threads), "cpu": (n_cpu, 1), "solo": (1, 1)}
    pools: Dict[str, ProcessPoolExecutor] = {}

    def pool(kind: str, renew: bool = False) -> ProcessPoolExecutor:
        """The worker pool of ``kind``; a new one if ``renew`` or if a worker process of the old one died
        (segfault, killed for lack of host memory...), which breaks the whole pool."""
        old = pools.get(kind)
        if old is None or renew or getattr(old, "_broken", False):
            if old is not None:
                old.shutdown(wait=False, cancel_futures=True)
            n, threads = sizes[kind]
            pools[kind] = ProcessPoolExecutor(n, mp_context=ctx, initializer=_init_worker, initargs=(threads,))
        return pools[kind]

    def submit(kind: str, fn, *args) -> Future:
        try:
            return pool(kind).submit(fn, *args)
        except BrokenProcessPool:
            return pool(kind, renew=True).submit(fn, *args)

    running: Dict[Future, Any] = {}
    # detector runs wait here and are handed to the CPU pool a few at a time, so that a dead worker process only
    # takes down those few; the runs it took down are then retried one by one in the "solo" pool
    det_queue: deque = deque()
    solo_queue: deque = deque()
    det_in_flight = {"cpu": 0, "solo": 0}
    max_in_flight = 2 * n_cpu

    def pump_detectors() -> None:
        while solo_queue and det_in_flight["solo"] == 0:
            order, args = solo_queue.popleft()
            running[submit("solo", _detector_job, *args)] = ("detector", order, args, "solo")
            det_in_flight["solo"] += 1
        while det_queue and det_in_flight["cpu"] < max_in_flight:
            order, args = det_queue.popleft()
            running[submit("cpu", _detector_job, *args)] = ("detector", order, args, "cpu")
            det_in_flight["cpu"] += 1

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
                    fut = submit("acc", _extract_job, job.payload["cls"], job.payload["cfg"], dev, datasets, cache_dir,
                                 token, job.payload["intensity"], detectors)
                else:
                    p = job.payload
                    fut = submit("acc", _anomalib_job, p["spec"], p["ds"], p["seed"], train_subsample, opts, dev, p["k"])
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

    def submit_detectors(job: _Job, feats: Dict[str, Any], text) -> int:
        n = 0
        for i_ds, ds in enumerate(datasets):
            Xtr, Xte, t_feat = feats[ds.name]
            for i_s, seed in enumerate(seeds):
                for i_k, k in enumerate(shots):
                    for i_pp, pp in enumerate(preprocessings):
                        for i_det, det_spec in enumerate(detectors):
                            if _skip_reason(det_spec, pp, k, text):
                                prog.skip("detector")
                                continue
                            args = (Xtr, Xte, ds.test_labels, ds.name, job.payload["name"], t_feat, det_spec, seed, pp,
                                    train_subsample, opts, k, text, on_error)
                            det_queue.append(((0, job.payload["i_ext"], i_ds, i_s, i_k, i_pp, i_det), args))
                            n += 1
        pump_detectors()
        return n

    runs_per_ds = len(seeds) * len(shots) * len(preprocessings) * len(detectors)

    def fail_job(job: _Job, e: Exception) -> None:
        """An accelerator job failed: record failed rows and drop the runs that depended on it."""
        _on_failure(e, on_error, f"{job.label} on {job.device}")
        if job.kind == "extract":
            for i_ds, ds in enumerate(datasets):
                rows.append(((0, job.payload["i_ext"], i_ds, -1),
                             failed_row(ds.name, job.payload["name"], "-", "-", None, None, "extract", e)))
            prog.skip("load")
            prog.skip("extract", len(datasets))
            prog.skip("detector", runs_per_ds * len(datasets))
        else:
            p = job.payload
            row = failed_row(p["ds"].name, "anomalib", "-", _anomalib_name(p["spec"]), p["seed"], p["k"], "anomalib", e)
            rows.append((p["order"], row))
            prog.add("anomalib", time.time() - job.started)
            _log_run(logger, prog, row, time.time() - job.started)

    try:
        launch()
        while running:
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in done:
                job = running.pop(fut)
                if isinstance(job, tuple):  # detector run
                    _, order, args, kind = job
                    det_in_flight[kind] -= 1
                    try:
                        row, dt = fut.result()
                    except BrokenProcessPool as e:
                        if kind == "cpu" and on_error == "continue":
                            # maybe another run killed the worker: retry it alone, where a crash can only be its own
                            logger.warning(f"A worker process died during detector run {order}; retrying it alone")
                            solo_queue.append((order, args))
                            pump_detectors()
                            continue
                        _on_failure(e, on_error, f"detector run {order} (its worker process died)")
                        row, dt = failed_row(args[3], args[4], "-", str(args[6]), args[7], args[11], "detector", e), 0.0
                    except Exception as e:  # the worker itself failed (not the detector, caught there)
                        _on_failure(e, on_error, f"detector run {order}")
                        row, dt = failed_row(args[3], args[4], "-", str(args[6]), args[7], args[11], "detector", e), 0.0
                    rows.append((order, row))
                    prog.add("detector", dt)
                    _log_run(logger, prog, row, dt)
                    pump_detectors()
                    continue
                rm.release(job.id)
                try:
                    res = fut.result()
                except BrokenProcessPool as e:
                    if job.crashes == 0 and on_error == "continue":
                        # the pool breaks for every job that was running in it: retry each once, alone on its
                        # device, so that only the job that really crashes the process is recorded as failed
                        job.crashes, job.exclusive = 1, True
                        logger.warning(f"A worker process died while running {job.label} on {job.device}; "
                                       "retrying it alone")
                        pending.appendleft(job)
                        continue
                    fail_job(job, RuntimeError(f"the worker process died (segfault or killed, e.g. out of host "
                                               f"memory): {e}"))
                    continue
                except Exception as e:
                    fail_job(job, e)
                    continue
                try:
                    if res.get("oom") and job.exclusive:
                        raise MemoryError(f"{job.label} does not fit on {job.device} even alone: {res['error']}")
                except Exception as e:
                    fail_job(job, e)
                    continue
                if res.get("oom"):
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
                    n = submit_detectors(job, feats, res.get("text"))
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
        for pl in pools.values():
            pl.shutdown(wait=False, cancel_futures=True)
        raise
    for pl in pools.values():
        pl.shutdown()
    return [r for _, r in sorted(rows, key=lambda kv: kv[0])]
