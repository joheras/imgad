"""Device discovery, memory checks and GPU memory admission for parallel benchmarks.

The parent process of a parallel benchmark never creates CUDA contexts: GPU memory is queried with
``nvidia-smi`` (or NVML through ``pynvml`` when installed), and ``torch.cuda`` is only used as a last
resort. Jobs reserve an *estimated* amount of memory on a device and only start when it fits both in
the budget (free memory at start × ``safety``, minus what running jobs reserved) and in the memory
that is free right now (other processes may have started meanwhile).

    imgad devices          # what the benchmark would see
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

GiB = 1024**3
CUDA_CONTEXT_BYTES = int(0.5 * GiB)  # per process, not reported by torch's allocator


@dataclass
class DeviceInfo:
    name: str                       # "cuda:0", "mps" or "cpu"
    kind: str                       # "cuda" | "mps" | "cpu"
    description: str = ""
    total_bytes: Optional[int] = None
    free_bytes: Optional[int] = None

    @property
    def is_gpu(self) -> bool:
        return self.kind in ("cuda", "mps")

    def __str__(self) -> str:
        mem = (f", {self.free_bytes / GiB:.1f}/{self.total_bytes / GiB:.1f} GiB free"
               if self.total_bytes else "")
        return f"{self.name} ({self.description}{mem})" if self.description else f"{self.name}{mem}"


# --------------------------------------------------------------------------------- discovery
def _visible_indices() -> Optional[List[int]]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is None:
        return None
    try:
        return [int(x) for x in env.split(",") if x.strip() != ""]
    except ValueError:  # UUIDs: let torch resolve them
        return []


def _query_nvidia_smi() -> Optional[List[Dict[str, Any]]]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        idx, name, total, free = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(idx), "name": name, "total": int(float(total)) * 2**20, "free": int(float(free)) * 2**20})
    return gpus


def _query_nvml() -> Optional[List[Dict[str, Any]]]:
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            gpus.append({"index": i, "name": name.decode() if isinstance(name, bytes) else name,
                         "total": int(mem.total), "free": int(mem.free)})
        return gpus
    except Exception:
        return None


def _query_torch() -> List[Dict[str, Any]]:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return []
    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)  # creates a context on device i
        gpus.append({"index": i, "name": torch.cuda.get_device_name(i), "total": int(total), "free": int(free)})
    return gpus


def query_cuda() -> List[DeviceInfo]:
    """CUDA devices as seen by this process (``cuda:i`` numbering honours ``CUDA_VISIBLE_DEVICES``)."""
    visible = _visible_indices()
    raw = _query_nvml() or _query_nvidia_smi()
    if raw is None or visible == []:
        raw, physical = _query_torch(), None
    else:
        physical = visible
    if physical is not None:
        by_idx = {g["index"]: g for g in raw}
        raw = [dict(by_idx[p], index=i) for i, p in enumerate(physical) if p in by_idx]
    return [DeviceInfo(f"cuda:{g['index']}", "cuda", g["name"], g["total"], g["free"]) for g in raw]


def _mps_available() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:  # pragma: no cover
        return False


def list_devices(include_cpu: bool = True) -> List[DeviceInfo]:
    """All usable devices: CUDA GPUs (with total/free memory), Apple MPS, and the CPU."""
    devs = query_cuda()
    if not devs and _mps_available():
        devs.append(DeviceInfo("mps", "mps", "Apple Metal"))
    if include_cpu:
        devs.append(DeviceInfo("cpu", "cpu", f"{os.cpu_count()} logical cores"))
    return devs


def resolve_devices(devices: Union[None, str, Sequence[str]]) -> List[DeviceInfo]:
    """``"auto"`` -> every GPU (or the CPU if there is none); ``"cpu"``; ``"cuda:0,cuda:1"`` or a list.
    Raises if a requested device does not exist."""
    available = {d.name: d for d in list_devices()}
    if devices is None or devices == "auto":
        gpus = [d for d in available.values() if d.is_gpu]
        return gpus or [available["cpu"]]
    names = devices.split(",") if isinstance(devices, str) else list(devices)
    out = []
    for n in (x.strip() for x in names):
        if n == "cuda":
            n = "cuda:0"
        if n not in available:
            raise ValueError(f"Device {n!r} is not available. Available: {', '.join(available)}")
        out.append(available[n])
    return out


# --------------------------------------------------------------------------------- estimates
def estimate_extractor_bytes(config: Dict[str, Any], default_bytes: int) -> int:
    """GPU memory needed to run a feature extractor: weights (size of the checkpoint files on the Hub or
    in the local HF cache) + activations for one batch + the CUDA context. Falls back to
    ``default_bytes`` when the model is not a Hub model or the size cannot be found."""
    weights = _hf_weights_bytes(config.get("model_name"))
    if weights is None:
        return default_bytes
    dtype = str(config.get("dtype") or "")
    if dtype in ("float16", "bfloat16", "half"):
        weights //= 2
    batch = int(config.get("batch_size") or 32)
    activations = int(0.25 * GiB) + batch * 24 * 2**20  # ViT-B/L at 224-336 px, no gradients
    return int(weights + activations + CUDA_CONTEXT_BYTES)


def _hf_weights_bytes(model_id: Optional[str]) -> Optional[int]:
    if not model_id or not isinstance(model_id, str) or "/" not in model_id or model_id.startswith(("hf-hub:", "/", ".")):
        return None

    def pick(files: Dict[str, int]) -> Optional[int]:
        st = sum(v for k, v in files.items() if k.endswith(".safetensors"))
        if st:
            return st
        pt = sum(v for k, v in files.items() if k.endswith(".bin") and "pytorch_model" in k)
        return pt or None

    try:  # local cache first: works offline and does not hit the network
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id == model_id and repo.repo_type == "model":
                files = {f.file_name: f.size_on_disk for rev in repo.revisions for f in rev.files}
                if (n := pick(files)) is not None:
                    return n
    except Exception:
        pass
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return None
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, files_metadata=True, timeout=10)
        return pick({s.rfilename: s.size for s in info.siblings if s.size})
    except Exception:
        return None


# --------------------------------------------------------------------------------- admission
@dataclass
class _Slot:
    info: DeviceInfo
    max_jobs: int
    budget: Optional[int]                       # bytes usable by the benchmark (None = unlimited)
    running: Dict[Any, int] = field(default_factory=dict)  # job id -> reserved bytes


class ResourceManager:
    """Decides on which device (if any) a job may start now.

    A job asks for ``est_bytes``. On a CUDA device it starts only if
    ``est_bytes <= min(budget - reserved, free_now - margin)``, where ``budget`` is the free memory
    when the benchmark started times ``safety``, ``reserved`` what running jobs asked for, and
    ``free_now`` a fresh query (refreshed at most every ``refresh_s`` seconds). Devices without
    memory information (CPU, MPS) only limit the number of concurrent jobs.
    """

    def __init__(self, devices: Sequence[DeviceInfo], max_jobs: Union[int, str] = "auto", safety: float = 0.9,
                 margin_bytes: int = int(0.3 * GiB), auto_cap: int = 4, refresh_s: float = 1.0, query=None):
        self.slots: Dict[str, _Slot] = {}
        for d in devices:
            if d.kind == "cuda":
                n = auto_cap if max_jobs == "auto" else int(max_jobs)
                budget = int((d.free_bytes or 0) * safety)
            else:  # CPU feature extraction / MPS: one job at a time unless asked otherwise
                n = 1 if max_jobs == "auto" else int(max_jobs)
                budget = None
            self.slots[d.name] = _Slot(d, max(1, n), budget)
        self.margin = margin_bytes
        self.refresh_s = refresh_s
        self._query = query or query_cuda
        self._free_cache: Dict[str, int] = {}
        self._free_t = 0.0

    def _free_now(self, name: str) -> Optional[int]:
        if self.slots[name].info.kind != "cuda":
            return None
        if time.time() - self._free_t > self.refresh_s:
            self._free_cache = {d.name: d.free_bytes for d in self._query() if d.free_bytes is not None}
            self._free_t = time.time()
        return self._free_cache.get(name)

    def available(self, name: str) -> Optional[int]:
        slot = self.slots[name]
        if slot.budget is None:
            return None
        avail = slot.budget - sum(slot.running.values())
        free = self._free_now(name)
        if free is not None:
            avail = min(avail, free - self.margin)
        return avail

    def fits_somewhere(self, est_bytes: int) -> bool:
        """Could the job ever start (on an idle device)?"""
        return any(s.budget is None or est_bytes <= s.budget for s in self.slots.values())

    def try_acquire(self, job_id, est_bytes: int, exclusive: bool = False) -> Optional[str]:
        """Device name for the job, or None if it must wait. ``exclusive`` waits for an idle device and
        skips the memory check (used to retry after an out-of-memory error, or for jobs larger than any
        budget)."""
        best, best_avail = None, None
        for name, slot in self.slots.items():
            if exclusive:
                if slot.running:
                    continue
                avail = float("inf") if slot.budget is None else (slot.info.total_bytes or 0)
            else:
                if len(slot.running) >= slot.max_jobs:
                    continue
                avail = self.available(name)
                if avail is None:
                    avail = float("inf")
                elif est_bytes > avail:
                    continue
            if best is None or avail > best_avail:
                best, best_avail = name, avail
        if best is not None:
            self.slots[best].running[job_id] = int(est_bytes)
            self._free_t = 0.0  # memory changes: refresh on the next query
        return best

    def release(self, job_id) -> None:
        for slot in self.slots.values():
            slot.running.pop(job_id, None)
        self._free_t = 0.0

    def n_running(self) -> int:
        return sum(len(s.running) for s in self.slots.values())

    def total_capacity(self) -> int:
        return sum(s.max_jobs for s in self.slots.values())

    def describe(self) -> List[str]:
        lines = []
        for name, s in self.slots.items():
            if s.budget is not None:
                lines.append(f"{s.info}: up to {s.max_jobs} concurrent GPU job(s), memory budget {s.budget / GiB:.1f} GiB")
            else:
                lines.append(f"{s.info}: up to {s.max_jobs} concurrent job(s)")
        return lines
