"""Feature-extractor interface, registry and on-disk feature cache."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np
from tqdm.auto import tqdm

from ..utils import ImageLike, as_list, batched, load_image

logger = logging.getLogger("imgad.features")
_EXTRACTORS: Dict[str, Type["FeatureExtractor"]] = {}


def register_extractor(name: str):
    """Class decorator to make an extractor buildable by name (e.g. from YAML configs)."""

    def deco(cls):
        _EXTRACTORS[name] = cls
        cls.registry_name = name
        return cls

    return deco


def list_extractors() -> List[str]:
    return sorted(_EXTRACTORS)


def build_extractor(config: Dict[str, Any] | str, token=None, **kwargs) -> "FeatureExtractor":
    """Build an extractor from ``{"type": "hf", "model_name": ...}`` or a bare HF model id. The optional
    key ``"intensity"`` sets the rule for images with more than 8 bits (see :mod:`imgad.imageio`).
    ``token``: Hugging Face access token for gated/private models (passed only to extractors that
    accept one; see :mod:`imgad.hub`)."""
    if isinstance(config, str):
        config = {"type": "hf", "model_name": config}
    config = {**config, **kwargs}
    kind = config.pop("type", "hf")
    if kind not in _EXTRACTORS:
        raise KeyError(f"Unknown extractor type {kind!r}. Available: {list_extractors()}")
    cls = _EXTRACTORS[kind]
    intensity = config.pop("intensity", None)
    if token is not None and "token" in inspect.signature(cls.__init__).parameters:
        config["token"] = token
    ext = cls(**config)
    if intensity is not None:
        ext.set_intensity(intensity)
    return ext


class FeatureExtractor(ABC):
    """Maps a list of images to an ``(n_images, dim)`` float32 array.

    Subclasses implement :meth:`_extract_batch` (list of PIL RGB images -> np.ndarray)
    and :meth:`get_config` (JSON-serialisable kwargs to rebuild the extractor).
    """

    registry_name: str = "base"
    batch_size: int = 32
    intensity: str = "auto"  # how images with more than 8 bits are mapped to 8 bits (imgad.imageio)

    @abstractmethod
    def _extract_batch(self, images: List) -> np.ndarray:
        ...

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        ...

    @property
    def identifier(self) -> str:
        """Stable id used for caching (type + config)."""
        cfg = {"type": self.registry_name, **self.get_config()}
        cfg.pop("device", None)
        cfg.pop("batch_size", None)
        if self.intensity != "auto":
            cfg["intensity"] = self.intensity
        return json.dumps(cfg, sort_keys=True, default=str)

    def set_intensity(self, intensity) -> "FeatureExtractor":
        """Rule mapping images with more than 8 bits to 8 bits: ``"auto"``, ``"minmax"``, ``"dicom"``,
        ``"percentile:1,99"`` or ``"window:lo,hi"`` (see :mod:`imgad.imageio`)."""
        from ..imageio import intensity_str

        self.intensity = intensity_str(intensity)
        return self

    def full_config(self) -> Dict[str, Any]:
        """``{"type": ..., **get_config()}`` plus the intensity rule when it is not the default; this is what
        saved pipelines store and :func:`build_extractor` accepts."""
        cfg = {"type": self.registry_name, **self.get_config()}
        if self.intensity != "auto":
            cfg["intensity"] = self.intensity
        return cfg

    def extract(
        self,
        images: Sequence[ImageLike] | ImageLike,
        cache_dir: Optional[str | os.PathLike] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Extract features. If ``cache_dir`` is set and all inputs are file paths,
        features are cached per image (keyed by extractor id + path + mtime + size)."""
        images = as_list(images)
        if cache_dir is not None and images and all(isinstance(i, (str, os.PathLike)) for i in images):
            return self._extract_cached(images, Path(cache_dir), show_progress)
        return self._extract_all(images, show_progress)

    def _extract_all(self, images: List, show_progress: bool) -> np.ndarray:
        feats = []
        it = batched(images, self.batch_size)
        if show_progress and len(images) > self.batch_size:
            it = tqdm(it, total=(len(images) + self.batch_size - 1) // self.batch_size, desc="features")
        for batch in it:
            pil = [load_image(x, self.intensity) for x in batch]
            feats.append(np.asarray(self._extract_batch(pil), dtype=np.float32))
        if not feats:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(feats, axis=0)

    def _cache_dir(self, cache_dir: Path) -> Path:
        return Path(cache_dir) / hashlib.sha1(self.identifier.encode()).hexdigest()[:16]

    def _cache_key(self, d: Path, p) -> Path:
        from ..imageio import split_ref

        path, axis, index = split_ref(p)  # slice references: stat the file, keep the slice in the key
        st = os.stat(path)
        ref = f"::{axis}={index}" if axis is not None else ""
        h = hashlib.sha1(f"{os.path.abspath(path)}{ref}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()
        return d / f"{h}.npy"

    def cached_count(self, images, cache_dir: Optional[str | os.PathLike]) -> int:
        """How many of ``images`` already have cached features (0 without a cache or for in-memory images)."""
        if cache_dir is None:
            return 0
        d = self._cache_dir(cache_dir)
        if not d.exists():
            return 0
        return sum(self._cache_key(d, p).exists() for p in as_list(images) if isinstance(p, (str, os.PathLike)))

    def _extract_cached(self, paths: List, cache_dir: Path, show_progress: bool) -> np.ndarray:
        d = self._cache_dir(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "extractor.json").write_text(self.identifier)
        keys = [self._cache_key(d, p) for p in paths]
        missing = [i for i, k in enumerate(keys) if not k.exists()]
        logger.debug(f"feature cache {d}: {len(paths) - len(missing)}/{len(paths)} cached, extracting {len(missing)}")
        if missing:
            new = self._extract_all([paths[i] for i in missing], show_progress)
            for i, f in zip(missing, new):
                np.save(keys[i], f)
        return np.stack([np.load(k) for k in keys]).astype(np.float32)

    def __call__(self, images, **kw) -> np.ndarray:
        return self.extract(images, **kw)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_config().items())
        return f"{type(self).__name__}({args})"
