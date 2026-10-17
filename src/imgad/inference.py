"""Inference with an already-trained pipeline.

    from imgad import AnomalyPredictor

    predictor = AnomalyPredictor.load("bottle.joblib")          # device auto (cuda/mps/cpu)
    r = predictor.predict_one("img.png")
    print(r.is_anomaly, r.score, r.normal_percentile)

    results = predictor.predict("folder/")                     # folder, list of paths, PIL images...
    predictor.save_results(results, "predictions.csv")         # .csv or .json

    # anomalib models (PatchCore, PaDiM...): pixel-level anomaly maps
    results = predictor.predict("folder/", return_maps=True)   # r.anomaly_map at original image size
    predictor.save_maps(results, "maps/")                      # heatmap overlays + binary masks
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

from .data import IMG_EXTENSIONS, list_images
from .pipeline import load_pipeline
from .utils import ImageLike, batched, is_single_array


@dataclass
class Prediction:
    """Result for one image.

    score:             raw anomaly score (higher = more anomalous; scale depends on the detector)
    is_anomaly:        score > threshold
    threshold:         decision threshold used
    normal_percentile: fraction of normal training images with a lower score (0-1). Comparable
                       across detectors: 0.5 ~ typical normal image, 0.999 ~ extremely unusual.
    anomaly_map:       (``return_maps=True``, anomalib models) float map, higher = more anomalous
    anomalous_area:    fraction of pixels above the pixel threshold (only with maps)
    heatmap / mask:    files written by :meth:`AnomalyPredictor.save_maps`
    """

    source: str
    score: float
    is_anomaly: bool
    threshold: float
    normal_percentile: float
    anomaly_map: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    anomalous_area: Optional[float] = None
    heatmap: Optional[str] = None
    mask: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable fields (the map array is left out; unset optional fields are omitted)."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "anomaly_map" or (f.default is None and v is None):
                continue
            out[f.name] = v
        return out


class AnomalyPredictor:
    """Thin inference wrapper around a trained :class:`AnomalyDetectionPipeline` or
    :class:`~imgad.anomalib_backend.AnomalibPipeline`."""

    def __init__(self, pipeline, threshold: Optional[float] = None):
        if pipeline.threshold_ is None:
            raise ValueError("The pipeline is not trained (no threshold). Call pipeline.fit(...) first.")
        self.pipeline = pipeline
        self.threshold = float(threshold) if threshold is not None else float(pipeline.threshold_)

    @classmethod
    def load(
        cls,
        path: Union[str, os.PathLike],
        device: Optional[str] = None,
        threshold: Optional[float] = None,
        batch_size: Optional[int] = None,
        token=None,
    ) -> "AnomalyPredictor":
        """``token``: Hugging Face access token for gated/private extractors (never stored)."""
        return cls(load_pipeline(path, device=device, batch_size=batch_size, token=token), threshold)

    # ------------------------------------------------------------------ core
    def predict(
        self,
        images: Union[ImageLike, Sequence[ImageLike]],
        recursive: bool = True,
        show_progress: bool = True,
        return_maps: bool = False,
    ) -> List[Prediction]:
        """Predict on a path to an image or folder, a PIL image / numpy array, or a list of any of these.
        ``return_maps=True`` (anomalib models) adds each image's anomaly map at its original size."""
        items, names = self._expand(images, recursive)
        if not items:
            return []
        return self._run(items, names, return_maps, show_progress)

    def predict_one(self, image: ImageLike, return_maps: bool = False) -> Prediction:
        if isinstance(image, (str, os.PathLike)) and Path(image).is_dir():
            raise ValueError("predict_one expects a single image; use predict() for folders")
        return self.predict([image], show_progress=False, return_maps=return_maps)[0]

    def predict_iter(
        self,
        images: Union[ImageLike, Sequence[ImageLike]],
        chunk_size: int = 256,
        recursive: bool = True,
        return_maps: bool = False,
    ) -> Iterator[Prediction]:
        """Stream predictions chunk by chunk (constant memory for very large folders)."""
        items, names = self._expand(images, recursive)
        for chunk in batched(list(range(len(items))), chunk_size):
            yield from self._run([items[i] for i in chunk], [names[i] for i in chunk], return_maps, False)

    __call__ = predict

    @property
    def supports_maps(self) -> bool:
        return bool(getattr(self.pipeline, "supports_maps", False))

    # ------------------------------------------------------------------ helpers
    def _run(self, items, names, return_maps: bool, show_progress: bool) -> List[Prediction]:
        if not return_maps:
            return self._to_predictions(names, self.pipeline.score(items, show_progress=show_progress))
        if not self.supports_maps:
            raise ValueError("This model gives image-level scores only; anomaly maps need an anomalib model")
        scores, maps = self.pipeline.score_and_maps(items, size="original", show_progress=show_progress)
        return self._to_predictions(names, scores, maps)

    def _to_predictions(self, names: List[str], scores: np.ndarray, maps=None) -> List[Prediction]:
        pct = self.pipeline.normal_percentile(scores)
        out = [
            Prediction(n, float(s), bool(s > self.threshold), self.threshold, float(p))
            for n, s, p in zip(names, scores, pct)
        ]
        if maps is not None:
            for r, m in zip(out, maps):
                r.anomaly_map = m
                r.anomalous_area = float((m > self.pipeline.pixel_threshold_).mean())
        return out

    def save_maps(
        self, results: Iterable[Prediction], out_dir: Union[str, os.PathLike], alpha: float = 0.5
    ) -> List[Prediction]:
        """Write, for each prediction with a map, ``<name>_heatmap.png`` (overlay on the image when the
        source is a file) and ``<name>_mask.png`` (pixels above the pixel threshold). Fills
        ``r.heatmap`` / ``r.mask`` so :meth:`save_results` lists the files."""
        from .maps import colorize, overlay

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        low, thr = self.pipeline.map_reference_, self.pipeline.pixel_threshold_
        used = set()
        results = list(results)
        for i, r in enumerate(results):
            if r.anomaly_map is None:
                continue
            from .imageio import split_ref

            path, axis, index = split_ref(r.source)
            src = Path(path)
            # parent folder in the name: MVTec-like layouts reuse file names (test/<type>/000.png)
            stem = f"{src.parent.name}_{src.stem}" if src.is_file() else f"image_{i}"
            if axis is not None and src.is_file():
                stem = f"{src.parent.name}_{src.name.split('.')[0]}_{axis}{index}"
            if stem in used:
                stem = f"{stem}_{i}"
            used.add(stem)
            if src.is_file():
                heat = overlay(r.source, r.anomaly_map, low, thr, alpha, getattr(self.pipeline, "intensity", None)
                               or getattr(getattr(self.pipeline, "extractor", None), "intensity", None))
            else:
                heat = Image.fromarray(colorize(r.anomaly_map, low, thr))
            r.heatmap = str(out_dir / f"{stem}_heatmap.png")
            r.mask = str(out_dir / f"{stem}_mask.png")
            heat.save(r.heatmap)
            Image.fromarray((r.anomaly_map > thr).astype(np.uint8) * 255).save(r.mask)
        return results

    @staticmethod
    def _expand(images, recursive: bool):
        if isinstance(images, (str, os.PathLike, Image.Image)) or is_single_array(images):
            images = [images]
        items, names = [], []
        for i, img in enumerate(images):
            if isinstance(img, (str, os.PathLike)):
                from .imageio import exists, is_nifti, split_ref

                path, axis, _ = split_ref(img)
                p = Path(path)
                if axis is None and p.is_dir():
                    found = list_images(p, recursive=recursive)
                    items += found
                    names += found
                    continue
                if not exists(img):
                    raise FileNotFoundError(p)
                if axis is None and is_nifti(path):
                    raise ValueError(f"{p} is a volume: use imgad.data.volume_slices() to get its slices")
                if axis is None and p.suffix.lower() not in IMG_EXTENSIONS:
                    raise ValueError(f"Not an image file: {p}")
                items.append(str(img))
                names.append(str(img))
            else:
                items.append(img)
                names.append(getattr(img, "filename", None) or f"<image {i}>")
        return items, names

    @staticmethod
    def save_results(results: Iterable[Prediction], path: Union[str, os.PathLike]) -> None:
        """Write predictions to ``.csv`` or ``.json`` (chosen by extension)."""
        rows = [r.to_dict() for r in results]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(rows, indent=2))
        else:
            keys = list(dict.fromkeys(k for r in rows for k in r)) or list(Prediction.__dataclass_fields__)[:5]
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)

    def info(self) -> Dict[str, Any]:
        return {**self.pipeline.info(), "threshold": self.threshold}

    def __repr__(self) -> str:
        return f"AnomalyPredictor(threshold={self.threshold:.6g}, pipeline={self.pipeline!r})"
