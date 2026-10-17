"""Anomalib backend: end-to-end models (PatchCore, PaDiM, EfficientAD, FastFlow, STFPM, ...) with the
same API as :class:`~imgad.pipeline.AnomalyDetectionPipeline`, plus pixel-level anomaly maps.

    pip install -e ".[anomalib]"

    from imgad import AnomalibPipeline
    pipe = AnomalibPipeline("patchcore", backbone="wide_resnet50_2", image_size=256)
    pipe.fit(ds.train_paths)                          # normal images only
    scores, maps = pipe.score_and_maps(ds.test_paths) # maps at model resolution
    pipe.save("bottle_patchcore.joblib")              # -> AnomalyPredictor.load(...) as usual

Anomalib is used for the model definitions and the training loop only. Its post-processing
(normalisation / threshold learnt on a labelled validation set) is disabled: scores and maps are
the raw model outputs, and the threshold follows imgad's one-class convention (a quantile of the
scores of held-out normal images, see ``calibration_split``).
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import inspect
import logging
import os
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .maps import MaskLike, load_mask
from .pipeline import ThresholdMixin
from .progress import step
from .utils import ImageLike, as_list, load_image, resolve_device

_FORMAT_VERSION = 1
MapSize = Union[None, str, Tuple[int, int]]


def _require_anomalib():
    try:
        import anomalib  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError('AnomalibPipeline needs anomalib: pip install -e ".[anomalib]"') from e


def list_anomalib_models() -> List[str]:
    """Anomalib model names (case/underscores are ignored when building: ``efficient_ad`` ==
    ``efficientad``). Only one-class models can be trained here. Empty if anomalib is missing."""
    try:
        return sorted(c.__name__.lower() for c in _model_classes())
    except ImportError:
        return []


def _model_classes():
    with _quiet():
        import anomalib.models as am
        from anomalib.models import AnomalibModule
    return [
        obj
        for obj in vars(am).values()
        if isinstance(obj, type) and issubclass(obj, AnomalibModule) and obj is not AnomalibModule
    ]


def _model_class(name: str):
    key = name.replace("_", "").replace("-", "").lower()
    for cls in _model_classes():
        if cls.__name__.lower() == key:
            return cls
    raise KeyError(f"Unknown anomalib model {name!r}. Available: {list_anomalib_models()}")


@contextlib.contextmanager
def _quiet():
    """Silence Lightning/anomalib banners and deprecation warnings during training."""
    names = (
        "lightning", "lightning.pytorch", "lightning.fabric", "pytorch_lightning", "anomalib",
        "lightning.pytorch.utilities.rank_zero", "lightning.fabric.utilities.rank_zero",
        "lightning_fabric", "lightning_fabric.utilities.rank_zero",
    )
    loggers = [logging.getLogger(n) for n in names]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        for lg, lv in zip(loggers, levels):
            lg.setLevel(lv)


def _accelerator(device: str) -> Tuple[str, Any]:
    if device.startswith("cuda"):
        _, _, idx = device.partition(":")
        return "gpu", [int(idx)] if idx else 1
    return ("mps", 1) if device == "mps" else ("cpu", 1)


class _Images(Dataset):
    """Loads and resizes images to the model input size; returns (CHW float tensor in [0,1], (h, w))."""

    def __init__(self, images: Sequence[ImageLike], size: Tuple[int, int], as_item: bool = False, intensity="auto"):
        self.images, self.size, self.as_item, self.intensity = list(images), size, as_item, intensity

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        from PIL import Image

        img = load_image(self.images[i], self.intensity)
        orig = (img.height, img.width)
        img = img.resize((self.size[1], self.size[0]), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
        if self.as_item:
            from anomalib.data import ImageItem

            src = self.images[i]
            return ImageItem(image=x, gt_label=torch.tensor(0), image_path=str(src) if isinstance(src, (str, os.PathLike)) else f"<image {i}>")
        return x, orig


def _collate_infer(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


class AnomalibPipeline(ThresholdMixin):
    """Anomalib model wrapped with imgad's API (``fit`` / ``score`` / ``predict`` / ``save`` / ``load``).

    Parameters
    ----------
    model:             anomalib model name: ``"patchcore"``, ``"padim"``, ``"efficient_ad"``,
                       ``"fastflow"``, ``"stfpm"``, ``"reverse_distillation"``, ... (one-class models)
    image_size:        model input size (int or ``(h, w)``); images are resized (no crop), so maps
                       stay aligned with the full image
    calibration_split: fraction of the normal training images held out (not used for training)
                       to set the image/pixel thresholds and the ``normal_percentile`` reference.
                       ``0`` -> use in-sample training scores (optimistic, e.g. for PatchCore).
    threshold_quantile: quantile of the held-out normal scores used as image threshold (0.95 ->
                       ~5% false alarms). The pixel threshold is the same quantile of the
                       per-image maximum of the held-out normal maps.
    batch_size / train_batch_size: inference / training batch sizes (``efficient_ad`` trains with 1)
    max_epochs:        epochs for trainable models (ignored by PatchCore/PaDiM, which use 1)
    trainer_kwargs:    extra ``lightning.Trainer`` arguments (e.g. ``{"max_steps": 1000}``)
    intensity:         rule mapping images with more than 8 bits (16-bit, DICOM...) to 8 bits; see
                       :mod:`imgad.imageio`
    **model_kwargs:    passed to the anomalib model, e.g. ``backbone="resnet18"``,
                       ``layers=["layer2", "layer3"]``, ``coreset_sampling_ratio=0.01``
    """

    supports_maps = True

    def __init__(
        self,
        model: str = "patchcore",
        image_size: Union[int, Tuple[int, int]] = 256,
        calibration_split: float = 0.2,
        threshold_quantile: float = 0.95,
        batch_size: int = 32,
        train_batch_size: Optional[int] = None,
        num_workers: int = 0,
        device: Optional[str] = None,
        max_epochs: Optional[int] = None,
        seed: Optional[int] = 0,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        intensity: str = "auto",
        **model_kwargs,
    ):
        _require_anomalib()
        self.model_name = model
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        self.calibration_split = calibration_split
        self.threshold_quantile = threshold_quantile
        self.batch_size = batch_size
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        self.device = device
        self.max_epochs = max_epochs
        self.seed = seed
        self.trainer_kwargs = dict(trainer_kwargs or {})
        from .imageio import intensity_str

        self.intensity = intensity_str(intensity)
        self.model_kwargs = model_kwargs
        self.threshold_: Optional[float] = None
        self.train_scores_: Optional[np.ndarray] = None
        self.pixel_threshold_: Optional[float] = None
        self.map_reference_: Optional[float] = None
        self.metadata: Dict[str, Any] = {}
        self.model = self._build_model()

    # ----------------------------------------------------------------- model
    def _build_model(self, pretrained: bool = True):
        from anomalib import LearningType

        cls = _model_class(self.model_name)
        kw = dict(self.model_kwargs)
        if not pretrained and "pre_trained" in inspect.signature(cls.__init__).parameters:
            kw["pre_trained"] = False  # weights come from the saved state_dict
        try:
            pre = cls.configure_pre_processor(image_size=self.image_size)
        except TypeError:
            pre = True
        with _quiet():
            model = cls(pre_processor=pre, post_processor=False, evaluator=False, visualizer=False, **kw)
        if model.learning_type != LearningType.ONE_CLASS:
            raise NotImplementedError(
                f"{cls.__name__} is a {model.learning_type.value} model; only one-class anomalib models are supported"
            )
        return model

    @property
    def _device(self) -> str:
        return resolve_device(self.device)

    def _train(self, images: List[ImageLike], show_progress: bool) -> None:
        from anomalib.engine import Engine

        bs = self.train_batch_size or (1 if self.model_name.replace("_", "").lower() == "efficientad" else self.batch_size)
        ds = _Images(images, self.image_size, as_item=True, intensity=self.intensity)
        from anomalib.data import ImageBatch

        loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=self.num_workers, collate_fn=ImageBatch.collate)
        accelerator, devices = _accelerator(self._device)
        args = dict(
            logger=False,
            enable_progress_bar=show_progress,
            enable_model_summary=False,
            accelerator=accelerator,
            devices=devices,
        )
        if self.max_epochs is not None:
            args["max_epochs"] = self.max_epochs
        args.update(self.trainer_kwargs)
        with tempfile.TemporaryDirectory() as tmp, _quiet():
            Engine(default_root_dir=tmp, **args).fit(self.model, train_dataloaders=loader)
        with contextlib.suppress(Exception):
            self.model.trainer = None  # drop the reference to the Lightning trainer
        self.model.eval()

    # ----------------------------------------------------------------- inference
    @torch.no_grad()
    def _infer(self, images, with_maps: bool, map_size: MapSize = None, show_progress: bool = True):
        items = as_list(images)
        if not items:
            return np.zeros(0), []
        dev = self._device
        model = self.model.to(dev).eval()
        loader = DataLoader(
            _Images(items, self.image_size, intensity=self.intensity), batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=_collate_infer
        )
        if show_progress and len(items) > self.batch_size:
            loader = tqdm(loader, desc=f"{self.model_name}")
        scores, maps = [], []
        for x, sizes in loader:
            x = x.to(dev)
            if model.pre_processor is not None:
                x = model.pre_processor(x)
            out = model.model(x)
            amap = getattr(out, "anomaly_map", None)
            if amap is not None:
                amap = amap.reshape(len(x), 1, *amap.shape[-2:]).float()
            s = getattr(out, "pred_score", None)
            if s is None:
                s = amap.flatten(1).max(dim=1).values
            scores.append(s.reshape(len(x), -1)[:, 0].float().cpu().numpy())
            if not with_maps:
                continue
            if amap is None:
                raise RuntimeError(f"{self.model_name} does not produce anomaly maps")
            if map_size == "original":
                maps += [
                    torch.nn.functional.interpolate(amap[i : i + 1], size=hw, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                    for i, hw in enumerate(sizes)
                ]
            else:
                if map_size is not None:
                    amap = torch.nn.functional.interpolate(amap, size=tuple(map_size), mode="bilinear", align_corners=False)
                maps += list(amap[:, 0].cpu().numpy())
        return np.concatenate(scores).astype(np.float64), maps

    # ----------------------------------------------------------------- public API
    def fit(self, images, show_progress: bool = True) -> "AnomalibPipeline":
        """Train on **normal** images only, then set thresholds on the held-out calibration split."""
        items = as_list(images)
        n_cal = int(round(len(items) * self.calibration_split)) if self.calibration_split else 0
        if n_cal and len(items) - n_cal < 1:
            raise ValueError("Not enough images for calibration_split; lower it or set it to 0")
        perm = np.random.default_rng(self.seed).permutation(len(items))
        train = [items[i] for i in perm[n_cal:]]
        cal = [items[i] for i in perm[:n_cal]] or train
        if self.seed is not None:  # before building: also fixes randomly initialised layers
            from lightning.pytorch import seed_everything

            seed_everything(self.seed, verbose=False)
        with step(f"  building {self.model_name} ({', '.join(f'{k}={v}' for k, v in self.model_kwargs.items()) or 'defaults'})"):
            self.model = self._build_model()
        epochs = f", max_epochs={self.max_epochs}" if self.max_epochs else ""
        with step(f"  training {self.model_name} on {len(train)} normal images (device {self._device}{epochs})", logging.INFO):
            self._train(train, show_progress)
        src = f"{len(cal)} held-out" if n_cal else f"{len(cal)} training"
        with step(f"  calibrating image/pixel thresholds on {src} normal images", logging.INFO):
            self._calibrate_from_normals(cal, show_progress)
        self.metadata.update(
            n_train=len(train),
            n_calibration=len(cal) if n_cal else 0,
            trained_at=_dt.datetime.now().isoformat(timespec="seconds"),
            threshold_source=(
                f"quantile {self.threshold_quantile} of {'held-out' if n_cal else 'in-sample'} normal scores"
            ),
        )
        return self

    def _calibrate_from_normals(self, images, show_progress: bool) -> None:
        scores, maps = self._infer(images, with_maps=True, show_progress=show_progress)
        self.train_scores_ = scores
        self.threshold_ = float(np.quantile(scores, self.threshold_quantile))
        self.pixel_threshold_ = float(np.quantile([m.max() for m in maps], self.threshold_quantile))
        self.map_reference_ = float(np.median([np.median(m) for m in maps]))

    def score(self, images, show_progress: bool = True) -> np.ndarray:
        """Image-level anomaly scores (higher = more anomalous)."""
        self._check_fitted()
        return self._infer(images, with_maps=False, show_progress=show_progress)[0]

    def predict(self, images, threshold: Optional[float] = None, show_progress: bool = True) -> np.ndarray:
        t = self.threshold_ if threshold is None else threshold
        return (self.score(images, show_progress=show_progress) > t).astype(int)

    def score_and_maps(self, images, size: MapSize = None, show_progress: bool = True):
        """``(scores, maps)``. ``size``: ``None`` = model resolution, ``"original"`` = each image's own
        size, or ``(h, w)``. Maps are raw (unnormalised) float arrays; higher = more anomalous."""
        self._check_fitted()
        return self._infer(images, with_maps=True, map_size=size, show_progress=show_progress)

    def anomaly_maps(self, images, size: MapSize = None, show_progress: bool = True) -> List[np.ndarray]:
        return self.score_and_maps(images, size, show_progress)[1]

    def predict_masks(self, maps: Sequence[np.ndarray], threshold: Optional[float] = None) -> List[np.ndarray]:
        """Binary defect masks: ``map > pixel_threshold_``."""
        t = self.pixel_threshold_ if threshold is None else threshold
        return [np.asarray(m) > t for m in maps]

    def calibrate_pixel_threshold(
        self,
        images=None,
        masks: Optional[Sequence[MaskLike]] = None,
        maps: Optional[Sequence[np.ndarray]] = None,
        max_pixels: int = 5_000_000,
        show_progress: bool = True,
    ) -> Dict[str, float]:
        """Set ``pixel_threshold_`` to maximise pixel-level F1 on a validation set with ground-truth
        masks (``None`` = defect-free image). Pass ``images`` or precomputed ``maps``."""
        from sklearn.metrics import precision_recall_curve

        if maps is None:
            maps = self.anomaly_maps(images, show_progress=show_progress)
        if masks is None or len(masks) != len(maps):
            raise ValueError("masks must be given, one per image/map")
        s = np.concatenate([np.ravel(m) for m in maps])
        y = np.concatenate([load_mask(g, np.shape(m)).ravel() for g, m in zip(masks, maps)])
        if not y.any():
            raise ValueError("calibrate_pixel_threshold needs at least one anomalous pixel")
        if len(s) > max_pixels:
            idx = np.random.default_rng(0).choice(len(s), max_pixels, replace=False)
            s, y = s[idx], y[idx]
        p, r, t = precision_recall_curve(y, s)
        f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
        i = int(np.argmax(f1))
        self.pixel_threshold_ = float(t[i])
        out = {"pixel_threshold": self.pixel_threshold_, "pixel_f1": float(f1[i])}
        self.metadata["pixel_calibration"] = out
        return out

    # ----------------------------------------------------------------- threshold hooks
    def _set_threshold_value(self, threshold: float) -> None:
        self.threshold_ = threshold

    @property
    def _reference_scores(self) -> np.ndarray:
        return self.train_scores_

    def _check_fitted(self):
        if self.threshold_ is None:
            raise RuntimeError("AnomalibPipeline is not fitted yet.")

    # ----------------------------------------------------------------- persistence
    def get_config(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "image_size": list(self.image_size),
            "calibration_split": self.calibration_split,
            "threshold_quantile": self.threshold_quantile,
            "batch_size": self.batch_size,
            "train_batch_size": self.train_batch_size,
            "num_workers": self.num_workers,
            "max_epochs": self.max_epochs,
            "seed": self.seed,
            "trainer_kwargs": self.trainer_kwargs,
            "intensity": self.intensity,
            **self.model_kwargs,
        }

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Save config + trained weights (state_dict, incl. e.g. the PatchCore memory bank) + thresholds.
        Unlike feature pipelines, the file contains the model weights: no download is needed to load it."""
        import anomalib

        from . import __version__

        self._check_fitted()
        joblib.dump(
            {
                "kind": "anomalib",
                "version": _FORMAT_VERSION,
                "config": self.get_config(),
                "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "threshold_": self.threshold_,
                "train_scores_": self.train_scores_,
                "pixel_threshold_": self.pixel_threshold_,
                "map_reference_": self.map_reference_,
                "metadata": {**self.metadata, "imgad_version": __version__, "anomalib_version": anomalib.__version__},
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, os.PathLike], device: Optional[str] = None, batch_size: Optional[int] = None):
        return cls._from_state(joblib.load(path), device=device, batch_size=batch_size)

    @classmethod
    def _from_state(cls, state: Dict[str, Any], device: Optional[str] = None, batch_size: Optional[int] = None):
        _require_anomalib()
        cfg = dict(state["config"])
        if batch_size is not None:
            cfg["batch_size"] = batch_size
        obj = cls.__new__(cls)
        model_kwargs = {k: v for k, v in cfg.items() if k not in inspect.signature(cls.__init__).parameters}
        for k in model_kwargs:
            cfg.pop(k)
        name = cfg.pop("model")
        obj.model_name = name
        obj.image_size = tuple(cfg.pop("image_size"))
        obj.trainer_kwargs = dict(cfg.pop("trainer_kwargs") or {})
        for k, v in cfg.items():
            setattr(obj, k, v)
        obj.device = device
        obj.model_kwargs = model_kwargs
        obj.model = obj._build_model(pretrained=False)
        _load_state_dict(obj.model, state["state_dict"])
        obj.model.eval()
        for k in ("threshold_", "train_scores_", "pixel_threshold_", "map_reference_"):
            setattr(obj, k, state[k])
        obj.metadata = state.get("metadata", {})
        return obj

    def info(self) -> Dict[str, Any]:
        return {
            "backend": "anomalib",
            **self.get_config(),
            "threshold": self.threshold_,
            "pixel_threshold": self.pixel_threshold_,
            **self.metadata,
        }

    def __repr__(self) -> str:
        extra = ", ".join(f"{k}={v!r}" for k, v in self.model_kwargs.items())
        return f"AnomalibPipeline(model={self.model_name!r}, image_size={self.image_size}" + (f", {extra}" if extra else "") + ")"


def _load_state_dict(model: torch.nn.Module, sd: Dict[str, torch.Tensor]) -> None:
    """``load_state_dict`` that first resizes lazily-shaped buffers (e.g. memory banks / Gaussian
    statistics that are empty until the model is fitted)."""
    own = model.state_dict()
    for k, v in sd.items():
        if k in own and own[k].shape != v.shape:
            mod_name, _, attr = k.rpartition(".")
            mod = model.get_submodule(mod_name) if mod_name else model
            if attr in mod._buffers:
                mod._buffers[attr] = torch.empty_like(v)
            elif attr in mod._parameters:
                mod._parameters[attr] = torch.nn.Parameter(torch.empty_like(v), requires_grad=mod._parameters[attr].requires_grad)
    model.load_state_dict(sd)


def build_anomalib_pipeline(spec: Union[str, Dict[str, Any], AnomalibPipeline], **kwargs) -> AnomalibPipeline:
    """``"patchcore"`` or ``{"name": "patchcore", "backbone": "resnet18", ...}`` (YAML-friendly)."""
    if isinstance(spec, AnomalibPipeline):
        return spec
    if isinstance(spec, str):
        return AnomalibPipeline(spec, **kwargs)
    spec = dict(spec)
    name = spec.pop("name", None) or spec.pop("model")
    return AnomalibPipeline(name, **{**spec, **kwargs})
