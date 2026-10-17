"""Zero-shot and few-shot anomaly detection with vision–language models.

:class:`ZeroShotDetector` scores an image by comparing its embedding with text prompts describing normal and
anomalous findings (CLIP-style zero-shot classification, as in WinCLIP). With a few normal images it adds the
distance to them (as in WinCLIP+), so the same detector covers the zero-shot and the few-shot settings::

    from imgad import AnomalyDetectionPipeline

    pipe = AnomalyDetectionPipeline("zoo:flair-clip", {"name": "zero_shot", "modality": "fundus"})
    pipe.fit([])                       # zero-shot: no training images
    pipe.fit(normal_images[:4])        # 4-shot: text prompts + the 4 normal images

It needs an extractor whose image embeddings share a space with text: CLIP/SigLIP models (``hf`` with
``pooling="projection"``, ``open_clip``), BiomedCLIP, MedSigLIP and the retinal models FLAIR and RetiZero
(``zoo:*-clip``). The pipeline and the benchmark encode the prompts with that extractor (:meth:`bind`);
the text embeddings are then stored in the detector, so a fitted detector needs no text encoder.

Scores:
  * text: probability of the anomalous prompts, ``softmax(logit_scale · cos(image, prompts))`` summed over the
    anomalous states (in [0, 1]);
  * visual (few-shot): ``(1 − mean cosine similarity to the k nearest normal images) / 2`` (in [0, 1]);
  * ``mode``: ``"text"``, ``"visual"``, ``"both"`` (``(1 − w) · text + w · visual``, ``w = visual_weight``) or
    ``"auto"``: text alone without training images, both with them.

Threshold: with training images, the usual quantile of their (leave-one-out) scores; zero-shot, 0.5 on the
text probability (the anomalous prompts win). Recalibrate it on normal images when you have some.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..prompts import default_prompts
from .base import BaseDetector, register_detector

_MODES = ("auto", "text", "visual", "both")


@register_detector("zero_shot", "zeroshot", "vlm_prompts")
class ZeroShotDetector(BaseDetector):
    """Prompt-based zero/few-shot detector (see the module docs).

    modality:  prompt set from :mod:`imgad.prompts` (``"fundus"``, ``"oct"``, ``"cxr"``...; ``None`` = generic).
    normal / anomalous: states that replace the modality's defaults (``["healthy"]``, ``["glaucoma", ...]``).
    templates: replace the modality's templates (each with one ``{}``).
    mode, visual_weight, k: see the module docs. temperature: overrides the model's logit scale.
    """

    needs_text = True
    calibration = "train"

    def __init__(self, modality: Optional[str] = None, normal: Optional[Sequence[str]] = None,
                 anomalous: Optional[Sequence[str]] = None, templates: Optional[Sequence[str]] = None,
                 mode: str = "auto", visual_weight: float = 0.5, k: int = 1, temperature: Optional[float] = None,
                 threshold_quantile: float = 0.95):
        super().__init__(threshold_quantile)
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.modality, self.mode, self.visual_weight, self.k, self.temperature = modality, mode, visual_weight, k, temperature
        d = default_prompts(modality)
        self.normal = list(normal) if normal else d["normal"]
        self.anomalous = list(anomalous) if anomalous else d["anomalous"]
        self.templates = list(templates) if templates else d["templates"]
        for t in self.templates:
            if t.count("{}") != 1:
                raise ValueError(f"template {t!r} must contain exactly one '{{}}'")
        self.text_ = None  # (n_states, dim) prototypes: normal states first
        self.logit_scale_ = None

    # ------------------------------------------------------------------ text
    def prompts(self) -> List[str]:
        """Every prompt string the detector needs encoded."""
        return [t.format(s) for s in self.normal + self.anomalous for t in self.templates]

    def bind(self, extractor) -> "ZeroShotDetector":
        """Encode the prompts with ``extractor`` (which must have ``encode_text``)."""
        if not (hasattr(extractor, "has_text_encoder") and extractor.has_text_encoder()):
            name = getattr(extractor, "model_name", type(extractor).__name__)
            raise TypeError(f"{name} cannot encode text: the zero-shot detector needs a vision–language "
                            "extractor (CLIP/SigLIP with pooling='projection', open_clip, zoo:*-clip)")
        prompts = self.prompts()
        return self.set_text(dict(zip(prompts, extractor.encode_text(prompts))), extractor.logit_scale)

    def set_text(self, embeddings: Dict[str, np.ndarray], logit_scale: float) -> "ZeroShotDetector":
        """Use precomputed prompt embeddings (``{prompt: vector}``, e.g. from a benchmark's text bank)."""
        protos = []
        for s in self.normal + self.anomalous:
            e = np.stack([np.asarray(embeddings[t.format(s)], dtype=np.float64) for t in self.templates])
            e = _l2(e).mean(0)
            protos.append(e / np.linalg.norm(e))
        self.text_ = np.stack(protos)
        self.logit_scale_ = float(logit_scale)
        return self

    # ------------------------------------------------------------------ fit / score
    def fit(self, X) -> "ZeroShotDetector":
        if self.text_ is None:
            raise RuntimeError("ZeroShotDetector has no text embeddings: call bind(extractor) first (pipelines "
                               "and benchmarks do it automatically)")
        X = np.asarray(X, dtype=np.float64).reshape(-1, self.text_.shape[1]) if np.size(X) == 0 else np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.text_.shape[1]:
            raise ValueError(f"image features have dimension {X.shape[-1]}, the text embeddings {self.text_.shape[1]}: "
                             "use the joint image embeddings of the same model, without preprocessing that "
                             "changes them (only 'l2' is compatible)")
        self.mode_ = ("both" if len(X) else "text") if self.mode == "auto" else self.mode
        if self.mode_ != "text" and not len(X):
            raise ValueError(f"mode={self.mode!r} needs normal training images (few-shot)")
        self.memory_ = _l2(X) if self.mode_ != "text" else None
        self.n_features_in_ = X.shape[1]
        if len(X):
            self.train_scores_ = self._combine(self._text(X), self._visual(X, loo=True) if self.memory_ is not None else None)
            self.threshold_ = float(np.quantile(self.train_scores_, self.threshold_quantile))
        else:
            self.train_scores_ = np.zeros(0)
            self.threshold_ = 0.5
        return self

    def _fit(self, X):  # pragma: no cover - fit is overridden
        raise NotImplementedError

    def _score(self, X):
        return self._combine(self._text(X), self._visual(X) if self.memory_ is not None else None)

    def _text(self, X) -> np.ndarray:
        scale = self.temperature or self.logit_scale_
        z = scale * (_l2(X) @ self.text_.T)
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return p[:, len(self.normal):].sum(1)

    def _visual(self, X, loo: bool = False) -> np.ndarray:
        sims = _l2(X) @ self.memory_.T
        if loo:  # training images: leave each one out of its own neighbours
            if len(self.memory_) < 2:
                return np.full(len(X), np.nan)
            np.fill_diagonal(sims, -np.inf)
        k = max(1, min(self.k, sims.shape[1] - (1 if loo else 0)))
        top = -np.sort(-sims, axis=1)[:, :k]
        return (1 - top.mean(1)) / 2

    def _combine(self, text: np.ndarray, visual: Optional[np.ndarray]) -> np.ndarray:
        if self.mode_ == "text":
            return text
        if visual is not None and np.isnan(visual).any():  # 1-shot calibration: no neighbour to compare with
            visual = np.zeros_like(text)
        if self.mode_ == "visual":
            return visual
        w = self.visual_weight
        return (1 - w) * text + w * visual

    def get_params(self) -> Dict[str, Any]:
        return dict(modality=self.modality, normal=self.normal, anomalous=self.anomalous, templates=self.templates,
                    mode=self.mode, visual_weight=self.visual_weight, k=self.k, temperature=self.temperature,
                    threshold_quantile=self.threshold_quantile)

    def __repr__(self) -> str:
        return (f"ZeroShotDetector(modality={self.modality!r}, mode={self.mode!r}, {len(self.normal)} normal / "
                f"{len(self.anomalous)} anomalous states × {len(self.templates)} templates)")


def _l2(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def needs_text(detector) -> bool:
    """True for detectors (instances or specs) that compare images with text prompts."""
    if isinstance(detector, BaseDetector):
        return bool(getattr(detector, "needs_text", False))
    from .base import _DETECTORS

    name = detector.get("name") if isinstance(detector, dict) else detector
    return bool(getattr(_DETECTORS.get(name), "needs_text", False))
