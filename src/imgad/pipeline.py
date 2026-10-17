"""End-to-end pipeline: images -> features -> preprocessing -> detector -> anomaly scores."""

from __future__ import annotations

import datetime as _dt
import os
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import joblib
import numpy as np

from .detectors import BaseDetector, build_detector
from .features import FeatureExtractor, build_extractor
from .preprocessing import Step, build_preprocessing

_FORMAT_VERSION = 2


class ThresholdMixin:
    """Decision-threshold logic shared by every pipeline type. Subclasses provide ``threshold_``,
    ``_set_threshold_value(t)``, ``_reference_scores`` (out-of-sample normal scores), ``score`` and
    a ``metadata`` dict."""

    def set_threshold(self, threshold: float, source: str = "manual") -> "ThresholdMixin":
        self._set_threshold_value(float(threshold))
        self.metadata["threshold_source"] = source
        return self

    def normal_percentile(self, scores: np.ndarray) -> np.ndarray:
        """Fraction of normal (out-of-sample) training scores below each score, in [0, 1].
        ~0.5 = typical normal image, >0.99 = more anomalous than 99% of normal images."""
        ref = np.sort(self._reference_scores)
        return np.searchsorted(ref, np.asarray(scores), side="right") / len(ref)

    def calibrate_threshold(
        self,
        images=None,
        labels: Optional[Sequence[int]] = None,
        scores: Optional[np.ndarray] = None,
        method: str = "f1",
        target_fpr: float = 0.05,
        target_tpr: float = 0.95,
        show_progress: bool = True,
    ) -> Dict[str, float]:
        """Choose the decision threshold from a **labelled validation set** (1 = anomaly).

        method:
          * ``"f1"``     -> threshold maximising F1 (needs both classes)
          * ``"youden"`` -> maximises TPR - FPR (needs both classes)
          * ``"fpr"``    -> highest recall with FPR <= ``target_fpr`` (normals alone are enough)
          * ``"tpr"``    -> lowest FPR with TPR >= ``target_tpr`` (needs anomalies)

        Returns a dict with the chosen threshold and the validation metrics at that threshold.
        """
        from sklearn.metrics import precision_recall_curve, roc_curve

        if scores is None:
            scores = self.score(images, show_progress=show_progress)
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=int) if labels is not None else np.zeros(len(scores), dtype=int)
        if len(y) != len(scores):
            raise ValueError("labels and images/scores must have the same length")
        has_both = len(np.unique(y)) == 2

        if method == "fpr":
            normal = np.sort(scores[y == 0])
            if len(normal) == 0:
                raise ValueError("method='fpr' needs normal validation samples")
            thr = float(np.quantile(normal, 1 - target_fpr))
        elif not has_both:
            raise ValueError(f"method={method!r} needs both normal and anomalous validation samples")
        elif method == "f1":
            p, r, t = precision_recall_curve(y, scores)
            f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
            thr = _just_below(scores, t[int(np.argmax(f1))])
        elif method in ("youden", "tpr"):
            fpr, tpr, t = roc_curve(y, scores)
            if method == "youden":
                i = int(np.argmax(tpr - fpr))
            else:
                i = int(np.argmax(tpr >= target_tpr))
            thr = _just_below(scores, t[i]) if np.isfinite(t[i]) else float(scores.max())
        else:
            raise ValueError(f"Unknown method {method!r}")

        self.set_threshold(thr, source=f"calibrated on {len(y)} validation images (method={method})")
        pred = (scores > thr).astype(int)
        out = {"threshold": thr, "fpr": float(pred[y == 0].mean()) if (y == 0).any() else float("nan")}
        if (y == 1).any():
            tp = int(((pred == 1) & (y == 1)).sum())
            out["tpr"] = float(pred[y == 1].mean())
            out["precision"] = tp / max(int(pred.sum()), 1)
            out["f1"] = 2 * tp / max(int(pred.sum()) + int(y.sum()), 1)
        self.metadata["calibration"] = {"method": method, "n_val": int(len(y)), **out}
        return out


class AnomalyDetectionPipeline(ThresholdMixin):
    """
    Parameters
    ----------
    extractor:  a :class:`FeatureExtractor`, a HF model id, or a config dict ``{"type": "hf", ...}``
    detector:   a :class:`BaseDetector`, a detector name, or a spec dict ``{"name": "knn", "k": 1}``
    preprocessing: list of steps (see :mod:`imgad.preprocessing`), e.g. ``["l2", "pca:0.95"]``
    cache_dir:  optional directory to cache per-image features (only for file-path inputs)
    """

    supports_maps = False  # image-level scores only (see AnomalibPipeline for anomaly maps)

    def __init__(
        self,
        extractor: Union[FeatureExtractor, str, Dict[str, Any]],
        detector: Union[BaseDetector, str, Dict[str, Any]] = "knn",
        preprocessing: Optional[Iterable[Step]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
    ):
        self.extractor = extractor if isinstance(extractor, FeatureExtractor) else build_extractor(extractor)
        self.detector = detector if isinstance(detector, BaseDetector) else build_detector(detector)
        self.preprocessing_steps = list(preprocessing) if preprocessing else []
        self.preprocessing = build_preprocessing(self.preprocessing_steps)
        self.cache_dir = cache_dir
        self.metadata: Dict[str, Any] = {}

    # ----------------------------------------------------------------- features
    def extract(self, images, show_progress: bool = True) -> np.ndarray:
        return self.extractor.extract(images, cache_dir=self.cache_dir, show_progress=show_progress)

    # ----------------------------------------------------------------- images API
    def fit(self, images, **kw) -> "AnomalyDetectionPipeline":
        """Fit on **normal** images only."""
        return self.fit_features(self.extract(images, **kw))

    def score(self, images, **kw) -> np.ndarray:
        return self.score_features(self.extract(images, **kw))

    def predict(self, images, threshold: Optional[float] = None, **kw) -> np.ndarray:
        return self.predict_features(self.extract(images, **kw), threshold)

    # ----------------------------------------------------------------- features API
    def fit_features(self, X: np.ndarray) -> "AnomalyDetectionPipeline":
        if self.preprocessing is not None:
            X = self.preprocessing.fit_transform(X)
        self.detector.fit(X)
        self.metadata.update(
            n_train=int(X.shape[0]),
            feature_dim=int(X.shape[1]),
            trained_at=_dt.datetime.now().isoformat(timespec="seconds"),
            threshold_source=f"quantile {self.detector.threshold_quantile} of normal training scores",
        )
        return self

    def _transform(self, X):
        return self.preprocessing.transform(X) if self.preprocessing is not None else X

    def score_features(self, X: np.ndarray) -> np.ndarray:
        return self.detector.score(self._transform(X))

    def predict_features(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        return self.detector.predict(self._transform(X), threshold)

    # ----------------------------------------------------------------- threshold
    @property
    def threshold_(self) -> Optional[float]:
        return self.detector.threshold_

    def _set_threshold_value(self, threshold: float) -> None:
        self.detector.threshold_ = threshold

    @property
    def _reference_scores(self) -> np.ndarray:
        return self.detector.train_scores_

    # ----------------------------------------------------------------- persistence
    def save(self, path: Union[str, os.PathLike]) -> None:
        """Save detector + preprocessing + extractor *config* (model weights are re-loaded from the Hub)."""
        from . import __version__

        joblib.dump(
            {
                "extractor": self.extractor.full_config(),
                "preprocessing": self.preprocessing,
                "preprocessing_steps": [s for s in self.preprocessing_steps if isinstance(s, str)],
                "detector": self.detector,
                "cache_dir": self.cache_dir,
                "metadata": {**self.metadata, "imgad_version": __version__},
                "version": _FORMAT_VERSION,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Union[str, os.PathLike],
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        token=None,
    ) -> "AnomalyDetectionPipeline":
        """Load a trained pipeline. ``device``/``batch_size``/``cache_dir`` override the saved values
        (e.g. a model trained on GPU can be used on CPU). Feature caching is off at inference unless
        ``cache_dir`` is given. ``token``: Hugging Face access token if the extractor is a gated/private
        model (otherwise ``HF_TOKEN`` or the stored login are used; tokens are never saved in the file).

        Files saved by :class:`~imgad.anomalib_backend.AnomalibPipeline` are also accepted (an
        ``AnomalibPipeline`` is returned), so every loader in imgad works with both kinds."""
        state = joblib.load(path)
        if state.get("kind") == "anomalib":
            from .anomalib_backend import AnomalibPipeline

            return AnomalibPipeline._from_state(state, device=device, batch_size=batch_size)
        ext_cfg = dict(state["extractor"])
        if "device" in ext_cfg:
            ext_cfg["device"] = device
        if batch_size is not None and "batch_size" in ext_cfg:
            ext_cfg["batch_size"] = batch_size
        obj = cls.__new__(cls)
        obj.extractor = build_extractor(ext_cfg, token=token)
        obj.detector = state["detector"]
        obj.preprocessing = state["preprocessing"]
        obj.preprocessing_steps = state.get("preprocessing_steps", [])
        obj.cache_dir = cache_dir
        obj.metadata = state.get("metadata", {})
        return obj

    def info(self) -> Dict[str, Any]:
        """Human-readable summary of a (trained) pipeline."""
        return {
            "extractor": self.extractor.full_config(),
            "preprocessing": self.preprocessing_steps,
            "detector": repr(self.detector),
            "threshold": self.threshold_,
            **self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"AnomalyDetectionPipeline(\n  extractor={self.extractor!r},\n"
            f"  preprocessing={self.preprocessing_steps!r},\n  detector={self.detector!r}\n)"
        )


def load_pipeline(path: Union[str, os.PathLike], device: Optional[str] = None, batch_size: Optional[int] = None,
                  token=None):
    """Load any saved imgad pipeline (feature + detector, or anomalib)."""
    return AnomalyDetectionPipeline.load(path, device=device, batch_size=batch_size, token=token)


def _just_below(scores: np.ndarray, t: float) -> float:
    """predict() uses ``score > threshold``; sklearn thresholds use ``>=``. Return a value just below
    ``t`` (midpoint to the next lower observed score) so the sample scoring exactly ``t`` is flagged."""
    lower = scores[scores < t]
    return float((t + lower.max()) / 2) if lower.size else float(np.nextafter(t, -np.inf))
