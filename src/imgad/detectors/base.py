"""Detector interface and registry.

Convention: ``score(X)`` returns an anomaly score where **higher = more anomalous**.
Detectors are trained only on normal samples (one-class / novelty-detection setting).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import numpy as np

_DETECTORS: Dict[str, Type["BaseDetector"]] = {}


def register_detector(*names: str):
    """Class decorator registering a detector under one or more names."""

    def deco(cls):
        for n in names:
            _DETECTORS[n] = cls
        cls.registry_name = names[0]
        return cls

    return deco


def list_detectors() -> List[str]:
    return sorted(_DETECTORS)


def build_detector(spec: Dict[str, Any] | str, **kwargs) -> "BaseDetector":
    """``build_detector("knn", k=5)`` or ``build_detector({"name": "knn", "k": 5})``."""
    if isinstance(spec, dict):
        spec = dict(spec)
        name = spec.pop("name")
        kwargs = {**spec, **kwargs}
    else:
        name = spec
    if name not in _DETECTORS:
        raise KeyError(f"Unknown detector {name!r}. Available: {list_detectors()}")
    return _DETECTORS[name](**kwargs)


class BaseDetector(ABC):
    """Base class for all detectors.

    Parameters
    ----------
    threshold_quantile:
        After fitting, the decision threshold is set to this quantile of the **out-of-sample**
        training scores; ``predict`` flags samples above it.
        E.g. 0.95 -> ~5% false-positive rate on normal data.

    Calibration (class attributes, can be overridden per instance):
        ``calibration = "cv"``   -> out-of-fold scores via K-fold (``n_calibration_splits``), unless the
                                    detector provides unbiased leave-one-out scores (kNN, LOF).
        ``calibration = "train"``-> in-sample scores (fast, but optimistic for KDE/OCSVM/PCA...).
    """

    registry_name: str = "base"
    calibration: str = "cv"
    n_calibration_splits: int = 5
    random_state_calibration: int = 0

    def __init__(self, threshold_quantile: float = 0.95):
        self.threshold_quantile = threshold_quantile
        self.threshold_: Optional[float] = None
        self.train_scores_: Optional[np.ndarray] = None

    # -- to implement -----------------------------------------------------
    @abstractmethod
    def _fit(self, X: np.ndarray) -> None:
        ...

    @abstractmethod
    def _score(self, X: np.ndarray) -> np.ndarray:
        ...

    def _train_scores(self, X: np.ndarray) -> np.ndarray:
        """Scores of the training set used for calibration. Override when in-sample scores are
        biased (e.g. kNN, where every point is its own nearest neighbour)."""
        return self._score(X)

    # -- public API -------------------------------------------------------
    def fit(self, X: np.ndarray) -> "BaseDetector":
        X = _check(X)
        self._fit(X)
        self.n_features_in_ = X.shape[1]
        self.train_scores_ = np.asarray(self._calibration_scores(X), dtype=np.float64)
        self.threshold_ = float(np.quantile(self.train_scores_, self.threshold_quantile))
        return self

    def _calibration_scores(self, X: np.ndarray) -> np.ndarray:
        has_loo = type(self)._train_scores is not BaseDetector._train_scores
        k = self.n_calibration_splits
        if has_loo or self.calibration != "cv" or len(X) < 2 * k:
            return self._train_scores(X)
        import copy

        from sklearn.model_selection import KFold

        oof = np.empty(len(X))
        for tr, va in KFold(k, shuffle=True, random_state=self.random_state_calibration).split(X):
            m = copy.deepcopy(self)
            m._fit(X[tr])
            oof[va] = m._score(X[va])
        return oof

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self._score(_check(X)), dtype=np.float64)

    # sklearn-style aliases
    decision_function = score

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """1 = anomaly, 0 = normal."""
        t = self.threshold_ if threshold is None else threshold
        return (self.score(X) > t).astype(int)

    def get_params(self) -> Dict[str, Any]:
        import inspect

        sig = inspect.signature(type(self).__init__)
        return {k: getattr(self, k) for k in sig.parameters if k != "self" and hasattr(self, k)}

    def _check_fitted(self):
        if self.threshold_ is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted yet.")

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"


def _check(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array (n_samples, n_features), got shape {X.shape}")
    return X
