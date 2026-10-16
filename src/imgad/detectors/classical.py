"""Classical anomaly detectors operating on fixed-length feature vectors."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import BaseDetector, register_detector


# --------------------------------------------------------------------------- distance based
@register_detector("knn")
class KNNDetector(BaseDetector):
    """Distance to the k nearest normal training samples (strong baseline with CLIP/DINO features).

    aggregation: "mean" (avg of the k distances), "max"/"kth" (k-th distance) or "min".
    metric: any sklearn metric; "cosine" is a good default for L2-normalised embeddings.
    """

    def __init__(self, k: int = 1, metric: str = "cosine", aggregation: str = "mean", threshold_quantile: float = 0.95):
        super().__init__(threshold_quantile)
        self.k, self.metric, self.aggregation = k, metric, aggregation

    def _fit(self, X):
        from sklearn.neighbors import NearestNeighbors

        if len(X) < 2:
            raise ValueError("KNNDetector needs at least 2 training samples")
        self._k = max(1, min(self.k, len(X) - 1))
        self.nn_ = NearestNeighbors(n_neighbors=self._k + 1, metric=self.metric).fit(X)

    def _agg(self, d):
        if self.aggregation == "mean":
            return d.mean(1)
        if self.aggregation in ("max", "kth"):
            return d.max(1)
        if self.aggregation == "min":
            return d.min(1)
        raise ValueError(self.aggregation)

    def _score(self, X):
        d, _ = self.nn_.kneighbors(X, n_neighbors=self._k)
        return self._agg(d)

    def _train_scores(self, X):  # leave-one-out: drop self-match
        d, _ = self.nn_.kneighbors(X, n_neighbors=self._k + 1)
        return self._agg(d[:, 1:])


@register_detector("mahalanobis", "gaussian")
class MahalanobisDetector(BaseDetector):
    """Mahalanobis distance to a single Gaussian fitted on normal features.

    estimator: "ledoit_wolf" (default, robust for d >> n), "oas", "empirical", "mcd".
    """

    def __init__(self, estimator: str = "ledoit_wolf", threshold_quantile: float = 0.95):
        super().__init__(threshold_quantile)
        self.estimator = estimator

    def _fit(self, X):
        from sklearn import covariance

        est = {
            "ledoit_wolf": covariance.LedoitWolf,
            "oas": covariance.OAS,
            "empirical": covariance.EmpiricalCovariance,
            "mcd": covariance.MinCovDet,
        }[self.estimator]()
        self.cov_ = est.fit(X)

    def _score(self, X):
        return np.sqrt(np.maximum(self.cov_.mahalanobis(X), 0))


@register_detector("centroid")
class CentroidDetector(BaseDetector):
    """Distance (cosine or euclidean) to the mean normal embedding. Very cheap baseline."""

    def __init__(self, metric: str = "cosine", threshold_quantile: float = 0.95):
        super().__init__(threshold_quantile)
        self.metric = metric

    def _fit(self, X):
        self.center_ = X.mean(0)

    def _score(self, X):
        if self.metric == "cosine":
            c = self.center_ / (np.linalg.norm(self.center_) + 1e-12)
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            return 1.0 - Xn @ c
        return np.linalg.norm(X - self.center_, axis=1)


@register_detector("kmeans")
class KMeansDetector(BaseDetector):
    """Distance to the nearest of ``n_clusters`` k-means centroids (handles multi-modal normality)."""

    def __init__(self, n_clusters: int = 8, random_state: Optional[int] = 0, threshold_quantile: float = 0.95):
        super().__init__(threshold_quantile)
        self.n_clusters, self.random_state = n_clusters, random_state

    def _fit(self, X):
        from sklearn.cluster import KMeans

        self.km_ = KMeans(n_clusters=min(self.n_clusters, len(X)), n_init="auto", random_state=self.random_state).fit(X)

    def _score(self, X):
        return self.km_.transform(X).min(1)


# --------------------------------------------------------------------------- density based
@register_detector("lof")
class LOFDetector(BaseDetector):
    """Local Outlier Factor in novelty mode."""

    def __init__(self, n_neighbors: int = 20, metric: str = "cosine", threshold_quantile: float = 0.95):
        super().__init__(threshold_quantile)
        self.n_neighbors, self.metric = n_neighbors, metric

    def _fit(self, X):
        from sklearn.neighbors import LocalOutlierFactor

        self.lof_ = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, len(X) - 1), metric=self.metric, novelty=True
        ).fit(X)

    def _score(self, X):
        return -self.lof_.score_samples(X)

    def _train_scores(self, X):
        return -self.lof_.negative_outlier_factor_


@register_detector("gmm")
class GMMDetector(BaseDetector):
    """Negative log-likelihood under a Gaussian mixture."""

    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = "diag",
        reg_covar: float = 1e-4,
        random_state: Optional[int] = 0,
        threshold_quantile: float = 0.95,
    ):
        super().__init__(threshold_quantile)
        self.n_components, self.covariance_type = n_components, covariance_type
        self.reg_covar, self.random_state = reg_covar, random_state

    def _fit(self, X):
        from sklearn.mixture import GaussianMixture

        self.gmm_ = GaussianMixture(
            n_components=min(self.n_components, len(X)),
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
        ).fit(X)

    def _score(self, X):
        return -self.gmm_.score_samples(X)


@register_detector("kde")
class KDEDetector(BaseDetector):
    """Negative log-density under a kernel density estimate (best after PCA to a few dims)."""

    def __init__(self, bandwidth: Any = "scott", kernel: str = "gaussian", threshold_quantile: float = 0.95):
        super().__init__(threshold_quantile)
        self.bandwidth, self.kernel = bandwidth, kernel

    def _fit(self, X):
        from sklearn.neighbors import KernelDensity

        self.kde_ = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(X)

    def _score(self, X):
        return -self.kde_.score_samples(X)


# --------------------------------------------------------------------------- boundary / ensemble
@register_detector("ocsvm")
class OCSVMDetector(BaseDetector):
    """One-Class SVM."""

    def __init__(self, kernel: str = "rbf", nu: float = 0.1, gamma: Any = "scale", threshold_quantile: float = 0.95):
        super().__init__(threshold_quantile)
        self.kernel, self.nu, self.gamma = kernel, nu, gamma

    def _fit(self, X):
        from sklearn.svm import OneClassSVM

        self.svm_ = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma).fit(X)

    def _score(self, X):
        return -self.svm_.score_samples(X)


@register_detector("iforest", "isolation_forest")
class IsolationForestDetector(BaseDetector):
    """Isolation Forest."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: Any = "auto",
        max_features: float = 1.0,
        random_state: Optional[int] = 0,
        threshold_quantile: float = 0.95,
    ):
        super().__init__(threshold_quantile)
        self.n_estimators, self.max_samples = n_estimators, max_samples
        self.max_features, self.random_state = max_features, random_state

    def _fit(self, X):
        from sklearn.ensemble import IsolationForest

        self.if_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            random_state=self.random_state,
        ).fit(X)

    def _score(self, X):
        return -self.if_.score_samples(X)


# --------------------------------------------------------------------------- subspace
@register_detector("pca", "pca_reconstruction")
class PCAReconstructionDetector(BaseDetector):
    """Reconstruction error after projecting onto the principal subspace of normal data.

    n_components: int, or float in (0, 1) = explained variance ratio to keep.
    """

    def __init__(self, n_components: Any = 0.9, whiten: bool = False, threshold_quantile: float = 0.95):
        super().__init__(threshold_quantile)
        self.n_components, self.whiten = n_components, whiten

    def _fit(self, X):
        from sklearn.decomposition import PCA

        nc = self.n_components
        if isinstance(nc, int):
            nc = min(nc, min(X.shape) - 1)
        self.pca_ = PCA(n_components=nc, whiten=self.whiten).fit(X)

    def _score(self, X):
        rec = self.pca_.inverse_transform(self.pca_.transform(X))
        return np.linalg.norm(X - rec, axis=1)


# --------------------------------------------------------------------------- adapters
@register_detector("sklearn")
class SklearnDetector(BaseDetector):
    """Wrap any sklearn-like estimator exposing ``score_samples`` (higher = more normal)
    or ``decision_function``. Pass an instance: ``SklearnDetector(estimator=EllipticEnvelope())``."""

    def __init__(self, estimator=None, threshold_quantile: float = 0.95):
        super().__init__(threshold_quantile)
        if estimator is None:
            raise ValueError("SklearnDetector needs an `estimator` instance")
        self.estimator = estimator

    def _fit(self, X):
        self.estimator.fit(X)

    def _score(self, X):
        if hasattr(self.estimator, "score_samples"):
            return -self.estimator.score_samples(X)
        return -self.estimator.decision_function(X)


@register_detector("pyod")
class PyODDetector(BaseDetector):
    """Adapter for any PyOD model (``pip install pyod``): ECOD, COPOD, HBOS, LODA, ABOD, DeepSVDD ...

    ``PyODDetector(model="ECOD")`` or ``PyODDetector(model=pyod.models.ecod.ECOD())``.
    PyOD's ``decision_function`` already follows the higher = more anomalous convention.
    """

    _PYOD_PATHS = {
        "ECOD": "pyod.models.ecod", "COPOD": "pyod.models.copod", "HBOS": "pyod.models.hbos",
        "LODA": "pyod.models.loda", "ABOD": "pyod.models.abod", "CBLOF": "pyod.models.cblof",
        "COF": "pyod.models.cof", "INNE": "pyod.models.inne", "LMDD": "pyod.models.lmdd",
        "MCD": "pyod.models.mcd", "SOD": "pyod.models.sod", "SOS": "pyod.models.sos",
        "KNN": "pyod.models.knn", "LOF": "pyod.models.lof", "IForest": "pyod.models.iforest",
        "OCSVM": "pyod.models.ocsvm", "PCA": "pyod.models.pca", "AutoEncoder": "pyod.models.auto_encoder",
        "VAE": "pyod.models.vae", "DeepSVDD": "pyod.models.deep_svdd", "LUNAR": "pyod.models.lunar",
        "DIF": "pyod.models.dif", "SUOD": "pyod.models.suod",
    }

    def __init__(self, model: Any = "ECOD", model_kwargs: Optional[dict] = None, threshold_quantile: float = 0.95):
        super().__init__(threshold_quantile)
        self.model, self.model_kwargs = model, dict(model_kwargs or {})

    def _make(self):
        if not isinstance(self.model, str):
            return self.model
        import importlib

        try:
            mod = importlib.import_module(self._PYOD_PATHS[self.model])
        except KeyError:
            raise KeyError(f"Unknown PyOD model {self.model!r}; pass an instance instead.")
        except ImportError as e:  # pragma: no cover
            raise ImportError("PyODDetector requires `pip install pyod`.") from e
        return getattr(mod, self.model)(**self.model_kwargs)

    def _fit(self, X):
        self.model_ = self._make()
        self.model_.fit(X)

    def _score(self, X):
        return self.model_.decision_function(X)


@register_detector("ensemble")
class EnsembleDetector(BaseDetector):
    """Average of member scores after standardising each by its training-score distribution.

    members: list of detector specs, e.g. ``[{"name": "knn", "k": 1}, "mahalanobis"]``.
    normalization: "zscore" (mean/std of train scores) or "rank" (empirical CDF of train scores).
    """

    def __init__(self, members=("knn", "mahalanobis"), normalization: str = "zscore", threshold_quantile: float = 0.95):
        super().__init__(threshold_quantile)
        self.members, self.normalization = list(members), normalization

    def _fit(self, X):
        from .base import build_detector

        self.detectors_ = [m if isinstance(m, BaseDetector) else build_detector(m) for m in self.members]
        for d in self.detectors_:
            d.fit(X)

    def _norm(self, d, s):
        ref = d.train_scores_
        if self.normalization == "rank":
            return np.searchsorted(np.sort(ref), s, side="right") / len(ref)
        return (s - ref.mean()) / (ref.std() + 1e-12)

    def _score(self, X):
        return np.mean([self._norm(d, d.score(X)) for d in self.detectors_], axis=0)

    def _train_scores(self, X):
        return np.mean([self._norm(d, d.train_scores_) for d in self.detectors_], axis=0)
