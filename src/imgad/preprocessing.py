"""Feature post-processing between the extractor and the detector.

Steps are given as strings (handy for YAML) or sklearn transformers:

* ``"l2"``            -> row-wise L2 normalisation
* ``"standard"``      -> StandardScaler
* ``"center"``        -> subtract the mean of normal features
* ``"pca:<n>"``       -> PCA with n components (int) or explained variance (float < 1)
* ``"pca_whiten:<n>"``-> whitened PCA
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler

Step = Union[str, TransformerMixin]


def _parse(step: Step):
    if not isinstance(step, str):
        return type(step).__name__.lower(), step
    name, _, arg = step.partition(":")
    if name == "l2":
        return "l2", Normalizer(norm="l2")
    if name == "standard":
        return "standard", StandardScaler()
    if name == "center":
        return "center", StandardScaler(with_std=False)
    if name in ("pca", "pca_whiten"):
        n = float(arg) if arg else 0.95
        n = int(n) if n >= 1 else n
        return name, PCA(n_components=n, whiten=name == "pca_whiten")
    if name in ("identity", "none"):
        return "identity", FunctionTransformer()
    raise ValueError(f"Unknown preprocessing step {step!r}")


def build_preprocessing(steps: Optional[Iterable[Step]]) -> Optional[Pipeline]:
    if not steps:
        return None
    parsed = [_parse(s) for s in steps]
    # make step names unique for sklearn
    named = [(f"{i}_{n}", t) for i, (n, t) in enumerate(parsed)]
    return Pipeline(named)
