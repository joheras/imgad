"""Extractor for timm models, including Hub checkpoints such as the pathology foundation models UNI,
UNI2-h, Virchow2, Prov-GigaPath or H-optimus-0 (``hf-hub:org/name``).

    TimmFeatureExtractor("hf-hub:MahmoodLab/uni", timm_kwargs={"init_values": 1e-5, "dynamic_img_size": True})
    TimmFeatureExtractor("hf-hub:paige-ai/Virchow2", pooling="cls+mean",
                         timm_kwargs={"mlp_layer": "timm.layers.SwiGLUPacked", "act_layer": "torch.nn.SiLU"})

``timm_kwargs`` are passed to ``timm.create_model``. Values that are classes (layers, activations) are given
as dotted import paths (``"torch.nn.SiLU"``) so that the configuration stays serialisable (saved pipelines,
YAML, feature-cache keys). The ready-made recipes for medical models are in :mod:`imgad.zoo`.

Pooling: ``"auto"`` = the model's own pooled output (``model(x)``; for models that return tokens, the class
token); ``"cls"``; ``"mean"`` (patch tokens, without class/register tokens); ``"cls+mean"`` (concatenation).

Requires ``pip install timm``.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..hub import raise_if_hub_error, token_env
from ..utils import resolve_device
from .base import FeatureExtractor, register_extractor

_POOLINGS = {"auto", "cls", "mean", "cls+mean"}


def _resolve(value: Any) -> Any:
    """``"torch.nn.SiLU"`` -> the class; other values unchanged."""
    if isinstance(value, str) and value.count(".") >= 1 and value.split(".")[0] in ("timm", "torch"):
        module, _, attr = value.rpartition(".")
        try:
            return getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError):
            return value
    return value


@register_extractor("timm")
class TimmFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        model_name: str,
        pooling: str = "auto",
        timm_kwargs: Optional[Dict[str, Any]] = None,
        pretrained: bool = True,
        normalize: bool = True,
        device: Optional[str] = None,
        batch_size: int = 32,
        token: Union[None, bool, str] = None,
    ):
        if pooling not in _POOLINGS:
            raise ValueError(f"pooling must be one of {_POOLINGS}, got {pooling!r}")
        self.model_name = model_name
        self.pooling = pooling
        self.timm_kwargs = dict(timm_kwargs or {})
        self.pretrained = pretrained
        self.normalize = normalize
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._token = token  # never part of get_config()
        self._model = None
        self._transform = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import timm
            from timm.data import create_transform, resolve_data_config
        except ImportError as e:  # pragma: no cover
            raise ImportError('TimmFeatureExtractor requires timm: pip install "imgad[medical]" (or timm)') from e
        kwargs = {k: _resolve(v) for k, v in self.timm_kwargs.items()}
        kwargs.setdefault("num_classes", 0)
        try:
            with token_env(self._token):  # timm downloads through huggingface_hub without a token argument
                model = timm.create_model(self.model_name, pretrained=self.pretrained, **kwargs)
        except Exception as e:
            if self.model_name.startswith(("hf-hub:", "hf_hub:")):
                raise_if_hub_error(e, self.model_name.replace("hf_hub:", "hf-hub:"), self._token)
            raise
        self._model = model.eval().to(self.device)
        self._transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model), is_training=False)

    def _extract_batch(self, images: List) -> np.ndarray:
        import torch

        self._load()
        x = torch.stack([self._transform(im) for im in images]).to(self.device)
        with torch.inference_mode():
            if self.pooling == "auto":
                f = self._model(x)
                if f.ndim == 3:  # models that return tokens (e.g. Virchow2): class token
                    f = f[:, 0]
            else:
                tokens = self._model.forward_features(x)
                if tokens.ndim != 3:
                    raise ValueError(f"{self.model_name} does not return tokens; use pooling='auto'")
                prefix = int(getattr(self._model, "num_prefix_tokens", 1))
                cls, patches = tokens[:, 0], tokens[:, prefix:]
                f = {"cls": cls, "mean": patches.mean(1), "cls+mean": torch.cat([cls, patches.mean(1)], dim=-1)}[self.pooling]
            f = f.float()
        if self.normalize:
            f = torch.nn.functional.normalize(f, dim=-1)
        return f.cpu().numpy()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_token"] = None
        return state

    def get_config(self) -> Dict[str, Any]:
        return dict(
            model_name=self.model_name,
            pooling=self.pooling,
            timm_kwargs=self.timm_kwargs,
            pretrained=self.pretrained,
            normalize=self.normalize,
            device=self.device,
            batch_size=self.batch_size,
        )
