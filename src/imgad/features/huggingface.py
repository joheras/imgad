"""Feature extractor for (almost) any vision / vision-language model on the HuggingFace Hub.

Works out of the box with:
  * CLIP-style dual encoders: CLIP, SigLIP, SigLIP2, ALIGN, (Meta)CLIP, EVA-CLIP ports, BLIP ...
    -> uses ``model.get_image_features`` (projected, multimodal embedding space)
  * Pure vision backbones: DINOv2, ViT, BEiT, ConvNeXt, Swin, ResNet, ...
    -> uses pooler output / CLS token / mean of patch tokens

Pooling options:
  * ``"auto"``        -> ``"projection"`` if the model has ``get_image_features`` else ``"pooler"``
  * ``"projection"``  -> ``get_image_features`` (CLIP-like image embedding)
  * ``"pooler"``      -> ``pooler_output`` (falls back to CLS)
  * ``"cls"``         -> first token of the selected hidden layer
  * ``"mean"``        -> mean of patch tokens (excluding CLS) of the selected hidden layer
  * ``"cls+mean"``    -> concatenation of both

``layer`` selects an intermediate hidden state (e.g. ``-2``, or ``[6, 9, 12]`` to concatenate
several layers), which often helps anomaly detection since low/mid-level features retain
more local detail than the final, semantically-aligned layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ..utils import resolve_device
from .base import FeatureExtractor, register_extractor

_POOLINGS = {"auto", "projection", "pooler", "cls", "mean", "cls+mean"}


@register_extractor("hf")
class HFFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        model_name: str,
        pooling: str = "auto",
        layer: Optional[Union[int, Sequence[int]]] = None,
        normalize: bool = True,
        device: Optional[str] = None,
        batch_size: int = 32,
        dtype: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        image_size: Optional[int] = None,
        hf_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if pooling not in _POOLINGS:
            raise ValueError(f"pooling must be one of {_POOLINGS}, got {pooling!r}")
        self.model_name = model_name
        self.pooling = pooling
        self.layer = list(layer) if isinstance(layer, (list, tuple)) else layer
        self.normalize = normalize
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.dtype = dtype
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.image_size = image_size
        self.hf_kwargs = dict(hf_kwargs or {})
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------ loading
    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor

        kw = dict(revision=self.revision, trust_remote_code=self.trust_remote_code, **self.hf_kwargs)
        torch_dtype = getattr(torch, self.dtype) if isinstance(self.dtype, str) else self.dtype
        load_kw = dict(kw)
        if torch_dtype is not None:
            import transformers
            from packaging.version import Version

            key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
            load_kw[key] = torch_dtype
        model = AutoModel.from_pretrained(self.model_name, **load_kw)
        model.eval().to(self.device)
        try:
            processor = AutoImageProcessor.from_pretrained(self.model_name, **kw)
        except Exception:
            processor = AutoProcessor.from_pretrained(self.model_name, **kw)
            processor = getattr(processor, "image_processor", processor)
        self._model, self._processor = model, processor

        pooling = self.pooling
        if pooling == "auto":
            pooling = "projection" if hasattr(model, "get_image_features") else "pooler"
        if pooling == "projection" and not hasattr(model, "get_image_features"):
            raise ValueError(f"{self.model_name} has no get_image_features(); use pooling='pooler'/'cls'/'mean'.")
        if pooling == "projection" and self.layer is not None:
            raise ValueError("`layer` cannot be combined with pooling='projection'.")
        self._pooling = pooling

    @property
    def model(self):
        self._load()
        return self._model

    @property
    def processor(self):
        self._load()
        return self._processor

    def _vision_module(self):
        """For dual encoders (CLIP, SigLIP, ...) return the vision tower, else the model itself."""
        m = self._model
        for attr in ("vision_model", "vision_tower", "visual", "vision_encoder"):
            if hasattr(m, attr) and not callable(getattr(type(m), attr, None)):
                return getattr(m, attr)
        return m

    # ------------------------------------------------------------------ forward
    def _extract_batch(self, images: List) -> np.ndarray:
        import torch

        self._load()
        proc_kw = {}
        if self.image_size is not None:
            proc_kw["size"] = {"height": self.image_size, "width": self.image_size}
        inputs = self._processor(images=images, return_tensors="pt", **proc_kw)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=next(self._model.parameters()).dtype)

        with torch.inference_mode():
            if self._pooling == "projection":
                out = self._model.get_image_features(pixel_values=pixel_values)
                feats = _to_tensor(out)
            else:
                vision = self._vision_module()
                need_hidden = self.layer is not None
                out = vision(pixel_values=pixel_values, output_hidden_states=need_hidden, return_dict=True)
                feats = self._pool(out)

        feats = feats.float()
        if self.normalize:
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.cpu().numpy()

    def _pool(self, out):
        import torch

        if self.layer is None:
            if self._pooling == "pooler":
                po = getattr(out, "pooler_output", None)
                if po is not None:
                    return po.flatten(1)
                return _token_pool(out.last_hidden_state, "cls")
            return _token_pool(out.last_hidden_state, self._pooling)

        layers = self.layer if isinstance(self.layer, list) else [self.layer]
        mode = "cls" if self._pooling == "pooler" else self._pooling
        return torch.cat([_token_pool(out.hidden_states[l], mode) for l in layers], dim=-1)

    def get_config(self) -> Dict[str, Any]:
        return dict(
            model_name=self.model_name,
            pooling=self.pooling,
            layer=self.layer,
            normalize=self.normalize,
            device=self.device,
            batch_size=self.batch_size,
            dtype=self.dtype,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            image_size=self.image_size,
            hf_kwargs=self.hf_kwargs,
        )


def _to_tensor(out):
    """``get_image_features`` returns a tensor in transformers 4.x and may return a ModelOutput in 5.x."""
    import torch

    if isinstance(out, torch.Tensor):
        return out
    for attr in ("image_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    return out[0]


def _token_pool(hidden, mode: str):
    import torch

    if hidden.ndim == 4:  # CNN feature map (B, C, H, W)
        return hidden.mean(dim=(2, 3))
    if mode == "cls":
        return hidden[:, 0]
    if mode == "mean":
        return hidden[:, 1:].mean(dim=1) if hidden.shape[1] > 1 else hidden[:, 0]
    if mode == "cls+mean":
        return torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=-1)
    raise ValueError(mode)
