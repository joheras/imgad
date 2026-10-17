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

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ..hub import GatedModelError, hub_error, is_cached, pop_token_kwargs, raise_if_hub_error
from ..utils import resolve_device
from .base import FeatureExtractor, register_extractor

logger = logging.getLogger("imgad.features")
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
        token: Union[None, bool, str] = None,
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
        # access token for gated/private models: kept out of get_config(), so it is never saved
        tok = pop_token_kwargs(self.hf_kwargs)
        self._token = token if token is not None else tok
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------ loading
    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor

        kw = dict(revision=self.revision, trust_remote_code=self.trust_remote_code, **self.hf_kwargs)
        if self._token is not None:
            kw["token"] = self._token
        torch_dtype = getattr(torch, self.dtype) if isinstance(self.dtype, str) else self.dtype
        load_kw = dict(kw)
        if torch_dtype is not None:
            import transformers
            from packaging.version import Version

            key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
            load_kw[key] = torch_dtype
        model = self._from_pretrained(AutoModel, load_kw)
        model.eval().to(self.device)
        try:
            processor = self._from_pretrained(AutoImageProcessor, kw)
        except GatedModelError:
            raise
        except Exception:
            processor = self._from_pretrained(AutoProcessor, kw)
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

    def _from_pretrained(self, auto_cls, kw: Dict[str, Any]):
        """``auto_cls.from_pretrained`` with clear errors for gated/private models. Without access but with
        the files in the local cache, it loads them without contacting the Hub (otherwise optional files
        that are not cached trigger a 401 and abort the load)."""
        try:
            return auto_cls.from_pretrained(self.model_name, **kw)
        except Exception as e:
            if not (hub_error(e) and not kw.get("local_files_only") and is_cached(self.model_name)):
                raise_if_hub_error(e, self.model_name, self._token)
                raise
        if not getattr(self, "_warned_cache", False):
            logger.warning(f"{self.model_name}: no access with the current credentials; using the local Hugging Face cache")
            self._warned_cache = True
        return auto_cls.from_pretrained(self.model_name, **{**kw, "local_files_only": True})

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

    # ------------------------------------------------------------------ text (zero-shot)
    def has_text_encoder(self) -> bool:
        """True for dual encoders (CLIP, SigLIP...) used with ``pooling="projection"`` (or ``"auto"``), whose
        image features share a space with text."""
        self._load()
        return self._pooling == "projection" and hasattr(self._model, "get_text_features")

    def encode_text(self, prompts: List[str]) -> np.ndarray:
        """L2-normalised text embeddings in the same space as the image features."""
        import torch
        from transformers import AutoProcessor, AutoTokenizer

        if not self.has_text_encoder():
            raise TypeError(f"{self.model_name} (pooling={self.pooling!r}) has no text encoder aligned with its "
                            "image features; zero-shot detection needs a CLIP/SigLIP-style model with "
                            "pooling='projection'.")
        if getattr(self, "_tokenizer", None) is None:
            kw = dict(revision=self.revision, trust_remote_code=self.trust_remote_code)
            if self._token is not None:
                kw["token"] = self._token
            try:
                tok = self._from_pretrained(AutoTokenizer, kw)
            except GatedModelError:
                raise
            except Exception:
                tok = getattr(self._from_pretrained(AutoProcessor, kw), "tokenizer")
            self._tokenizer = tok
        cfg = self._model.config
        tkw: Dict[str, Any] = dict(padding=True, truncation=True, return_tensors="pt")
        if getattr(cfg, "model_type", "") in ("siglip", "siglip2"):  # trained with fixed-length padding
            tkw.update(padding="max_length", max_length=getattr(cfg.text_config, "max_position_embeddings", 64))
        inputs = self._tokenizer(list(prompts), **tkw).to(self.device)
        with torch.inference_mode():
            t = _to_tensor(self._model.get_text_features(**inputs), text=True).float()
        return torch.nn.functional.normalize(t, dim=-1).cpu().numpy()

    @property
    def logit_scale(self) -> float:
        self._load()
        s = getattr(self._model, "logit_scale", None)
        return float(s.detach().exp()) if s is not None else 100.0

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_token"] = None  # pickles (e.g. sent to worker processes) never carry the access token
        state["_tokenizer"] = None
        return state

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


def _to_tensor(out, text: bool = False):
    """``get_image_features`` / ``get_text_features`` return a tensor in transformers 4.x and may return a
    ModelOutput in 5.x."""
    import torch

    if isinstance(out, torch.Tensor):
        return out
    for attr in ("text_embeds" if text else "image_embeds", "pooler_output"):
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
