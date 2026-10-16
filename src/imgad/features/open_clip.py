"""Extractor for OpenCLIP checkpoints (many LAION / DataComp / EVA CLIP models on the HF Hub are
published in open_clip format, e.g. ``hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K``).

Requires ``pip install open_clip_torch``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..hub import raise_if_hub_error, token_env
from ..utils import resolve_device
from .base import FeatureExtractor, register_extractor


@register_extractor("open_clip")
class OpenCLIPFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        model_name: str,
        pretrained: Optional[str] = None,
        normalize: bool = True,
        device: Optional[str] = None,
        batch_size: int = 32,
        token: Union[None, bool, str] = None,
    ):
        self.model_name = model_name
        self._token = token  # never part of get_config()
        self.pretrained = pretrained
        self.normalize = normalize
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._model = None
        self._preprocess = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import open_clip
        except ImportError as e:  # pragma: no cover
            raise ImportError("OpenCLIPFeatureExtractor requires `pip install open_clip_torch`.") from e
        try:
            with token_env(self._token):  # open_clip has no token argument
                model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
        except Exception as e:
            if self.model_name.startswith("hf-hub:"):
                raise_if_hub_error(e, self.model_name, self._token)
            raise
        self._model = model.eval().to(self.device)
        self._preprocess = preprocess

    def _extract_batch(self, images: List) -> np.ndarray:
        import torch

        self._load()
        x = torch.stack([self._preprocess(im) for im in images]).to(self.device)
        with torch.inference_mode():
            f = self._model.encode_image(x).float()
        if self.normalize:
            f = torch.nn.functional.normalize(f, dim=-1)
        return f.cpu().numpy()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_token"] = None  # pickles (e.g. sent to worker processes) never carry the access token
        return state

    def get_config(self) -> Dict[str, Any]:
        return dict(
            model_name=self.model_name,
            pretrained=self.pretrained,
            normalize=self.normalize,
            device=self.device,
            batch_size=self.batch_size,
        )
