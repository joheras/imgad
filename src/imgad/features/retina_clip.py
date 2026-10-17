"""Retinal vision–language models with a FLAIR-style architecture: FLAIR and RetiZero.

Both pair an image encoder with a Bio_ClinicalBERT text encoder and two linear projections into a 512-d joint
space, trained contrastively on fundus photographs and expert text. :class:`RetinaCLIPExtractor` gives the
joint image embeddings (for any detector) and encodes text (for the zero-shot detector,
:mod:`imgad.detectors.zero_shot`)::

    ext = build_extractor({"type": "retina_clip", "model": "flair"})   # or "zoo:flair-clip"
    ext.encode_text(["A fundus photograph of glaucoma"])

The weights come from the authors' checkpoints (``hf-hub:jusiro2/FLAIR``; RetiZero from Google Drive) and are
read without executing code (see :mod:`imgad.features.timm_models`). Preprocessing follows each model's code:
FLAIR scales the longest side to 512 px, pads with black and uses intensities in [0, 1]; RetiZero resizes to
224×224 with ImageNet normalisation.

Text embeddings are the mean of the token states of BERT layers 1, 2 and 12, as in the authors' code. Each
prompt is encoded on its own, so the result does not depend on the padding of a batch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..utils import resolve_device
from .base import FeatureExtractor, register_extractor
from .timm_models import checkpoint_file, load_checkpoint, read_state_dict

_IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
RECIPES: Dict[str, Dict[str, Any]] = {
    "flair": dict(
        checkpoint="hf-hub:jusiro2/FLAIR/model.safetensors", arch="resnet50", prefix="vision_model.model.",
        pooling="avg", size=512, letterbox=True, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0),
        template="A fundus photograph of {}",
    ),
    "retizero": dict(
        checkpoint="gdrive:14bMmnefO73_NL1Xc4x0A5qFNbuI7GqKM/retizero.pth", arch="vit_large_patch16_224",
        prefix="vision_model.model.lora_vit.", pooling="cls", size=224, letterbox=False,
        mean=_IMAGENET[0], std=_IMAGENET[1], template="A fundus photograph of {}",
    ),
}
BERT = "emilyalsentzer/Bio_ClinicalBERT"


@register_extractor("retina_clip")
class RetinaCLIPExtractor(FeatureExtractor):
    def __init__(self, model: str = "flair", device: Optional[str] = None, batch_size: int = 16,
                 checkpoint: Optional[str] = None):
        if model not in RECIPES:
            raise ValueError(f"model must be one of {sorted(RECIPES)}, got {model!r}")
        self.model = model
        self.model_name = f"{model}-clip"
        self.checkpoint = checkpoint  # local copy of the weights, if the default download is not possible
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._vision = self._text = self._tokenizer = None

    # ------------------------------------------------------------------ loading
    def _load(self):
        if self._vision is not None:
            return
        import timm
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        r = RECIPES[self.model]
        ckpt = self.checkpoint or r["checkpoint"]
        state = read_state_dict(checkpoint_file(ckpt))
        vision = timm.create_model(r["arch"], pretrained=False, num_classes=0)
        missing = load_checkpoint(vision, ckpt, prefix=r["prefix"], state=state)
        if missing:
            raise ValueError(f"{self.model_name}: image-encoder weights missing from {ckpt}: {missing[:5]}")
        self._vision = vision.eval().to(self.device)
        self._proj_v = state["vision_model.projection_head_vision.projection.weight"].float().to(self.device)
        self._proj_t = state["text_model.projection_head_text.projection.weight"].float().to(self.device)
        self._logit_scale = float(torch.clamp(state["logit_scale"].float(), 0, 4.6052).exp())

        text = AutoModel.from_config(AutoConfig.from_pretrained(BERT))
        tstate = {k[len("text_model.model."):]: v for k, v in state.items() if k.startswith("text_model.model.")}
        res = text.load_state_dict(tstate, strict=False)
        missing_t = [k for k in res.missing_keys if not k.endswith("position_ids")]
        if missing_t:
            raise ValueError(f"{self.model_name}: text-encoder weights missing from {ckpt}: {missing_t[:5]}")
        self._text = text.eval().to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(BERT)
        self._tokenizer.model_max_length = 77

    def _preprocess(self, im):
        import torch
        import torchvision.transforms.functional as F

        r = RECIPES[self.model]
        if r["letterbox"]:  # FLAIR: longest side to `size`, black padding right/bottom
            w, h = im.size
            scale = r["size"] / max(w, h)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), resample=2)
            x = F.to_tensor(im)
            x = torch.nn.functional.pad(x, (0, r["size"] - x.shape[2], 0, r["size"] - x.shape[1]))
        else:
            x = F.to_tensor(im.resize((r["size"], r["size"]), resample=2))
        return F.normalize(x, r["mean"], r["std"])

    # ------------------------------------------------------------------ images
    def _extract_batch(self, images: List) -> np.ndarray:
        import torch

        self._load()
        x = torch.stack([self._preprocess(im) for im in images]).to(self.device)
        with torch.inference_mode():
            if RECIPES[self.model]["pooling"] == "cls":
                f = self._vision.forward_features(x)[:, 0]
            else:
                f = self._vision(x)
            f = torch.nn.functional.normalize(f.float(), dim=-1) @ self._proj_v.T
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()

    # ------------------------------------------------------------------ text (zero-shot)
    def has_text_encoder(self) -> bool:
        return True

    @property
    def default_template(self) -> str:
        return RECIPES[self.model]["template"]

    def encode_text(self, prompts: List[str]) -> np.ndarray:
        import torch

        self._load()
        out = []
        with torch.inference_mode():
            for p in prompts:
                tok = self._tokenizer([p], truncation=True, return_tensors="pt").to(self.device)
                hs = self._text(**tok, output_hidden_states=True).hidden_states
                e = torch.stack([hs[1], hs[2], hs[-1]]).mean(dim=(0, 2))  # layers 1, 2, last; mean over tokens
                e = torch.nn.functional.normalize(e.float(), dim=-1) @ self._proj_t.T
                out.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.cat(out).cpu().numpy()

    @property
    def logit_scale(self) -> float:
        self._load()
        return self._logit_scale

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_vision=None, _text=None, _tokenizer=None)  # rebuilt lazily in worker processes
        return state

    def get_config(self) -> Dict[str, Any]:
        return dict(model=self.model, device=self.device, batch_size=self.batch_size, checkpoint=self.checkpoint)
