"""Extractor for timm models, including Hub checkpoints such as the pathology foundation models UNI,
UNI2-h, Virchow2, Prov-GigaPath or H-optimus-0 (``hf-hub:org/name``).

    TimmFeatureExtractor("hf-hub:MahmoodLab/uni", timm_kwargs={"init_values": 1e-5, "dynamic_img_size": True})
    TimmFeatureExtractor("hf-hub:paige-ai/Virchow2", pooling="cls+mean",
                         timm_kwargs={"mlp_layer": "timm.layers.SwiGLUPacked", "act_layer": "torch.nn.SiLU"})

``timm_kwargs`` are passed to ``timm.create_model``. Values that are classes (layers, activations) are given
as dotted import paths (``"torch.nn.SiLU"``) so that the configuration stays serialisable (saved pipelines,
YAML, feature-cache keys). The ready-made recipes for medical models are in :mod:`imgad.zoo`.

Checkpoints that are not timm models on the Hub (raw ``.pth`` state dicts such as RETFound, UrFound or FLAIR) are
loaded into a timm architecture with ``checkpoint``::

    TimmFeatureExtractor("vit_large_patch16_224", pooling="mean",
                         checkpoint="hf-hub:YukunZhou/RETFound_mae_natureCFP/RETFound_mae_natureCFP.pth",
                         checkpoint_key="model", data_config={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})

``checkpoint`` is ``hf-hub:org/repo/path/in/repo`` (downloaded with the Hub token, so gated repos work), an
``https://`` URL or ``gdrive:<file id>/<name>`` for a public Google Drive file (both cached under the torch hub
directory), or a local path. ``checkpoint_key`` selects a nested state
dict (``"model"``, ``"teacher"``…; by default the usual ones are tried), ``checkpoint_prefix`` is stripped from the
keys (``"module."`` and ``"backbone."`` always are). Heads, decoders and keys with another shape are ignored;
position embeddings are resampled to the model's size, and LoRA adapters on ``qkv`` (``linear_a_q`` /
``linear_b_q`` / ``linear_a_v`` / ``linear_b_v``, as in RetiZero) are merged into the weights. Checkpoints are read with ``torch.load(weights_only=True)``:
no code from the file is executed. ``data_config`` overrides the preprocessing (``mean``, ``std``,
``input_size``, ``crop_pct``, ``interpolation``), needed when the checkpoint was trained with other constants
than the timm architecture's defaults.

Pooling: ``"auto"`` = the model's own pooled output (``model(x)``; for models that return tokens, the class
token); ``"cls"``; ``"mean"`` (patch tokens, without class/register tokens); ``"cls+mean"`` (concatenation).

Requires ``pip install timm``.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..hub import raise_if_hub_error, token_env
from ..utils import resolve_device
from .base import FeatureExtractor, register_extractor

_POOLINGS = {"auto", "cls", "mean", "cls+mean"}
_NESTED_KEYS = ("state_dict", "model", "teacher", "module", "student")
_SKIP_PREFIXES = ("head.", "fc.", "decoder", "mask_token", "dino_head", "ibot_head", "projection_head", "fc_norm.")
logger = logging.getLogger("imgad")


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
        checkpoint: Optional[str] = None,
        checkpoint_key: Optional[str] = None,
        checkpoint_prefix: Optional[str] = None,
        data_config: Optional[Dict[str, Any]] = None,
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
        self.checkpoint = checkpoint
        self.checkpoint_key = checkpoint_key
        self.checkpoint_prefix = checkpoint_prefix
        self.data_config = dict(data_config or {})
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
                model = timm.create_model(self.model_name, pretrained=self.pretrained and not self.checkpoint, **kwargs)
        except Exception as e:
            if self.model_name.startswith(("hf-hub:", "hf_hub:")):
                raise_if_hub_error(e, self.model_name.replace("hf_hub:", "hf-hub:"), self._token)
            raise
        if self.checkpoint:
            load_checkpoint(model, self.checkpoint, key=self.checkpoint_key, prefix=self.checkpoint_prefix,
                            token=self._token)
        self._model = model.eval().to(self.device)
        cfg = resolve_data_config(model.pretrained_cfg, model=model)
        cfg.update({k: tuple(v) if isinstance(v, list) else v for k, v in self.data_config.items()})
        self._transform = create_transform(**cfg, is_training=False)

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
            checkpoint=self.checkpoint,
            checkpoint_key=self.checkpoint_key,
            checkpoint_prefix=self.checkpoint_prefix,
            data_config=self.data_config,
        )


def checkpoint_file(checkpoint: str, token: Union[None, bool, str] = None) -> str:
    """Local path of a checkpoint given as ``hf-hub:org/repo/file``, an ``https://`` URL, ``gdrive:<id>/<name>``
    or a path (downloads are cached)."""
    if checkpoint.startswith(("hf-hub:", "hf_hub:")):
        from huggingface_hub import hf_hub_download

        parts = checkpoint.split(":", 1)[1].split("/")
        if len(parts) < 3:
            raise ValueError(f"checkpoint {checkpoint!r} must be hf-hub:org/repo/path/to/file")
        repo_id, filename = "/".join(parts[:2]), "/".join(parts[2:])
        try:
            return hf_hub_download(repo_id, filename, token=token)
        except Exception as e:
            raise_if_hub_error(e, repo_id, token)
            raise
    if checkpoint.startswith("gdrive:"):
        file_id, _, name = checkpoint[len("gdrive:"):].partition("/")
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        return _download(url, name or f"gdrive-{file_id}.pth", checkpoint)
    if checkpoint.startswith(("https://", "http://")):
        return _download(checkpoint, checkpoint.rstrip("/").rsplit("/", 1)[-1], checkpoint)
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    return str(path)


def _download(url: str, name: str, label: str) -> str:
    import torch

    target = Path(torch.hub.get_dir()) / "checkpoints" / name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {label}")
        torch.hub.download_url_to_file(url, str(target), progress=False)
        with open(target, "rb") as f:
            head = f.read(64).lstrip().lower()
        if head.startswith((b"<!doctype", b"<html")):  # an error / sign-in / quota page instead of the file
            target.unlink()
            raise RuntimeError(f"{label} returned a web page instead of a checkpoint (file not public, moved, "
                               "or download quota exceeded). Download it manually and pass the local path.")
    return str(target)


def _safe_globals() -> list:
    """Non-tensor objects that training checkpoints commonly store next to the weights (argparse namespace,
    numpy scalars); allowed in ``weights_only`` loading because they cannot run code."""
    import numpy as np

    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # numpy < 2
        from numpy.core.multiarray import scalar
    dtypes = [type(np.dtype(t)) for t in (np.float64, np.float32, np.float16, np.int64, np.int32, np.bool_)]
    return [argparse.Namespace, scalar, (scalar, "numpy.core.multiarray.scalar"), np.dtype, *dtypes]




def read_state_dict(path: str, key: Optional[str] = None) -> Dict[str, Any]:
    """Tensors of a ``.safetensors`` / ``.pth`` / ``.bin`` file, without executing code from it."""
    import torch

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        obj = load_file(path)
    else:
        with torch.serialization.safe_globals(_safe_globals()):
            obj = torch.load(path, map_location="cpu", weights_only=True)
    if key:
        if key not in obj:
            raise KeyError(f"checkpoint has no {key!r} entry (entries: {', '.join(map(str, obj))})")
        obj = obj[key]
    else:
        while isinstance(obj, dict) and not any(isinstance(v, torch.Tensor) for v in obj.values()):
            nested = next((k for k in _NESTED_KEYS if isinstance(obj.get(k), dict)), None)
            if nested is None:
                break
            obj = obj[nested]
    return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}


def load_checkpoint(model, checkpoint: str, key: Optional[str] = None, prefix: Optional[str] = None,
                    token: Union[None, bool, str] = None) -> List[str]:
    """Load ``checkpoint`` into a timm ``model`` (non-strict, see the module docs); returns the model keys that
    stayed uninitialised. Raises if almost nothing matched (wrong architecture, key or prefix)."""
    state = read_state_dict(checkpoint_file(checkpoint, token), key)
    if prefix:  # explicit: keep only that sub-model (e.g. the vision tower of a vision-language checkpoint)
        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    for p in ("module.", "backbone."):
        if sum(k.startswith(p) for k in state) > len(state) / 2:
            state = {k[len(p):]: v for k, v in state.items() if k.startswith(p)}
    state = _merge_lora(state)
    own = model.state_dict()
    if "pos_embed" in state and "pos_embed" in own and state["pos_embed"].shape != own["pos_embed"].shape:
        from timm.layers import resample_abs_pos_embed

        n_prefix = 0 if getattr(model, "no_embed_class", False) else int(getattr(model, "num_prefix_tokens", 1))
        grid = model.patch_embed.grid_size
        state["pos_embed"] = resample_abs_pos_embed(state["pos_embed"], list(grid), num_prefix_tokens=n_prefix)
    usable = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    wanted = [k for k in own if not k.startswith(_SKIP_PREFIXES) and not k.endswith("num_batches_tracked")]
    missing = [k for k in wanted if k not in usable]
    if len(usable) < 0.5 * len(wanted):
        raise ValueError(f"checkpoint {checkpoint} does not match {type(model).__name__}: {len(usable)} of "
                         f"{len(wanted)} weights found (first missing: {missing[:5]}). Check the architecture, "
                         "checkpoint_key and checkpoint_prefix.")
    model.load_state_dict(usable, strict=False)
    if missing:
        logger.warning(f"checkpoint {checkpoint}: {len(missing)} weight(s) not in the file and left as initialised: "
                       f"{', '.join(missing[:8])}{'…' if len(missing) > 8 else ''}")
    return missing


def _merge_lora(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fold LoRA_ViT adapters (``attn.qkv.qkv`` + ``linear_{a,b}_{q,v}``) into plain ``attn.qkv`` weights."""
    lora = [k[: -len("qkv.qkv.weight")] for k in state if k.endswith("attn.qkv.qkv.weight")]
    if not lora:
        return state
    state = dict(state)
    for base in lora:
        w = state.pop(base + "qkv.qkv.weight").clone()
        dim = w.shape[1]
        for part, rows in (("q", slice(0, dim)), ("v", slice(-dim, None))):
            a, b = state.pop(f"{base}qkv.linear_a_{part}.weight", None), state.pop(f"{base}qkv.linear_b_{part}.weight", None)
            if a is not None and b is not None:
                w[rows] += b.float().mm(a.float()).to(w.dtype)
        state[base + "qkv.weight"] = w
        if base + "qkv.qkv.bias" in state:
            state[base + "qkv.bias"] = state.pop(base + "qkv.qkv.bias")
    return state
