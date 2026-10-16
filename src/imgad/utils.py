"""Small shared helpers: image loading, batching, device selection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Union

import numpy as np
from PIL import Image

ImageLike = Union[str, os.PathLike, Image.Image, np.ndarray]


def load_image(img: ImageLike) -> Image.Image:
    """Return an RGB PIL image from a path, PIL image or HxW(xC) numpy array."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, (str, os.PathLike)):
        with Image.open(img) as im:
            return im.convert("RGB")
    if isinstance(img, np.ndarray):
        arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)!r}")


def batched(items: Sequence, batch_size: int) -> Iterator[Sequence]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def resolve_device(device: str | None = None) -> str:
    if device is not None:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


def as_list(images: Union[ImageLike, Iterable[ImageLike]]) -> List[ImageLike]:
    if isinstance(images, (str, os.PathLike, Image.Image)):
        return [images]
    if isinstance(images, np.ndarray) and images.ndim in (2, 3) and images.dtype == np.uint8:
        return [images]
    return list(images)


def is_path(x) -> bool:
    return isinstance(x, (str, os.PathLike)) and Path(x).exists()
