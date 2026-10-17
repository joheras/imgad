"""Small shared helpers: image loading, batching, device selection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Union

import numpy as np
from PIL import Image

ImageLike = Union[str, os.PathLike, Image.Image, np.ndarray]


def load_image(img: ImageLike, intensity=None) -> Image.Image:
    """Return an RGB 8-bit PIL image from a path (raster, DICOM, NIfTI slice reference), PIL image or
    HxW(xC) numpy array. Images with more than 8 bits are mapped with the ``intensity`` rule (see
    :mod:`imgad.imageio`; default ``"auto"``); 8-bit images are returned unchanged."""
    from .imageio import read_image

    return read_image(img, intensity)


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


def is_single_array(x) -> bool:
    """One image given as an array: HxW, or HxWxC with 1, 3 or 4 channels (any dtype, e.g. 16-bit)."""
    return isinstance(x, np.ndarray) and (x.ndim == 2 or (x.ndim == 3 and x.shape[-1] in (1, 3, 4)))


def as_list(images: Union[ImageLike, Iterable[ImageLike]]) -> List[ImageLike]:
    if isinstance(images, (str, os.PathLike, Image.Image)) or is_single_array(images):
        return [images]
    return list(images)


def is_path(x) -> bool:
    from .imageio import exists

    return isinstance(x, (str, os.PathLike)) and exists(x)
