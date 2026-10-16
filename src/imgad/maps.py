"""Helpers for pixel-level anomaly maps: resizing, ground-truth masks and heatmap overlays."""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

from .utils import ImageLike, load_image

MaskLike = Union[None, str, os.PathLike, np.ndarray]


def resize_map(m: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a float map to ``size = (height, width)``."""
    if tuple(m.shape) == tuple(size):
        return m
    im = Image.fromarray(np.ascontiguousarray(m, dtype=np.float32))  # mode "F"
    return np.asarray(im.resize((size[1], size[0]), Image.BILINEAR), dtype=np.float32)


def load_mask(mask: MaskLike, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Boolean ground-truth mask from a path or array (``None`` = no defect, all False).
    Resized with nearest neighbour to ``size = (height, width)`` when given."""
    if mask is None:
        if size is None:
            raise ValueError("size is required for an empty (None) mask")
        return np.zeros(size, dtype=bool)
    if isinstance(mask, (str, os.PathLike)):
        with Image.open(mask) as im:
            arr = np.asarray(im.convert("L"))
    else:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr.max(axis=-1)
    if size is not None and tuple(arr.shape) != tuple(size):
        im = Image.fromarray((arr > 0).astype(np.uint8) * 255)
        arr = np.asarray(im.resize((size[1], size[0]), Image.NEAREST))
    return arr > 0


def colorize(m: np.ndarray, low: float, threshold: float) -> np.ndarray:
    """Map scores to a jet-like RGB uint8 image with a fixed scale: ``low`` -> blue,
    ``threshold`` -> mid (green/yellow), ``2*threshold - low`` or more -> red. Using a fixed scale
    (instead of per-image min/max) keeps normal images blue."""
    v = np.clip(0.5 * (m - low) / max(threshold - low, 1e-12), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(image: ImageLike, m: np.ndarray, low: float, threshold: float, alpha: float = 0.5) -> Image.Image:
    """Heatmap of ``m`` blended over ``image`` (map is resized to the image size)."""
    img = load_image(image)
    m = resize_map(m, (img.height, img.width))
    heat = Image.fromarray(colorize(m, low, threshold))
    return Image.blend(img, heat, alpha)
