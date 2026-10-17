"""Image reading for 8-bit, high-bit-depth (16-bit PNG/TIFF, float) and medical images (DICOM, NIfTI).

Every model in imgad receives 8-bit RGB images. Images with more than 8 bits (X-ray, CT, MRI, 16-bit
industrial cameras...) are mapped to 8 bits with an **intensity rule**:

========================  =========================================================================
``"auto"`` (default)      8-bit images unchanged; DICOM with a window -> that window; anything else
                          with more than 8 bits -> per-image percentiles 0.5-99.5
``"minmax"``              per-image minimum and maximum
``"percentile:1,99"``     per-image percentiles
``"window:-1000,400"``    fixed range (e.g. CT Hounsfield units, or ``window:0,4095`` for a 12-bit
                          detector). The same mapping for every image: recommended for anomaly
                          detection, since per-image rules hide global intensity anomalies
``"dicom"``               the DICOM WindowCenter/WindowWidth of each file (else ``"auto"``)
========================  =========================================================================

Volumes (NIfTI ``.nii`` / ``.nii.gz``, multi-frame DICOM) are read slice by slice through references
``"<path>::<axis>=<index>"`` (``axis`` x/y/z for NIfTI, ``frame`` for DICOM), e.g. ``"ct.nii.gz::z=42"``;
see :func:`imgad.data.volume_slices`. Only pixel data is read: DICOM/NIfTI metadata (patient name, IDs,
dates...) is never stored or logged.

DICOM needs ``pydicom`` and NIfTI ``nibabel`` (``pip install "imgad[medical]"``).
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

IntensityLike = Union[None, str, Dict[str, Any]]
DICOM_EXTENSIONS = (".dcm", ".dicom")
NIFTI_EXTENSIONS = (".nii", ".nii.gz")
_REF = re.compile(r"^(?P<path>.+)::(?P<axis>[xyz]|frame)=(?P<index>\d+)$")


# ----------------------------------------------------------------------------- intensity rules
def parse_intensity(spec: IntensityLike) -> Dict[str, Any]:
    """Normalise an intensity rule: ``"auto"``, ``"minmax"``, ``"dicom"``, ``"percentile:1,99"``,
    ``"window:lo,hi"`` or the dict forms ``{"percentile": [1, 99]}`` / ``{"window": [lo, hi]}``."""
    if spec is None or spec == "auto":
        return {"mode": "auto"}
    if isinstance(spec, dict):
        if "mode" in spec:
            return dict(spec)
        (mode, value), = spec.items()
        return _with_range(mode, value)
    s = str(spec).strip().lower()
    if s in ("minmax", "dicom"):
        return {"mode": s}
    mode, _, rest = s.partition(":")
    if mode in ("percentile", "window") and rest:
        return _with_range(mode, [float(v) for v in rest.split(",")])
    raise ValueError(f"Unknown intensity rule {spec!r}: use auto, minmax, dicom, percentile:lo,hi or window:lo,hi")


def _with_range(mode: str, value) -> Dict[str, Any]:
    if mode not in ("percentile", "window") or len(value) != 2:
        raise ValueError(f"Intensity rule {mode!r} needs two values, got {value!r}")
    lo, hi = float(value[0]), float(value[1])
    if hi <= lo:
        raise ValueError(f"Intensity range must have low < high, got {value!r}")
    if mode == "percentile" and not (0 <= lo < hi <= 100):
        raise ValueError("percentiles must be within [0, 100]")
    return {"mode": mode, "low": lo, "high": hi}


def intensity_str(spec: IntensityLike) -> str:
    """Canonical string of a rule (used in configs, cache keys and reports)."""
    r = parse_intensity(spec)
    if r["mode"] in ("percentile", "window"):
        return f"{r['mode']}:{r['low']:g},{r['high']:g}"
    return r["mode"]


def to_uint8(arr: np.ndarray, spec: IntensityLike = "auto", dicom_window: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Map an image array (any dtype; HxW, HxWxC) to uint8 with the given intensity rule."""
    r = parse_intensity(spec)
    mode = r["mode"]
    if arr.dtype == np.uint8 and mode in ("auto", "dicom") and dicom_window is None:
        return arr
    a = np.asarray(arr, dtype=np.float32)
    if mode == "dicom" or (mode == "auto" and dicom_window is not None):
        if dicom_window is not None:
            lo, hi = dicom_window
        else:
            lo, hi = np.percentile(a, [0.5, 99.5])
    elif mode == "auto":
        if np.issubdtype(arr.dtype, np.floating) and a.size and 0 <= a.min() and a.max() <= 1:
            lo, hi = 0.0, 1.0  # float image in [0, 1]
        else:
            lo, hi = np.percentile(a, [0.5, 99.5])
    elif mode == "minmax":
        lo, hi = float(a.min()), float(a.max())
    elif mode == "percentile":
        lo, hi = np.percentile(a, [r["low"], r["high"]])
    else:  # window
        lo, hi = r["low"], r["high"]
    if hi <= lo:  # constant image
        return np.zeros(a.shape, dtype=np.uint8)
    return (np.clip((a - lo) / (hi - lo), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_rgb(arr8: np.ndarray) -> Image.Image:
    if arr8.ndim == 3 and arr8.shape[-1] == 1:
        arr8 = arr8[..., 0]
    if arr8.ndim == 3 and arr8.shape[-1] == 4:
        arr8 = arr8[..., :3]
    return Image.fromarray(np.ascontiguousarray(arr8)).convert("RGB")


# ----------------------------------------------------------------------------- references to slices
def split_ref(path: Union[str, os.PathLike]) -> Tuple[str, Optional[str], Optional[int]]:
    """``"ct.nii.gz::z=42"`` -> ``("ct.nii.gz", "z", 42)``; plain paths -> ``(path, None, None)``."""
    s = os.fspath(path)
    m = _REF.match(s)
    if m is None:
        return s, None, None
    return m["path"], m["axis"], int(m["index"])


def file_of(path: Union[str, os.PathLike]) -> str:
    """The file behind a path or a slice reference."""
    return split_ref(path)[0]


def is_dicom(path: str) -> bool:
    return path.lower().endswith(DICOM_EXTENSIONS)


def is_nifti(path: str) -> bool:
    return path.lower().endswith(NIFTI_EXTENSIONS)


# ----------------------------------------------------------------------------- readers
def read_image(img, intensity: IntensityLike = "auto") -> Image.Image:
    """RGB 8-bit PIL image from a path (raster, DICOM, NIfTI slice reference), a PIL image or an array."""
    if isinstance(img, Image.Image):
        return _from_pil(img, intensity)
    if isinstance(img, np.ndarray):
        return to_rgb(to_uint8(img, intensity))
    if isinstance(img, (str, os.PathLike)):
        path, axis, index = split_ref(img)
        if is_dicom(path):
            arr, window = _read_dicom(path, index if axis == "frame" else None)
            return to_rgb(to_uint8(arr, intensity, window))
        if is_nifti(path):
            if axis not in ("x", "y", "z"):
                raise ValueError(f"{path} is a volume: refer to a slice, e.g. '{path}::z=0' "
                                 "(see imgad.data.volume_slices)")
            return to_rgb(to_uint8(_nifti_slice(path, axis, index), intensity))
        with Image.open(path) as im:
            return _from_pil(im, intensity)
    raise TypeError(f"Unsupported image type: {type(img)!r}")


def _from_pil(im: Image.Image, intensity: IntensityLike) -> Image.Image:
    if im.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
        return to_rgb(to_uint8(np.asarray(im), intensity))
    if parse_intensity(intensity)["mode"] in ("window", "percentile", "minmax"):
        return to_rgb(to_uint8(np.asarray(im.convert("RGB")), intensity))  # explicit rule: apply it
    return im.convert("RGB")


def _read_dicom(path: str, frame: Optional[int]) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    try:
        import pydicom
    except ImportError as e:  # pragma: no cover
        raise ImportError('Reading DICOM needs pydicom: pip install "imgad[medical]"') from e
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper()
    n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if n_frames > 1:
        if frame is None:
            frame = n_frames // 2
        arr = arr[frame]
    elif frame not in (None, 0):
        raise IndexError(f"{path} has a single frame")
    if photometric.startswith("MONOCHROME"):
        arr = arr.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
        window = _dicom_window(ds)
        if photometric == "MONOCHROME1":  # high values are dark: invert so that the image looks like MONOCHROME2
            arr = -arr
            window = (-window[1], -window[0]) if window else None
        return arr, window
    if photometric != "RGB":
        try:
            from pydicom.pixels import convert_color_space
        except ImportError:  # pragma: no cover - pydicom < 3
            from pydicom.pixel_data_handlers.util import convert_color_space
        arr = convert_color_space(arr, photometric, "RGB")
    return arr, None


def _dicom_window(ds) -> Optional[Tuple[float, float]]:
    c, w = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
    if c is None or w is None:
        return None
    c, w = _first(c), _first(w)
    return (c - w / 2.0, c + w / 2.0) if w > 0 else None


def _first(v) -> float:
    """DICOM values may be multi-valued (several windows): use the first one."""
    try:
        return float(v)
    except TypeError:
        return float(v[0])


@functools.lru_cache(maxsize=4)
def _nifti_volume(path: str, mtime_ns: int) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as e:  # pragma: no cover
        raise ImportError('Reading NIfTI needs nibabel: pip install "imgad[medical]"') from e
    img = nib.as_closest_canonical(nib.load(path))  # RAS orientation: axes x, y, z are comparable across files
    vol = np.asarray(img.dataobj, dtype=np.float32)
    if vol.ndim == 4:  # time series / channels: first volume
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D volume, got shape {vol.shape}")
    return vol


def nifti_shape(path: str) -> Tuple[int, int, int]:
    return _nifti_volume(path, os.stat(path).st_mtime_ns).shape


def _nifti_slice(path: str, axis: str, index: int) -> np.ndarray:
    vol = _nifti_volume(path, os.stat(path).st_mtime_ns)
    ax = "xyz".index(axis)
    if not 0 <= index < vol.shape[ax]:
        raise IndexError(f"{path}: slice {axis}={index} out of range (size {vol.shape[ax]})")
    sl = np.take(vol, index, axis=ax)
    return np.rot90(sl)  # display convention: anterior/superior up


def dicom_frames(path: str) -> int:
    try:
        import pydicom
    except ImportError as e:  # pragma: no cover
        raise ImportError('Reading DICOM needs pydicom: pip install "imgad[medical]"') from e
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    return int(getattr(ds, "NumberOfFrames", 1) or 1)


def exists(path: Union[str, os.PathLike]) -> bool:
    """``Path.exists`` that understands slice references."""
    return Path(file_of(path)).exists()
