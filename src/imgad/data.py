"""Dataset helpers. Datasets are represented as plain lists of image paths + labels
(0 = normal, 1 = anomaly) so they plug directly into extractors and pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".dcm", ".dicom")
PathLike = Union[str, Path]


def list_images(root: PathLike, recursive: bool = True, extensions: Sequence[str] = IMG_EXTENSIONS) -> List[str]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    it = root.rglob("*") if recursive else root.glob("*")
    exts = tuple(e.lower() for e in extensions)
    return sorted(str(p) for p in it if p.is_file() and p.suffix.lower() in exts)


@dataclass
class AnomalyDataset:
    """Train (normal only) / test (normal + anomalous) split."""

    train_paths: List[str]
    test_paths: List[str]
    test_labels: np.ndarray
    test_types: List[str] = field(default_factory=list)  # e.g. defect type per test image
    name: str = "dataset"
    test_masks: List[Optional[str]] = field(default_factory=list)  # ground-truth mask per test image (None = no defect)

    @property
    def has_masks(self) -> bool:
        return len(self.test_masks) == len(self.test_paths) and any(m is not None for m in self.test_masks)

    def __post_init__(self):
        self.test_labels = np.asarray(self.test_labels, dtype=int)
        if not self.test_types:
            self.test_types = ["anomaly" if l else "good" for l in self.test_labels]

    def summary(self) -> str:
        n_anom = int(self.test_labels.sum())
        return (
            f"{self.name}: train={len(self.train_paths)} normal | test={len(self.test_paths)} "
            f"({len(self.test_paths) - n_anom} normal, {n_anom} anomalous)"
        )


def from_folders(
    train_normal: PathLike,
    test_normal: PathLike,
    test_anomalous: Union[PathLike, Sequence[PathLike]],
    name: str = "folders",
) -> AnomalyDataset:
    """Generic layout: one folder of normal training images, one of normal test images and one
    or more folders of anomalous test images."""
    if isinstance(test_anomalous, (str, Path)):
        test_anomalous = [test_anomalous]
    tn = list_images(test_normal)
    test, labels, types = list(tn), [0] * len(tn), ["good"] * len(tn)
    for folder in test_anomalous:
        imgs = list_images(folder)
        test += imgs
        labels += [1] * len(imgs)
        types += [Path(folder).name] * len(imgs)
    return AnomalyDataset(list_images(train_normal), test, np.array(labels), types, name)


def mvtec_ad(root: PathLike, category: str, normal_name: str = "good") -> AnomalyDataset:
    """MVTec AD / MVTec LOCO / VisA-in-MVTec-format layout::

        root/<category>/train/good/*.png
        root/<category>/test/good/*.png
        root/<category>/test/<defect_type>/*.png
        root/<category>/ground_truth/<defect_type>/<name>_mask.png   (optional, for pixel metrics)
    """
    base = Path(root) / category
    train = list_images(base / "train" / normal_name)
    test, labels, types, masks = [], [], [], []
    for sub in sorted(p for p in (base / "test").iterdir() if p.is_dir()):
        imgs = list_images(sub)
        test += imgs
        labels += [0 if sub.name == normal_name else 1] * len(imgs)
        types += [sub.name] * len(imgs)
        masks += [None if sub.name == normal_name else _find_mask(base / "ground_truth" / sub.name, p) for p in imgs]
    return AnomalyDataset(train, test, np.array(labels), types, f"mvtec/{category}", masks)


def _find_mask(folder: Path, image: str) -> Optional[str]:
    stem = Path(image).stem
    for name in (f"{stem}_mask.png", f"{stem}.png"):
        if (folder / name).exists():
            return str(folder / name)
    return None


def volume_slices(
    volumes: Union[PathLike, Sequence[PathLike]],
    axis: str = "z",
    step: int = 1,
    middle_only: bool = False,
    skip_empty: bool = True,
) -> List[str]:
    """Slice references (``"vol.nii.gz::z=42"``) for NIfTI volumes or multi-frame DICOM files; every imgad
    function accepts them as image paths. ``volumes``: files and/or folders (searched recursively).

    ``axis`` x / y / z (NIfTI, RAS orientation; ignored for DICOM, which uses its frames), ``step`` keeps
    every n-th slice, ``middle_only`` one slice per volume, ``skip_empty`` drops constant slices (e.g. the
    background around a skull-stripped brain).

    Scores are per slice: aggregate them per volume (e.g. the maximum) for a volume-level decision.
    """
    from .imageio import _nifti_slice, dicom_frames, is_dicom, is_nifti, nifti_shape

    if isinstance(volumes, (str, Path)):
        volumes = [volumes]
    files: List[str] = []
    for v in volumes:
        v = Path(v)
        if v.is_dir():
            files += sorted(str(p) for p in v.rglob("*") if p.is_file() and (is_nifti(str(p)) or is_dicom(str(p))))
        else:
            files.append(str(v))
    refs = []
    for f in files:
        if is_nifti(f):
            n, key = nifti_shape(f)["xyz".index(axis)], axis
        elif is_dicom(f):
            n, key = dicom_frames(f), "frame"
        else:
            raise ValueError(f"Not a volume (NIfTI or DICOM): {f}")
        idx = [n // 2] if middle_only else list(range(0, n, step))
        for i in idx:
            ref = f"{f}::{key}={i}"
            if skip_empty and key != "frame" and not middle_only:
                arr = _nifti_slice(f, axis, i)  # cheap: the volume is cached
                if float(arr.max()) == float(arr.min()):
                    continue
            refs.append(ref)
    return refs


def mvtec_categories(root: PathLike) -> List[str]:
    return sorted(p.name for p in Path(root).iterdir() if (p / "train").is_dir())


def one_vs_rest(
    root: PathLike,
    normal_class: str,
    train_fraction: float = 0.8,
    max_anomalies: Optional[int] = None,
    seed: int = 0,
) -> AnomalyDataset:
    """Semantic AD from a classification folder (``root/<class>/*.jpg``): one class is normal,
    all others are anomalies. Normal images are split train/test."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    normal = np.array(list_images(root / normal_class))
    rng.shuffle(normal)
    n_tr = int(len(normal) * train_fraction)
    train, test_n = list(normal[:n_tr]), list(normal[n_tr:])
    anom, types = [], []
    for c in classes:
        if c == normal_class:
            continue
        imgs = list_images(root / c)
        anom += imgs
        types += [c] * len(imgs)
    if max_anomalies is not None and len(anom) > max_anomalies:
        idx = rng.choice(len(anom), max_anomalies, replace=False)
        anom, types = [anom[i] for i in idx], [types[i] for i in idx]
    test = test_n + anom
    labels = np.array([0] * len(test_n) + [1] * len(anom))
    return AnomalyDataset(train, test, labels, ["good"] * len(test_n) + types, f"{root.name}/{normal_class}")
