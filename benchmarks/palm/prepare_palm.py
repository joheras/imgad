"""Lay out the PALM pathologic-myopia dataset for an imgad one-class benchmark.

PALM (Fang et al., Sci. Data 11, 99, 2024; https://doi.org/10.6084/m9.figshare.c.6224616.v1) has 1,200
colour fundus photographs split by patient into training, validation and test sets of 400 images, each
labelled 1 = pathologic myopia (PM) or 0 = no PM. This script reads the release as downloaded (the
Figshare release with ``Classification Labels.xlsx`` per set, or the 2019 challenge release whose
training images are named ``P*``/``N*``/``H*``) and writes it in the MVTec layout that
``data.type: mvtec`` reads, one "category" per evaluation:

    <out>/palm-val/train/good/      training non-PM images   (normal data used to fit every model)
    <out>/palm-val/test/good/       validation non-PM images
    <out>/palm-val/test/pm/         validation PM images      -> choose the configuration here
    <out>/palm-test/train/good/     same normal training images
    <out>/palm-test/test/good/      test non-PM images
    <out>/palm-test/test/pm/        test PM images            -> report the chosen configuration here
    <out>/palm-lesion/...           (--lesion-masks) test PM images that have lesion masks, with
                                    ground_truth/lesion/<name>_mask.png for pixel-level metrics
    <out>/manifest.csv              one row per image: split, file, label, device, centring, mask

Normal = no PM (this includes high myopia without PM); anomalous = PM. Images are symlinked by default
(``--copy`` copies them). ``--crop-fov`` writes PNG copies cropped to the circular field of view, with masks
cropped identically.

Usage:
    python benchmarks/palm/prepare_palm.py --palm /data/PALM --out /data/palm_imgad [--lesion-masks] [--crop-fov]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
EXPECTED = {"train": 400, "val": 400, "test": 400}  # Sci. Data, Table 1


def split_of(path: Path) -> Optional[str]:
    """Split named by any folder of the path ('Training set', 'PALM-Validation400', 'test'...)."""
    for part in reversed([p.lower() for p in path.parts]):
        if "train" in part:
            return "train"
        if "valid" in part or re.fullmatch(r"val(\W.*)?", part):
            return "val"
        if "test" in part:
            return "test"
    return None


def is_mask_path(path: Path) -> bool:
    return any(k in part.lower() for part in path.parts for k in ("mask", "disc", "lesion", "atrophy", "detachment"))


def find_images(root: Path) -> Dict[str, Dict[str, Path]]:
    """{split: {file name: path}} for fundus images (masks excluded)."""
    out: Dict[str, Dict[str, Path]] = {s: {} for s in SPLITS}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT and not is_mask_path(p.relative_to(root)):
            s = split_of(p.relative_to(root))
            if s is not None:
                out[s][p.name] = p
    return out


def read_table(path: Path) -> List[Dict[str, object]]:
    if path.suffix.lower() in (".xlsx", ".xls"):
        import pandas as pd

        return pd.read_excel(path).to_dict("records")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _norm_name(v: object) -> str:
    return Path(str(v).strip()).name.lower()


def name_column(rows: List[Dict[str, object]], names: set) -> Optional[str]:
    """Column whose values best match the image file names (with or without extension)."""
    stems = {Path(n).stem for n in names}
    best, hits = None, 0
    for col in rows[0]:
        vals = [_norm_name(r[col]) for r in rows]
        h = sum(v in names or Path(v).stem in stems for v in vals)
        if h > hits:
            best, hits = col, h
    return best if hits >= max(1, len(rows) // 2) else None


def label_column(rows: List[Dict[str, object]], skip: str) -> Optional[str]:
    """First column (other than the name) whose values are all 0/1, preferring one called *label*."""
    cands = []
    for col in rows[0]:
        if col == skip:
            continue
        try:
            vals = {int(float(r[col])) for r in rows}
        except (TypeError, ValueError):
            continue
        if vals <= {0, 1}:
            cands.append(col)
    cands.sort(key=lambda c: "label" not in str(c).lower())
    return cands[0] if cands else None


def read_labels(root: Path, images: Dict[str, Dict[str, Path]]) -> Dict[Tuple[str, str], int]:
    """{(split, file name): 0/1} from the label tables; falls back to P/N/H file-name prefixes."""
    labels: Dict[Tuple[str, str], int] = {}
    tables = [p for p in root.rglob("*") if p.suffix.lower() in (".xlsx", ".xls", ".csv")
              and ("label" in p.name.lower() or "classification" in p.name.lower())]
    for t in tables:
        rows = read_table(t)
        if not rows:
            continue
        s = split_of(t.relative_to(root))
        pool = {n.lower(): (sp, n) for sp in ([s] if s else SPLITS) for n in images[sp]}
        ncol = name_column(rows, set(pool))
        lcol = label_column(rows, ncol) if ncol else None
        if not (ncol and lcol):
            print(f"  skipped {t.relative_to(root)}: no image-name or 0/1 label column", file=sys.stderr)
            continue
        by_stem = {Path(k).stem: v for k, v in pool.items()}
        n = 0
        for r in rows:
            v = _norm_name(r[ncol])
            key = pool.get(v) or by_stem.get(Path(v).stem)
            if key:
                labels[key] = int(float(r[lcol]))
                n += 1
        print(f"  {t.relative_to(root)}: {n} labels (name column {ncol!r}, label column {lcol!r})")
    for sp, files in images.items():  # 2019 challenge training set: P = PM, N = normal, H = high myopia
        for n in files:
            if (sp, n) not in labels and n[:1].upper() in "PNH" and re.fullmatch(r"[PNH]\d+", Path(n).stem, re.I):
                labels[(sp, n)] = int(n[:1].upper() == "P")
    return labels


def read_supplementary(root: Path, images: Dict[str, Dict[str, Path]]) -> Dict[Tuple[str, str], Dict[str, object]]:
    """Device (1 Zeiss, 2 Canon) and centring (1 disc, 2 fovea, 3 between) when the release has them."""
    info: Dict[Tuple[str, str], Dict[str, object]] = {}
    for t in root.rglob("*"):
        if t.suffix.lower() not in (".xlsx", ".xls", ".csv") or "supplement" not in t.name.lower():
            continue
        rows = read_table(t)
        s = split_of(t.relative_to(root))
        pool = {n.lower(): (sp, n) for sp in ([s] if s else SPLITS) for n in images[sp]}
        ncol = name_column(rows, set(pool)) if rows else None
        if not ncol:
            continue
        by_stem = {Path(k).stem: v for k, v in pool.items()}
        for r in rows:
            v = _norm_name(r[ncol])
            key = pool.get(v) or by_stem.get(Path(v).stem)
            if key:
                info[key] = {"device": next((r[c] for c in r if "device" in str(c).lower()), ""),
                             "centring": next((r[c] for c in r if "cent" in str(c).lower()), "")}
    return info


def find_lesion_masks(root: Path, split: str) -> Dict[str, List[Path]]:
    """{image stem: [mask paths]} from the split's lesion-mask folders (atrophy, detachment)."""
    out: Dict[str, List[Path]] = {}
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if (p.is_file() and p.suffix.lower() in IMG_EXT and split_of(rel) == split
                and any(k in part.lower() for part in rel.parts for k in ("lesion", "atrophy", "detachment"))):
            out.setdefault(p.stem.lower(), []).append(p)
    return out


def binary_mask(path: Path) -> np.ndarray:
    """PALM masks mark the structure in black on white; take the minority side if a mask is inverted."""
    a = np.asarray(Image.open(path).convert("L"))
    fg = a < 128 if (a >= 128).mean() > 0.5 else a >= 128
    return fg


def fov_box(img: Image.Image, thr: int = 15, margin: int = 4) -> Tuple[int, int, int, int]:
    a = np.asarray(img.convert("RGB")).max(axis=2) > thr
    ys, xs = np.where(a)
    if len(xs) == 0:
        return (0, 0, img.width, img.height)
    return (max(0, xs.min() - margin), max(0, ys.min() - margin),
            min(img.width, xs.max() + 1 + margin), min(img.height, ys.max() + 1 + margin))


class Writer:
    def __init__(self, copy: bool, crop: bool):
        self.copy, self.crop, self.boxes = copy, crop, {}

    def image(self, src: Path, dst_dir: Path) -> Path:
        dst_dir.mkdir(parents=True, exist_ok=True)
        if self.crop:
            with Image.open(src) as im:
                box = self.boxes.setdefault(src, fov_box(im))
                dst = dst_dir / (src.stem + ".png")
                if not dst.exists():
                    im.convert("RGB").crop(box).save(dst)
            return dst
        dst = dst_dir / src.name
        if not dst.exists() and not dst.is_symlink():
            shutil.copy2(src, dst) if self.copy else os.symlink(src.resolve(), dst)
        return dst

    def mask(self, srcs: List[Path], image: Path, dst: Path) -> None:
        m = np.zeros(Image.open(srcs[0]).size[::-1], dtype=bool)
        for s in srcs:
            b = binary_mask(s)
            if b.shape == m.shape:
                m |= b
        if self.crop:
            if image not in self.boxes:
                with Image.open(image) as im:
                    self.boxes[image] = fov_box(im)
            x0, y0, x1, y1 = self.boxes[image]
            m = m[y0:y1, x0:x1]
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((m * 255).astype(np.uint8)).save(dst)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--palm", required=True, type=Path, help="folder where the PALM release was extracted")
    ap.add_argument("--out", required=True, type=Path, help="output folder (MVTec layout)")
    ap.add_argument("--lesion-masks", action="store_true", help="also write palm-lesion with lesion masks")
    ap.add_argument("--crop-fov", action="store_true", help="crop every image to its field of view (PNG copies)")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking them")
    ap.add_argument("--force", action="store_true", help="write even if split sizes differ from 400/400/400")
    a = ap.parse_args(argv)

    root = a.palm.resolve()
    if not root.is_dir():
        print(f"ERROR: --palm {a.palm} does not exist or is not a folder. Pass the folder where you extracted "
              "PALM (the one that contains the training, validation and test sets).", file=sys.stderr)
        return 1
    try:
        a.out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"ERROR: cannot create --out {a.out} ({e.strerror}). Choose a folder you can write to, "
              "e.g. --out ~/palm_imgad.", file=sys.stderr)
        return 1

    print(f"Reading {root}")
    images = find_images(root)
    labels = read_labels(root, images)
    info = read_supplementary(root, images)

    if not labels:
        n_img = sum(len(v) for v in images.values())
        print(f"ERROR: no labelled images found under {root} ({n_img} images in training/validation/test "
              "folders).", file=sys.stderr)
        zips = sorted(p.relative_to(root) for p in root.rglob("*") if p.suffix.lower() == ".zip")
        if zips:
            print("  The release is still compressed; extract these files and run again:", file=sys.stderr)
            for z in zips[:10]:
                print(f"    {z}", file=sys.stderr)
        top = sorted(p.name + ("/" if p.is_dir() else "") for p in root.iterdir())
        print("  Contents of --palm: " + (", ".join(top[:20]) or "(empty)"), file=sys.stderr)
        return 1

    problems = []
    for sp in SPLITS:
        n, nl = len(images[sp]), sum(1 for k in labels if k[0] == sp)
        pm = sum(v for k, v in labels.items() if k[0] == sp)
        print(f"  {sp:5s}: {n} images, {nl} labelled ({pm} PM, {nl - pm} non-PM)")
        if n != EXPECTED[sp]:
            problems.append(f"{sp} has {n} images, expected {EXPECTED[sp]}")
        if nl < n:
            problems.append(f"{sp}: {n - nl} images without a label")
    if problems:
        print("\n".join("WARNING: " + p for p in problems), file=sys.stderr)
        if not a.force:
            print("Stopping (check --palm, or pass --force to write what was found).", file=sys.stderr)
            return 1

    w = Writer(a.copy, a.crop_fov)
    out = a.out
    manifest = []

    def put(category: str, sp: str, name: str, sub: str) -> Path:
        return w.image(images[sp][name], out / category / sub)

    for name in sorted(images["train"]):
        if labels.get(("train", name)) == 0:
            for cat in ("palm-val", "palm-test") + (("palm-lesion",) if a.lesion_masks else ()):
                put(cat, "train", name, "train/good")
    for cat, sp in (("palm-val", "val"), ("palm-test", "test")):
        for name in sorted(images[sp]):
            if (sp, name) in labels:
                put(cat, sp, name, "test/pm" if labels[(sp, name)] else "test/good")

    masks_by_stem = find_lesion_masks(root, "test") if a.lesion_masks else {}
    n_lesion = 0
    for sp in SPLITS:
        for name in sorted(images[sp]):
            lab = labels.get((sp, name))
            mask = ""
            if a.lesion_masks and sp == "test" and lab is not None:
                srcs = masks_by_stem.get(Path(name).stem.lower(), [])
                if lab == 0:
                    put("palm-lesion", sp, name, "test/good")
                elif srcs:
                    dst = put("palm-lesion", sp, name, "test/lesion")
                    mpath = out / "palm-lesion" / "ground_truth" / "lesion" / f"{dst.stem}_mask.png"
                    w.mask(srcs, images[sp][name], mpath)
                    mask, n_lesion = str(mpath), n_lesion + 1
            extra = info.get((sp, name), {})
            manifest.append({"split": sp, "file": name, "label": "" if lab is None else lab,
                             "device": extra.get("device", ""), "centring": extra.get("centring", ""), "mask": mask})

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=list(manifest[0]))
        cw.writeheader()
        cw.writerows(manifest)

    print(f"\nWrote {out}:")
    for cat in sorted(p.name for p in out.iterdir() if p.is_dir()):
        parts = [f"{sub.relative_to(out / cat)}={sum(1 for _ in sub.iterdir())}"
                 for sub in sorted((out / cat).glob("*/*")) if sub.is_dir() and "ground_truth" not in sub.parts]
        print(f"  {cat}: " + ", ".join(parts))
    if a.lesion_masks:
        print(f"  palm-lesion: {n_lesion} PM test images with lesion masks")
    print(f"  manifest.csv: {len(manifest)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
