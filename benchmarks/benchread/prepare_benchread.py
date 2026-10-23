"""Lay out the BenchReAD fundus benchmark for an imgad one-class benchmark.

BenchReAD (Lian et al., MICCAI 2025; https://github.com/DopamineLcy/BenchReAD) trains on EDDFS + BRSET and tests
on two other datasets, RIADD (RFMiD) and JSIEC. It does not distribute images or split lists: its notebooks
rebuild the splits from each dataset's label files with pandas ``sample(random_state=0)``. This script repeats
those steps exactly (``fundus/EDDFS_split.ipynb``, ``BRSET_split.ipynb``, ``RIADD_split.ipynb``, ``JSIEC.ipynb``)
and writes the one-class setting in the MVTec layout that ``data.type: mvtec`` reads:

    <out>/benchread-val/train/good/          labelled normal training images (1/3 of the normals of EDDFS
                                             and BRSET, 4,983 + 2,803): what BenchReAD's one-class methods see
    <out>/benchread-val/test/good/           100 validation normals (50 EDDFS + 50 BRSET)
    <out>/benchread-val/test/abnormal/       100 validation anomalies          -> choose the configuration here
    <out>/benchread-riadd/train/good/        same normal training images
    <out>/benchread-riadd/test/good/         669 RIADD normals
    <out>/benchread-riadd/test/<type>/       1,634 RIADD anomalies, one folder per category (DR, AMD, MH, ...)
    <out>/benchread-jsiec/train/good/        same normal training images
    <out>/benchread-jsiec/test/good/         38 JSIEC normals
    <out>/benchread-jsiec/test/<type>/       686 JSIEC anomalies, one folder per category (DR, LOC, PM, ...)
    <out>/manifest.csv                       one row per written image: category, part, type, source, file

Anomaly-type folders keep BenchReAD's abbreviations, so per-type AUROCs line up with its Tables 1-2.

Preprocessing follows BenchReAD: EDDFS, BRSET and RIADD images are cropped to the field of view (square around
the largest bright region, 5 px margin) and resized to 512 x 512; JSIEC images are used as they are. Cropped images
are written as PNG; ``--no-crop`` symlinks (or ``--copy`` copies) the originals instead.

Usage:
    python benchmarks/benchread/prepare_benchread.py --eddfs /data/EDDFS --brset /data/brset/1.0.1 \\
        --riadd /data/RIADD --jsiec /data/JSIEC --out /data/benchread_imgad [-j 8]
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

SEED = 0
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# BenchReAD's counts (paper Sec. 2.1 and the splits its notebooks produce)
EXPECTED = {"train": 7786, "val_normal": 100, "val_abnormal": 100,
            "riadd_normal": 669, "riadd_abnormal": 1634, "jsiec_normal": 38, "jsiec_abnormal": 686}

EDDFS_EXCLUDED = ["RVO", "LS", "hyper", "others"]
BRSET_DISEASES = ["diabetic_retinopathy", "macular_edema", "scar", "nevus", "amd", "vascular_occlusion",
                  "hypertensive_retinopathy", "drusens", "hemorrhage", "retinal_detachment", "myopic_fundus",
                  "increased_cup_disc", "other"]
BRSET_EXCLUDED = ["macular_edema", "scar", "nevus", "vascular_occlusion", "hypertensive_retinopathy", "hemorrhage",
                  "retinal_detachment", "other"]
RIADD_CATEGORIES = ["DR", "ARMD", "ODC", "MYA", "DN", "MH", "BRVO", "TSLN", "CSR", "CRVO", "ODP", "ODE", "RS", "CRS",
                    "RPEC"]
RIADD_NAMES = {"ARMD": "AMD"}  # BenchReAD's tables call it AMD
RIADD_CSVS = {"train": ("RFMiD_Training_Labels.csv", "Training"),
              "val": ("RFMiD_Validation_Labels.csv", "Validation"),
              "test": ("RFMiD_Testing_Labels.csv", "Test")}
# JSIEC folder -> BenchReAD abbreviation (None = normal)
JSIEC_FOLDERS = {"0.0.Normal": None, "0.2.Large optic cup": "LOC", "0.3.DR1": "DR", "1.0.DR2": "DR",
                 "1.1.DR3": "DR", "2.0.BRVO": "BRVO", "2.1.CRVO": "CRVO", "4.Rhegmatogenous RD": "RRD",
                 "6.Maculopathy": "MAC", "7.ERM": "ERM", "8.MH": "MH", "9.Pathological myopia": "PM",
                 "15.0.Retinitis pigmentosa": "RP", "21.Yellow-white spots-flecks": "YWSF", "27.Laser Spots": "LS",
                 "29.0.Blur fundus without PDR": "BF_wo_PDR", "29.1.Blur fundus with suspected PDR": "BF_w_PDR"}


class SetupError(Exception):
    """A missing or unexpected input; the message says what to pass instead."""


# --------------------------------------------------------------------------------------------- finding files
def find_files(root: Path, names: Iterable[str]) -> Dict[str, Path]:
    """{file name: path} for the given names anywhere under root (case-insensitive; first match wins)."""
    want = {n.lower(): n for n in names}
    found: Dict[str, Path] = {}
    for p in sorted(root.rglob("*")):
        n = want.get(p.name.lower())
        if n is not None and n not in found and p.is_file():
            found[n] = p
    return found


def find_one(root: Path, candidates: List[str], what: str, flag: str) -> Path:
    for name in candidates:
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    raise SetupError(f"{flag} {root}: no {what} found (looked for {', '.join(candidates)}). Pass the folder where "
                     f"you extracted the dataset.")


def check_dir(path: Optional[Path], flag: str, hint: str) -> Path:
    if path is None:
        raise SetupError(f"{flag} is required: {hint}")
    if not path.is_dir():
        raise SetupError(f"{flag} {path} does not exist or is not a folder: {hint}")
    return path.resolve()


def missing_images(root: Path, flag: str, missing: List[str], n: int) -> SetupError:
    zips = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.suffix.lower() in (".zip", ".rar", ".7z"))
    msg = f"{flag} {root}: {len(missing)} of {n} images listed in the labels were not found (e.g. {missing[:3]})."
    if zips:
        msg += " The release is still compressed; extract " + ", ".join(zips[:5]) + " and run again."
    return SetupError(msg)


# --------------------------------------------------------------------------------------------- splits
def split_source(normal: pd.DataFrame, abnormal: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """BenchReAD's split of one training source: 50 + 50 validation, then 1/3 of each class labelled."""
    nv = normal.sample(n=50, random_state=SEED)
    av = abnormal.sample(n=50, random_state=SEED)
    n_rest = normal[~normal.index.isin(nv.index)]
    a_rest = abnormal[~abnormal.index.isin(av.index)]
    nl = n_rest.sample(n=len(n_rest) // 3, random_state=SEED)
    al = a_rest.sample(n=len(a_rest) // 3, random_state=SEED)
    return {"val_normal": nv, "val_abnormal": av, "train_normal": nl, "train_abnormal": al}


def eddfs_split(root: Path) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Path]]:
    tr = find_one(root, ["train.csv"], "EDDFS label file train.csv (from datas/EDDFS/Annotation of the EDDFS "
                  "GitHub repository)", "--eddfs")
    te = tr.parent / "test.csv"
    if not te.exists():
        raise SetupError(f"--eddfs: found {tr} but not test.csv next to it. Put both EDDFS annotation files "
                         "(train.csv, test.csv) in the same folder.")
    df = pd.concat([pd.read_csv(tr), pd.read_csv(te)], ignore_index=True)
    need = {"fnames", "normal", *EDDFS_EXCLUDED}
    if not need <= set(df.columns):
        raise SetupError(f"--eddfs: {tr} lacks columns {sorted(need - set(df.columns))}; is it the EDDFS annotation?")
    normal = df[df["normal"] == 1]
    abnormal = df[df["normal"] == 0]
    abnormal = abnormal[(abnormal[EDDFS_EXCLUDED] == 0).all(axis=1)]
    parts = split_source(normal, abnormal)
    names = set().union(*(set(p["fnames"]) for p in parts.values()))
    files = {k: v for k, v in find_files(root, names).items() if v.suffix.lower() in IMG_EXT}
    return parts, files


def brset_split(root: Path) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Path]]:
    lab = find_one(root, ["labels_brset.csv", "labels.csv"], "BRSET label file (labels_brset.csv or labels.csv)",
                   "--brset")
    df = pd.read_csv(lab)
    missing = [c for c in BRSET_DISEASES + ["image_id"] if c not in df.columns]
    if missing:
        raise SetupError(f"--brset: {lab} lacks columns {missing}; is it BRSET's label file?")
    if list(df.columns[20:33]) != BRSET_DISEASES:
        # BenchReAD selects the disease columns by position (iloc[:, 20:33]); warn if this release differs
        print(f"  WARNING: {lab.name} columns 20-32 are {list(df.columns[20:33])}, not the 13 disease columns "
              "BenchReAD's notebook assumes; using the columns by name.", file=sys.stderr)
    any_disease = df[BRSET_DISEASES].sum(axis=1) > 0
    normal = df[~any_disease]
    abnormal = df[any_disease]
    abnormal = abnormal[(abnormal[BRSET_EXCLUDED] == 0).all(axis=1)]
    parts = split_source(normal, abnormal)
    names = {f"{i}.jpg" for p in parts.values() for i in p["image_id"]}
    files = find_files(root, names)
    return parts, files


def riadd_split(root: Path) -> Tuple[List[Tuple[str, Path, Optional[str]]], List[str]]:
    """[(name, path, type or None for normal)] in BenchReAD's order, plus image names not found."""
    out, missing = [], []
    for sp, (csv_name, folder) in RIADD_CSVS.items():
        cands = [csv_name, csv_name.replace("Labels", "Label")]
        lab = find_one(root, cands, f"RFMiD {sp} labels", "--riadd")
        df = pd.read_csv(lab)
        if "Disease_Risk" not in df.columns or "ID" not in df.columns:
            raise SetupError(f"--riadd: {lab} lacks ID/Disease_Risk columns; is it an RFMiD label file?")
        dis = df.iloc[:, 2:]
        normal = df[df["Disease_Risk"] == 0]
        single = df[(df["Disease_Risk"] == 1) & (dis.sum(axis=1) == 1)]
        single = single[single[RIADD_CATEGORIES].sum(axis=1) == 1]
        # images sit in a folder named Training / Validation / Test (IDs repeat across splits); prefer the one
        # next to the label file
        dirs = sorted((d for d in root.rglob("*") if d.is_dir() and d.name.lower() == folder.lower()),
                      key=lambda d: lab.parent not in d.parents)
        if not dirs:
            raise SetupError(f"--riadd {root}: no '{folder}' image folder found for {lab.name}. Extract the RFMiD "
                             f"{sp} set so its images are in a folder named {folder}.")
        imgs = {p.name: p for p in dirs[0].glob("*.png")}
        for rows, is_normal in ((normal, True), (single, False)):
            for _, r in rows.iterrows():
                name = f"{r['ID']}.png"
                t = None if is_normal else RIADD_NAMES.get(r[RIADD_CATEGORIES].idxmax(), r[RIADD_CATEGORIES].idxmax())
                if name in imgs:
                    out.append((f"{sp}_{name}", imgs[name], t))
                else:
                    missing.append(f"{folder}/{name}")
    return out, missing


def jsiec_split(root: Path) -> List[Tuple[str, Path, Optional[str]]]:
    normal_dirs = sorted(p for p in root.rglob("0.0.Normal") if p.is_dir())
    if not normal_dirs:
        raise SetupError(f"--jsiec {root}: no '0.0.Normal' folder found. Pass the folder of the Kaggle download "
                         "'1000 Fundus images with 39 categories' (it contains 1000images/0.0.Normal, ...).")
    base = normal_dirs[0].parent
    out = []
    for folder, t in JSIEC_FOLDERS.items():
        d = base / folder
        if not d.is_dir():
            raise SetupError(f"--jsiec: folder {d} is missing; BenchReAD uses these 17 folders: "
                             + ", ".join(JSIEC_FOLDERS))
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                out.append((f"{folder.split('.')[0]}.{folder.split('.')[1]}_{p.name}".replace(" ", "_"), p, t))
    return out


# --------------------------------------------------------------------------------------------- writing
def fundus_crop(img: Image.Image, size: int = 512, margin: int = 5) -> Image.Image:
    """BenchReAD's ``fundus_crop``: square crop centred on the largest region with RGB sum > 30, resized."""
    from scipy import ndimage

    a = np.asarray(img.convert("RGB"))
    lab, n = ndimage.label(a.astype(np.int32).sum(axis=-1) > 30, structure=np.ones((3, 3)))
    if n == 0:
        return img.convert("RGB").resize((size, size), Image.BILINEAR)
    areas = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    k = int(np.argmax(areas)) + 1
    rows, cols = np.nonzero(lab == k)
    side = max(rows.max() + 1 - rows.min(), cols.max() + 1 - cols.min())
    cy, cx = rows.mean(), cols.mean()
    y0, x0 = int(cy - side / 2) - margin, int(cx - side / 2) - margin   # int16 cast in BenchReAD truncates too
    y1, x1 = int(cy + side / 2) + margin, int(cx + side / 2) + margin
    out = np.zeros((y1 - y0, x1 - x0, 3), dtype=a.dtype)
    sy0, sx0, sy1, sx1 = max(y0, 0), max(x0, 0), min(y1, a.shape[0]), min(x1, a.shape[1])
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
    return Image.fromarray(out).resize((size, size), Image.BILINEAR)


class Writer:
    def __init__(self, copy: bool, crop: bool):
        self.copy, self.crop = copy, crop
        self.cache: Dict[Path, Path] = {}

    def prepare(self, src: Path, cache_dir: Path, name: str, crop: bool) -> Path:
        """Cropped PNG (written once under cache_dir) or the original file."""
        if not (crop and self.crop):
            return src
        dst = cache_dir / (Path(name).stem + ".png")
        if not dst.exists():
            with Image.open(src) as im:
                fundus_crop(im).save(dst)
        return dst

    def place(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            return
        if self.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(src.resolve(), dst)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eddfs", type=Path, help="EDDFS folder: original images and the annotation train.csv/test.csv")
    ap.add_argument("--brset", type=Path, help="BRSET 1.0.1 folder from PhysioNet (fundus_photos/, labels csv)")
    ap.add_argument("--riadd", type=Path, help="RIADD/RFMiD folder (Training, Validation, Test and their label CSVs)")
    ap.add_argument("--jsiec", type=Path, help="JSIEC Kaggle folder (1000images/0.0.Normal, ...)")
    ap.add_argument("--out", required=True, type=Path, help="output folder (MVTec layout)")
    ap.add_argument("--no-crop", action="store_true", help="do not crop EDDFS/BRSET/RIADD to the field of view")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking them")
    ap.add_argument("--force", action="store_true", help="write even if the counts differ from BenchReAD's")
    ap.add_argument("-j", "--jobs", type=int, default=8, help="threads for cropping")
    a = ap.parse_args(argv)
    try:
        return run(a)
    except SetupError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def run(a) -> int:
    eddfs = check_dir(a.eddfs, "--eddfs", "the folder with the EDDFS images and annotation CSVs "
                      "(https://github.com/xia-xx-cv/EDDFS_dataset).")
    brset = check_dir(a.brset, "--brset", "the BRSET 1.0.1 folder downloaded from PhysioNet "
                      "(https://physionet.org/content/brazilian-ophthalmological/1.0.1/).")
    riadd = check_dir(a.riadd, "--riadd", "the RIADD/RFMiD folder (https://riadd.grand-challenge.org/).")
    jsiec = check_dir(a.jsiec, "--jsiec", "the JSIEC folder (https://www.kaggle.com/datasets/linchundan/fundusimage1000).")
    try:
        a.out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise SetupError(f"cannot create --out {a.out} ({e.strerror}). Choose a folder you can write to.")

    print("Rebuilding BenchReAD's splits")
    e_parts, e_files = eddfs_split(eddfs)
    b_parts, b_files = brset_split(brset)
    e_need = [f for p in e_parts.values() for f in p["fnames"]]
    b_need = [f"{i}.jpg" for p in b_parts.values() for i in p["image_id"]]
    e_miss = [f for f in e_need if f not in e_files]
    b_miss = [f for f in b_need if f not in b_files]
    for src, parts in (("EDDFS", e_parts), ("BRSET", b_parts)):
        print(f"  {src}: train normal {len(parts['train_normal'])} (abnormal labelled {len(parts['train_abnormal'])}, "
              f"not used), val {len(parts['val_normal'])} + {len(parts['val_abnormal'])}")
    riadd_items, r_miss = riadd_split(riadd)
    jsiec_items = jsiec_split(jsiec)

    errors = []
    if e_miss:
        errors.append(missing_images(eddfs, "--eddfs", e_miss, len(e_need)))
    if b_miss:
        errors.append(missing_images(brset, "--brset", b_miss, len(b_need)))
    if r_miss:
        errors.append(missing_images(riadd, "--riadd", r_miss, len(riadd_items) + len(r_miss)))
    if errors and not a.force:
        raise SetupError("\n  ".join(str(e) for e in errors) + "\n  (--force writes the images that were found.)")
    for e in errors:
        print(f"WARNING: {e}", file=sys.stderr)

    # (category part, type, source, name, path, crop?)
    items: List[Tuple[str, str, str, str, Path, bool]] = []
    for src, parts, files, col, fmt in (("EDDFS", e_parts, e_files, "fnames", "{}"),
                                        ("BRSET", b_parts, b_files, "image_id", "{}.jpg")):
        for key, part, t in (("train_normal", "train", "good"), ("val_normal", "val", "good"),
                             ("val_abnormal", "val", "abnormal")):
            for v in parts[key][col]:
                name = fmt.format(v)
                if name in files:
                    items.append((part, t, src, f"{src.lower()}_{name}", files[name], True))
    for name, p, t in riadd_items:
        items.append(("riadd", t or "good", "RIADD", f"riadd_{name}", p, True))
    for name, p, t in jsiec_items:
        items.append(("jsiec", t or "good", "JSIEC", f"jsiec_{name}", p, False))

    counts = {"train": sum(i[0] == "train" for i in items),
              "val_normal": sum(i[0] == "val" and i[1] == "good" for i in items),
              "val_abnormal": sum(i[0] == "val" and i[1] != "good" for i in items),
              "riadd_normal": sum(i[0] == "riadd" and i[1] == "good" for i in items),
              "riadd_abnormal": sum(i[0] == "riadd" and i[1] != "good" for i in items),
              "jsiec_normal": sum(i[0] == "jsiec" and i[1] == "good" for i in items),
              "jsiec_abnormal": sum(i[0] == "jsiec" and i[1] != "good" for i in items)}
    bad = [f"{k}: {v} (BenchReAD: {EXPECTED[k]})" for k, v in counts.items() if v != EXPECTED[k]]
    for k, v in counts.items():
        print(f"  {k:15s} {v}")
    if bad:
        print("WARNING: counts differ from BenchReAD's:\n  " + "\n  ".join(bad), file=sys.stderr)
        if not a.force:
            print("Stopping (check the input folders and dataset versions, or pass --force).", file=sys.stderr)
            return 1

    w = Writer(a.copy, not a.no_crop)
    cache = a.out / "_cropped"
    if w.crop:
        cache.mkdir(parents=True, exist_ok=True)
        print(f"Cropping {sum(i[5] for i in items)} images to the field of view (512 px) into {cache}")
    with ThreadPoolExecutor(max(1, a.jobs)) as ex:
        prepared = list(ex.map(lambda i: w.prepare(i[4], cache, i[3], i[5]), items))

    tests = {"val": "benchread-val", "riadd": "benchread-riadd", "jsiec": "benchread-jsiec"}
    manifest = []
    for (part, t, src, name, orig, _), path in zip(items, prepared):
        fname = Path(name).stem + path.suffix if path != orig else name
        cats = list(tests.values()) if part == "train" else [tests[part]]
        for cat in cats:
            sub = "train/good" if part == "train" else f"test/{t}"
            w.place(path, a.out / cat / sub / fname)
        manifest.append({"part": part, "type": t, "source": src, "file": fname, "original": str(orig)})

    with open(a.out / "manifest.csv", "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=list(manifest[0]))
        cw.writeheader()
        cw.writerows(manifest)

    print(f"\nWrote {a.out}:")
    for cat in tests.values():
        subs = sorted(p for p in (a.out / cat).glob("*/*") if p.is_dir())
        print(f"  {cat}: " + ", ".join(f"{s.relative_to(a.out / cat)}={sum(1 for _ in s.iterdir())}" for s in subs))
    print(f"  manifest.csv: {len(manifest)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
