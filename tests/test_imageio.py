"""High-bit-depth and medical images: 16-bit PNG/TIFF, intensity rules, DICOM and NIfTI (synthetic data only)."""

import json

import numpy as np
import pytest
from PIL import Image

from imgad import AnomalyDetectionPipeline, AnomalyPredictor, benchmark, build_extractor, data
from imgad.imageio import intensity_str, parse_intensity, split_ref, to_uint8
from imgad.utils import as_list, load_image

GRADIENT = np.linspace(0, 4095, 256, dtype=np.uint16)[None].repeat(32, 0)  # 12-bit detector values


def _levels(img) -> int:
    return len(np.unique(np.asarray(img)[..., 0]))


# ----------------------------------------------------------------------------- 16-bit rasters
@pytest.mark.parametrize("ext", ["png", "tif"])
def test_16bit_files_keep_their_dynamic_range(tmp_path, ext):
    p = tmp_path / f"xray.{ext}"
    Image.fromarray(GRADIENT).save(p)
    assert Image.open(p).mode.startswith("I;16")
    img = load_image(p)
    assert img.mode == "RGB" and _levels(img) >= 200
    assert (np.asarray(img)[..., 0] == 255).mean() < 0.05     # the old loader saturated ~94% of pixels


def test_8bit_images_are_unchanged(tmp_path):
    rgb = (np.random.default_rng(0).random((20, 30, 3)) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(tmp_path / "a.png")
    assert np.array_equal(np.asarray(load_image(tmp_path / "a.png")), rgb)
    assert np.array_equal(np.asarray(load_image(rgb)), rgb)
    gray = rgb[..., 0]
    assert np.array_equal(np.asarray(load_image(gray))[..., 1], gray)


def test_single_16bit_array_is_one_image():
    assert len(as_list(GRADIENT)) == 1 and _levels(load_image(GRADIENT)) >= 200
    assert len(as_list(np.zeros((5, 8, 8), np.uint16))) == 5     # a stack of 5 grayscale images


def test_intensity_rules():
    assert parse_intensity("window:-1000,400") == {"mode": "window", "low": -1000.0, "high": 400.0}
    assert intensity_str({"percentile": [1, 99]}) == "percentile:1,99"
    assert intensity_str(None) == "auto"
    for bad in ("window:5,1", "percentile:-1,50", "magic", "window:1"):
        with pytest.raises(ValueError):
            parse_intensity(bad)
    a = np.array([[0, 1000, 2000, 4095]], np.uint16)
    assert to_uint8(a, "window:0,4095").tolist() == [[0, 62, 125, 255]]
    assert to_uint8(a, "window:1000,2000").tolist() == [[0, 0, 255, 255]]
    assert to_uint8(a, "minmax")[0, -1] == 255 and to_uint8(a, "minmax")[0, 0] == 0
    assert to_uint8(np.full((4, 4), 7, np.uint16), "minmax").max() == 0            # constant image
    assert to_uint8(np.array([[0.0, 0.5, 1.0]], np.float32)).tolist() == [[0, 128, 255]]  # float in [0, 1]


def test_fixed_window_is_the_same_mapping_for_every_image():
    dim, bright = np.full((8, 8), 1000, np.uint16), np.full((8, 8), 3000, np.uint16)
    # per-image rules hide a globally brighter image; a fixed window keeps the difference
    assert to_uint8(dim, "minmax").max() == to_uint8(bright, "minmax").max() == 0
    assert to_uint8(dim, "window:0,4095")[0, 0] < to_uint8(bright, "window:0,4095")[0, 0]


def test_intensity_is_part_of_extractor_identity_and_saved_pipelines(tmp_path):
    a = build_extractor({"type": "stats"})
    b = build_extractor({"type": "stats", "intensity": "window:0,4095"})
    assert a.intensity == "auto" and b.intensity == "window:0,4095"
    assert "intensity" not in json.loads(a.identifier)                     # old feature caches stay valid
    assert a.identifier != b.identifier
    pipe = AnomalyDetectionPipeline(b, "knn")
    pipe.save(tmp_path / "p.joblib")
    assert AnomalyDetectionPipeline.load(tmp_path / "p.joblib").extractor.intensity == "window:0,4095"


def test_benchmark_detects_anomalies_hidden_in_high_bits(tmp_path):
    """12-bit images: normal ~1000, anomalies ~1400. Saturating to 8 bits made both identical."""
    rng = np.random.default_rng(0)

    def save(folder, n, anomalous):
        folder.mkdir(parents=True)
        for i in range(n):
            a = (1000 + rng.normal(0, 30, (40, 40))).astype(np.uint16)
            if anomalous:
                a[10:30, 10:30] += 400
            Image.fromarray(a).save(folder / f"{i:03d}.png")

    save(tmp_path / "c" / "train" / "good", 20, False)
    save(tmp_path / "c" / "test" / "good", 8, False)
    save(tmp_path / "c" / "test" / "lesion", 8, True)
    ds = data.mvtec_ad(tmp_path, "c")
    rows = benchmark([{"type": "stats"}], ["mahalanobis"], [ds], intensity="window:0,4095", verbose=False)
    assert rows[0]["auroc"] == pytest.approx(1.0)


# ----------------------------------------------------------------------------- DICOM
def _write_dicom(path, pixels, photometric="MONOCHROME2", slope=1.0, intercept=0.0, window=None, frames=1):
    pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName, ds.PatientID = "SECRET^PATIENT", "ID-12345"
    ds.Modality, ds.SOPClassUID, ds.SOPInstanceUID = "OT", meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, photometric
    ds.Rows, ds.Columns = pixels.shape[-2:]
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    if frames > 1:
        ds.NumberOfFrames = frames
    ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
    if window is not None:
        ds.WindowCenter, ds.WindowWidth = window
    ds.PixelData = np.ascontiguousarray(pixels.astype(np.uint16)).tobytes()
    try:
        ds.save_as(str(path), enforce_file_format=True)           # pydicom >= 3
    except TypeError:
        ds.save_as(str(path), write_like_original=False)          # pydicom 2.x
    return path


def test_dicom_rescale_window_and_monochrome1(tmp_path):
    raw = np.tile(np.arange(0, 2000, 10, dtype=np.uint16), (16, 1))       # 200 levels
    p = _write_dicom(tmp_path / "ct.dcm", raw, slope=1.0, intercept=-1000.0, window=(0, 400))   # HU = raw - 1000
    img = np.asarray(load_image(p, "dicom"))[..., 0]
    hu = raw.astype(float) - 1000
    assert img[0][hu[0] <= -200].max() == 0 and img[0][hu[0] >= 200].min() == 255   # window [-200, 200]
    assert _levels(load_image(p)) > 30                                    # auto uses the file window too
    full = np.asarray(load_image(p, "window:-1000,1000"))[..., 0]
    assert full[0, 0] == 0 and full[0, -1] > 240

    inv = _write_dicom(tmp_path / "cr.dcm", raw, photometric="MONOCHROME1")
    a = np.asarray(load_image(inv, "minmax"))[..., 0]
    assert a[0, 0] == 255 and a[0, -1] == 0                               # high values shown dark


def test_dicom_multiframe_slices_and_no_patient_data(tmp_path, monkeypatch):
    frames = np.stack([np.full((8, 8), 100 * (i + 1), np.uint16) for i in range(4)])
    p = _write_dicom(tmp_path / "multi.dcm", frames, frames=4)
    refs = data.volume_slices(p)
    assert refs == [f"{p}::frame={i}" for i in range(4)]
    assert split_ref(refs[2]) == (str(p), "frame", 2)
    vals = [np.asarray(load_image(r, "window:0,400"))[0, 0, 0] for r in refs]
    assert vals == sorted(vals) and vals[0] < vals[-1]

    single = _write_dicom(tmp_path / "one.dcm", np.full((8, 8), 50, np.uint16))
    pipe = AnomalyDetectionPipeline({"type": "stats"}, "knn").fit(refs + [str(single)] * 3)
    res = AnomalyPredictor(pipe).predict([str(single), refs[0]], show_progress=False)
    out = tmp_path / "preds.csv"
    AnomalyPredictor.save_results(res, out)
    assert "SECRET" not in out.read_text() and "ID-12345" not in out.read_text()


# ----------------------------------------------------------------------------- NIfTI
def test_nifti_slices(tmp_path):
    nib = pytest.importorskip("nibabel")
    vol = np.zeros((10, 12, 6), np.float32)
    vol[2:8, 3:9, 1:5] = np.arange(4, dtype=np.float32)[None, None, :] * 100 + 500   # empty first/last z slices
    p = tmp_path / "brain.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(p))
    refs = data.volume_slices(p)
    assert [split_ref(r)[2] for r in refs] == [1, 2, 3, 4]               # constant slices skipped
    assert len(data.volume_slices(p, skip_empty=False)) == 6
    assert data.volume_slices(tmp_path, middle_only=True) == [f"{p}::z=3"]
    assert len(data.volume_slices(p, axis="x", skip_empty=False)) == 10
    img = load_image(refs[0], "window:0,1000")
    assert img.size == (10, 12) and np.asarray(img).max() == 128          # 500 in [0, 1000]
    with pytest.raises(ValueError, match="volume"):
        load_image(p)
    # features through the cache: slice references get distinct cache entries
    ext = build_extractor({"type": "stats", "intensity": "window:0,1000"})
    X = ext.extract(refs, cache_dir=tmp_path / "cache", show_progress=False)
    X2 = ext.extract(refs, cache_dir=tmp_path / "cache", show_progress=False)
    assert np.allclose(X, X2) and len({tuple(r) for r in X}) == 4
