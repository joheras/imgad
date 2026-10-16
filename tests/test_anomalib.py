import csv
import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("anomalib")

from imgad import AnomalibPipeline, AnomalyDetectionPipeline, AnomalyPredictor, benchmark, data, evaluate_maps, load_pipeline
from imgad.cli import main
from imgad.evaluation import aupro

# Offline and fast: randomly initialised ResNet-18, small input size.
PATCHCORE = {"name": "patchcore", "backbone": "resnet18", "layers": ["layer2", "layer3"], "pre_trained": False,
             "image_size": 64, "device": "cpu"}


def _patchcore(**kw):
    spec = {k: v for k, v in PATCHCORE.items() if k != "name"}
    return AnomalibPipeline("patchcore", **{**spec, **kw})


@pytest.fixture
def ds(image_folder):
    return data.mvtec_ad(image_folder, "cat")


def test_mvtec_masks(ds):
    assert ds.has_masks
    assert all((m is None) == (l == 0) for m, l in zip(ds.test_masks, ds.test_labels))


def test_evaluate_maps_perfect_and_random():
    rng = np.random.default_rng(0)
    masks = [np.zeros((32, 32), bool) for _ in range(6)]
    for g in masks[3:]:
        g[8:20, 10:22] = True
    perfect = [g.astype(float) + 0.01 * rng.random(g.shape) for g in masks]
    m = evaluate_maps(masks, perfect)
    assert m["pixel_auroc"] == pytest.approx(1.0) and m["aupro"] == pytest.approx(1.0, abs=1e-3)
    random = [rng.random(g.shape) for g in masks]
    assert aupro(masks, random) == pytest.approx(0.15, abs=0.05)  # PRO ~ FPR -> 0.3**2 / 2 / 0.3
    # masks as paths / None and maps at another resolution
    assert evaluate_maps([None] * 3 + masks[3:], perfect)["pixel_auroc"] == pytest.approx(1.0)


def test_fit_score_maps_and_metrics(ds):
    pipe = _patchcore().fit(ds.train_paths, show_progress=False)
    assert pipe.metadata["n_train"] == 16 and pipe.metadata["n_calibration"] == 4
    scores, maps = pipe.score_and_maps(ds.test_paths, show_progress=False)
    assert maps[0].shape == (64, 64)
    pred = pipe.predict(ds.test_paths, show_progress=False)
    assert pred[ds.test_labels == 1].all() and pred[ds.test_labels == 0].sum() <= 1
    pix = evaluate_maps(ds.test_masks, maps)
    assert pix["pixel_auroc"] > 0.9 and pix["aupro"] > 0.5
    orig = pipe.anomaly_maps(ds.test_paths[:2], size="original", show_progress=False)
    assert orig[0].shape == (48, 48)
    masks = pipe.predict_masks(maps)
    assert sum(m.any() for m, l in zip(masks, ds.test_labels) if l == 0) <= 2  # threshold = q0.95 of normal maxima
    assert all(m.any() for m, l in zip(masks, ds.test_labels) if l == 1)
    res = pipe.calibrate_pixel_threshold(masks=ds.test_masks, maps=maps)
    assert res["pixel_f1"] > 0.5 and pipe.pixel_threshold_ == res["pixel_threshold"]
    # threshold logic shared with feature pipelines
    assert pipe.calibrate_threshold(scores=scores, labels=ds.test_labels)["f1"] == pytest.approx(1.0)


def test_save_load_predictor_and_maps(ds, tmp_path):
    pipe = _patchcore().fit(ds.train_paths, show_progress=False)
    path = tmp_path / "pc.joblib"
    pipe.save(path)
    loaded = load_pipeline(path)
    assert isinstance(loaded, AnomalibPipeline) and loaded.threshold_ == pipe.threshold_
    assert isinstance(AnomalyDetectionPipeline.load(path), AnomalibPipeline)
    np.testing.assert_allclose(loaded.score(ds.test_paths, show_progress=False), pipe.score(ds.test_paths, show_progress=False), rtol=1e-3)

    pred = AnomalyPredictor.load(path)
    assert pred.supports_maps
    res = pred.predict(ds.test_paths, return_maps=True, show_progress=False)
    assert res[0].anomaly_map.shape == (48, 48)
    assert all(r.is_anomaly for r, l in zip(res, ds.test_labels) if l == 1)
    assert all(r.anomalous_area > 0 for r in res if r.is_anomaly)
    pred.save_maps(res, tmp_path / "maps")
    assert Image.open(res[-1].heatmap).size == (48, 48)
    assert np.asarray(Image.open(res[-1].mask)).max() == 255
    out = tmp_path / "p.csv"
    pred.save_results(res, out)
    row = next(csv.DictReader(open(out)))
    assert {"heatmap", "mask", "anomalous_area"} <= set(row) and "anomaly_map" not in row
    # in-memory image + streaming
    one = pred.predict_one(np.asarray(Image.open(ds.test_paths[-1])), return_maps=True)
    assert one.is_anomaly and one.anomaly_map is not None
    assert len(list(pred.predict_iter(ds.test_paths, chunk_size=5, return_maps=True))) == len(ds.test_paths)


def test_feature_pipeline_has_no_maps(ds):
    pipe = AnomalyDetectionPipeline({"type": "stats"}, {"name": "knn", "k": 1}).fit(ds.train_paths)
    with pytest.raises(ValueError, match="anomalib"):
        AnomalyPredictor(pipe).predict(ds.test_paths, return_maps=True)


@pytest.mark.parametrize("name,kw", [("padim", {"pre_trained": False}), ("fastflow", {"pre_trained": False, "max_epochs": 1})])
def test_other_models_roundtrip(ds, tmp_path, name, kw):
    pipe = AnomalibPipeline(name, backbone="resnet18", image_size=64, device="cpu", **kw).fit(ds.train_paths, show_progress=False)
    pipe.save(tmp_path / "m.joblib")
    s1 = pipe.score(ds.test_paths, show_progress=False)
    s2 = AnomalibPipeline.load(tmp_path / "m.joblib").score(ds.test_paths, show_progress=False)
    np.testing.assert_allclose(s1, s2, rtol=1e-3)


def test_benchmark_with_anomalib(ds):
    rows = benchmark(detectors=[{"name": "knn", "k": 1}], extractors=[{"type": "stats"}], datasets=[ds],
                     anomalib_models=[PATCHCORE], verbose=False)
    assert len(rows) == 2
    ab = rows[-1]
    assert ab["extractor"] == "anomalib" and ab["detector"].startswith("patchcore")
    assert ab["auroc"] == pytest.approx(1.0) and ab["pixel_auroc"] > 0.9 and "aupro" in ab


def test_cli_fit_predict_maps_info(image_folder, tmp_path, capsys):
    model = tmp_path / "pc.joblib"
    main(["fit", "--anomalib", "patchcore", "-p", "backbone=resnet18", "pre_trained=false", "layers=[layer2,layer3]",
          "--image-size", "64", "--device", "cpu", "--train", str(image_folder / "cat" / "train" / "good"), "--out", str(model)])
    out = tmp_path / "p.json"
    maps = tmp_path / "maps"
    main(["predict", "--model", str(model), str(image_folder / "cat" / "test"), "--maps-dir", str(maps),
          "--only-anomalies", "--out", str(out), "-q"])
    rows = json.loads(out.read_text())
    assert all(r["is_anomaly"] and r["heatmap"] for r in rows)
    n_good = sum("/good/" in r["source"] for r in rows)
    assert len(rows) - n_good == 8 and n_good <= 1  # every defect found, at most one false alarm
    assert len(list(maps.glob("scratch_*_heatmap.png"))) == 8 and len(list(maps.glob("good_*_heatmap.png"))) == n_good
    capsys.readouterr()
    main(["info", "--model", str(model)])
    info = json.loads(capsys.readouterr().out)
    assert info["backend"] == "anomalib" and info["model"] == "patchcore" and info["pixel_threshold"] is not None
