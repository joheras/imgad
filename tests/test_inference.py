import csv
import json

import numpy as np
import pytest
from PIL import Image

from imgad import AnomalyDetectionPipeline, AnomalyPredictor, data
from imgad.cli import main


@pytest.fixture
def trained(image_folder, tmp_path):
    ds = data.mvtec_ad(image_folder, "cat")
    pipe = AnomalyDetectionPipeline({"type": "stats"}, {"name": "knn", "k": 1, "metric": "euclidean"},
                                    preprocessing=["standard"]).fit(ds.train_paths)
    path = tmp_path / "model.joblib"
    pipe.save(path)
    return path, ds


def test_predict_folder_list_and_single(trained, image_folder):
    path, ds = trained
    pred = AnomalyPredictor.load(path)
    res = pred.predict(image_folder / "cat" / "test", show_progress=False)
    assert len(res) == 16
    flags = {r.source: r.is_anomaly for r in res}
    assert all(flags[p] for p in ds.test_paths if "scratch" in p)
    assert sum(flags[p] for p in ds.test_paths if "/good/" in p) <= 2
    assert all(0.0 <= r.normal_percentile <= 1.0 for r in res)

    one = pred.predict_one(ds.test_paths[-1])
    assert one.source == ds.test_paths[-1] and one.is_anomaly
    # mixed list: path + PIL + numpy
    arr = np.asarray(Image.open(ds.test_paths[0]))
    mixed = pred.predict([ds.test_paths[0], Image.open(ds.test_paths[0]), arr], show_progress=False)
    assert np.allclose([m.score for m in mixed], mixed[0].score)
    # streaming gives the same as batch
    assert [r.score for r in pred.predict_iter(image_folder / "cat" / "test", chunk_size=5)] == pytest.approx([r.score for r in res])


def test_threshold_override_and_metadata(trained):
    path, ds = trained
    pred = AnomalyPredictor.load(path, threshold=1e9)
    assert not any(r.is_anomaly for r in pred.predict(ds.test_paths, show_progress=False))
    info = pred.info()
    assert info["n_train"] == 20 and "trained_at" in info and info["threshold"] == 1e9


@pytest.mark.parametrize("method", ["f1", "youden", "fpr", "tpr"])
def test_calibrate_threshold(trained, method):
    path, ds = trained
    pipe = AnomalyDetectionPipeline.load(path)
    res = pipe.calibrate_threshold(ds.test_paths, ds.test_labels, method=method, target_fpr=0.0, show_progress=False)
    preds = pipe.predict(ds.test_paths, show_progress=False)
    if method in ("f1", "youden", "tpr"):
        assert res["f1"] == pytest.approx(1.0)
        assert (preds == ds.test_labels).all()
    assert res["fpr"] == 0.0
    assert pipe.metadata["calibration"]["method"] == method


def test_calibrate_needs_both_classes(trained):
    path, ds = trained
    pipe = AnomalyDetectionPipeline.load(path)
    with pytest.raises(ValueError):
        pipe.calibrate_threshold(scores=np.array([0.1, 0.2]), labels=[0, 0], method="f1")


def test_untrained_pipeline_rejected():
    with pytest.raises(ValueError):
        AnomalyPredictor(AnomalyDetectionPipeline({"type": "stats"}, "knn"))


def test_cli_predict_calibrate_info(trained, image_folder, tmp_path, capsys):
    path, ds = trained
    out_json = tmp_path / "p.json"
    main(["predict", "--model", str(path), str(image_folder / "cat" / "test"), "--out", str(out_json), "-q"])
    rows = json.loads(out_json.read_text())
    assert len(rows) == 16 and {"score", "is_anomaly", "normal_percentile", "source", "threshold"} <= set(rows[0])

    out_csv = tmp_path / "p.csv"
    main(["predict", "--model", str(path), ds.test_paths[0], ds.test_paths[-1], "--out", str(out_csv), "--only-anomalies", "-q"])
    assert len(list(csv.DictReader(open(out_csv)))) == 1

    cal = tmp_path / "cal.joblib"
    main(["calibrate", "--model", str(path), "--normal", str(image_folder / "cat" / "test" / "good"),
          "--anomalous", str(image_folder / "cat" / "test" / "scratch"), "--out", str(cal)])
    main(["info", "--model", str(cal)])
    info = json.loads(capsys.readouterr().out.split("Saved to")[-1].split("\n", 1)[1])
    assert info["calibration"]["method"] == "f1"

    # legacy alias
    main(["score", "--pipeline", str(path), "--images", str(image_folder / "cat" / "test"), "-q"])
