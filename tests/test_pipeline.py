"""End-to-end tests. No internet needed: a tiny randomly-initialised CLIP model is created and saved
locally, so the HF code path (AutoModel / AutoProcessor / get_image_features) is exercised."""

import numpy as np
import pytest

from imgad import AnomalyDetectionPipeline, FeatureExtractor, HFFeatureExtractor, data
from imgad.evaluation import benchmark, evaluate_scores, per_type_auroc


@pytest.fixture(scope="session")
def tiny_clip(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPTokenizer

    d = tmp_path_factory.mktemp("tiny_clip")
    cfg = CLIPConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_attention_heads=2, num_hidden_layers=2, vocab_size=99),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_attention_heads=2, num_hidden_layers=3,
                           image_size=32, patch_size=8),
        projection_dim=16,
    )
    CLIPModel(cfg).save_pretrained(d)
    CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}).save_pretrained(d)
    return str(d)


def test_pipeline_with_stats_extractor(image_folder, tmp_path):
    ds = data.mvtec_ad(image_folder, "cat")
    assert len(ds.train_paths) == 20 and ds.test_labels.sum() == 8
    pipe = AnomalyDetectionPipeline({"type": "stats"}, {"name": "knn", "k": 1, "metric": "euclidean"},
                                    preprocessing=["standard"], cache_dir=tmp_path / "cache")
    pipe.fit(ds.train_paths)
    s = pipe.score(ds.test_paths)
    assert evaluate_scores(ds.test_labels, s)["auroc"] > 0.95
    assert per_type_auroc(ds.test_labels, s, ds.test_types)["scratch"] > 0.95
    # cache is reused and gives identical features
    s2 = pipe.score(ds.test_paths)
    np.testing.assert_allclose(s, s2)

    path = tmp_path / "pipe.joblib"
    pipe.save(path)
    loaded = AnomalyDetectionPipeline.load(path)
    np.testing.assert_allclose(loaded.score(ds.test_paths), s)


@pytest.mark.parametrize(
    "kwargs, dim",
    [
        ({}, 16),                                  # auto -> projection (get_image_features)
        ({"pooling": "pooler"}, 32),
        ({"pooling": "cls"}, 32),
        ({"pooling": "mean"}, 32),
        ({"pooling": "cls+mean"}, 64),
        ({"pooling": "cls", "layer": [1, 2]}, 64),  # multi-layer concatenation
    ],
)
def test_hf_extractor_poolings(tiny_clip, image_folder, kwargs, dim):
    ext = HFFeatureExtractor(tiny_clip, device="cpu", batch_size=4, **kwargs)
    paths = data.list_images(image_folder / "cat" / "test")
    X = ext.extract(paths, show_progress=False)
    assert X.shape == (len(paths), dim)
    assert X.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)


def test_hf_pipeline_end_to_end(tiny_clip, image_folder, tmp_path):
    ds = data.mvtec_ad(image_folder, "cat")
    pipe = AnomalyDetectionPipeline(
        HFFeatureExtractor(tiny_clip, device="cpu", batch_size=8),
        "knn",
        preprocessing=["l2"],
    ).fit(ds.train_paths)
    s = pipe.score(ds.test_paths)
    assert s.shape == (len(ds.test_paths),)  # random weights: only check it runs end-to-end
    pipe.save(tmp_path / "p.joblib")
    np.testing.assert_allclose(AnomalyDetectionPipeline.load(tmp_path / "p.joblib", device="cpu").score(ds.test_paths), s, rtol=1e-5)


def test_benchmark_grid(image_folder):
    ds = data.mvtec_ad(image_folder, "cat")
    rows = benchmark(
        extractors=[{"type": "stats"}],
        detectors=["mahalanobis", {"name": "knn", "k": 3, "metric": "euclidean"}, "iforest"],
        datasets=[ds],
        preprocessings=[None, ["standard"]],
        verbose=False,
    )
    assert len(rows) == 6
    assert all(0.0 <= r["auroc"] <= 1.0 for r in rows)


def test_cli_fit_and_score(image_folder, tmp_path):
    import csv

    from imgad.cli import main

    out = tmp_path / "m.joblib"
    main(["fit", "--model", "stats", "--extractor-type", "stats", "--detector", "knn", "-p", "k=1", "metric=euclidean",
          "--prep", "standard", "--train", str(image_folder / "cat" / "train" / "good"), "--out", str(out)])
    csv_path = tmp_path / "scores.csv"
    main(["score", "--pipeline", str(out), "--images", str(image_folder / "cat" / "test"), "--out", str(csv_path)])
    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 16
    flagged = {r["source"].split("/")[-2] for r in rows if r["is_anomaly"] == "True"}
    assert "scratch" in flagged
