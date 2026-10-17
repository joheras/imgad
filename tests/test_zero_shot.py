"""Zero/few-shot detection with text prompts and the few-shot benchmark protocol (offline: a toy
vision–language extractor stands in for CLIP-style models)."""

import numpy as np
import pytest

from imgad import (AnomalyDetectionPipeline, BaseDetector, FeatureExtractor, ZeroShotDetector, benchmark,
                   build_detector, data, register_detector)
from imgad.detectors.zero_shot import needs_text
from imgad.features import register_extractor
from imgad.pipeline import load_pipeline
from imgad.prompts import default_prompts

NORMAL_WORDS = ("flawless", "perfect", "unblemished", "normal", "healthy")


@register_extractor("toy_vlm")
class ToyVLM(FeatureExtractor):
    """Image embedding = (5·fraction of bright pixels, 1 − fraction, small noise); text embedding = (0, 1, …)
    for normal prompts and (1, 0, …) for anomalous ones: bright squares look like the anomalous prompts."""

    def __init__(self, model_name: str = "toy_vlm", batch_size: int = 8):
        self.model_name, self.batch_size = model_name, batch_size
        self.encoded = 0

    def _extract_batch(self, images):
        out = []
        for im in images:
            a = np.asarray(im, dtype=np.float32) / 255.0
            b = float((a > 0.9).mean())
            v = np.r_[5 * b, 1 - b, a.mean((0, 1)) * 0.01]
            out.append(v / np.linalg.norm(v))
        return np.array(out)

    def has_text_encoder(self):
        return True

    def encode_text(self, prompts):
        self.encoded += len(prompts)
        normal = [any(w in p for w in NORMAL_WORDS) for p in prompts]
        return np.array([[0.0, 1.0, 0, 0, 0] if n else [1.0, 0.0, 0, 0, 0] for n in normal])

    @property
    def logit_scale(self):
        return 100.0

    def get_config(self):
        return {"model_name": self.model_name, "batch_size": self.batch_size}


@register_detector("needs_three")
class NeedsThree(BaseDetector):
    """Distance to the mean that refuses to fit on fewer than 3 images (a detector that fails in few-shot)."""

    def _fit(self, X):
        if len(X) < 3:
            raise ValueError("needs at least 3 training images")
        self.mu_ = X.mean(0)

    def _score(self, X):
        return np.linalg.norm(X - self.mu_, axis=1)


@pytest.fixture
def toy(image_folder):
    return data.mvtec_ad(image_folder, "cat")


def _auroc(y, s):
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y, s)


def test_prompts_and_detector_basics():
    d = default_prompts("fundus")
    assert "A fundus photograph of {}" in d["templates"] and "glaucoma" in d["anomalous"]
    assert default_prompts(None)["normal"][0] == "a flawless object"
    det = ZeroShotDetector(modality="oct", anomalous=["drusen"])
    assert det.anomalous == ["drusen"] and det.prompts()[0] == "An OCT scan of a normal retina"
    assert needs_text(det) and needs_text({"name": "zero_shot"}) and not needs_text("knn")
    with pytest.raises(ValueError, match="exactly one"):
        ZeroShotDetector(templates=["no placeholder"])
    with pytest.raises(RuntimeError, match="bind"):
        ZeroShotDetector().fit(np.zeros((0, 5)))
    with pytest.raises(TypeError, match="cannot encode text"):
        ZeroShotDetector().bind(build_detector("knn"))


def test_zero_shot_and_few_shot_pipeline(toy, tmp_path):
    pipe = AnomalyDetectionPipeline({"type": "toy_vlm"}, {"name": "zero_shot"})
    pipe.fit([])                                                           # zero-shot: no training images
    assert pipe.detector.mode_ == "text" and pipe.threshold_ == 0.5
    s = pipe.score(toy.test_paths, show_progress=False)
    assert _auroc(toy.test_labels, s) == 1.0 and ((s > 0.5) == np.asarray(toy.test_labels)).mean() > 0.9
    assert "zero-shot" in pipe.metadata["threshold_source"]

    pipe.save(tmp_path / "zs.joblib")                                      # no text encoder needed after fitting
    loaded = load_pipeline(tmp_path / "zs.joblib")
    assert np.allclose(loaded.score_features(pipe.extract(toy.test_paths, show_progress=False)), s)

    few = AnomalyDetectionPipeline({"type": "toy_vlm"}, {"name": "zero_shot", "k": 2}, preprocessing=["l2"])
    few.fit(toy.train_paths[:4], show_progress=False)                      # 4-shot: text + nearest normal images
    assert few.detector.mode_ == "both" and few.detector.memory_.shape[0] == 4
    assert len(few.detector.train_scores_) == 4 and few.threshold_ > 0
    assert _auroc(toy.test_labels, few.score(toy.test_paths, show_progress=False)) == 1.0

    with pytest.raises(ValueError, match="only the 'l2' preprocessing"):
        AnomalyDetectionPipeline({"type": "toy_vlm"}, "zero_shot", preprocessing=["pca:2"]).fit([])
    with pytest.raises(ValueError, match="needs normal training images"):
        AnomalyDetectionPipeline({"type": "toy_vlm"}, {"name": "zero_shot", "mode": "visual"}).fit([])


def test_score_combination_and_one_shot():
    det = ZeroShotDetector(mode="both", visual_weight=0.5)
    prompts = det.prompts()
    emb = {p: (np.array([0.0, 1.0]) if any(w in p for w in NORMAL_WORDS) else np.array([1.0, 0.0])) for p in prompts}
    det.set_text(emb, 10.0)
    det.fit(np.array([[0.0, 1.0]]))                                        # 1 shot: no leave-one-out neighbour
    t = det._text(np.array([[1.0, 0.0]]))[0]
    v = det._visual(np.array([[1.0, 0.0]]))[0]
    assert v == pytest.approx(0.5) and det.score(np.array([[1.0, 0.0]]))[0] == pytest.approx(0.5 * t + 0.5 * v)
    assert np.isfinite(det.train_scores_).all()
    with pytest.raises(ValueError, match="dimension"):
        det.fit(np.zeros((3, 7)))


def test_benchmark_few_shot_protocol(toy):
    dets = [{"name": "zero_shot"}, {"name": "knn", "k": 1}, "centroid", "needs_three"]
    rows = benchmark([{"type": "toy_vlm"}, {"type": "stats"}], dets, [toy], k_shots=[0, 1, 4], seeds=[0, 1],
                     preprocessings=[None, ["standard"]], verbose=False)
    ok = [r for r in rows if not r.get("error")]
    key = {(r["extractor"], r["preprocessing"], r["detector"], r["k_shot"]) for r in ok}
    # zero-shot runs only with the text-capable extractor and without space-changing preprocessing
    assert ("toy_vlm", "-", "zero_shot", 0) in key and ("toy_vlm", "standard", "zero_shot", 0) not in key
    assert not any(e == "stats" and d == "zero_shot" for e, _, d, _ in key)
    # k = 0 is zero-shot only; classical detectors run from k = 1 (needs_three needs 3 images:
    # recorded as failed runs, the rest goes on)
    assert not any(k == 0 and d != "zero_shot" for _, _, d, k in key)
    assert ("toy_vlm", "-", "centroid", 1) in key and ("toy_vlm", "-", "knn(k=1)", 1) in key
    failed = [r for r in rows if r.get("error")]
    assert failed and all(r["detector"] == "needs_three" and r["k_shot"] == 1 for r in failed)
    assert all(r["n_train"] == r["k_shot"] for r in ok)
    zs = [r for r in ok if r["detector"] == "zero_shot" and r["k_shot"] == 0]
    assert len(zs) == 2 and all(r["auroc"] == 1.0 for r in zs) and zs[0]["config"].endswith("| 0-shot")


def test_parallel_few_shot_matches_sequential(toy):
    args = ([{"type": "toy_vlm"}], [{"name": "zero_shot"}, "centroid"], [toy])
    kw = dict(k_shots=[0, 2], seeds=[0, 1], verbose=False)
    seq = benchmark(*args, **kw)
    par = benchmark(*args, n_jobs=2, devices="cpu", **kw)
    pick = lambda rows: [(r["config"], r["seed"], round(r["auroc"], 6)) for r in rows]  # noqa: E731
    assert pick(seq) == pick(par) and len(seq) == 6                         # zero_shot ×2 k + centroid at k=2


def test_report_few_shot_and_failures(toy, tmp_path):
    pytest.importorskip("matplotlib")
    from imgad.report import generate_report

    rows = benchmark([{"type": "toy_vlm"}], [{"name": "zero_shot"}, "needs_three"], [toy],
                     k_shots=[0, 1, 2, 4], seeds=[0, 1], keep_scores=True, verbose=False)
    md = generate_report(rows, [toy], tmp_path / "rep").read_text()
    assert "## Few-shot learning curves" in md and "| k=0 | k=1 | k=2 | k=4 |" in md.replace("Configuration ", "")
    assert "## Failed runs" in md and "needs_three" in md.split("## Failed runs")[1]
    assert (tmp_path / "rep" / "figures" / "few_shot.png").exists()
    assert "error" in (tmp_path / "rep" / "results.csv").read_text().splitlines()[0]
