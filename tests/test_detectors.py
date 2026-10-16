import numpy as np
import pytest

from imgad import build_detector, list_detectors
from imgad.detectors import EnsembleDetector, SklearnDetector
from imgad.evaluation import evaluate_scores

SIMPLE = ["knn", "mahalanobis", "centroid", "kmeans", "lof", "gmm", "kde", "ocsvm", "iforest", "pca", "ensemble"]


@pytest.mark.parametrize("name", SIMPLE)
def test_detector_separates_shifted_gaussian(name, gaussian_data):
    Xtr, Xte, y = gaussian_data
    metric_kw = {"metric": "euclidean"} if name in ("knn", "lof", "centroid") else {}
    det = build_detector(name, **metric_kw).fit(Xtr)
    s = det.score(Xte)
    assert s.shape == (len(Xte),)
    assert evaluate_scores(y, s)["auroc"] > 0.85, name
    pred = det.predict(Xte)
    assert set(np.unique(pred)) <= {0, 1}
    # threshold calibrated at the 95% quantile -> few false positives on normal data
    assert pred[y == 0].mean() < 0.25


def test_registry_lists_all():
    for n in SIMPLE + ["pyod", "sklearn"]:
        assert n in list_detectors()


def test_build_from_dict():
    det = build_detector({"name": "knn", "k": 3, "metric": "euclidean"})
    assert det.k == 3 and det.metric == "euclidean"


def test_knn_train_scores_are_leave_one_out(gaussian_data):
    Xtr, _, _ = gaussian_data
    det = build_detector("knn", k=1, metric="euclidean").fit(Xtr)
    assert (det.train_scores_ > 0).all()  # self-match excluded


def test_sklearn_adapter(gaussian_data):
    from sklearn.covariance import EllipticEnvelope

    Xtr, Xte, y = gaussian_data
    det = SklearnDetector(estimator=EllipticEnvelope(random_state=0)).fit(Xtr)
    assert evaluate_scores(y, det.score(Xte))["auroc"] > 0.85


def test_ensemble_rank(gaussian_data):
    Xtr, Xte, y = gaussian_data
    det = EnsembleDetector(members=[{"name": "knn", "metric": "euclidean"}, "mahalanobis"], normalization="rank")
    det.fit(Xtr)
    assert evaluate_scores(y, det.score(Xte))["auroc"] > 0.85


def test_unfitted_raises():
    with pytest.raises(RuntimeError):
        build_detector("knn").score(np.zeros((2, 3)))


def test_metrics_perfect():
    m = evaluate_scores([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["auroc"] == 1.0 and m["aupr"] == 1.0 and m["fpr@95tpr"] == 0.0
