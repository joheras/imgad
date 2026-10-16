import logging
import shutil

import pytest
import yaml

from imgad import BaseDetector, benchmark, data, register_detector
from imgad.cli import main
from imgad.progress import BenchmarkProgress, fmt_duration


def test_fmt_duration():
    assert fmt_duration(5) == "5s"
    assert fmt_duration(125) == "2m05s"
    assert fmt_duration(3 * 3600 + 120) == "3h02m"


def test_eta_per_kind():
    p = BenchmarkProgress({"extract": 2, "detector": 10, "anomalib": 2})
    assert p.eta() == (None, ["extract", "detector", "anomalib"])
    p.add("extract", 10.0)                       # 1 extraction left at 10 s
    p.add("detector", 1.0)                       # 9 detector runs left at 1 s
    eta, untimed = p.eta()
    assert eta == pytest.approx(10 + 9) and untimed == ["anomalib"]
    assert "anomalib runs (not timed yet)" in p.timing()
    assert p.counter() == "[ 1/12   8%]"
    for _ in range(9):
        p.add("detector", 1.0)
    p.add("extract", 10.0)
    assert p.eta() == (None, ["anomalib"]) and "ETA after the first of the anomalib runs" in p.timing()


@pytest.fixture
def two_categories(image_folder):
    shutil.copytree(image_folder / "cat", image_folder / "dog")
    return image_folder


def test_benchmark_logs_plan_runs_and_summary(two_categories, caplog):
    dss = [data.mvtec_ad(two_categories, c) for c in ("cat", "dog")]
    with caplog.at_level(logging.DEBUG, logger="imgad"):
        rows = benchmark([{"type": "stats"}], [{"name": "knn", "k": 1}, "mahalanobis"], dss, seeds=[0, 1],
                         train_subsample=0.8, verbose=False)
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Benchmark plan: 2 dataset(s), 2 seed(s), 8 runs") for m in msgs)
    assert any("= 8 runs" in m for m in msgs)
    assert sum(m.startswith("Extracting features") for m in msgs) == 2
    runs = [m for m in msgs if m.startswith("[")]
    assert len(runs) == len(rows) == 8
    assert runs[0].startswith("[1/8  12%]") and runs[-1].startswith("[8/8 100%]")
    assert "elapsed" in runs[-1] and "AUROC=" in runs[-1]
    assert any(m.startswith("fit mvtec/cat · seed 1") for m in msgs)          # DEBUG steps
    assert any(m.startswith("Benchmark finished: 8 runs") for m in msgs)


@register_detector("broken_test_detector")
class _Broken(BaseDetector):
    def _fit(self, X):
        raise ValueError("boom")

    def _score(self, X):  # pragma: no cover
        return X[:, 0]


def test_failed_step_is_logged(two_categories, caplog):
    ds = data.mvtec_ad(two_categories, "cat")
    with caplog.at_level(logging.ERROR, logger="imgad"), pytest.raises(ValueError, match="boom"):
        benchmark([{"type": "stats"}], ["broken_test_detector"], [ds], verbose=False)
    failed = [r.getMessage() for r in caplog.records if r.getMessage().startswith("FAILED")]
    assert failed == ["FAILED: fit mvtec/cat · seed 0 · stats | - | broken_test_detector (ValueError: boom)"]


def _cfg(tmp_path, root):
    cfg = tmp_path / "b.yaml"
    cfg.write_text(yaml.safe_dump({"data": {"type": "mvtec", "root": str(root)}, "extractors": [{"type": "stats"}],
                                   "detectors": [{"name": "knn", "k": 1}, "centroid"]}))
    return cfg


def test_cli_quiet_and_log_file(two_categories, tmp_path, capsys):
    cfg = _cfg(tmp_path, two_categories)
    main(["benchmark", "--config", str(cfg), "-q"])
    err = capsys.readouterr().err
    assert "Benchmark plan" not in err and "[1/4" not in err

    log = tmp_path / "run.log"
    main(["benchmark", "--config", str(cfg), "--log-file", str(log)])
    err = capsys.readouterr().err
    assert "Benchmark plan" in err and "[4/4 100%]" in err and "DEBUG" not in err
    text = log.read_text()
    assert "INFO    imgad: Benchmark plan" in text and "DEBUG   imgad: fit mvtec/cat" in text
