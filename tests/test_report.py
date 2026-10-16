import shutil

import pytest
import yaml

from imgad import benchmark, data
from imgad.cli import main
from imgad.report import generate_report, short_extractor

SECTIONS = ["## Summary", "## Data analysis", "## Methods", "## Results", "## Discussion", "## Limitations",
            "## Reproducibility"]


@pytest.fixture
def two_categories(image_folder):
    shutil.copytree(image_folder / "cat", image_folder / "dog")
    return image_folder


def test_short_extractor():
    assert short_extractor("openai/clip-vit-base-patch16") == "clip-vit-base-patch16"
    assert short_extractor("facebook/dinov2-base[mean@[6, 9, 12]]") == "dinov2-base · mean@6/9/12"


def test_generate_report_with_figures(two_categories, tmp_path):
    pytest.importorskip("matplotlib")
    dss = [data.mvtec_ad(two_categories, c) for c in ("cat", "dog")]
    rows = benchmark([{"type": "stats"}], [{"name": "knn", "k": 1}, "mahalanobis", "centroid"], dss,
                     preprocessings=[["standard"]], seeds=[0, 1], train_subsample=0.8, bootstrap=30,
                     keep_scores=True, verbose=False)
    path = generate_report(rows, dss, tmp_path / "rep", config={"seeds": [0, 1]}, title="Toy")
    md = path.read_text()
    assert md.startswith("# Toy")
    assert all(s in md for s in SECTIONS)
    for sub in ("### Ranking of configurations", "### ROC curves", "### Results per defect type",
                "### Variability across seeds", "### Statistical comparison of configurations"):
        assert sub in md
    figs = sorted(p.name for p in (tmp_path / "rep" / "figures").glob("*.png"))
    assert {"data_sizes.png", "results_ranking.png", "results_roc.png", "results_scores.png", "results_cd.png"} <= set(figs)
    for f in figs:  # every figure is referenced
        assert f"figures/{f}" in md
    assert (tmp_path / "rep" / "results.csv").exists() and (tmp_path / "rep" / "config.yaml").exists()


def test_report_without_scores_or_figures(two_categories, tmp_path):
    ds = data.mvtec_ad(two_categories, "cat")
    rows = benchmark([{"type": "stats"}], [{"name": "knn", "k": 1}, "mahalanobis"], [ds], verbose=False)
    md = generate_report(rows, [], tmp_path / "rep", figures=False).read_text()
    assert all(s in md for s in ["## Summary", "## Methods", "## Results", "## Discussion"])
    assert "![" not in md and "ROC curves" not in md


def test_cli_benchmark_report_and_rebuild(two_categories, tmp_path):
    pytest.importorskip("matplotlib")
    cfg = tmp_path / "b.yaml"
    cfg.write_text(yaml.safe_dump({
        "data": {"type": "mvtec", "root": str(two_categories)},
        "extractors": [{"type": "stats"}],
        "preprocessing": [["standard"]],
        "detectors": [{"name": "knn", "k": 1}, "centroid", "mahalanobis"],
    }))
    main(["benchmark", "--config", str(cfg), "--report", str(tmp_path / "rep"), "--scores-out", str(tmp_path / "s.joblib")])
    md = (tmp_path / "rep" / "report.md").read_text()
    assert "imgad benchmark" in md and "## Discussion" in md
    main(["report", "--scores", str(tmp_path / "s.joblib"), "--config", str(cfg), "--out", str(tmp_path / "rep2"), "--title", "Again"])
    assert (tmp_path / "rep2" / "report.md").read_text().startswith("# Again")
