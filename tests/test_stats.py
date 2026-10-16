import shutil

import numpy as np
import pytest
import yaml
from sklearn.metrics import roc_auc_score

from imgad import benchmark, data, stats
from imgad.cli import main


def _scores(rng, y, sep):
    return rng.normal(size=len(y)) + sep * y


def test_bootstrap_ci_contains_value_and_shrinks():
    rng = np.random.default_rng(0)
    widths = []
    for n in (60, 600):
        y = np.r_[np.zeros(n), np.ones(n // 2)].astype(int)
        ci = stats.bootstrap_ci(y, _scores(rng, y, 1.0), n_boot=300)
        a = ci["auroc"]
        assert a["low"] <= a["value"] <= a["high"]
        assert a["value"] == pytest.approx(roc_auc_score(y, _scores(np.random.default_rng(0), y, 1.0)), abs=0.2)
        widths.append(a["high"] - a["low"])
    assert widths[1] < widths[0] / 2


def test_delong():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(150), np.ones(100)].astype(int)
    good, bad = _scores(rng, y, 2.0), _scores(rng, y, 0.3)
    t = stats.delong_test(y, good, bad)
    assert t["auroc_a"] == pytest.approx(roc_auc_score(y, good))
    assert t["auroc_b"] == pytest.approx(roc_auc_score(y, bad))
    assert t["p_value"] < 1e-6 and t["diff_low"] > 0
    assert stats.delong_test(y, good, good)["p_value"] == 1.0
    # two noisy copies of the same model: no significant difference
    base = _scores(rng, y, 1.0)
    t2 = stats.delong_test(y, base + 0.05 * rng.normal(size=len(y)), base + 0.05 * rng.normal(size=len(y)))
    assert t2["p_value"] > 0.05 and t2["diff_low"] < 0 < t2["diff_high"]
    # paired bootstrap agrees in direction
    pb = stats.paired_bootstrap_test(y, good, bad, metric="auroc", n_boot=300)
    assert pb["p_value"] < 0.01 and pb["diff"] == pytest.approx(t["diff"])


def test_holm_and_nemenyi_reference_values():
    np.testing.assert_allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    # Demšar (2006), table 5: q_0.05 = 2.343 for k=3
    assert stats.nemenyi_cd(3, 10) == pytest.approx(2.343 * np.sqrt(3 * 4 / 60), rel=1e-3)


def _rows(n_datasets=10, seeds=(0, 1)):
    rng = np.random.default_rng(0)
    rows = []
    for d in range(n_datasets):
        for seed in seeds:
            for name, mu in (("A", 0.95), ("B", 0.90), ("C", 0.80)):
                rows.append({"dataset": f"d{d}", "config": name, "seed": seed,
                             "auroc": mu + 0.01 * rng.normal(), "fpr@95tpr": 1 - mu})
    return rows


def test_compare_configs_and_report(tmp_path):
    res = stats.compare_configs(_rows(), metric="auroc")
    assert res["configs"] == ["A", "B", "C"] and res["avg_rank"][0] == pytest.approx(1.0)
    assert res["friedman"]["p_value"] < 0.001
    ac = next(p for p in res["pairs"] if p["a"] == "A" and p["b"] == "C")
    assert ac["significant"] and ac["wins"] == 10 and ac["nemenyi_p"] < 0.05
    # lower-is-better metric ranks the same way
    assert stats.compare_configs(_rows(), metric="fpr@95tpr")["configs"][0] == "A"
    md = stats.report_markdown(res)
    assert "Friedman" in md and "| A | C |" in md
    assert stats.md_escape("clip | l2 | knn") == "clip \\| l2 \\| knn"
    pytest.importorskip("matplotlib")
    stats.plot_cd_diagram(res, str(tmp_path / "cd.png"))
    assert (tmp_path / "cd.png").stat().st_size > 0


def test_summarize_seeds():
    summ = stats.summarize_seeds(_rows(n_datasets=2, seeds=(0, 1, 2)), metrics=("auroc",))
    r = next(s for s in summ if s["dataset"] == "d0" and s["config"] == "A")
    assert r["n_seeds"] == 3 and r["auroc_ci_low"] < r["auroc_mean"] < r["auroc_ci_high"]
    assert any(s["dataset"] == "mean" for s in summ)


@pytest.fixture
def two_categories(image_folder):
    shutil.copytree(image_folder / "cat", image_folder / "dog")
    return image_folder


def test_benchmark_seeds_bootstrap_and_delong(two_categories):
    dss = [data.mvtec_ad(two_categories, c) for c in ("cat", "dog")]
    rows = benchmark([{"type": "stats"}], [{"name": "knn", "k": 1}, "iforest", "centroid"], dss,
                     preprocessings=[["standard"]], seeds=[0, 1, 2], train_subsample=0.6, bootstrap=50,
                     keep_scores=True, verbose=False)
    assert len(rows) == 2 * 3 * 3
    assert {r["n_train"] for r in rows} == {12}
    knn = [r for r in rows if r["detector"].startswith("knn") and r["dataset"] == "mvtec/cat"]
    assert len({tuple(r["scores"]) for r in knn}) == 3  # different training subsets per seed
    assert all(r["auroc_ci_low"] <= r["auroc"] <= r["auroc_ci_high"] for r in rows)
    base = rows[0]["config"]
    tests = stats.delong_vs_baseline(rows, base)
    assert len(tests) == 2 * 3 * 2 and all(0 <= t["p_holm"] <= 1 for t in tests)
    res = stats.compare_configs(rows)
    assert len(res["configs"]) == 3


def test_cli_benchmark_and_stats(two_categories, tmp_path, capsys):
    cfg = tmp_path / "b.yaml"
    cfg.write_text(yaml.safe_dump({
        "data": {"type": "mvtec", "root": str(two_categories)},
        "extractors": [{"type": "stats"}],
        "preprocessing": [["standard"]],
        "detectors": [{"name": "knn", "k": 1}, "centroid", "iforest"],
        "seeds": [0, 1],
        "train_subsample": 0.8,
    }))
    csv_path, scores = tmp_path / "r.csv", tmp_path / "s.joblib"
    main(["benchmark", "--config", str(cfg), "--out", str(csv_path), "--bootstrap", "30",
          "--scores-out", str(scores), "--stats"])
    out = capsys.readouterr().out
    assert "Repeticiones" in out and "Friedman" in out and "DeLong" in out
    rows = stats.load_results(str(csv_path))
    assert len(rows) == 12 and "scores" not in rows[0] and isinstance(rows[0]["auroc_ci_low"], float)
    report = tmp_path / "informe.md"
    main(["stats", "--results", str(csv_path), "--metric", "aupr", "--out", str(report)])
    assert "aupr" in report.read_text()
    main(["stats", "--scores", str(scores), "--baseline", rows[0]["config"]])
    assert "DeLong" in capsys.readouterr().out
