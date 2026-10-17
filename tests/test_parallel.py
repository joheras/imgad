import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from imgad import FeatureExtractor, benchmark, data
from imgad.cli import main
from imgad.devices import GiB, DeviceInfo, ResourceManager, resolve_devices
from imgad.features import register_extractor

TIMING = {"feat_time_s", "det_time_s", "fit_time_s", "test_time_s"}


def _strip(rows):
    return [{k: v for k, v in r.items() if k not in TIMING} for r in rows]


# ----------------------------------------------------------------------------- memory admission
def _gpu(name="cuda:0", total=8, free=8):
    return DeviceInfo(name, "cuda", "fake GPU", int(total * GiB), int(free * GiB))


def test_resource_manager_admission():
    free = {"cuda:0": 8 * GiB}
    rm = ResourceManager([_gpu()], max_jobs="auto", safety=1.0, margin_bytes=0, refresh_s=0,
                         query=lambda: [DeviceInfo("cuda:0", "cuda", "", 8 * GiB, free["cuda:0"])])
    assert rm.try_acquire(1, 3 * GiB) == "cuda:0"
    assert rm.try_acquire(2, 3 * GiB) == "cuda:0"
    assert rm.try_acquire(3, 3 * GiB) is None            # budget: 8 - 6 = 2 GiB left
    assert rm.try_acquire(4, 2 * GiB) == "cuda:0"        # a smaller job still fits
    rm.release(1)
    free["cuda:0"] = 1 * GiB                              # another process took the memory meanwhile
    assert rm.try_acquire(5, 2 * GiB) is None
    free["cuda:0"] = 8 * GiB
    assert rm.try_acquire(5, 2 * GiB) == "cuda:0"
    assert not rm.fits_somewhere(9 * GiB)
    assert rm.try_acquire(6, 9 * GiB, exclusive=True) is None   # exclusive waits for an idle device
    for j in (2, 4, 5):
        rm.release(j)
    assert rm.try_acquire(6, 9 * GiB, exclusive=True) == "cuda:0"


def test_resource_manager_picks_device_with_most_room_and_caps_jobs():
    devs = [_gpu("cuda:0", free=4), _gpu("cuda:1", free=8)]
    rm = ResourceManager(devs, max_jobs=1, safety=1.0, margin_bytes=0, query=lambda: devs)
    assert rm.try_acquire(1, GiB) == "cuda:1"
    assert rm.try_acquire(2, GiB) == "cuda:0"
    assert rm.try_acquire(3, GiB) is None                 # max_jobs=1 per device
    cpu = ResourceManager([DeviceInfo("cpu", "cpu")])
    assert cpu.try_acquire(1, 100 * GiB) == "cpu" and cpu.try_acquire(2, 1) is None


def test_resolve_devices():
    assert resolve_devices("cpu")[0].name == "cpu"
    assert resolve_devices("auto")                        # GPUs, or the CPU when there is none
    with pytest.raises(ValueError, match="not available"):
        resolve_devices("cuda:99")


# ----------------------------------------------------------------------------- parallel == sequential
@pytest.fixture
def two_categories(image_folder):
    shutil.copytree(image_folder / "cat", image_folder / "dog")
    return image_folder


KW = dict(extractors=[{"type": "stats"}, {"type": "stats", "model_name": "stats2"}],
          detectors=[{"name": "knn", "k": 1}, "mahalanobis", "iforest"], preprocessings=[["standard"], None],
          seeds=[0, 1], train_subsample=0.8, bootstrap=20, keep_scores=True, verbose=False)


def test_parallel_matches_sequential(two_categories):
    dss = [data.mvtec_ad(two_categories, c) for c in ("cat", "dog")]
    seq = benchmark(datasets=dss, **KW)
    par = benchmark(datasets=dss, n_jobs=3, devices="cpu", **KW)
    assert len(seq) == len(par) == 2 * 2 * 2 * 2 * 3
    for a, b in zip(_strip(seq), _strip(par)):
        sa, sb = a.pop("scores"), b.pop("scores")
        a.pop("labels"), b.pop("labels")
        assert a == b
        np.testing.assert_allclose(sa, sb)


@register_extractor("oom_once")
class OOMOnceExtractor(FeatureExtractor):
    """Raises a CUDA-like out-of-memory error the first time it is loaded (marker file shared by processes)."""

    def __init__(self, marker: str, batch_size: int = 8):
        self.marker, self.batch_size = marker, batch_size

    def _load(self):
        if not Path(self.marker).exists():
            Path(self.marker).write_text("1")
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    def _extract_batch(self, images):
        return np.array([np.asarray(im, dtype=np.float32).mean((0, 1)) for im in images])

    def get_config(self):
        return {"marker": self.marker, "batch_size": self.batch_size}


def test_out_of_memory_job_is_retried_alone(two_categories, tmp_path, caplog):
    ds = data.mvtec_ad(two_categories, "cat")
    marker = tmp_path / "oom_marker"
    with caplog.at_level("INFO", logger="imgad"):
        rows = benchmark([{"type": "oom_once", "marker": str(marker)}], ["mahalanobis"], [ds], n_jobs=2,
                         devices="cpu", verbose=False)
    assert len(rows) == 1 and marker.exists()
    assert any("Out of memory" in r.getMessage() and "retrying it alone" in r.getMessage() for r in caplog.records)


def test_parallel_failure_is_recorded_and_the_rest_continues(two_categories, caplog):
    ds = data.mvtec_ad(two_categories, "cat")
    with caplog.at_level("ERROR", logger="imgad"):
        rows = benchmark([{"type": "stats"}], [{"name": "knn", "metric": "no-such-metric"}, "mahalanobis"], [ds],
                         n_jobs=2, devices="cpu", verbose=False)
    failed = [r for r in rows if r.get("error")]
    assert len(rows) == 2 and len(failed) == 1 and "auroc" not in failed[0] and failed[0]["stage"] == "detector"
    assert [r["detector"] for r in rows if not r.get("error")] == ["mahalanobis"]
    assert any("FAILED" in r.getMessage() for r in caplog.records)
    with pytest.raises(Exception):
        benchmark([{"type": "stats"}], [{"name": "knn", "metric": "no-such-metric"}], [ds], n_jobs=2, devices="cpu",
                  verbose=False, on_error="raise")


def test_dead_worker_process_is_recorded_and_the_rest_continues(two_categories, caplog):
    from conftest import CrashingDetector

    ds = data.mvtec_ad(two_categories, "cat")
    with caplog.at_level("WARNING", logger="imgad"):
        rows = benchmark([{"type": "stats"}], [CrashingDetector(), "centroid", "mahalanobis"], [ds], n_jobs=2,
                         devices="cpu", verbose=False)
    ok = [r["detector"] for r in rows if not r.get("error")]
    failed = [r for r in rows if r.get("error")]
    assert sorted(ok) == ["centroid", "mahalanobis"]
    assert len(failed) == 1 and "BrokenProcessPool" in failed[0]["error"] and failed[0]["dataset"] == ds.name
    assert any("worker process died" in r.getMessage() for r in caplog.records)


# ----------------------------------------------------------------------------- CLI
def test_cli_parallel_options_and_devices(two_categories, tmp_path, capsys):
    cfg = tmp_path / "b.yaml"
    cfg.write_text(yaml.safe_dump({"data": {"type": "mvtec", "root": str(two_categories)},
                                   "extractors": [{"type": "stats"}], "detectors": ["knn", "mahalanobis"],
                                   "parallel": {"jobs": 2, "devices": "cpu"}}))
    out = tmp_path / "r.csv"
    main(["benchmark", "--config", str(cfg), "--out", str(out)])
    assert "Parallel execution: 2 CPU worker(s)" in capsys.readouterr().err
    assert len(out.read_text().strip().splitlines()) == 1 + 4
    main(["benchmark", "--config", str(cfg), "-j", "3", "--devices", "cpu", "-q"])   # flags override the YAML
    main(["devices", "--devices", "cpu"])
    text = capsys.readouterr().out
    assert "Devices:" in text and "cpu" in text and "up to 1 concurrent job(s)" in text
