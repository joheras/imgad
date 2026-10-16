import numpy as np
import pytest
from PIL import Image

from imgad import FeatureExtractor
from imgad.features import register_extractor


@register_extractor("stats")
class ColorStatsExtractor(FeatureExtractor):
    """Model-free extractor (per-channel mean/std/max) used to test the pipeline logic."""

    def __init__(self, model_name: str = "stats", batch_size: int = 8):
        self.model_name, self.batch_size = model_name, batch_size

    def _extract_batch(self, images):
        out = []
        for im in images:
            a = np.asarray(im, dtype=np.float32) / 255.0
            out.append(np.r_[a.mean((0, 1)), a.std((0, 1)), a.max((0, 1)), (a > 0.9).mean()])
        return np.array(out)

    def get_config(self):
        return {"model_name": self.model_name, "batch_size": self.batch_size}


@pytest.fixture
def gaussian_data():
    """Normal data: isotropic Gaussian; anomalies: shifted Gaussian."""
    rng = np.random.default_rng(0)
    d = 32
    Xtr = rng.normal(size=(200, d))
    Xte_n = rng.normal(size=(100, d))
    Xte_a = rng.normal(loc=1.5, size=(50, d))
    Xte = np.vstack([Xte_n, Xte_a])
    y = np.r_[np.zeros(100), np.ones(50)].astype(int)
    return Xtr, Xte, y


@pytest.fixture
def image_folder(tmp_path):
    """Tiny MVTec-style dataset: normal = dark noisy images, anomalies = dark images with a bright square
    (with ground-truth masks in ``ground_truth/scratch``)."""
    rng = np.random.default_rng(0)
    root = tmp_path / "toy"

    def save(folder, n, anomalous):
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = (rng.random((48, 48, 3)) * 60).astype(np.uint8)
            if anomalous:
                x, y = rng.integers(0, 28, size=2)
                arr[y : y + 20, x : x + 20] = 255
                mask = np.zeros((48, 48), dtype=np.uint8)
                mask[y : y + 20, x : x + 20] = 255
                gt = root / "cat" / "ground_truth" / folder.name
                gt.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(gt / f"{i:03d}_mask.png")
            Image.fromarray(arr).save(folder / f"{i:03d}.png")

    save(root / "cat" / "train" / "good", 20, False)
    save(root / "cat" / "test" / "good", 8, False)
    save(root / "cat" / "test" / "scratch", 8, True)
    return root
