"""How to plug in your own detector or feature extractor.

Both become available by name in Python, YAML configs and the CLI once registered.
"""

import numpy as np

from imgad import AnomalyDetectionPipeline, BaseDetector, FeatureExtractor, register_detector, register_extractor


# --------------------------------------------------------------------------- custom detector
@register_detector("max_cosine")
class MaxCosineDetector(BaseDetector):
    """1 - max cosine similarity to any normal sample (== kNN k=1 with cosine, written by hand)."""

    def _fit(self, X):
        self.bank_ = X / np.linalg.norm(X, axis=1, keepdims=True)

    def _score(self, X):
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        return 1.0 - (Xn @ self.bank_.T).max(1)

    def _train_scores(self, X):  # leave-one-out so the threshold is not biased
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        S = Xn @ self.bank_.T
        np.fill_diagonal(S, -np.inf)
        return 1.0 - S.max(1)


# --------------------------------------------------------------------------- custom extractor
@register_extractor("timm")
class TimmExtractor(FeatureExtractor):
    """Any timm model (pip install timm), e.g. 'hf-hub:timm/vit_base_patch16_clip_224.openai'."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32):
        import timm

        self.model_name, self.device, self.batch_size = model_name, device, batch_size
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(device)
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**cfg)

    def _extract_batch(self, images):
        import torch

        x = torch.stack([self.transform(im) for im in images]).to(self.device)
        with torch.inference_mode():
            return torch.nn.functional.normalize(self.model(x), dim=-1).cpu().numpy()

    def get_config(self):
        return {"model_name": self.model_name, "device": self.device, "batch_size": self.batch_size}


if __name__ == "__main__":
    pipe = AnomalyDetectionPipeline("openai/clip-vit-base-patch32", "max_cosine")
    print(pipe)
