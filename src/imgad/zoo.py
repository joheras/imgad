"""Curated feature extractors for medical imaging, with recommendations per modality.

    from imgad import zoo
    zoo.recommend("chest-xray")                     # ranked ModelCards (domain models, then general baselines)
    zoo.recommend("histopathology", commercial=True, public_only=True)
    build_extractor("zoo:rad-dino")                 # any card by name, also in benchmark YAML files

    imgad models --modality histopathology --check --yaml

Every card records how to load the model (an extractor spec), its access (public / gated), license and
intended use. These facts come from the model cards on the Hugging Face Hub; verify them on the model page
before relying on them, especially for any use beyond research.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

MODALITIES: Dict[str, str] = {
    "chest-xray": "Chest radiographs",
    "ct": "Computed tomography (2-D slices)",
    "mri": "Magnetic resonance (2-D slices)",
    "histopathology": "H&E (and IHC) histology tiles",
    "dermatology": "Dermoscopic and clinical skin images",
    "ophthalmology": "Retinal fundus photographs",
    "oct": "Optical coherence tomography",
}
ALIASES = {
    "xray": "chest-xray", "x-ray": "chest-xray", "cxr": "chest-xray", "radiography": "chest-xray",
    "chest": "chest-xray", "pathology": "histopathology", "histology": "histopathology", "wsi": "histopathology",
    "h&e": "histopathology", "he": "histopathology", "skin": "dermatology", "derm": "dermatology",
    "dermoscopy": "dermatology", "fundus": "ophthalmology", "retina": "ophthalmology", "retinal": "ophthalmology",
    "eye": "ophthalmology", "brain-mri": "mri", "mr": "mri", "tomography": "ct",
}
NON_COMMERCIAL = ("cc-by-nc", "non-commercial", "research only", "research use")


@dataclass(frozen=True)
class ModelCard:
    name: str                          # short name, usable as "zoo:<name>"
    spec: Dict[str, Any]               # extractor spec for build_extractor / benchmark YAML
    modalities: Tuple[str, ...]        # MODALITIES keys; ("*",) for general-purpose baselines
    hub_id: str
    access: str                        # "public" | "gated"
    license: str
    use: str                           # intended use as stated by the authors
    dim: Optional[int] = None          # embedding dimension (None if not documented)
    intensity: str = "auto"            # suggested rule for >8-bit inputs (see imgad.imageio)
    notes: str = ""
    reference: str = ""
    general: bool = False

    @property
    def commercial_ok(self) -> bool:
        text = f"{self.license} {self.use}".lower()
        return not any(k in text for k in NON_COMMERCIAL)

    def extractor_spec(self) -> Dict[str, Any]:
        spec = dict(self.spec)
        if self.intensity != "auto":
            spec.setdefault("intensity", self.intensity)
        return spec


def _hf(model_name: str, **kw) -> Dict[str, Any]:
    return {"type": "hf", "model_name": model_name, **kw}


def _timm(model_name: str, pooling: str = "auto", **timm_kwargs) -> Dict[str, Any]:
    spec = {"type": "timm", "model_name": model_name, "pooling": pooling}
    if timm_kwargs:
        spec["timm_kwargs"] = timm_kwargs
    return spec


CARDS: List[ModelCard] = [
    # ---------------------------------------------------------------- radiology / multi-modality
    ModelCard(
        "rad-dino", _hf("microsoft/rad-dino"), ("chest-xray",), "microsoft/rad-dino", "public", "MIT",
        "research only, not for clinical practice", 768,
        notes="DINOv2-B continued on chest X-rays from five public datasets (MIMIC-CXR, CheXpert, NIH-CXR, PadChest…); "
              "no text supervision.",
        reference="Pérez-García et al., Nature Machine Intelligence 2025",
    ),
    ModelCard(
        "medsiglip", _hf("google/medsiglip-448"),
        ("chest-xray", "ct", "mri", "dermatology", "ophthalmology", "histopathology"),
        "google/medsiglip-448", "gated", "Health AI Developer Foundations terms",
        "development tool for research; outputs not for clinical decisions without validation", None,
        notes="SigLIP image encoder (448 px) trained on medical image–text pairs: chest X-ray, CT/MRI slices, "
              "dermatology, ophthalmology, histopathology.",
        reference="google/medsiglip-448 model card",
    ),
    ModelCard(
        "biomedclip",
        {"type": "open_clip", "model_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"},
        ("chest-xray", "ct", "mri", "dermatology", "ophthalmology", "oct", "histopathology"),
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", "public", "MIT",
        "research only; deployed use out of scope", 512,
        notes="CLIP ViT-B/16 trained on PMC-15M (15M figure–caption pairs from PubMed Central): broad but "
              "figure-style images. Needs open_clip.",
        reference="Zhang et al., BiomedCLIP (arXiv 2303.00915)",
    ),
    # ---------------------------------------------------------------- histopathology
    ModelCard(
        "uni2-h",
        _timm("hf-hub:MahmoodLab/UNI2-h", img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
              embed_dim=1536, mlp_ratio=2.66667 * 2, no_embed_class=True, mlp_layer="timm.layers.SwiGLUPacked",
              act_layer="torch.nn.SiLU", reg_tokens=8, dynamic_img_size=True),
        ("histopathology",), "MahmoodLab/UNI2-h", "gated", "CC-BY-NC-ND-4.0",
        "non-commercial academic research; institutional e-mail required", 1536,
        notes="ViT-H/14 on >200M H&E and IHC tiles from >300k slides.", reference="Chen et al., Nature Medicine 2024 (UNI)",
    ),
    ModelCard(
        "uni", _timm("hf-hub:MahmoodLab/uni", init_values=1e-5, dynamic_img_size=True),
        ("histopathology",), "MahmoodLab/UNI", "gated", "CC-BY-NC-ND-4.0",
        "non-commercial academic research; institutional e-mail required", 1024,
        notes="ViT-L/16 on ~100M tiles from ~100k H&E slides (Mass-100K).", reference="Chen et al., Nature Medicine 2024",
    ),
    ModelCard(
        "virchow2", _timm("hf-hub:paige-ai/Virchow2", pooling="cls+mean", mlp_layer="timm.layers.SwiGLUPacked",
                          act_layer="torch.nn.SiLU"),
        ("histopathology",), "paige-ai/Virchow2", "gated", "CC-BY-NC-ND-4.0",
        "non-commercial academic research; not for clinical or diagnostic use", 2560,
        notes="ViT-H/14 with 4 register tokens; embedding = class token + mean patch token, as in the model card.",
        reference="paige-ai/Virchow2 model card",
    ),
    ModelCard(
        "h-optimus-0", _timm("hf-hub:bioptimus/H-optimus-0", init_values=1e-5, dynamic_img_size=False),
        ("histopathology",), "bioptimus/H-optimus-0", "gated", "Apache-2.0", "see model card", 1536,
        notes="ViT-g/14; expects 224×224 tiles at 0.5 µm/px (its own normalisation is applied by timm).",
        reference="bioptimus/H-optimus-0 model card",
    ),
    ModelCard(
        "prov-gigapath", _timm("hf-hub:prov-gigapath/prov-gigapath"),
        ("histopathology",), "prov-gigapath/prov-gigapath", "gated", "Apache-2.0", "research (see model card)", 1536,
        notes="Tile encoder (ViT-g) of Prov-GigaPath.", reference="Xu et al., Nature 2024",
    ),
    ModelCard(
        "phikon-v2", _hf("owkin/phikon-v2", pooling="cls"),
        ("histopathology",), "owkin/phikon-v2", "public", "Owkin non-commercial license",
        "non-commercial research", 1024,
        notes="DINOv2 ViT-L/16 on ~456M tiles from 60k H&E slides (TCGA, CPTAC, GTEx…). Public, no request needed.",
        reference="owkin/phikon-v2 model card",
    ),
    # ---------------------------------------------------------------- general baselines
    ModelCard(
        "dinov2-base", _hf("facebook/dinov2-base"), ("*",), "facebook/dinov2-base", "public", "Apache-2.0",
        "general purpose", 768, notes="Strong general self-supervised baseline.", general=True,
        reference="Oquab et al., TMLR 2024",
    ),
    ModelCard(
        "dinov3-vitb16", _hf("facebook/dinov3-vitb16-pretrain-lvd1689m"), ("*",),
        "facebook/dinov3-vitb16-pretrain-lvd1689m", "gated", "DINOv3 license", "see model card", 768,
        notes="Newer self-supervised baseline (gated).", general=True, reference="Siméoni et al., 2025",
    ),
    ModelCard(
        "siglip2-base", _hf("google/siglip2-base-patch16-224"), ("*",), "google/siglip2-base-patch16-224", "public",
        "Apache-2.0", "general purpose", 768, notes="Vision–language baseline.", general=True,
        reference="Tschannen et al., 2025",
    ),
]
_BY_NAME = {c.name: c for c in CARDS}


def modality(name: str) -> str:
    """Canonical modality key (accepts aliases such as ``cxr``, ``pathology``, ``fundus``)."""
    key = name.strip().lower().replace("_", "-").replace(" ", "-")
    key = ALIASES.get(key, key)
    if key not in MODALITIES:
        raise ValueError(f"Unknown modality {name!r}. Known: {', '.join(MODALITIES)} (aliases: {', '.join(sorted(ALIASES))})")
    return key


def get(name: str) -> ModelCard:
    """Card by short name (``"rad-dino"`` or ``"zoo:rad-dino"``)."""
    name = name[4:] if name.startswith("zoo:") else name
    if name not in _BY_NAME:
        raise KeyError(f"Unknown zoo model {name!r}. Available: {', '.join(_BY_NAME)}")
    return _BY_NAME[name]


def list_cards() -> List[ModelCard]:
    return list(CARDS)


def recommend(
    modality_name: Optional[str] = None,
    include_general: bool = True,
    public_only: bool = False,
    commercial: bool = False,
) -> List[ModelCard]:
    """Models to benchmark for a modality, most specific first: domain models trained on that modality
    (single-modality models before multi-modality ones), then general-purpose baselines.

    ``public_only`` drops gated models; ``commercial`` drops models whose license or terms are
    non-commercial / research-only. Without a modality, every card is returned (filters still apply).
    A recommendation is a starting point to benchmark on your own data, not a ranking of quality.
    """
    m = modality(modality_name) if modality_name else None
    picked = []
    for i, c in enumerate(CARDS):
        if public_only and c.access != "public":
            continue
        if commercial and not c.commercial_ok:
            continue
        if c.general:
            if include_general:
                picked.append((2, i, c))
        elif m is None or m in c.modalities:
            picked.append((0 if len(c.modalities) == 1 else 1, i, c))
    return [c for _, _, c in sorted(picked, key=lambda t: (t[0], t[1]))]


def benchmark_yaml(cards: Sequence[ModelCard]) -> str:
    """``extractors:`` block for a benchmark YAML with these models (intensity rules included)."""
    import yaml

    return yaml.safe_dump({"extractors": [c.extractor_spec() for c in cards]}, sort_keys=False, default_flow_style=None)
