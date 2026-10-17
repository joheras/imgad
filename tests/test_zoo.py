"""Medical model zoo, recommendations and the timm extractor (offline: no weights are downloaded)."""

import numpy as np
import pytest
import yaml

from imgad import build_extractor, zoo
from imgad.cli import main
from imgad.evaluation import _hub_name

timm = pytest.importorskip("timm")


def test_cards_are_consistent():
    names = [c.name for c in zoo.list_cards()]
    assert len(names) == len(set(names))
    for c in zoo.list_cards():
        assert c.access in ("public", "gated") and c.license and c.use
        assert c.general == (c.modalities == ("*",))
        assert c.general or set(c.modalities) <= set(zoo.MODALITIES)
        assert c.spec["type"] in ("hf", "open_clip", "timm")
        ext = build_extractor(f"zoo:{c.name}", device="cpu")          # lazy: weights are not loaded
        if "checkpoint" in c.spec:                                     # raw checkpoint into a timm architecture
            assert ext.checkpoint == c.spec["checkpoint"] and ext.data_config == c.spec.get("data_config", {})
        else:
            assert ext.model_name.endswith(c.hub_id.split("/")[-1]) or c.hub_id.lower() in ext.model_name.lower()
        assert _hub_name(f"zoo:{c.name}") == c.hub_id


def test_recommend_order_and_filters():
    cxr = zoo.recommend("cxr")
    assert cxr[0].name == "rad-dino"                                    # single-modality model first
    assert [c.general for c in cxr] == sorted(c.general for c in cxr)   # general baselines last
    path = [c.name for c in zoo.recommend("pathology", include_general=False)]
    assert path[:3] == ["uni2-h", "uni", "virchow2"] and "rad-dino" not in path
    assert all(c.access == "public" for c in zoo.recommend("histopathology", public_only=True))
    commercial = zoo.recommend("histopathology", commercial=True)
    assert "uni" not in [c.name for c in commercial] and "h-optimus-0" in [c.name for c in commercial]
    assert {c.name for c in zoo.recommend()} == {c.name for c in zoo.list_cards()}
    with pytest.raises(ValueError, match="Unknown modality"):
        zoo.recommend("teeth")


def test_ophthalmology_recommendations():
    fundus = [c.name for c in zoo.recommend("fundus", include_general=False)]
    assert fundus[0] == "retfound-cfp" and {"visionfm-fundus", "retizero", "flair", "retfound-green"} <= set(fundus)
    assert fundus.index("urfound") < fundus.index("medsiglip") < fundus.index("biomedclip")   # fewer modalities first
    oct_ = [c.name for c in zoo.recommend("oct", include_general=False)]
    assert oct_[:3] == ["retfound-oct", "visionfm-oct", "retfound-oct-mirror"] and "retfound-cfp" not in oct_
    assert [c.name for c in zoo.recommend("cfp", include_general=False, commercial=True)][:2] == ["flair", "urfound"]
    retizero = zoo.get("retizero")
    assert not retizero.commercial_ok and retizero.spec["checkpoint"].startswith("gdrive:")
    assert _hub_name("zoo:retfound-cfp") == "YukunZhou/RETFound_mae_natureCFP"                # gated: access checked
    assert _hub_name({"type": "timm", "model_name": "vit_base_patch16_224",
                      "checkpoint": "hf-hub:yyyyk/UrFound/urfound_mm.pth"}) == "hf-hub:yyyyk/UrFound"


def test_benchmark_yaml_and_cli(capsys):
    block = yaml.safe_load(zoo.benchmark_yaml(zoo.recommend("cxr", public_only=True)))
    assert block["extractors"][0] == {"type": "hf", "model_name": "microsoft/rad-dino"}
    main(["models", "-m", "fundus", "--no-general"])
    out = capsys.readouterr().out
    assert "ophthalmology" in out and "zoo:medsiglip" in out and "zoo:dinov2-base" not in out
    main(["models", "-m", "pathology", "--public-only", "--yaml"])
    assert yaml.safe_load(capsys.readouterr().out)["extractors"][0]["model_name"] == "owkin/phikon-v2"


def test_virchow2_style_timm_recipe_serialises():
    spec = zoo.get("uni2-h").spec
    assert spec["timm_kwargs"]["act_layer"] == "torch.nn.SiLU"            # string, so configs stay serialisable
    ext = build_extractor("zoo:uni2-h")
    assert "SiLU" in ext.identifier


@pytest.mark.parametrize("pooling,dim", [("auto", 192), ("cls", 192), ("mean", 192), ("cls+mean", 384)])
def test_timm_extractor_poolings(pooling, dim):
    ext = build_extractor({"type": "timm", "model_name": "vit_tiny_patch16_224", "pretrained": False,
                           "pooling": pooling, "device": "cpu",
                           "timm_kwargs": {"reg_tokens": 4, "no_embed_class": True, "act_layer": "torch.nn.GELU"}})
    imgs = [(np.random.default_rng(i).random((50, 70, 3)) * 255).astype(np.uint8) for i in range(3)]
    X = ext.extract(imgs, show_progress=False)
    assert X.shape == (3, dim) and np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-5)
    # cls+mean ignores the 4 register tokens: its second half is the mean of the patch tokens only
    if pooling == "cls+mean":
        mean = build_extractor({"type": "timm", "model_name": "vit_tiny_patch16_224", "pretrained": False,
                                "pooling": "mean", "normalize": False, "device": "cpu",
                                "timm_kwargs": {"reg_tokens": 4, "no_embed_class": True}})
        assert mean._load() is None and mean._model.num_prefix_tokens == 5
