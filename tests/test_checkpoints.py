"""Raw checkpoints loaded into timm architectures (RETFound, VisionFM, RetiZero, FLAIR style), offline."""

import argparse

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from imgad import build_extractor  # noqa: E402
from imgad.features import timm_models  # noqa: E402
from imgad.features.timm_models import load_checkpoint, read_state_dict  # noqa: E402


def _vit(**kw):
    return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0, **kw)


def _same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def test_dino_style_checkpoint_with_metadata(tmp_path):
    src = _vit()
    teacher = {f"backbone.{k}": v for k, v in src.state_dict().items()}
    teacher["head.mlp.0.weight"] = torch.zeros(4, 4)                      # DINO/iBOT projection head: ignored
    path = tmp_path / "ckpt.pth"
    # training metadata next to the weights, as in MAE / iBOT checkpoints
    torch.save({"teacher": teacher, "student": {}, "epoch": 400, "args": argparse.Namespace(arch="vit_tiny"),
                "loss": np.float64(0.5), "dtype": np.dtype("float32")}, path)
    dst = _vit()
    assert load_checkpoint(dst, str(path), key="teacher") == []
    assert _same(src, dst)
    assert set(read_state_dict(str(path))) == set(teacher)              # "teacher" found without a key


def test_prefix_selects_a_sub_model_and_pos_embed_is_resampled(tmp_path):
    src = _vit()
    sd = {f"vision_model.model.{k}": v for k, v in src.state_dict().items()}
    sd["text_model.embeddings.weight"] = torch.zeros(10, 4)
    path = tmp_path / "vlm.pth"
    torch.save(sd, path)
    dst = _vit(img_size=320)                                               # 20x20 patches instead of 14x14
    assert load_checkpoint(dst, str(path), prefix="vision_model.model.") == []
    assert dst.pos_embed.shape[1] == 1 + 20 * 20
    assert torch.equal(dst.blocks[0].attn.qkv.weight, src.blocks[0].attn.qkv.weight)


def test_lora_adapters_are_merged(tmp_path):
    src = _vit()
    sd, dim, r = {}, 192, 4
    g = torch.Generator().manual_seed(0)
    for k, v in src.state_dict().items():
        sd[k.replace("attn.qkv.", "attn.qkv.qkv.")] = v
    for i in range(len(src.blocks)):
        for part in "qv":
            sd[f"blocks.{i}.attn.qkv.linear_a_{part}.weight"] = torch.randn(r, dim, generator=g)
            sd[f"blocks.{i}.attn.qkv.linear_b_{part}.weight"] = torch.randn(dim, r, generator=g)
    path = tmp_path / "lora.pth"
    torch.save(sd, path)
    dst = _vit()
    assert load_checkpoint(dst, str(path)) == []
    x = torch.randn(2, 7, dim)
    ref = src.blocks[3].attn.qkv(x)
    ref[..., :dim] += x @ sd["blocks.3.attn.qkv.linear_a_q.weight"].T @ sd["blocks.3.attn.qkv.linear_b_q.weight"].T
    ref[..., -dim:] += x @ sd["blocks.3.attn.qkv.linear_a_v.weight"].T @ sd["blocks.3.attn.qkv.linear_b_v.weight"].T
    assert torch.allclose(dst.blocks[3].attn.qkv(x), ref, atol=1e-4)


def test_wrong_architecture_is_an_error(tmp_path):
    path = tmp_path / "resnet.pth"
    torch.save(timm.create_model("resnet18", pretrained=False).state_dict(), path)
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(_vit(), str(path))


def test_pickled_code_is_refused(tmp_path):
    path = tmp_path / "evil.pth"
    torch.save({"model": _vit().state_dict(), "hook": argparse.ArgumentParser}, path)   # not allow-listed
    with pytest.raises(Exception):
        read_state_dict(str(path))


def test_web_page_instead_of_file(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(torch.hub, "download_url_to_file",
                        lambda url, dst, progress=False: open(dst, "w").write("<!DOCTYPE html><html>quota</html>"))
    with pytest.raises(RuntimeError, match="web page"):
        timm_models.checkpoint_file("gdrive:abc123/model.pth")
    assert not (tmp_path / "checkpoints" / "model.pth").exists()           # the bad file is not cached


def test_extractor_with_checkpoint_and_data_config(tmp_path):
    path = tmp_path / "w.safetensors"
    from safetensors.torch import save_file

    save_file(_vit().state_dict(), str(path))
    ext = build_extractor({"type": "timm", "model_name": "vit_tiny_patch16_224", "pooling": "mean", "device": "cpu",
                           "checkpoint": str(path),
                           "data_config": {"mean": [0, 0, 0], "std": [1, 1, 1], "crop_pct": 1.0}})
    imgs = [(np.random.default_rng(i).random((40, 60, 3)) * 255).astype(np.uint8) for i in range(2)]
    assert ext.extract(imgs, show_progress=False).shape == (2, 192)
    assert "Normalize(mean=tensor([0, 0, 0])" in str(ext._transform)
    assert ext.get_config()["checkpoint"] == str(path) and "w.safetensors" in ext.identifier
