"""Tests for the S57/S58 gate variants (entropy-gated d1 refinement, multi-scale texture gates, noise augmentation)."""

from __future__ import annotations

import pytest
import torch

from matblast.models import build_model


def _cfg(**model):
    return {
        "data": {"num_classes": 5},
        "model": {"name": "matblast_unet", "base_channels": 4, "deep_supervision": True, **model},
    }


def _forward(model, seed=0):
    torch.manual_seed(seed)
    return model(torch.rand(2, 3, 32, 32))


def test_gate_variants_off_is_baseline():
    torch.manual_seed(0)
    base = build_model(_cfg())
    torch.manual_seed(0)
    off = build_model(_cfg(entropy_gated_refinement=False, multiscale_texture_gate=False))
    assert list(base.state_dict().keys()) == list(off.state_dict().keys())
    base.eval()
    off.eval()
    assert torch.equal(_forward(base)["logits"], _forward(off)["logits"])


def test_entropy_gate_contract_gradients_and_range():
    model = build_model(_cfg(entropy_gated_refinement=True, entropy_gate_hidden_channels=8))
    out = _forward(model)
    assert out["logits"].shape == (2, 5, 32, 32)
    gate = out["entropy_gate"]
    assert gate.shape == (2, 1, 32, 32) and gate.min() >= 0 and gate.max() <= 1
    out["logits"].sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.entropy_gate.parameters()
    )
    # the gate itself carries no gradient (detached coarse logits)
    assert not gate.requires_grad


def test_entropy_gate_requires_deep_supervision():
    with pytest.raises(ValueError):
        build_model(_cfg(deep_supervision=False, entropy_gated_refinement=True))


def test_entropy_gate_is_max_at_uniform_prediction():
    from matblast.modules.gated_texture import EntropyGatedRefiner

    ref = EntropyGatedRefiner(4, 5)
    uniform = torch.zeros(1, 5, 4, 4)
    peaked = torch.zeros(1, 5, 4, 4)
    peaked[:, 0] = 20.0
    assert torch.allclose(ref.gate(uniform, (8, 8)), torch.ones(1, 1, 8, 8), atol=1e-5)
    assert ref.gate(peaked, (8, 8)).max() < 1e-3
    with pytest.raises(ValueError):
        ref.gate(torch.zeros(1, 3, 4, 4), (8, 8))


def test_multiscale_texture_gate_contract_and_gradients():
    model = build_model(_cfg(multiscale_texture_gate=True))
    keys = model.state_dict().keys()
    assert any(k.startswith("texture_gate3.") for k in keys) and any(
        k.startswith("texture_gate2.") for k in keys
    )
    out = _forward(model)
    assert out["logits"].shape == (2, 5, 32, 32)
    out["logits"].sum().backward()
    for name in ("texture_gate3", "texture_gate2"):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters()
        )


def test_train_noise_option(tmp_path):
    import numpy as np
    from PIL import Image

    from matblast.data.dataset import BlastocystDataset

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.fromarray((np.random.default_rng(0).random((16, 16)) * 255).astype(np.uint8)).save(
        img_dir / "a.png"
    )
    clean = BlastocystDataset(str(img_dir), None, 16, ("icm",), augment=False)
    noisy = BlastocystDataset(
        str(img_dir), None, 16, ("icm",), augment="default", train_noise_sigma_max=0.05
    )
    torch.manual_seed(0)
    a = clean[0]["image"]
    torch.manual_seed(0)
    b = noisy[0]["image"]
    assert a.shape == b.shape == (3, 16, 16)
    assert b[0].min() >= 0 and b[0].max() <= 1 and not torch.equal(a[0], b[0])
    # z-score channel is rebuilt from the noisy raw channel
    z = (b[0] - b[0].mean()) / (b[0].std() + 1e-6)
    assert torch.allclose(b[1], z, atol=1e-4)
    with pytest.raises(ValueError):
        BlastocystDataset(str(img_dir), None, 16, ("icm",), train_noise_sigma_max=-1)
