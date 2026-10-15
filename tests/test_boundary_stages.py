"""Tests for the S48+ boundary / sparse refinement stages."""

from __future__ import annotations

import pytest
import torch

from matblast.losses import CombinedLoss
from matblast.models import build_model


def _cfg(**model):
    return {
        "data": {"num_classes": 5},
        "model": {"name": "matblast_unet", "base_channels": 4, **model},
    }


# --------------------------------------------------------------------------- S48
def test_deep_strip_normals_are_unit_and_strip_shapes():
    from matblast.modules.deep_strip import DeepStripRefiner, boundary_normals

    torch.manual_seed(0)
    logits = torch.randn(2, 5, 16, 16)
    coords = torch.rand(2, 7, 2)
    normals = boundary_normals(logits, coords, (1, 2, 4))
    assert normals.shape == (2, 7, 2)
    assert torch.allclose(normals.norm(dim=-1), torch.ones(2, 7), atol=1e-5)
    refiner = DeepStripRefiner(4, 5, num_strips=7, strip_length=5, hidden_channels=8)
    strips = refiner._strip_coords(coords, normals, (16, 16))
    assert strips.shape == (2, 35, 2) and strips.min() >= 0 and strips.max() <= 1
    with pytest.raises(ValueError):
        DeepStripRefiner(4, 5, strip_length=4)
    with pytest.raises(ValueError):
        DeepStripRefiner(4, 5, classes_of_interest=(4,))


def test_deep_strip_train_contract_and_gradients():
    model = build_model(
        _cfg(deep_strip=True, deep_strip_num_strips=32, deep_strip_hidden_channels=8)
    )
    model.train()
    x, y = torch.randn(2, 3, 32, 32), torch.randint(0, 5, (2, 32, 32))
    out = model(x)
    assert out["logits"].shape == (2, 5, 32, 32)
    assert out["point_logits"].shape == (2, 5, 32) and out["point_coords"].shape == (2, 32, 2)
    loss = CombinedLoss(point_loss_weight=1.0)(out, y)
    assert float(loss["point"]) > 0
    loss["total_loss"].backward()
    assert all(p.grad is not None for p in model.deep_strip.parameters())
    assert all(p.grad is not None for p in model.dec1.parameters())


def test_deep_strip_eval_refines_only_selected_pixels_and_is_deterministic():
    torch.manual_seed(1)
    model = build_model(
        _cfg(deep_strip=True, deep_strip_hidden_channels=8, deep_strip_inference_num_strips=40)
    ).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
        again = model(x)["logits"]
    assert out["logits"].shape == (1, 5, 32, 32) and "point_logits" not in out
    changed = (out["logits"] != out["dense_logits"]).any(dim=1).sum()
    assert 0 < int(changed) <= 40
    assert torch.equal(out["logits"], again)


def test_deep_strip_toggle_off_is_baseline_and_exclusive_with_pointrend():
    image = torch.randn(1, 3, 32, 32)
    torch.manual_seed(17)
    baseline = build_model(_cfg())
    torch.manual_seed(17)
    disabled = build_model(_cfg(deep_strip=False))
    assert set(baseline.state_dict()) == set(disabled.state_dict())
    with torch.no_grad():
        assert torch.equal(baseline(image)["logits"], disabled(image)["logits"])
    with pytest.raises(ValueError, match="only one"):
        build_model(_cfg(deep_supervision=True, pointrend=True, deep_strip=True))


# --------------------------------------------------------------------------- S49
def test_refine_mask_contract_stages_and_gradients():
    model = build_model(_cfg(refine_mask=True, refine_mask_hidden_channels=8))
    x, y = torch.randn(2, 3, 32, 32), torch.randint(0, 5, (2, 32, 32))
    out = model(x)
    assert out["logits"].shape == (2, 5, 32, 32)
    assert out["refine_logits_1"].shape == (2, 5, 32, 32) and out["refine_logits_2"].shape == (
        2,
        5,
        32,
        32,
    )
    assert "refine_logits_3" not in out  # the final stage is the main output
    loss = CombinedLoss(refine_loss_weight=0.5)(out, y)
    assert float(loss["refine"]) > 0
    loss["total_loss"].backward()
    assert all(p.grad is not None for p in model.refine_mask.parameters())
    assert float(CombinedLoss()(out, y)["refine"]) == 0.0  # weight 0 -> ignored
    with pytest.raises(ValueError):
        build_model(_cfg(refine_mask=True, refine_mask_hidden_channels=0))


def test_refine_mask_boundary_aware_toggle_changes_output_and_off_is_baseline():
    image = torch.randn(1, 3, 32, 32)
    torch.manual_seed(5)
    aware = build_model(_cfg(refine_mask=True, refine_mask_hidden_channels=8)).eval()
    torch.manual_seed(5)
    plain = build_model(
        _cfg(refine_mask=True, refine_mask_hidden_channels=8, refine_mask_boundary_aware=False)
    ).eval()
    assert set(aware.state_dict()) == set(plain.state_dict())
    with torch.no_grad():
        assert not torch.equal(aware(image)["logits"], plain(image)["logits"])
    torch.manual_seed(17)
    baseline = build_model(_cfg())
    torch.manual_seed(17)
    disabled = build_model(_cfg(refine_mask=False))
    assert set(baseline.state_dict()) == set(disabled.state_dict())
    with torch.no_grad():
        assert torch.equal(baseline(image)["logits"], disabled(image)["logits"])
    with pytest.raises(ValueError, match="only one"):
        build_model(_cfg(refine_mask=True, deep_strip=True))


# --------------------------------------------------------------------------- S50
def test_transfiner_sampling_and_validation():
    from matblast.modules.mask_transfiner import MaskTransfinerLite, sample_by_score

    torch.manual_seed(0)
    score = torch.rand(2, 1, 8, 8)
    coords = sample_by_score(score, 16, 3.0, 0.75)
    assert coords.shape == (2, 16, 2) and coords.min() >= 0 and coords.max() <= 1
    with pytest.raises(ValueError):
        MaskTransfinerLite((4, 8, 16), 5, nodes_per_level=(8, 8))
    with pytest.raises(ValueError):
        MaskTransfinerLite((4, 8, 16), 5, d_model=30, nhead=4)


def test_transfiner_train_contract_and_gradients():
    model = build_model(
        _cfg(
            mask_transfiner=True,
            transfiner_nodes_per_level=[32, 16, 8],
            transfiner_d_model=16,
            transfiner_dim_feedforward=32,
        )
    )
    model.train()
    x, y = torch.randn(2, 3, 32, 32), torch.randint(0, 5, (2, 32, 32))
    out = model(x)
    assert out["point_logits"].shape == (2, 5, 56) and out["point_coords"].shape == (2, 56, 2)
    loss = CombinedLoss(point_loss_weight=1.0)(out, y)
    loss["total_loss"].backward()
    assert all(p.grad is not None for p in model.mask_transfiner.parameters())
    assert all(p.grad is not None for p in model.dec3.parameters())


def test_transfiner_eval_rewrites_only_fine_nodes_and_is_deterministic():
    torch.manual_seed(2)
    model = build_model(
        _cfg(
            mask_transfiner=True,
            transfiner_nodes_per_level=[24, 8, 4],
            transfiner_d_model=16,
            transfiner_dim_feedforward=32,
        )
    ).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
        again = model(x)["logits"]
    changed = (out["logits"] != out["dense_logits"]).any(dim=1).sum()
    assert 0 < int(changed) <= 24 and torch.equal(out["logits"], again)
    image = torch.randn(1, 3, 32, 32)
    torch.manual_seed(17)
    baseline = build_model(_cfg())
    torch.manual_seed(17)
    disabled = build_model(_cfg(mask_transfiner=False))
    assert set(baseline.state_dict()) == set(disabled.state_dict())
    with torch.no_grad():
        assert torch.equal(baseline(image)["logits"], disabled(image)["logits"])
    with pytest.raises(ValueError, match="only one"):
        build_model(_cfg(mask_transfiner=True, refine_mask=True))


# --------------------------------------------------------------------------- S52
def test_segfix_targets_and_refine_on_toy_labels():
    from matblast.modules.segfix import IGNORE_INDEX, SegFixHead, segfix_targets

    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :, 4:] = 2  # vertical class boundary between columns 3 and 4
    boundary, direction = segfix_targets(target, offset_px=2)
    assert boundary.shape == (1, 1, 8, 8) and direction.shape == (1, 8, 8)
    assert (
        boundary[0, 0, :, 3].all() and boundary[0, 0, :, 4].all() and not boundary[0, 0, :, 0].any()
    )
    assert (direction[0, :, 0] == IGNORE_INDEX).all()  # interior pixels are ignored
    assert (direction[0, :, 3] == 1).all()  # left of the boundary -> walk west (index 1)
    assert (direction[0, :, 4] == 0).all()  # right of the boundary -> walk east (index 0)
    head = SegFixHead(4, hidden_channels=8, offset_px=2, steps=1)
    # perfect boundary/direction predictions should copy interior logits onto the boundary
    logits = torch.zeros(1, 5, 8, 8)
    logits[:, 0, :, :4] = 5.0
    logits[:, 2, :, 4:] = 5.0
    logits[:, 1, :, 3:5] = 9.0  # wrong class on the boundary columns
    boundary_logits = (boundary * 2 - 1) * 10
    direction_logits = F_one_hot_directions(direction)
    refined = head.refine(logits, boundary_logits, direction_logits)
    assert refined.argmax(1)[0, :, 3].eq(0).all() and refined.argmax(1)[0, :, 4].eq(2).all()
    with pytest.raises(ValueError):
        SegFixHead(4, steps=0)


def F_one_hot_directions(direction):
    import torch.nn.functional as F

    safe = direction.clamp_min(0)
    return F.one_hot(safe, 8).permute(0, 3, 1, 2).float() * 10


def test_segfix_model_contract_loss_and_toggle():
    model = build_model(_cfg(segfix=True, segfix_hidden_channels=8))
    x = torch.randn(2, 3, 32, 32)
    y = torch.zeros(2, 32, 32, dtype=torch.long)  # structured labels so interior pixels exist
    y[:, 8:24, 8:24] = 2
    y[:, 12:20, 12:20] = 4
    model.train()
    out = model(x)
    assert out["segfix_boundary"].shape == (2, 1, 32, 32) and out["segfix_direction"].shape == (
        2,
        8,
        32,
        32,
    )
    loss = CombinedLoss(segfix_loss_weight=0.5)(out, y)
    assert float(loss["segfix"]) > 0
    loss["total_loss"].backward()
    assert all(p.grad is not None for p in model.segfix.parameters())
    model.eval()
    with torch.no_grad():
        eval_out = model(x)
        again = model(x)["logits"]
    assert eval_out["logits"].shape == (2, 5, 32, 32) and "dense_logits" in eval_out
    assert torch.equal(eval_out["logits"], again)
    image = torch.randn(1, 3, 32, 32)
    torch.manual_seed(17)
    baseline = build_model(_cfg())
    torch.manual_seed(17)
    disabled = build_model(_cfg(segfix=False))
    assert set(baseline.state_dict()) == set(disabled.state_dict())
    with torch.no_grad():
        assert torch.equal(baseline(image)["logits"], disabled(image)["logits"])
    with pytest.raises(ValueError, match="only one"):
        build_model(_cfg(segfix=True, deep_strip=True))


# --------------------------------------------------------------------------- S53
def test_cascade_psp_contract_gradients_and_patch_bounds():
    from matblast.modules.cascade_psp import CascadePSPLite

    module = CascadePSPLite(3, 5, hidden_channels=8, patch_size=16, num_patches=3)
    origins = module._patch_origins(torch.randn(2, 5, 32, 48))
    assert origins.shape == (2, 3, 2)
    assert (
        (origins[..., 0] <= 32 - 16).all()
        and (origins[..., 1] <= 48 - 16).all()
        and (origins >= 0).all()
    )
    model = build_model(
        _cfg(
            cascade_psp=True,
            cascade_hidden_channels=8,
            cascade_patch_size=16,
            cascade_num_patches=2,
        )
    )
    x, y = torch.randn(2, 3, 32, 32), torch.randint(0, 5, (2, 32, 32))
    out = model(x)
    assert out["logits"].shape == (2, 5, 32, 32) and out["refine_logits_1"].shape == (2, 5, 32, 32)
    loss = CombinedLoss(refine_loss_weight=0.5)(out, y)
    loss["total_loss"].backward()
    assert all(p.grad is not None for p in model.cascade_psp.parameters())
    with pytest.raises(ValueError):
        CascadePSPLite(3, 5, global_scale=0.0)


def test_cascade_psp_deterministic_and_off_is_baseline():
    torch.manual_seed(4)
    model = build_model(
        _cfg(
            cascade_psp=True,
            cascade_hidden_channels=8,
            cascade_patch_size=16,
            cascade_num_patches=2,
        )
    ).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        first = model(x)["logits"]
        assert torch.equal(first, model(x)["logits"])
    torch.manual_seed(17)
    baseline = build_model(_cfg())
    torch.manual_seed(17)
    disabled = build_model(_cfg(cascade_psp=False))
    assert set(baseline.state_dict()) == set(disabled.state_dict())
    with torch.no_grad():
        assert torch.equal(baseline(x)["logits"], disabled(x)["logits"])
    with pytest.raises(ValueError, match="only one"):
        build_model(_cfg(cascade_psp=True, segfix=True))


# ------------------------------------------------------------------ S51/S55/S56
def test_polar_contour_ray_casting_and_rasterisation_roundtrip():
    import math

    from matblast.modules.contour_refinement import cast_radii, rasterise, region_centroids

    # a centred ring: inner radius 0.15, outer 0.30 (normalised), 32x32
    ys, xs = torch.meshgrid(
        (torch.arange(32) + 0.5) / 32, (torch.arange(32) + 0.5) / 32, indexing="ij"
    )
    radius = torch.sqrt((xs - 0.5) ** 2 + (ys - 0.5) ** 2)
    ring = ((radius >= 0.15) & (radius < 0.30)).float().view(1, 1, 32, 32)
    centers = region_centroids(ring)
    assert torch.allclose(centers, torch.tensor([[[0.5, 0.5]]]), atol=0.02)
    angles = torch.arange(64) * (2 * math.pi / 64)
    outer, inner = cast_radii(ring, centers, angles, 128, 0.75)
    assert torch.allclose(outer.mean(), torch.tensor(0.30), atol=0.03)
    assert torch.allclose(inner.mean(), torch.tensor(0.15), atol=0.03)
    raster = rasterise(centers, angles, outer, inner, (32, 32))
    agreement = (raster == ring).float().mean()
    assert agreement > 0.95


@pytest.mark.parametrize("variant", ["snake", "sharp", "poly"])
def test_contour_refiner_contract_loss_and_gradients(variant):
    model = build_model(
        _cfg(
            contour_refiner=variant,
            contour_num_vertices=16,
            contour_hidden_channels=8,
            contour_ray_steps=16,
            contour_iterations=2,
        )
    )
    x = torch.randn(2, 3, 32, 32)
    y = torch.zeros(2, 32, 32, dtype=torch.long)
    y[:, 6:26, 6:26] = 1
    y[:, 9:23, 9:23] = 2
    y[:, 13:19, 13:19] = 4
    out = model(x)
    assert out["logits"].shape == (2, 5, 32, 32) and out["dense_logits"].shape == (2, 5, 32, 32)
    assert out["contour_outer"].shape == (2, 3, 16) and out["contour_centers"].shape == (2, 3, 2)
    loss = CombinedLoss(contour_loss_weight=1.0, contour_ray_steps=16)(out, y)
    assert float(loss["contour"]) > 0
    loss["total_loss"].backward()
    assert all(p.grad is not None for p in model.contour_refiner.parameters())
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x)["logits"], model(x)["logits"])


def test_contour_refiner_off_is_baseline_and_validates():
    x = torch.randn(1, 3, 32, 32)
    torch.manual_seed(17)
    baseline = build_model(_cfg())
    torch.manual_seed(17)
    disabled = build_model(_cfg(contour_refiner=None))
    assert set(baseline.state_dict()) == set(disabled.state_dict())
    with torch.no_grad():
        assert torch.equal(baseline(x)["logits"], disabled(x)["logits"])
    with pytest.raises(ValueError, match="variant"):
        build_model(_cfg(contour_refiner="spline"))
    with pytest.raises(ValueError, match="only one"):
        build_model(_cfg(contour_refiner="snake", segfix=True))
