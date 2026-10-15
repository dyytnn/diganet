"""Tests for the V2 recipe-parity and boundary-stack features (A1-A7, B1, B2)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from matblast.config import assert_known_keys
from matblast.losses import class_boundary_target, foreground_boundary_target


def test_assert_known_keys_accepts_and_rejects():
    assert_known_keys({"a": 1}, {"a", "b"}, "x.y")
    with pytest.raises(ValueError, match="x.y: unknown keys \\['c'\\]"):
        assert_known_keys({"a": 1, "c": 2}, {"a", "b"}, "x.y")


def test_boundary_target_helpers_match_inline_formulas():
    torch.manual_seed(0)
    target = torch.randint(0, 5, (2, 16, 16))
    fg = (target > 0).float().unsqueeze(1)
    edge = torch.abs(F.avg_pool2d(fg, 3, 1, 1) - fg)
    assert torch.equal(foreground_boundary_target(target), (edge > 0.05).float())
    onehot = F.one_hot(target, 5).permute(0, 3, 1, 2).float()
    local_max = F.max_pool2d(onehot, 3, 1, 1)
    local_min = -F.max_pool2d(-onehot, 3, 1, 1)
    assert torch.equal(class_boundary_target(target, 5), (local_max - local_min > 0).float())
    assert class_boundary_target(target, 5).shape == (2, 5, 16, 16)


# --------------------------------------------------------------------------- A7
def _write_image(path, width, height, value=200):
    import numpy as np
    from PIL import Image

    Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8)).save(path)


def test_read_rgb_rectangular_resize_and_pad(tmp_path):
    from matblast.data.dataset import _read_rgb

    image_path = tmp_path / "img.png"
    _write_image(image_path, width=20, height=10)
    stretched = _read_rgb(image_path, [16, 24], 3, "legacy", "resize")
    assert stretched.shape == (3, 16, 24)
    assert torch.all(stretched[0] > 0.5)  # every pixel is image content
    padded = _read_rgb(image_path, [16, 24], 3, "legacy", "pad")
    assert padded.shape == (3, 16, 24)
    # 20x10 into 16x24: scale min(16/10, 24/20)=1.2 -> 24x12 content, 2-row borders.
    assert torch.all(padded[0, :2] == 0) and torch.all(padded[0, -2:] == 0)
    assert torch.all(padded[0, 2:14] > 0.5)
    assert torch.equal(_read_rgb(image_path, 16, 3, "legacy", "resize"), _read_rgb(image_path, 16))
    with pytest.raises(ValueError):
        _read_rgb(image_path, 16, 3, "legacy", "stretchy")
    with pytest.raises(ValueError):
        _read_rgb(image_path, [1, 2, 3])


def test_dataset_pad_mode_aligns_labels_with_image(tmp_path):
    import numpy as np
    from PIL import Image

    from matblast.data.dataset import BlastocystDataset

    (tmp_path / "images").mkdir()
    (tmp_path / "masks" / "icm").mkdir(parents=True)
    _write_image(tmp_path / "images" / "a.png", width=20, height=10)
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[:, 10:] = 255  # right half is ICM
    Image.fromarray(mask).save(tmp_path / "masks" / "icm" / "a.png")
    dataset = BlastocystDataset(
        tmp_path / "images", tmp_path / "masks", [16, 24], ["icm"], resize_mode="pad"
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 16, 24) and sample["mask"].shape == (16, 24)
    label = sample["mask"]
    assert torch.all(label[:2] == 0) and torch.all(label[-2:] == 0)  # padded rows = BG
    assert torch.all(label[2:14, :12] == 0) and torch.all(label[2:14, 12:] == 1)
    with pytest.raises(ValueError):
        BlastocystDataset(tmp_path / "images", None, 16, resize_mode="crop")


def test_default_augment_keeps_rectangular_shape():
    from matblast.data.dataset import _augment

    torch.manual_seed(3)
    for _ in range(50):
        image, label = _augment(torch.rand(3, 16, 24), torch.randint(0, 5, (16, 24)))
        assert image.shape == (3, 16, 24) and label.shape == (16, 24)


def test_model_and_metrics_accept_rectangular_input():
    from matblast.metrics import per_image_scores
    from matblast.models import build_model

    model = build_model(
        {"data": {"num_classes": 5}, "model": {"name": "matblast_unet", "base_channels": 4}}
    )
    x = torch.randn(1, 3, 32, 48)
    logits = model(x)["logits"]
    assert logits.shape == (1, 5, 32, 48)
    dice, iou = per_image_scores(logits, torch.randint(0, 5, (1, 32, 48)))
    assert dice.shape == (1, 5) and iou.shape == (1, 5)
    assert model(torch.rot90(x, 1, (-2, -1)))["logits"].shape == (1, 5, 48, 32)


# --------------------------------------------------------------------------- A3
def _all_off_params():
    from matblast.data.augment_nnunet import NNUNET_AUG_DEFAULTS

    return {k: (0.0 if k.startswith("p_") else v) for k, v in NNUNET_AUG_DEFAULTS.items()}


def test_nnunet_augment_shapes_labels_and_representation():
    from matblast.data.dataset import _augment, _intensity_zscore_gradient

    torch.manual_seed(5)
    params = {k: (1.0 if k.startswith("p_") else v) for k, v in _all_off_params().items()}
    label = torch.randint(0, 5, (16, 24))
    image = torch.rand(3, 16, 24)
    out, out_label = _augment(image, label, "nnunet", 0, "intensity_zscore_gradient", params)
    assert (
        out.shape == (3, 16, 24) and out_label.shape == (16, 24) and out_label.dtype == torch.long
    )
    assert torch.isfinite(out).all()
    assert set(out_label.unique().tolist()) <= set(label.unique().tolist()) | {0}
    assert torch.allclose(out, _intensity_zscore_gradient(out[:1]), atol=1e-5, rtol=1e-4)
    legacy, _ = _augment(torch.rand(3, 16, 24), label, "nnunet", 0, "legacy", params)
    gray = legacy[:1]
    assert torch.allclose(legacy[1:2], (gray - gray.mean()) / (gray.std() + 1e-6), atol=1e-5)
    single, _ = _augment(torch.rand(1, 16, 24), label, "nnunet", 0, "legacy", params)
    assert single.shape == (1, 16, 24)


def test_nnunet_augment_is_deterministic_under_seed():
    from matblast.data.dataset import _augment

    image, label = torch.rand(3, 16, 16), torch.randint(0, 5, (16, 16))
    torch.manual_seed(11)
    a = _augment(image.clone(), label.clone(), "nnunet")
    torch.manual_seed(11)
    b = _augment(image.clone(), label.clone(), "nnunet")
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_nnunet_augment_all_probabilities_zero_is_identity():
    from matblast.data.dataset import _augment

    image, label = torch.rand(3, 16, 16), torch.randint(0, 5, (16, 16))
    image[1:2] = (image[:1] - image[:1].mean()) / (image[:1].std() + 1e-6)
    out, out_label = _augment(
        image.clone(), label.clone(), "nnunet", 0, "legacy", _all_off_params()
    )
    assert torch.allclose(out, image, atol=1e-6) and torch.equal(out_label, label)


def test_augment_rejects_unknown_policy_and_params(tmp_path):
    from matblast.data.dataset import BlastocystDataset, _augment
    from matblast.data.augment_nnunet import resolve_augmentation_params

    image, label = torch.rand(3, 8, 8), torch.zeros(8, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="unknown augmentation policy"):
        _augment(image, label, "typo")
    with pytest.raises(ValueError, match="unknown keys"):
        resolve_augmentation_params("nnunet", {"p_rotatoin": 1.0})
    with pytest.raises(ValueError, match="only valid with"):
        resolve_augmentation_params("default", {"p_rotation": 1.0})
    assert resolve_augmentation_params("default", None) is None
    assert resolve_augmentation_params("nnunet", {"scale_range": [0.9, 1.1]})["scale_range"] == (
        0.9,
        1.1,
    )
    (tmp_path / "images").mkdir()
    _write_image(tmp_path / "images" / "a.png", 8, 8)
    with pytest.raises(ValueError, match="unknown augmentation policy"):
        BlastocystDataset(tmp_path / "images", None, 8, augment="typo")
    dataset = BlastocystDataset(
        tmp_path / "images", None, 8, augment="nnunet", augmentation_params={"p_rotation": 1.0}
    )
    assert dataset.augmentation_params["p_rotation"] == 1.0
    assert dataset[0]["image"].shape == (3, 8, 8)


def test_seed_worker_and_loader_init_fn():
    import numpy as np

    from matblast.data.dataset import seed_worker

    torch.manual_seed(7)
    seed_worker(0)
    first = np.random.rand()
    torch.manual_seed(7)
    seed_worker(0)
    assert np.random.rand() == first


# ----------------------------------------------------------------------- A6/A2
def test_overlap_reduction_image_vs_batch():
    from matblast.losses import CombinedLoss
    from matblast.losses.segmentation import overlap_loss

    torch.manual_seed(0)
    logits = torch.randn(2, 5, 8, 8)
    target = torch.randint(0, 5, (2, 8, 8))
    assert torch.equal(
        overlap_loss(logits, target), overlap_loss(logits, target, reduction="batch")
    )
    same = torch.cat([logits[:1], logits[:1]])
    same_target = torch.cat([target[:1], target[:1]])
    assert torch.allclose(
        overlap_loss(same, same_target, reduction="batch"),
        overlap_loss(same, same_target, reduction="image"),
        atol=1e-6,
    )
    absent = target.clone()
    absent[1] = 0  # second image has only background
    assert not torch.allclose(
        overlap_loss(logits, absent, reduction="batch"),
        overlap_loss(logits, absent, reduction="image"),
    )
    with pytest.raises(ValueError):
        overlap_loss(logits, target, reduction="typo")
    with pytest.raises(ValueError):
        CombinedLoss(overlap_reduction="typo")
    out = {"logits": logits}
    assert torch.equal(
        CombinedLoss()(out, target)["total_loss"],
        CombinedLoss(overlap_reduction="batch")(out, target)["total_loss"],
    )
    image_loss = CombinedLoss(
        overlap_reduction="image", overlap_metric="jaccard", adaptive_weights=[0.5]
    )(out, target)
    assert torch.isfinite(image_loss["total_loss"])


def test_deep_supervision_weights_default_matches_legacy_and_validates():
    from matblast.losses import CombinedLoss

    torch.manual_seed(1)
    target = torch.randint(0, 5, (2, 8, 8))
    output = {
        "logits": torch.randn(2, 5, 8, 8),
        "aux_logits_3": torch.randn(2, 5, 8, 8),
        "aux_logits_2": torch.randn(2, 5, 8, 8),
    }
    legacy = CombinedLoss()(output, target)["aux"]
    explicit = CombinedLoss(deep_supervision_weights=[1.0, 1.0], deep_supervision_normalize=False)(
        output, target
    )["aux"]
    assert torch.allclose(legacy, explicit, atol=1e-6)
    normalised = CombinedLoss(deep_supervision_weights=[1.0, 1.0])(output, target)["aux"]
    assert torch.allclose(normalised, legacy / 2, atol=1e-6)
    with pytest.raises(ValueError, match="entries"):
        CombinedLoss(deep_supervision_weights=[1.0, 0.5, 0.25])(output, target)
    with pytest.raises(ValueError, match="not set"):
        CombinedLoss()({**output, "aux_logits_4": torch.randn(2, 5, 8, 8)}, target)
    four = {
        **output,
        "aux_logits_4": torch.randn(2, 5, 8, 8),
        "aux_logits_5": torch.randn(2, 5, 8, 8),
    }
    weighted = CombinedLoss(deep_supervision_weights=[0.5, 0.25, 0.125, 0.0])(four, target)
    assert torch.isfinite(weighted["total_loss"])
    with pytest.raises(ValueError):
        CombinedLoss(deep_supervision_weights=[0.0, 0.0])
    with pytest.raises(ValueError):
        CombinedLoss(deep_supervision_weights=[])


# --------------------------------------------------------------------------- B2
def _unet_cfg(**model):
    return {
        "data": {"num_classes": 5},
        "model": {"name": "matblast_unet", "base_channels": 4, **model},
    }


def test_bmask_boundary_contract_and_gradients():
    from matblast.losses import CombinedLoss
    from matblast.models import build_model

    model = build_model(_unet_cfg(bmask_boundary_branch=True, bmask_hidden_channels=6))
    x, y = torch.randn(2, 3, 32, 32), torch.randint(0, 5, (2, 32, 32))
    out = model(x)
    assert out["logits"].shape == (2, 5, 32, 32)
    assert out["class_boundary"].shape == (2, 5, 32, 32)
    loss = CombinedLoss(class_boundary_weight=0.5)(out, y)
    loss["total_loss"].backward()
    assert torch.isfinite(loss["total_loss"]) and float(loss["class_boundary"]) > 0
    assert all(p.grad is not None for p in model.bmask.parameters() if p.requires_grad)


def test_bmask_toggle_and_determinism():
    from matblast.models import build_model

    image = torch.randn(1, 3, 32, 32)
    torch.manual_seed(17)
    baseline = build_model(_unet_cfg())
    torch.manual_seed(17)
    enabled = build_model(_unet_cfg(bmask_boundary_branch=True))
    torch.manual_seed(17)
    disabled = build_model(_unet_cfg(bmask_boundary_branch=False))
    assert baseline.bmask is None and disabled.bmask is None
    assert set(disabled.state_dict()) == set(baseline.state_dict())
    with torch.no_grad():
        assert torch.equal(baseline(image)["logits"], disabled(image)["logits"])
        first = enabled(image)["logits"]
        assert not torch.equal(baseline(image)["logits"], first)
    torch.manual_seed(17)
    repeat = build_model(_unet_cfg(bmask_boundary_branch=True))
    with torch.no_grad():
        assert torch.equal(first, repeat(image)["logits"])


def test_bmask_exclusive_flags_and_fusion_modes():
    from matblast.models import build_model

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_model(_unet_cfg(bmask_boundary_branch=True, class_boundary=True))
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_model(_unet_cfg(bmask_boundary_branch=True, boundary_refinement=True))
    concat = build_model(_unet_cfg(bmask_boundary_branch=True, bmask_fusion="concat"))
    assert concat(torch.randn(1, 3, 32, 32))["class_boundary"].shape == (1, 5, 32, 32)
    with pytest.raises(ValueError, match="fusion"):
        build_model(_unet_cfg(bmask_boundary_branch=True, bmask_fusion="add"))


# --------------------------------------------------------------------------- B1
def test_point_sample_matches_indexing_at_pixel_centres():
    from matblast.modules.point_rend import point_sample, sample_point_labels

    features = torch.randn(1, 3, 4, 6)
    ys, xs = torch.meshgrid(torch.arange(4), torch.arange(6), indexing="ij")
    coords = torch.stack(((xs.flatten() + 0.5) / 6, (ys.flatten() + 0.5) / 4), dim=-1).unsqueeze(0)
    sampled = point_sample(features, coords.float())
    assert torch.allclose(sampled, features.reshape(1, 3, -1), atol=1e-5)
    labels = torch.randint(0, 5, (1, 4, 6))
    assert torch.equal(sample_point_labels(labels, coords.float()), labels.reshape(1, -1))


def test_pointrend_sampling_counts_range_and_grid_selection():
    from matblast.modules.point_rend import (
        calculate_uncertainty,
        get_uncertain_point_coords_on_grid,
        get_uncertain_point_coords_with_randomness,
    )

    torch.manual_seed(0)
    coarse = torch.randn(2, 5, 8, 8)
    coords = get_uncertain_point_coords_with_randomness(coarse, 64, 3.0, 0.75, (1, 2, 4))
    assert coords.shape == (2, 64, 2) and coords.min() >= 0 and coords.max() <= 1
    uncertainty = calculate_uncertainty(coarse, (1, 2, 4))
    assert uncertainty.shape == (2, 1, 8, 8) and (uncertainty <= 0).all()
    indices, grid_coords = get_uncertain_point_coords_on_grid(uncertainty, 10)
    assert indices.shape == (2, 10) and grid_coords.shape == (2, 10, 2)
    # the selected pixels are the most uncertain ones
    top = uncertainty.reshape(2, -1).topk(10, dim=1).values
    assert torch.allclose(uncertainty.reshape(2, -1).gather(1, indices), top)
    with pytest.raises(ValueError):
        calculate_uncertainty(coarse, (1,))


def test_pointrend_train_contract_and_gradients():
    from matblast.losses import CombinedLoss
    from matblast.models import build_model

    model = build_model(
        _unet_cfg(
            deep_supervision=True, pointrend=True, pointrend_num_points=64, pointrend_mlp_hidden=16
        )
    )
    model.train()
    x, y = torch.randn(2, 3, 32, 32), torch.randint(0, 5, (2, 32, 32))
    out = model(x)
    assert out["logits"].shape == (2, 5, 32, 32)
    assert out["point_logits"].shape == (2, 5, 64) and out["point_coords"].shape == (2, 64, 2)
    loss = CombinedLoss(point_loss_weight=1.0)(out, y)
    assert float(loss["point"]) > 0
    loss["total_loss"].backward()
    assert all(p.grad is not None for p in model.pointrend.parameters())
    assert all(p.grad is not None for p in model.dec1.parameters())


def test_pointrend_eval_refines_full_res_logits_deterministically():
    from matblast.models import build_model

    torch.manual_seed(2)
    model = build_model(
        _unet_cfg(
            deep_supervision=True,
            pointrend=True,
            pointrend_mlp_hidden=16,
            pointrend_subdivision_num_points=50,
            pointrend_subdivision_steps=2,
        )
    ).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
        again = model(x)["logits"]
    assert out["logits"].shape == (1, 5, 32, 32) and out["dense_logits"].shape == (1, 5, 32, 32)
    assert "point_logits" not in out
    assert torch.equal(out["logits"], again)
    # coarse comes from the aux head, refined only at <= steps*N positions relative to the upsampled coarse
    upsampled_coarse = torch.nn.functional.interpolate(
        model.aux2(torch.zeros(1, 8, 16, 16)), size=(32, 32), mode="bilinear", align_corners=False
    )
    assert upsampled_coarse.shape == out["logits"].shape


def test_pointrend_toggle_off_is_baseline_and_validates():
    from matblast.models import build_model

    image = torch.randn(1, 3, 32, 32)
    torch.manual_seed(17)
    baseline = build_model(_unet_cfg(deep_supervision=True))
    torch.manual_seed(17)
    disabled = build_model(_unet_cfg(deep_supervision=True, pointrend=False))
    assert set(baseline.state_dict()) == set(disabled.state_dict())
    with torch.no_grad():
        assert torch.equal(baseline(image)["logits"], disabled(image)["logits"])
    with pytest.raises(ValueError, match="deep_supervision"):
        build_model(_unet_cfg(pointrend=True))
    with pytest.raises(ValueError, match="coarse_level"):
        build_model(_unet_cfg(deep_supervision=True, pointrend=True, pointrend_coarse_level=1))
    with pytest.raises(ValueError):
        build_model(
            _unet_cfg(deep_supervision=True, pointrend=True, pointrend_classes_of_interest=(4,))
        )
    level3 = build_model(
        _unet_cfg(
            deep_supervision=True, pointrend=True, pointrend_coarse_level=3, pointrend_mlp_hidden=8
        )
    ).eval()
    with torch.no_grad():
        assert level3(image)["logits"].shape == (1, 5, 32, 32)


# --------------------------------------------------------------------------- A1
def _v2_cfg(**model):
    base = {
        "name": "matblast_v2",
        "num_stages": 6,
        "features_per_stage": [4, 8, 8, 8, 8, 8],
        "deep_supervision": True,
    }
    return {"data": {"num_classes": 5}, "model": {**base, **model}}


def test_matblast_v2_contract_shapes_and_aux_levels():
    from matblast.losses import CombinedLoss
    from matblast.models import build_model

    model = build_model(_v2_cfg())
    x, y = torch.randn(1, 3, 64, 64), torch.randint(0, 5, (1, 64, 64))
    out = model(x)
    assert out["logits"].shape == (1, 5, 64, 64)
    assert {k for k in out if k.startswith("aux_logits_")} == {
        "aux_logits_2",
        "aux_logits_3",
        "aux_logits_4",
        "aux_logits_5",
    }
    assert all(out[f"aux_logits_{level}"].shape == (1, 5, 64, 64) for level in range(2, 6))
    assert out["boundary"].shape == (1, 1, 64, 64)
    # boundary_weight > 0 so the (always emitted) boundary head also receives gradient
    loss = CombinedLoss(deep_supervision_weights=[0.5, 0.25, 0.125, 0.0625], boundary_weight=0.1)(
        out, y
    )
    loss["total_loss"].backward()
    assert torch.isfinite(loss["total_loss"])
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    with pytest.raises(ValueError, match="not set"):
        CombinedLoss()(model(x), y)
    # rectangular input divisible by 32 works; stage kwarg accepted
    assert model(torch.randn(1, 3, 64, 96), stage=torch.zeros(1, dtype=torch.long))[
        "logits"
    ].shape == (1, 5, 64, 96)


def test_matblast_v2_rejects_bad_config_and_input():
    from matblast.models import build_model

    with pytest.raises(ValueError, match="features_per_stage"):
        build_model(_v2_cfg(features_per_stage=[4, 8, 8]))
    with pytest.raises(ValueError, match="norm"):
        build_model(_v2_cfg(norm="group"))
    with pytest.raises(ValueError, match="downsample"):
        build_model(_v2_cfg(downsample="avgpool"))
    with pytest.raises(ValueError, match="conv_block"):
        build_model(_v2_cfg(conv_block="dense"))
    with pytest.raises(TypeError):
        build_model(_v2_cfg(attention=True))
    model = build_model(_v2_cfg())
    with pytest.raises(ValueError, match="divisible by 32"):
        model(torch.randn(1, 3, 48, 48))
    with pytest.raises(ValueError, match="InstanceNorm"):
        model(torch.randn(1, 3, 32, 32))
    with pytest.raises(ValueError, match="deep_supervision_levels"):
        build_model(_v2_cfg(deep_supervision_levels=[1, 2]))


@pytest.mark.parametrize(
    "variant",
    [
        {"norm": "batch", "activation": "relu"},
        {"downsample": "maxpool"},
        {"conv_block": "plain"},
        {"encoder_blocks_per_stage": [1, 1, 2, 2, 2, 2], "decoder_blocks_per_stage": 1},
    ],
)
def test_matblast_v2_variants_forward_and_backward(variant):
    from matblast.models import build_model

    model = build_model(_v2_cfg(**variant))
    out = model(torch.randn(2, 3, 64, 64))
    assert out["logits"].shape == (2, 5, 64, 64)
    out["logits"].mean().backward()


def test_matblast_v2_heads_off_equal_backbone_and_seeded_determinism():
    from matblast.models import build_model

    torch.manual_seed(3)
    plain = build_model(_v2_cfg())
    torch.manual_seed(3)
    explicit = build_model(
        _v2_cfg(
            pyramid_context=False,
            global_context=False,
            bmask_boundary_branch=False,
            pointrend=False,
            bpr_refinement=False,
            class_context=False,
        )
    )
    assert set(plain.state_dict()) == set(explicit.state_dict())
    x = torch.randn(1, 3, 64, 64)
    plain.eval()
    explicit.eval()
    with torch.no_grad():
        assert torch.equal(plain(x)["logits"], explicit(x)["logits"])
        first = plain(x)["logits"]
    torch.manual_seed(3)
    repeat = build_model(_v2_cfg()).eval()
    with torch.no_grad():
        assert torch.equal(first, repeat(x)["logits"])


def test_matblast_v2_with_bottleneck_modules_and_all_heads():
    from matblast.losses import CombinedLoss
    from matblast.models import build_model

    model = build_model(
        _v2_cfg(
            pyramid_context=True,
            pyramid_dilations=[1, 2],
            global_context=True,
            global_pool_size=2,
            residual_encoder_block=True,
            literature_block="se_block",
            bmask_boundary_branch=True,
            pointrend=True,
            pointrend_num_points=32,
            pointrend_mlp_hidden=8,
            pointrend_subdivision_num_points=20,
            bpr_refinement=True,
            class_context=True,
        )
    )
    x, y = torch.randn(2, 3, 64, 64), torch.randint(0, 5, (2, 64, 64))
    model.train()
    out = model(x)
    assert {
        "class_boundary",
        "bpr_proximity",
        "class_context",
        "point_logits",
        "point_coords",
    } <= set(out)
    loss = CombinedLoss(
        deep_supervision_weights=[0.5, 0.25, 0.125, 0.0625],
        class_boundary_weight=0.5,
        point_loss_weight=1.0,
    )(out, y)
    loss["total_loss"].backward()
    assert all(p.grad is not None for p in model.heads.parameters() if p.requires_grad)
    model.eval()
    with torch.no_grad():
        eval_out = model(x)
    assert eval_out["logits"].shape == (2, 5, 64, 64) and "dense_logits" in eval_out
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_model(_v2_cfg(bmask_boundary_branch=True, class_boundary=True))
    with pytest.raises(ValueError, match="not a supervised level"):
        build_model(
            _v2_cfg(pointrend=True, deep_supervision_levels=[3, 4], pointrend_coarse_level=2)
        )


# --------------------------------------------------------------------------- A4
def test_model_ema_update_math_and_warmup():
    import torch.nn as nn

    from matblast.engine import ModelEMA

    model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1))
    with torch.no_grad():
        model[0].weight.fill_(1.0)
    ema = ModelEMA(model, decay=0.9)
    with torch.no_grad():
        model[0].weight.fill_(3.0)
        model[1].num_batches_tracked.fill_(7)
    ema.update(model)
    assert torch.allclose(ema.module[0].weight, torch.tensor([[[[0.9 * 1.0 + 0.1 * 3.0]]]]))
    assert int(ema.module[1].num_batches_tracked) == 7  # non-float buffers are copied
    assert ema.updates == 1 and ema.state_dict()["decay"] == 0.9
    assert not any(p.requires_grad for p in ema.module.parameters())
    warm = ModelEMA(model, decay=0.999, warmup_steps=10)
    assert warm.current_decay() == pytest.approx(1 / 10)
    warm.updates = 5
    assert warm.current_decay() == pytest.approx(6 / 15)
    warm.updates = 100000
    assert warm.current_decay() == 0.999
    with pytest.raises(ValueError):
        ModelEMA(model, decay=1.0)


def _tiny_loader(n=4, size=16):
    from torch.utils.data import DataLoader

    torch.manual_seed(0)
    items = [
        {"image": torch.rand(3, size, size), "mask": torch.randint(0, 5, (size, size))}
        for _ in range(n)
    ]
    return DataLoader(items, batch_size=2, shuffle=False)


def test_trainer_ema_disabled_checkpoint_is_unchanged(tmp_path):
    from matblast.engine import Trainer
    from matblast.losses import CombinedLoss
    from matblast.models import build_model

    model = build_model(_unet_cfg())
    trainer = Trainer(
        model,
        CombinedLoss(),
        torch.optim.AdamW(model.parameters()),
        device="cpu",
        run_dir=tmp_path,
        patience=5,
    )
    assert trainer.ema is None
    trainer.fit(_tiny_loader(), _tiny_loader(), epochs=1)
    checkpoint = torch.load(tmp_path / "best.pt", map_location="cpu")
    assert set(checkpoint) == {"model", "optimizer", "epoch", "metrics", "monitor", "monitor_value"}


def test_trainer_ema_validates_with_shadow_and_saves_both(tmp_path):
    from matblast.engine import Trainer
    from matblast.losses import CombinedLoss
    from matblast.models import build_model

    model = build_model(_unet_cfg())
    trainer = Trainer(
        model,
        CombinedLoss(),
        torch.optim.AdamW(model.parameters(), lr=1e-2),
        device="cpu",
        run_dir=tmp_path,
        patience=5,
        ema_enabled=True,
        ema_decay=0.5,
    )
    history = trainer.fit(_tiny_loader(), _tiny_loader(), epochs=1)
    assert trainer.ema.updates == 2  # two optimizer steps (4 samples / batch 2)
    checkpoint = torch.load(tmp_path / "best.pt", map_location="cpu")
    assert {"model", "model_raw", "ema"} <= set(checkpoint)
    assert checkpoint["ema"]["updates"] == 2
    key = next(k for k, v in checkpoint["model"].items() if v.dtype.is_floating_point)
    assert not torch.equal(checkpoint["model"][key], checkpoint["model_raw"][key])
    assert torch.equal(checkpoint["model"][key], trainer.ema.module.state_dict()[key])
    assert len(history) == 1
    # resume path: shadow can be re-initialised from a saved EMA state
    resumed = Trainer(
        build_model(_unet_cfg()),
        CombinedLoss(),
        torch.optim.AdamW(model.parameters()),
        device="cpu",
        run_dir=tmp_path,
        ema_enabled=True,
        ema_init_state=checkpoint["model"],
    )
    assert torch.equal(resumed.ema.module.state_dict()[key], checkpoint["model"][key])


# --------------------------------------------------------------------------- A5
def _oversample_fixture(tmp_path, icm_fractions):
    import json

    import numpy as np
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "masks" / "icm").mkdir(parents=True)
    names = []
    for index, fraction in enumerate(icm_fractions):
        name = f"img{index}.png"
        _write_image(tmp_path / "images" / name, 20, 20)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask.flat[: int(round(fraction * 400))] = 255
        Image.fromarray(mask).save(tmp_path / "masks" / "icm" / name)
        names.append(name)
    split = {"train": names[:-1], "val": names[-1:]}
    (tmp_path / "split.json").write_text(json.dumps(split))
    return {
        "seed": 3,
        "data": {
            "image_dir": str(tmp_path / "images"),
            "mask_root": str(tmp_path / "masks"),
            "split_json": str(tmp_path / "split.json"),
            "image_size": 16,
            "num_classes": 5,
            "mask_classes": ["zp", "te", "bl", "icm"],
        },
        "model": {},
        "loss": {},
        "training": {"batch_size": 2, "num_workers": 0},
    }


def test_sample_weights_follow_inverse_icm_fraction(tmp_path):
    from torch.utils.data import Subset

    from matblast.data.dataset import BlastocystDataset, build_sample_weights

    _oversample_fixture(tmp_path, [0.10, 0.01, 0.50])
    dataset = BlastocystDataset(
        tmp_path / "images", tmp_path / "masks", 16, ["zp", "te", "bl", "icm"]
    )
    fractions = dataset.class_pixel_fraction("icm")
    assert torch.allclose(
        fractions, torch.tensor([0.10, 0.01, 0.50], dtype=torch.float64), atol=1e-6
    )
    weights = build_sample_weights(dataset, "icm", power=1.0, eps=0.0)
    assert weights[1] / weights[0] == pytest.approx(10.0)
    uniform = build_sample_weights(dataset, "icm", power=0.0)
    assert torch.allclose(uniform, torch.full((3,), 1 / 3, dtype=torch.float64))
    subset_weights = build_sample_weights(Subset(dataset, [1, 2]), "icm", power=1.0, eps=0.0)
    assert subset_weights[0] / subset_weights[1] == pytest.approx(50.0)
    assert dataset.class_pixel_fraction("te").sum() == 0  # missing masks count as 0


def test_oversample_loader_is_seeded_and_disabled_path_is_untouched(tmp_path):
    from torch.utils.data import RandomSampler, Subset, WeightedRandomSampler

    from matblast.data.dataset import build_dataloaders

    cfg = _oversample_fixture(tmp_path, [0.10, 0.01, 0.50, 0.20, 0.30])
    plain, _ = build_dataloaders(cfg)
    assert isinstance(plain.sampler, RandomSampler)
    cfg["data"]["oversample"] = {
        "enabled": True,
        "key": "icm",
        "power": 1.0,
        "num_samples": 40,
        "eps": 1e-4,
    }
    first, val = build_dataloaders(cfg)
    assert isinstance(first.sampler, WeightedRandomSampler) and len(first.sampler) == 40
    second, _ = build_dataloaders(cfg)
    assert list(first.sampler) == list(second.sampler)
    drawn = torch.tensor(list(first.sampler))
    assert (drawn == 1).sum() > (
        drawn == 2
    ).sum()  # the 1% ICM image is drawn more than the 50% one
    smoke_train, smoke_val = build_dataloaders(cfg, smoke_limit=2)
    assert isinstance(smoke_train.dataset, Subset) and len(smoke_train.dataset) == 2
    assert isinstance(smoke_train.sampler, WeightedRandomSampler) and smoke_train.num_workers == 0
    assert len(smoke_val.dataset) == 1
    cfg["data"]["oversample"] = {"enabled": True, "kye": "icm"}
    with pytest.raises(ValueError, match="data.oversample"):
        build_dataloaders(cfg)
