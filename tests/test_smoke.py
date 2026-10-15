import math

import numpy as np
import pytest
import torch
from PIL import Image

from matblast.data.dataset import (
    _augment,
    _intensity_zscore_dualraw,
    _intensity_zscore_gradient,
    _read_rgb,
)
from matblast.losses import (
    CombinedLoss,
    GeneralizedDiceLoss,
    I2CNetCrossEntropyLoss,
    WeightedSoftJaccardLoss,
)
from matblast.models import build_model
from matblast.engine import Trainer
from matblast.eval import multiseed_paired_summary, paired_summary, per_image_iou
from matblast.metrics import per_image_scores, segmentation_metrics, summarize_scores
from matblast.metrics import betti_error, boundary_f, ece, hd95, morphometry


def test_paper_baseline_contracts_and_gradients():
    specs = [
        ("btformer", {"base_channels": 4, "n_angles": 4}, WeightedSoftJaccardLoss()),
        ("i2cnet", {"channels": [4, 8, 12, 16, 20]}, I2CNetCrossEntropyLoss()),
        ("mimics_net", {"base_channels": 4, "groups": 4}, GeneralizedDiceLoss()),
        ("massnet", {}, CombinedLoss()),
        (
            "blastnet_reconstruction",
            {
                "in_channels": 1,
                "stem_channels": 8,
                "stem_kernel_size": 3,
                "stage_planes": [8, 16, 32, 64],
                "stage_blocks": [1, 1, 1, 1],
                "stage_strides": [1, 2, 2, 1],
                "stage_dilations": [1, 1, 1, 2],
                "bottleneck_expansion": 2,
                "capp_channels": 16,
                "capp_rates": [1, 2, 4, 8, 16],
                "decoder_channels": [16, 8, 4, 2],
            },
            CombinedLoss(),
        ),
    ]
    x = torch.randn(2, 1, 64, 64)
    y = torch.randint(0, 5, (2, 64, 64))
    for name, kwargs, loss_fn in specs:
        model = build_model({"data": {"num_classes": 5}, "model": {"name": name, **kwargs}})
        output = model(x)
        loss = loss_fn(output, y)["total_loss"]
        loss.backward()
        assert output["logits"].shape == (2, 5, 64, 64)
        assert torch.isfinite(loss)
        assert all(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )


def test_model_loss_contract():
    cfg = {
        "data": {"num_classes": 4},
        "model": {
            "name": "matblast_unet",
            "base_channels": 4,
            "attention": True,
            "multimodal": True,
        },
    }
    model = build_model(cfg)
    x, y = torch.randn(2, 3, 32, 32), torch.randint(0, 4, (2, 32, 32))
    out = model(x)
    loss = CombinedLoss(boundary_weight=0.3)(out, y)["total_loss"]
    assert out["logits"].shape == (2, 4, 32, 32)
    assert torch.isfinite(loss)


def test_residual_encoder_toggle_and_gradients():
    """The residual encoder block changes features and receives gradients."""
    base_cfg = {
        "data": {"num_classes": 5},
        "model": {"name": "matblast_unet", "base_channels": 4},
    }
    enabled_cfg = {
        "data": {"num_classes": 5},
        "model": {
            "name": "matblast_unet",
            "base_channels": 4,
            "residual_encoder_block": True,
            "residual_expansion": 2,
        },
    }
    torch.manual_seed(0)
    baseline = build_model(base_cfg)
    torch.manual_seed(0)
    enabled = build_model(enabled_cfg)
    x = torch.randn(2, 3, 32, 32)
    out = enabled(x)["logits"]
    out.sum().backward()
    assert out.shape == (2, 5, 32, 32)
    assert torch.isfinite(out).all()
    assert all(
        p.grad is not None
        for p in enabled.residual_encoder_block.parameters()
        if p.requires_grad
    )
    assert enabled.residual_encoder_block.__class__.__name__ == "ResidualEncoderBlock"
    assert baseline.residual_encoder_block.__class__.__name__ == "Identity"


def test_adaptive_receptive_field_toggle_and_gradients():
    """Adaptive dilation branches preserve shape and propagate gradients."""
    cfg = {
        "data": {"num_classes": 5},
        "model": {
            "name": "matblast_unet",
            "base_channels": 4,
            "adaptive_receptive_field": True,
            "adaptive_receptive_dilations": [1, 2, 4],
        },
    }
    model = build_model(cfg)
    output = model(torch.randn(2, 3, 32, 32))["logits"]
    output.square().mean().backward()
    assert output.shape == (2, 5, 32, 32)
    assert torch.isfinite(output).all()
    assert all(
        parameter.grad is not None
        for parameter in model.adaptive_receptive_field.parameters()
        if parameter.requires_grad
    )


def test_anti_aliased_downsampling_toggle_and_shape():
    """Anti-aliased pooling keeps the encoder contract and changes the pool type."""
    cfg = {
        "data": {"num_classes": 5},
        "model": {
            "name": "matblast_unet",
            "base_channels": 4,
            "anti_aliased_downsampling": True,
        },
    }
    model = build_model(cfg)
    output = model(torch.randn(2, 3, 32, 32))["logits"]
    assert output.shape == (2, 5, 32, 32)
    assert model.pool.__class__.__name__ == "AntiAliasedDownsample"


def test_cnn_ssm_toggle_and_gradients():
    """The CNN--SSM-inspired block preserves shape and receives gradients."""
    model = build_model(
        {
            "data": {"num_classes": 5},
            "model": {"name": "matblast_unet", "base_channels": 4, "cnn_ssm_block": True},
        }
    )
    output = model(torch.randn(2, 3, 32, 32))["logits"]
    output.mean().backward()
    assert output.shape == (2, 5, 32, 32)
    assert torch.isfinite(output).all()
    assert all(
        parameter.grad is not None
        for parameter in model.cnn_ssm_block.parameters()
        if parameter.requires_grad
    )


def test_token_moe_toggle_and_gradients():
    """Token MoE routing preserves shape and propagates gradients."""
    model = build_model(
        {
            "data": {"num_classes": 5},
            "model": {"name": "matblast_unet", "base_channels": 4, "token_moe_adapter": True},
        }
    )
    output = model(torch.randn(2, 3, 32, 32))["logits"]
    output.mean().backward()
    assert output.shape == (2, 5, 32, 32)
    assert torch.isfinite(output).all()
    assert all(
        parameter.grad is not None
        for parameter in model.token_moe_adapter.parameters()
        if parameter.requires_grad
    )


def test_dynamic_snake_conv_toggle_and_gradients():
    """Dynamic snake paths preserve shape and propagate gradients."""
    model = build_model(
        {
            "data": {"num_classes": 5},
            "model": {
                "name": "matblast_unet",
                "base_channels": 4,
                "dynamic_snake_conv": True,
                "dynamic_snake_kernel_size": 5,
            },
        }
    )
    output = model(torch.randn(2, 3, 32, 32))["logits"]
    output.square().mean().backward()
    assert output.shape == (2, 5, 32, 32)
    assert torch.isfinite(output).all()
    assert all(p.grad is not None for p in model.snake1.parameters() if p.requires_grad)
    assert all(p.grad is not None for p in model.snake2.parameters() if p.requires_grad)


def test_literature_block_selector_contracts_and_gradients():
    """Every 2-D literature block preserves the model contract and gradients."""
    from matblast.models.matblast_unet import LITERATURE_BLOCKS

    for name in LITERATURE_BLOCKS:
        model = build_model(
            {
                "data": {"num_classes": 5},
                "model": {
                    "name": "matblast_unet",
                    "base_channels": 2,
                    "literature_block": name,
                },
            }
        )
        output = model(torch.randn(1, 3, 32, 32))["logits"]
        output.mean().backward()
        assert output.shape == (1, 5, 32, 32), name
        assert torch.isfinite(output).all(), name
        assert all(
            parameter.grad is not None
            for parameter in model.literature_block.parameters()
            if parameter.requires_grad
        ), name


def test_literature_block_selector_toggle_and_determinism():
    """Each selector changes the bottleneck and is reproducible under a seed."""
    from matblast.models.matblast_unet import LITERATURE_BLOCKS

    image = torch.randn(1, 3, 32, 32)
    for name in LITERATURE_BLOCKS:
        torch.manual_seed(17)
        baseline = build_model(
            {"data": {"num_classes": 5}, "model": {"name": "matblast_unet", "base_channels": 2}}
        )
        torch.manual_seed(17)
        enabled = build_model(
            {
                "data": {"num_classes": 5},
                "model": {
                    "name": "matblast_unet",
                    "base_channels": 2,
                    "literature_block": name,
                },
            }
        )
        with torch.no_grad():
            baseline_output = baseline(image)["logits"]
            enabled_output = enabled(image)["logits"]
        assert not torch.equal(baseline_output, enabled_output), name

        torch.manual_seed(17)
        repeat = build_model(
            {
                "data": {"num_classes": 5},
                "model": {
                    "name": "matblast_unet",
                    "base_channels": 2,
                    "literature_block": name,
                },
            }
        )
        with torch.no_grad():
            repeat_output = repeat(image)["logits"]
        assert torch.equal(enabled_output, repeat_output), name


def test_2p5d_and_token_literature_contracts_and_gradients():
    """CSAM, Bio2Vol CSDHA, and SegMoTE preserve explicit input axes."""
    from matblast.modules import Bio2VolCSDHA, CSAM, SegMoTE

    volume = torch.randn(2, 3, 8, 8, 8, requires_grad=True)
    for module in (CSAM(8, heads=2), Bio2VolCSDHA(8, heads=2)):
        output = module(volume)
        assert output.shape == volume.shape
        assert torch.isfinite(output).all()
        output.mean().backward(retain_graph=True)
        assert all(parameter.grad is not None for parameter in module.parameters())

    tokens = torch.randn(2, 11, 16, requires_grad=True)
    adapter = SegMoTE(16, experts=3, expansion=2)
    output = adapter(tokens)
    assert output.shape == tokens.shape
    assert torch.isfinite(output).all()
    output.mean().backward()
    assert all(parameter.grad is not None for parameter in adapter.parameters())


def test_extended_metric_hand_calculation_contracts():
    """Extended geometry and calibration metrics return finite contract values."""
    truth = np.zeros((4, 4), dtype=bool)
    truth[1:3, 1:3] = True
    prediction = truth.copy()
    assert hd95(prediction, truth) == 0.0
    assert boundary_f(prediction, truth, tolerance=2) == 1.0
    assert betti_error(prediction, truth) == {"components": 0, "holes": 0, "total": 0}
    measures = morphometry(np.array([[0, 1, 1, 0], [0, 2, 3, 0], [0, 2, 4, 0], [0, 0, 0, 0]]))
    assert measures["bl_area"] == 1.0
    assert measures["icm_area"] == 1.0
    probs = np.zeros((2, 2, 2), dtype=np.float32)
    probs[0] = 0.9
    probs[1] = 0.1
    assert ece(probs, np.zeros((2, 2), dtype=np.int64), bins=2) < 0.11


def test_attention_unet_contract_and_gradients():
    """Attention U-Net keeps the common dictionary output contract."""
    model = build_model(
        {"data": {"num_classes": 5}, "model": {"name": "attention_unet", "base_channels": 4}}
    )
    image = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 5, (2, 64, 64))
    output = model(image)
    loss = CombinedLoss(boundary_weight=0.1)(output, target)["total_loss"]
    loss.backward()
    assert output["logits"].shape == (2, 5, 64, 64)
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
    )


def test_unetpp_contract_and_gradients():
    """U-Net++ exposes the common segmentation output contract."""
    model = build_model(
        {"data": {"num_classes": 5}, "model": {"name": "unetpp", "base_channels": 4}}
    )
    image = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 5, (2, 64, 64))
    output = model(image)
    loss = CombinedLoss(boundary_weight=0.1)(output, target)["total_loss"]
    loss.backward()
    assert output["logits"].shape == (2, 5, 64, 64)
    assert torch.isfinite(loss)


def test_blasto_net_segmentation_contract_and_gradients():
    """The segmentation-only Blasto-Net reconstruction is differentiable."""
    model = build_model({"data": {"num_classes": 5}, "model": {"name": "blasto_net_segmentation"}})
    image = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 5, (2, 64, 64))
    output = model(image)
    loss = CombinedLoss(boundary_weight=0.1)(output, target)["total_loss"]
    loss.backward()
    assert output["logits"].shape == (2, 5, 64, 64)
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
    )


def test_class_boundary_contract():
    cfg = {
        "data": {"num_classes": 5},
        "model": {"name": "matblast_unet", "base_channels": 4, "class_boundary": True},
    }
    model = build_model(cfg)
    x, y = torch.randn(2, 3, 32, 32), torch.randint(0, 5, (2, 32, 32))
    out = model(x)
    loss = CombinedLoss(class_boundary_weight=0.2)(out, y)["total_loss"]
    loss.backward()
    assert out["class_boundary"].shape == (2, 5, 32, 32)
    assert torch.isfinite(loss)


def test_pyramid_context_contract():
    cfg = {
        "data": {"num_classes": 5},
        "model": {
            "name": "matblast_unet",
            "base_channels": 4,
            "pyramid_context": True,
            "pyramid_dilations": [1, 2, 4],
            "pyramid_reduction": 2,
        },
    }
    model = build_model(cfg)
    x, y = torch.randn(2, 3, 32, 32), torch.randint(0, 5, (2, 32, 32))
    out = model(x)
    loss = CombinedLoss(boundary_weight=0.1)(out, y)["total_loss"]
    loss.backward()
    assert out["logits"].shape == (2, 5, 32, 32)
    assert torch.isfinite(loss)


def test_convnext_block_contract_toggle_and_gradients():
    """ConvNeXt bottleneck toggle preserves the segmentation contract."""
    base_cfg = {
        "data": {"num_classes": 5},
        "model": {"name": "matblast_unet", "base_channels": 4},
    }
    block_cfg = {
        "data": {"num_classes": 5},
        "model": {
            "name": "matblast_unet",
            "base_channels": 4,
            "convnext_block": True,
            "convnext_expansion": 2,
            "convnext_kernel_size": 5,
        },
    }
    torch.manual_seed(7)
    baseline = build_model(base_cfg)
    torch.manual_seed(7)
    candidate = build_model(block_cfg)
    image = torch.randn(2, 3, 32, 32)
    target = torch.randint(0, 5, (2, 32, 32))
    output = candidate(image)
    loss = CombinedLoss()(output, target)["total_loss"]
    loss.backward()
    assert output["logits"].shape == (2, 5, 32, 32)
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None
        for parameter in candidate.convnext_block.parameters()
        if parameter.requires_grad
    )
    assert not isinstance(candidate.convnext_block, torch.nn.Identity)
    assert isinstance(baseline.convnext_block, torch.nn.Identity)


def test_adaptive_skip_fusion_contract_and_gradients():
    """Adaptive skip gates preserve the decoder concatenation contract."""
    cfg = {
        "data": {"num_classes": 5},
        "model": {
            "name": "matblast_unet",
            "base_channels": 4,
            "adaptive_skip_fusion": True,
            "adaptive_skip_hidden_channels": 3,
        },
    }
    model = build_model(cfg)
    x, y = torch.randn(2, 3, 32, 32), torch.randint(0, 5, (2, 32, 32))
    out = model(x)
    loss = CombinedLoss(boundary_weight=0.1)(out, y)["total_loss"]
    loss.backward()
    assert out["logits"].shape == (2, 5, 32, 32)
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
    )


def test_frequency_multiscale_contract_and_gradients():
    """Frequency multi-scale refinement preserves shape and gradient flow."""
    model = build_model(
        {
            "data": {"num_classes": 5},
            "model": {
                "name": "matblast_unet",
                "base_channels": 4,
                "frequency_multiscale": True,
                "frequency_hidden_channels": 8,
                "frequency_dilations": [1, 2, 4],
            },
        }
    )
    image = torch.randn(2, 3, 32, 32)
    target = torch.randint(0, 5, (2, 32, 32))
    output = model(image)
    loss = CombinedLoss(boundary_weight=0.1)(output, target)["total_loss"]
    loss.backward()
    assert output["logits"].shape == (2, 5, 32, 32)
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
    )


def test_stage_conditioned_graph_contract_and_requires_stage():
    """Stage graph consumes explicit stage IDs and rejects missing metadata."""
    model = build_model(
        {
            "data": {"num_classes": 5},
            "model": {
                "name": "matblast_unet",
                "base_channels": 4,
                "stage_conditioned_graph": True,
            },
        }
    )
    image = torch.randn(2, 3, 32, 32)
    target = torch.randint(0, 5, (2, 32, 32))
    try:
        model(image)
    except ValueError as error:
        assert "stage is required" in str(error)
    else:
        raise AssertionError("stage-conditioned graph accepted missing stage metadata")
    output = model(image, stage=torch.tensor([0, 3]))
    loss = CombinedLoss(boundary_weight=0.1)(output, target)["total_loss"]
    loss.backward()
    assert output["logits"].shape == (2, 5, 32, 32)
    assert torch.isfinite(loss)


def test_adaptive_skip_fusion_toggle_keeps_baseline_parameters_and_changes_output():
    """The disabled path keeps the baseline initialization bit-wise stable."""
    torch.manual_seed(0)
    baseline = build_model(
        {"data": {"num_classes": 5}, "model": {"name": "matblast_unet", "base_channels": 4}}
    )
    torch.manual_seed(0)
    adaptive = build_model(
        {
            "data": {"num_classes": 5},
            "model": {
                "name": "matblast_unet",
                "base_channels": 4,
                "adaptive_skip_fusion": True,
                "adaptive_skip_hidden_channels": 3,
            },
        }
    )
    baseline_state = baseline.state_dict()
    adaptive_state = adaptive.state_dict()
    assert all(torch.equal(baseline_state[key], adaptive_state[key]) for key in baseline_state)
    x = torch.randn(2, 3, 32, 32)
    assert not torch.equal(baseline(x)["logits"], adaptive(x)["logits"])


def test_auxiliary_cross_entropy_can_be_disabled_independently():
    """A WSJA run can keep auxiliary overlap loss without hidden CE supervision."""
    torch.manual_seed(0)
    target = torch.randint(0, 5, (2, 8, 8))
    output = {
        "logits": torch.randn(2, 5, 8, 8),
        "aux_logits_3": torch.randn(2, 5, 8, 8),
        "aux_logits_2": torch.randn(2, 5, 8, 8),
    }
    with_aux_ce = CombinedLoss(
        ce_weight=0.0, dice_weight=1.0, overlap_metric="jaccard", overlap_include_background=True
    )(output, target)
    without_aux_ce = CombinedLoss(
        ce_weight=0.0,
        dice_weight=1.0,
        overlap_metric="jaccard",
        overlap_include_background=True,
        aux_ce_weight=0.0,
    )(output, target)
    assert float(without_aux_ce["aux"]) < float(with_aux_ce["aux"])
    assert float(without_aux_ce["total_loss"]) < float(with_aux_ce["total_loss"])


def test_boundary_refinement_contract():
    cfg = {
        "data": {"num_classes": 5},
        "model": {
            "name": "matblast_unet",
            "base_channels": 4,
            "boundary_refinement": True,
            "class_boundary": True,
            "boundary_hidden_channels": 6,
        },
    }
    model = build_model(cfg)
    x, y = torch.randn(2, 3, 32, 32), torch.randint(0, 5, (2, 32, 32))
    out = model(x)
    loss = CombinedLoss(class_boundary_weight=0.1)(out, y)["total_loss"]
    loss.backward()
    assert out["logits"].shape == (2, 5, 32, 32)
    assert out["class_boundary"].shape == (2, 5, 32, 32)
    assert torch.isfinite(loss)


def _bpr_model_config(enabled: bool) -> dict:
    """Config for the S47 BPR-inspired refiner, enabled or disabled."""
    model = {"name": "matblast_unet", "base_channels": 4}
    if enabled:
        model.update(
            bpr_refinement=True,
            bpr_hidden_channels=6,
            bpr_proximity_kernel_size=5,
        )
    return {"data": {"num_classes": 5}, "model": model}


def test_bpr_refinement_contract_and_gradients():
    """S47 preserves the output contract and routes gradients to every parameter."""
    model = build_model(_bpr_model_config(enabled=True))
    x, y = torch.randn(2, 3, 32, 32), torch.randint(0, 5, (2, 32, 32))
    out = model(x)
    assert out["logits"].shape == (2, 5, 32, 32)
    assert out["bpr_proximity"].shape == (2, 1, 32, 32)
    assert torch.isfinite(out["logits"]).all()
    assert ((out["bpr_proximity"] >= 0.0) & (out["bpr_proximity"] <= 1.0)).all()

    loss = CombinedLoss()(out, y)["total_loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None
        for parameter in model.bpr_refiner.parameters()
        if parameter.requires_grad
    )


def test_bpr_refinement_toggle_and_determinism():
    """S47 changes the output when enabled and is reproducible under a fixed seed."""
    image = torch.randn(1, 3, 32, 32)

    torch.manual_seed(17)
    baseline = build_model(_bpr_model_config(enabled=False))
    torch.manual_seed(17)
    enabled = build_model(_bpr_model_config(enabled=True))
    with torch.no_grad():
        baseline_output = baseline(image)["logits"]
        enabled_output = enabled(image)["logits"]
    assert baseline.bpr_refiner is None
    assert not torch.equal(baseline_output, enabled_output)

    torch.manual_seed(17)
    repeat = build_model(_bpr_model_config(enabled=True))
    with torch.no_grad():
        repeat_output = repeat(image)["logits"]
    assert torch.equal(enabled_output, repeat_output)


def test_bpr_refinement_rejects_even_proximity_kernel():
    """An even proximity kernel cannot be centred, so it must raise."""
    with pytest.raises(ValueError):
        build_model(
            {
                "data": {"num_classes": 5},
                "model": {
                    "name": "matblast_unet",
                    "base_channels": 4,
                    "bpr_refinement": True,
                    "bpr_proximity_kernel_size": 4,
                },
            }
        )


def test_soft_anatomy_relation_contract():
    cfg = {
        "data": {"num_classes": 5},
        "model": {"name": "matblast_unet", "base_channels": 4},
    }
    model = build_model(cfg)
    x, y = torch.randn(2, 3, 32, 32), torch.randint(0, 5, (2, 32, 32))
    out = model(x)
    loss = CombinedLoss(
        relation_weight=0.05,
        relation_pairs=[(1, 2), (2, 3), (2, 4), (3, 4)],
        relation_kernel_size=5,
    )(out, y)["total_loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_trainer_rejects_invalid_monitor_mode():
    cfg = {"data": {"num_classes": 5}, "model": {"name": "matblast_unet", "base_channels": 2}}
    model = build_model(cfg)
    try:
        Trainer(model, CombinedLoss(), torch.optim.AdamW(model.parameters()), mode="sideways")
    except ValueError:
        return
    raise AssertionError("invalid monitor mode was accepted")


def test_paired_summary_detects_known_gain():
    report = paired_summary(
        torch.tensor([0.2, 0.4, 0.6]).numpy(),
        torch.tensor([0.3, 0.5, 0.7]).numpy(),
        bootstrap_samples=100,
        seed=0,
    )
    assert math.isclose(report["delta"], 0.1, rel_tol=1e-6, abs_tol=1e-6)
    assert report["ci95"][0] > 0.0


def test_multiseed_paired_summary_resamples_seed_and_image_axes():
    """A consistent gain remains positive after hierarchical resampling."""
    reference = np.array([[0.2, 0.4, np.nan], [0.3, 0.5, np.nan], [0.4, 0.6, np.nan]])
    candidate = np.array([[0.3, 0.5, np.nan], [0.4, 0.6, np.nan], [0.5, 0.7, np.nan]])
    report = multiseed_paired_summary(reference, candidate, bootstrap_samples=100, seed=0)
    assert report["n_seeds"] == 3
    assert report["n_paired_scores"] == 6
    assert report["n_paired_images"] == 2
    assert report["p_image_wilcoxon_greater"] <= 1.0
    assert math.isclose(report["delta"], 0.1, rel_tol=1e-6, abs_tol=1e-6)
    assert report["ci95_hierarchical"][0] > 0.0


def test_multiseed_paired_summary_reports_image_level_significance():
    """A positive gain on every independent image has a small paired p-value."""
    reference = np.zeros((3, 12), dtype=np.float64)
    candidate = np.full((3, 12), 0.1, dtype=np.float64)

    report = multiseed_paired_summary(reference, candidate, bootstrap_samples=100, seed=0)

    assert report["n_paired_images"] == 12
    assert report["p_image_wilcoxon_greater"] < 0.05


def test_gradient_input_representation_is_finite_and_nonredundant(tmp_path):
    """The S17 third channel must encode edges, not a duplicate raw channel."""
    ramp = np.tile(np.arange(16, dtype=np.uint8), (16, 1)) * 16
    image_path = tmp_path / "ramp.png"
    Image.fromarray(ramp).save(image_path)
    image = _read_rgb(
        image_path, size=16, in_channels=3, input_representation="intensity_zscore_gradient"
    )
    assert image.shape == (3, 16, 16)
    assert torch.isfinite(image).all()
    assert not torch.allclose(image[0], image[2])


def test_gradient_representation_is_recomputed_after_augmentation():
    """Intensity, z-score, and gradient channels stay physically consistent."""
    image = torch.rand(3, 16, 16)
    label = torch.zeros(16, 16, dtype=torch.long)
    augmented, _ = _augment(
        image,
        label,
        input_representation="intensity_zscore_gradient",
    )
    assert torch.allclose(
        augmented, _intensity_zscore_gradient(augmented[:1]), atol=1e-6, rtol=1e-5
    )


def test_dualraw_representation_is_recomputed_after_augmentation():
    """The S19 raw/z-score/raw representation remains photometrically coherent."""
    image = torch.rand(3, 16, 16)
    label = torch.zeros(16, 16, dtype=torch.long)
    augmented, _ = _augment(
        image,
        label,
        input_representation="intensity_zscore_dualraw",
    )
    assert torch.allclose(augmented, _intensity_zscore_dualraw(augmented[:1]), atol=1e-6, rtol=1e-5)


def test_dualraw_representation_duplicates_raw_channel_at_load(tmp_path):
    """S19 starts from two identical raw channels before any augmentation."""
    ramp = np.tile(np.arange(16, dtype=np.uint8), (16, 1)) * 16
    image_path = tmp_path / "ramp.png"
    Image.fromarray(ramp).save(image_path)
    image = _read_rgb(
        image_path, size=16, in_channels=3, input_representation="intensity_zscore_dualraw"
    )
    assert torch.allclose(image[0], image[2])


def test_segmentation_metrics_ignore_classes_absent_from_each_ground_truth():
    """A missing GT class must be NaN per image, not a false zero contribution."""
    target = torch.tensor([[[0, 1], [0, 1]], [[0, 2], [0, 2]]])
    logits = torch.full((2, 3, 2, 2), -10.0)
    logits.scatter_(1, target.unsqueeze(1), 10.0)
    metrics = segmentation_metrics(logits, target)
    assert math.isclose(metrics["iou_class_1"], 1.0)
    assert math.isclose(metrics["iou_class_2"], 1.0)
    assert math.isclose(metrics["miou_fg"], 1.0)


def test_summarize_scores_weights_images_not_batches():
    """A partial final batch must not receive the weight of a full batch."""
    dice = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    iou = dice.clone()
    report = summarize_scores(dice, iou)
    assert math.isclose(report["miou_fg"], 2.0 / 3.0, abs_tol=1e-6)
    logits = torch.full((3, 2, 1, 1), -10.0)
    target = torch.ones((3, 1, 1), dtype=torch.long)
    logits[:2, 1] = 10.0
    logits[2, 0] = 10.0
    _, values = per_image_scores(logits, target)
    assert math.isclose(summarize_scores(values, values)["miou_fg"], 2.0 / 3.0, abs_tol=1e-6)


def test_paired_iou_ignores_absent_ground_truth_class():
    """False positives for an absent class cannot fabricate a paired score."""
    target = torch.tensor([[[0, 1], [0, 1]]])
    logits = torch.full((1, 3, 2, 2), -10.0)
    logits.scatter_(1, target.unsqueeze(1), 10.0)
    logits[0, 2, 0, 0] = 20.0
    values = per_image_iou(logits, target)
    assert math.isnan(values[0, 2])
