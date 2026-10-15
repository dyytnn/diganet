"""Every new config must build its model and loss and run a small forward pass.

There is no YAML schema, so this is the safety net against misspelled keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from matblast.config import load_config
from matblast.losses import CombinedLoss
from matblast.models import build_model

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    "V2_0_recipe_parity_grouped_v1",
    "V2_1_backbone_v2_grouped_v1",
    "V2_2_nnunet_augmentation_grouped_v1",
    "V2_3_ema_grouped_v1",
    "V2_4_icm_oversample_grouped_v1",
    "V2_5_per_image_overlap_grouped_v1",
    "V2_6_pad_384x448_grouped_v1",
    "S46_matblast_pointrend_faithful_grouped_v1",
    "S57_matblast_entropy_gate_d1_grouped_v1",
    "S58_matblast_multiscale_texture_gate_noise_grouped_v1",
    "S54_matblast_bmask_boundary_grouped_v1",
    "S48_matblast_deep_strip_grouped_v1",
    "S49_matblast_refine_mask_grouped_v1",
    "S50_matblast_mask_transfiner_grouped_v1",
    "S52_matblast_segfix_grouped_v1",
    "S53_matblast_cascade_psp_grouped_v1",
    "S55_matblast_deep_snake_grouped_v1",
    "S51_matblast_sharp_contour_grouped_v1",
    "S56_matblast_poly_transform_grouped_v1",
]


@pytest.mark.parametrize("name", CONFIGS)
def test_new_config_builds_model_and_loss(name):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    model = build_model(cfg)
    size = cfg["data"]["image_size"]
    height, width = (size, size) if isinstance(size, int) else size
    # keep the forward cheap but divisible by the deepest stride (32)
    height, width = max(64, height // 4), max(64, width // 4)
    x = torch.randn(1, cfg["model"].get("in_channels", 3), height, width)
    y = torch.randint(0, cfg["data"]["num_classes"], (1, height, width))
    model.train()
    out = model(x)
    assert out["logits"].shape == (1, cfg["data"]["num_classes"], height, width)
    loss = CombinedLoss(**{k: v for k, v in cfg["loss"].items() if k != "name"})(out, y)
    assert torch.isfinite(loss["total_loss"])
    model.eval()
    with torch.no_grad():
        assert model(x)["logits"].shape == out["logits"].shape
    for section, allowed in (
        (
            "training",
            {
                "epochs",
                "batch_size",
                "num_workers",
                "lr",
                "lr_policy",
                "poly_power",
                "weight_decay",
                "patience",
                "monitor",
                "monitor_mode",
                "optimizer",
                "optimizer_eps",
                "drop_last",
                "ema_enabled",
                "ema_decay",
                "ema_warmup_steps",
            },
        ),
        (
            "data",
            {
                "image_dir",
                "mask_root",
                "split_json",
                "image_size",
                "num_classes",
                "mask_classes",
                "augment",
                "augmentation",
                "augmentation_params",
                "augmentation_repeats",
                "resize_mode",
                "oversample",
                "in_channels",
                "input_representation",
                "stage_metadata_csv",
                "train_noise_sigma_max",
            },
        ),
    ):
        unknown = set(cfg[section]) - allowed
        assert not unknown, f"{name}: unknown {section} keys {sorted(unknown)}"
