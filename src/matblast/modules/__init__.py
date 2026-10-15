"""Anatomy-aware feature modules."""

from __future__ import annotations

from typing import TypeVar

import torch
from torch import Tensor, nn


ModuleType = TypeVar("ModuleType", bound=type[nn.Module])
MODULES: dict[str, type[nn.Module]] = {}


def register_module(name: str):
    """Register a config-selectable feature module."""

    def decorator(module: ModuleType) -> ModuleType:
        if name in MODULES:
            raise ValueError(f"module '{name}' already registered")
        MODULES[name] = module
        return module

    return decorator


class ClassContext(nn.Module):
    """Lightweight intra-class context refinement used by the S2 ablation."""

    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, 1)
        self.classes = classes

    def forward(self, features: Tensor, probabilities: Tensor) -> Tensor:
        pooled = torch.einsum("bkhw,bchw->bck", probabilities, features)
        pooled = pooled / (probabilities.flatten(2).sum(-1).unsqueeze(1) + 1e-6)
        context = pooled.mean(-1).unsqueeze(-1).unsqueeze(-1)
        return features + self.proj(features) * torch.sigmoid(context)


@register_module("stage_conditioned_graph")
class StageConditionedGraph(nn.Module):
    """Learnable inter-class relation graph with optional stage conditioning."""

    def __init__(self, classes: int, stages: int = 5, hidden: int = 16):
        super().__init__()
        del hidden
        self.adjacency = nn.Parameter(torch.zeros(classes, classes))
        self.stage = nn.Embedding(stages, classes * classes)
        self.gate = nn.Linear(classes * classes, classes * classes)

    def forward(self, class_features: Tensor, stage: Tensor | None = None) -> Tensor:
        logits = self.adjacency.unsqueeze(0).expand(class_features.shape[0], -1, -1)
        if stage is not None:
            logits = logits + self.stage(stage).view(
                -1, self.adjacency.shape[0], self.adjacency.shape[1]
            )
        weights = torch.softmax(logits, -1)
        return torch.bmm(weights, class_features)


from .adaptive_skip_fusion import AdaptiveSkipFusion  # noqa: E402
from .adaptive_receptive_field import AdaptiveReceptiveField  # noqa: E402
from .anti_alias_downsample import AntiAliasedDownsample  # noqa: E402
from .cnn_ssm_block import CNNSSMBlock  # noqa: E402
from .token_moe_adapter import TokenMoEAdapter  # noqa: E402
from .boundary_refinement import BoundaryGuidedRefiner  # noqa: E402
from .boundary_patch_refinement import BoundaryProximityRefiner  # noqa: E402
from .bmask_boundary import BMaskBoundaryBranch  # noqa: E402
from .point_rend import PointRendHead  # noqa: E402
from .deep_strip import DeepStripRefiner  # noqa: E402
from .refine_mask import RefineMaskHead  # noqa: E402
from .mask_transfiner import MaskTransfinerLite  # noqa: E402
from .segfix import SegFixHead  # noqa: E402
from .cascade_psp import CascadePSPLite  # noqa: E402
from .contour_refinement import PolarContourRefiner  # noqa: E402
from .frequency_multiscale import FrequencyMultiScale  # noqa: E402
from .convnext_block import ConvNeXtBlock  # noqa: E402
from .pyramid_context import PyramidContext  # noqa: E402
from .residual_encoder import ResidualEncoderBlock  # noqa: E402
from .dynamic_snake_conv import DynamicSnakeConv  # noqa: E402
from .literature_blocks import (  # noqa: E402
    ADGBCBlock,
    BilateralGraphBlock,
    BiFormerBlock,
    CARAFEBlock,
    CBAMBlock,
    CENetBlock,
    ConvNeXtV2Block,
    CoordinateAttention,
    DCNv4Block,
    DLKABlock,
    DySample,
    ECABlock,
    MFMSABlock,
    MambaLiteFusion,
    MaxViTBlock,
    MedNeXtBlock,
    MSCABlock,
    ODConvBlock,
    PhaseAlignedFusion,
    PointRendRefiner,
    SEBlock,
    TokKANBlock,
)
from .volume_foundation_blocks import Bio2VolCSDHA, CSAM, SegMoTE  # noqa: E402

__all__ = [
    "AdaptiveSkipFusion",
    "AdaptiveReceptiveField",
    "AntiAliasedDownsample",
    "CNNSSMBlock",
    "TokenMoEAdapter",
    "BoundaryGuidedRefiner",
    "BoundaryProximityRefiner",
    "BMaskBoundaryBranch",
    "PointRendHead",
    "DeepStripRefiner",
    "RefineMaskHead",
    "MaskTransfinerLite",
    "SegFixHead",
    "CascadePSPLite",
    "PolarContourRefiner",
    "ClassContext",
    "ConvNeXtBlock",
    "FrequencyMultiScale",
    "MODULES",
    "PyramidContext",
    "ResidualEncoderBlock",
    "DynamicSnakeConv",
    "StageConditionedGraph",
    "register_module",
    "ADGBCBlock",
    "BilateralGraphBlock",
    "BiFormerBlock",
    "CARAFEBlock",
    "CBAMBlock",
    "CENetBlock",
    "ConvNeXtV2Block",
    "CoordinateAttention",
    "DCNv4Block",
    "DLKABlock",
    "DySample",
    "ECABlock",
    "MFMSABlock",
    "MambaLiteFusion",
    "MaxViTBlock",
    "MedNeXtBlock",
    "MSCABlock",
    "ODConvBlock",
    "PhaseAlignedFusion",
    "PointRendRefiner",
    "SEBlock",
    "TokKANBlock",
    "Bio2VolCSDHA",
    "CSAM",
    "SegMoTE",
]
