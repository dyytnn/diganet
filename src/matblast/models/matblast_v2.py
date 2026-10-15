"""MAT-BLAST v2: nnU-Net-style residual encoder-decoder with configurable depth.

Motivation (docs/V2_roadmap.md): the 4-stage, 1.2M-parameter MATBlastUNet trails
nnU-Net mainly on recipe. This backbone reproduces the nnU-Net structural recipe
(6 stages, residual blocks, InstanceNorm + LeakyReLU, strided-conv downsampling,
deep supervision at every decoder level) while keeping the MAT-BLAST bottleneck
modules and decoder heads available through the same flags.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn

from ..modules.pyramid_context import PyramidContext
from ..modules.residual_encoder import ResidualEncoderBlock
from .heads import DecoderHeads
from .matblast_unet import LITERATURE_BLOCKS, PooledGlobalContext

NORMS = ("instance", "batch")
ACTIVATIONS = ("leaky_relu", "relu")
DOWNSAMPLES = ("strided_conv", "maxpool")
CONV_BLOCKS = ("residual", "plain")


def _make_norm(norm: str) -> Callable[[int], nn.Module]:
    if norm == "instance":
        return lambda channels: nn.InstanceNorm2d(channels, affine=True)
    if norm == "batch":
        return nn.BatchNorm2d
    raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")


def _make_activation(activation: str, slope: float) -> Callable[[], nn.Module]:
    if activation == "leaky_relu":
        return lambda: nn.LeakyReLU(slope, inplace=False)
    if activation == "relu":
        return lambda: nn.ReLU(inplace=False)
    raise ValueError(f"activation must be one of {ACTIVATIONS}, got {activation!r}")


class ResidualStageBlock(nn.Module):
    """conv3x3(stride)-norm-act-conv3x3-norm with a projected shortcut.

    Input: (B, C_in, H, W). Output: (B, C_out, H/stride, W/stride).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm_layer: Callable[[int], nn.Module],
        act_layer: Callable[[], nn.Module],
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            norm_layer(out_channels),
            act_layer(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                norm_layer(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(x) + self.shortcut(x))


class PlainStageBlock(nn.Module):
    """conv3x3(stride)-norm-act-conv3x3-norm-act (the classic U-Net double conv)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm_layer: Callable[[int], nn.Module],
        act_layer: Callable[[], nn.Module],
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            norm_layer(out_channels),
            act_layer(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels),
            act_layer(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def _as_list(value: int | Sequence[int], count: int, name: str) -> list[int]:
    values = [int(value)] * count if isinstance(value, int) else [int(v) for v in value]
    if len(values) != count or any(v < 1 for v in values):
        raise ValueError(f"{name} must be an int or a list of {count} positive ints, got {value!r}")
    return values


class MATBlastResUNet(nn.Module):
    """nnU-Net-style residual U-Net with MAT-BLAST bottleneck modules and heads.

    Args:
        in_channels: Input channels.
        num_classes: K.
        num_stages: Encoder stages (the last one is the bottleneck).
        features_per_stage: Channel width of each stage (length ``num_stages``).
        encoder_blocks_per_stage / decoder_blocks_per_stage: Blocks per stage, int or list.
        conv_block: ``"residual"`` or ``"plain"``.
        norm: ``"instance"`` or ``"batch"``.
        activation: ``"leaky_relu"`` or ``"relu"``.
        downsample: ``"strided_conv"`` (stride-2 first conv) or ``"maxpool"``.
        dropout: Dropout2d probability at the bottleneck.
        deep_supervision / deep_supervision_levels: See :class:`DecoderHeads`.
        pyramid_*, global_*, residual_encoder_*, literature_block*: Bottleneck modules.
        Remaining keyword arguments are decoder-head flags forwarded to DecoderHeads.

    Input: (B, in_channels, H, W) with H and W divisible by ``2 ** (num_stages - 1)``.
    Output: dict with ``logits`` (B, K, H, W) and auxiliary keys.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        num_stages: int = 6,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 384, 384),
        encoder_blocks_per_stage: int | Sequence[int] = 2,
        decoder_blocks_per_stage: int | Sequence[int] = 2,
        conv_block: str = "residual",
        norm: str = "instance",
        activation: str = "leaky_relu",
        leaky_relu_slope: float = 0.01,
        downsample: str = "strided_conv",
        dropout: float = 0.0,
        deep_supervision: bool = True,
        deep_supervision_levels: Sequence[int] | None = None,
        pyramid_context: bool = False,
        pyramid_dilations: Sequence[int] = (1, 2, 4),
        pyramid_reduction: int = 4,
        global_context: bool = False,
        global_pool_size: int = 8,
        residual_encoder_block: bool = False,
        residual_expansion: int = 2,
        residual_dropout: float = 0.0,
        literature_block: str | None = None,
        literature_block_kwargs: dict | None = None,
        class_boundary: bool = False,
        boundary_refinement: bool = False,
        boundary_hidden_channels: int | None = None,
        class_context: bool = False,
        bpr_refinement: bool = False,
        bpr_hidden_channels: int | None = None,
        bpr_proximity_kernel_size: int = 5,
        bmask_boundary_branch: bool = False,
        bmask_hidden_channels: int | None = None,
        bmask_fusion: str = "gate",
        pointrend: bool = False,
        pointrend_coarse_level: int = 2,
        pointrend_num_points: int = 2048,
        pointrend_oversample_ratio: float = 3.0,
        pointrend_importance_sample_ratio: float = 0.75,
        pointrend_mlp_hidden: int = 256,
        pointrend_mlp_layers: int = 3,
        pointrend_classes_of_interest: Sequence[int] = (1, 2, 4),
        pointrend_subdivision_steps: int = 2,
        pointrend_subdivision_num_points: int = 8192,
    ) -> None:
        super().__init__()
        if num_stages < 2:
            raise ValueError(f"num_stages must be >= 2, got {num_stages}")
        features = [int(f) for f in features_per_stage]
        if len(features) != num_stages or any(f < 1 for f in features):
            raise ValueError(
                f"features_per_stage must list {num_stages} positive widths, got {features_per_stage!r}"
            )
        if conv_block not in CONV_BLOCKS:
            raise ValueError(f"conv_block must be one of {CONV_BLOCKS}, got {conv_block!r}")
        if downsample not in DOWNSAMPLES:
            raise ValueError(f"downsample must be one of {DOWNSAMPLES}, got {downsample!r}")
        norm_layer = _make_norm(norm)
        act_layer = _make_activation(activation, leaky_relu_slope)
        block_cls = ResidualStageBlock if conv_block == "residual" else PlainStageBlock
        encoder_blocks = _as_list(encoder_blocks_per_stage, num_stages, "encoder_blocks_per_stage")
        decoder_blocks = _as_list(
            decoder_blocks_per_stage, num_stages - 1, "decoder_blocks_per_stage"
        )
        self.num_stages = num_stages
        self.features = features
        self.stride = 2 ** (num_stages - 1)
        self.downsample_mode = downsample
        self.norm_name = norm

        stages = []
        previous = in_channels
        for index, (width, count) in enumerate(zip(features, encoder_blocks)):
            first_stride = 2 if (index > 0 and downsample == "strided_conv") else 1
            blocks = [block_cls(previous, width, first_stride, norm_layer, act_layer)]
            blocks += [block_cls(width, width, 1, norm_layer, act_layer) for _ in range(count - 1)]
            stages.append(nn.Sequential(*blocks))
            previous = width
        self.encoder = nn.ModuleList(stages)
        self.pool = nn.MaxPool2d(2) if downsample == "maxpool" else None

        bottleneck = features[-1]
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.pyramid_context = (
            PyramidContext(bottleneck, tuple(pyramid_dilations), pyramid_reduction)
            if pyramid_context
            else nn.Identity()
        )
        self.global_context = (
            PooledGlobalContext(bottleneck, global_pool_size) if global_context else nn.Identity()
        )
        self.residual_encoder_block = (
            ResidualEncoderBlock(bottleneck, residual_expansion, residual_dropout)
            if residual_encoder_block
            else nn.Identity()
        )
        if literature_block is not None:
            if literature_block not in LITERATURE_BLOCKS:
                raise ValueError(
                    f"unknown literature_block {literature_block!r}; choose from {sorted(LITERATURE_BLOCKS)}"
                )
            self.literature_block: nn.Module = LITERATURE_BLOCKS[literature_block](
                bottleneck, **dict(literature_block_kwargs or {})
            )
        else:
            self.literature_block = nn.Identity()

        ups, decoders = [], []
        for index in range(num_stages - 1, 0, -1):
            skip_width = features[index - 1]
            ups.append(nn.ConvTranspose2d(features[index], skip_width, kernel_size=2, stride=2))
            count = decoder_blocks[num_stages - 1 - index]
            blocks = [block_cls(skip_width * 2, skip_width, 1, norm_layer, act_layer)]
            blocks += [
                block_cls(skip_width, skip_width, 1, norm_layer, act_layer)
                for _ in range(count - 1)
            ]
            decoders.append(nn.Sequential(*blocks))
        self.ups = nn.ModuleList(ups)
        self.decoders = nn.ModuleList(decoders)

        self.heads = DecoderHeads(
            features[: num_stages - 1],
            num_classes,
            norm_layer,
            deep_supervision=deep_supervision,
            deep_supervision_levels=deep_supervision_levels,
            class_boundary=class_boundary,
            boundary_refinement=boundary_refinement,
            boundary_hidden_channels=boundary_hidden_channels,
            class_context=class_context,
            bpr_refinement=bpr_refinement,
            bpr_hidden_channels=bpr_hidden_channels,
            bpr_proximity_kernel_size=bpr_proximity_kernel_size,
            bmask_boundary_branch=bmask_boundary_branch,
            bmask_hidden_channels=bmask_hidden_channels,
            bmask_fusion=bmask_fusion,
            pointrend=pointrend,
            pointrend_coarse_level=pointrend_coarse_level,
            pointrend_num_points=pointrend_num_points,
            pointrend_oversample_ratio=pointrend_oversample_ratio,
            pointrend_importance_sample_ratio=pointrend_importance_sample_ratio,
            pointrend_mlp_hidden=pointrend_mlp_hidden,
            pointrend_mlp_layers=pointrend_mlp_layers,
            pointrend_classes_of_interest=pointrend_classes_of_interest,
            pointrend_subdivision_steps=pointrend_subdivision_steps,
            pointrend_subdivision_num_points=pointrend_subdivision_num_points,
        )

    def forward(
        self, x: torch.Tensor, stage: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Segment one batch; ``stage`` is accepted for trainer compatibility and unused."""
        del stage
        height, width = x.shape[-2:]
        if height % self.stride or width % self.stride:
            raise ValueError(
                f"input size {height}x{width} must be divisible by {self.stride} for {self.num_stages} stages"
            )
        if self.norm_name == "instance" and min(height, width) // self.stride < 2:
            raise ValueError(
                f"input size {height}x{width} leaves a {height // self.stride}x{width // self.stride} "
                "bottleneck; InstanceNorm needs at least 2x2 (use a larger input or fewer stages)"
            )
        skips = []
        feature = x
        for index, encoder in enumerate(self.encoder):
            if index > 0 and self.pool is not None:
                feature = self.pool(feature)
            feature = encoder(feature)
            skips.append(feature)
        bottleneck = self.literature_block(
            self.residual_encoder_block(
                self.global_context(self.pyramid_context(self.dropout(skips[-1])))
            )
        )
        decoder_features = []
        feature = bottleneck
        for step, (up, decoder) in enumerate(zip(self.ups, self.decoders)):
            skip = skips[self.num_stages - 2 - step]
            feature = decoder(torch.cat((up(feature), skip), dim=1))
            decoder_features.append(feature)
        decoder_features.reverse()  # fine -> coarse: [d1, d2, ..., d_{S-1}]
        return self.heads(decoder_features)
