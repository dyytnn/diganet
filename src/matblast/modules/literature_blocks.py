"""Config-selectable 2-D reimplementations of reviewed feature blocks.

The classes in this file provide dependency-light PyTorch implementations
with a common feature-map contract.  They are reimplementations of the
published ideas, not claims of bitwise reproduction of every public paper
repository (some of those require custom CUDA operators or a different
backbone).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import register_module


def _validate_channels(channels: int) -> None:
    if channels <= 0:
        raise ValueError("channels must be positive")


class _ResidualFeatureBlock(nn.Module):
    """Base class for shape-preserving residual feature blocks."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        _validate_channels(channels)
        self.channels = channels
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def _residual(self, x: Tensor, update: Tensor) -> Tensor:
        return x + self.residual_scale.tanh() * update


class _ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


@register_module("mednext_block")
class MedNeXtBlock(_ResidualFeatureBlock):
    """2-D MedNeXt-style depthwise large-kernel residual block.

    Args:
        channels: Number of feature channels.
        expansion: Pointwise channel expansion ratio.
        kernel_size: Odd depthwise kernel size.

    Shape:
        Input and output are ``(B, C, H, W)``.
    """

    def __init__(self, channels: int, expansion: int = 4, kernel_size: int = 7) -> None:
        super().__init__(channels)
        if expansion <= 0 or kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("expansion must be positive and kernel_size must be odd >= 3")
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.norm = _ChannelLayerNorm(channels)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.project = nn.Conv2d(hidden, channels, 1)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the block to ``(B, C, H, W)`` features."""
        update = self.project(self.activation(self.expand(self.norm(self.depthwise(x)))))
        return self._residual(x, update)


@register_module("convnext_v2_block")
class ConvNeXtV2Block(_ResidualFeatureBlock):
    """ConvNeXt V2-style block with global response normalization."""

    def __init__(self, channels: int, expansion: int = 4, kernel_size: int = 7) -> None:
        super().__init__(channels)
        if expansion <= 0 or kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("expansion must be positive and kernel_size must be odd >= 3")
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.norm = _ChannelLayerNorm(channels)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.project = nn.Conv2d(hidden, channels, 1)
        self.gamma = nn.Parameter(torch.ones(hidden))
        self.beta = nn.Parameter(torch.zeros(hidden))

    def forward(self, x: Tensor) -> Tensor:
        """Apply ConvNeXt V2 response competition to ``(B, C, H, W)``."""
        hidden = self.expand(self.norm(self.depthwise(x)))
        norm = torch.norm(hidden, p=2, dim=(2, 3), keepdim=True)
        norm = norm / (norm.mean(dim=1, keepdim=True) + 1e-6)
        hidden = hidden + self.gamma.view(1, -1, 1, 1) * (hidden * norm) + self.beta.view(1, -1, 1, 1)
        return self._residual(x, self.project(F.gelu(hidden)))


@register_module("msca_block")
class MSCABlock(_ResidualFeatureBlock):
    """Multi-scale convolutional spatial attention block (MSCA/MSCAN)."""

    def __init__(self, channels: int, dilations: tuple[int, ...] = (1, 2, 3)) -> None:
        super().__init__(channels)
        if not dilations or any(d <= 0 for d in dilations):
            raise ValueError("dilations must be positive and non-empty")
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=d, dilation=d, groups=channels),
                    nn.Conv2d(channels, channels, 1),
                    nn.GELU(),
                )
                for d in dilations
            ]
        )
        self.attention = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Fuse multi-scale depthwise responses for ``(B, C, H, W)``."""
        mixed = sum(self.branches[i](x) for i in range(len(self.branches))) / len(self.branches)
        return self._residual(x, self.project(mixed) * self.attention(mixed))


@register_module("dlka_block")
class DLKABlock(_ResidualFeatureBlock):
    """Large-kernel attention block with learned spatial gating.

    This dependency-free 2-D variant captures the D-LKA design principle.
    It is reported as a reimplementation, not as the original custom-op code.
    """

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__(channels)
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd >= 3")
        self.large = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.local = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply large-kernel gated refinement to ``(B, C, H, W)``."""
        large = self.large(x)
        update = self.project(large * self.gate(self.local(large)))
        return self._residual(x, update)


@register_module("dcnv4_block")
class DCNv4Block(_ResidualFeatureBlock):
    """CUDA-free deformable-convolution approximation using ``grid_sample``."""

    def __init__(self, channels: int, max_offset: float = 2.0) -> None:
        super().__init__(channels)
        if max_offset <= 0:
            raise ValueError("max_offset must be positive")
        self.max_offset = float(max_offset)
        self.offset = nn.Conv2d(channels, 2, 3, padding=1)
        self.value = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Sample a learned offset field from ``(B, C, H, W)``."""
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        grid = torch.stack((xx, yy), -1).unsqueeze(0).expand(b, -1, -1, -1).clone()
        offsets = torch.tanh(self.offset(x))
        offsets = offsets.permute(0, 2, 3, 1)
        offsets = offsets * torch.tensor(
            [2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)], device=x.device, dtype=x.dtype
        ) * self.max_offset
        sampled = F.grid_sample(self.value(x), grid + offsets, align_corners=True, padding_mode="border")
        return self._residual(x, self.project(sampled))


@register_module("odconv_block")
class ODConvBlock(_ResidualFeatureBlock):
    """Input-conditioned mixture of depthwise convolution kernels."""

    def __init__(self, channels: int, kernels: int = 4) -> None:
        super().__init__(channels)
        if kernels <= 0:
            raise ValueError("kernels must be positive")
        self.experts = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=1, groups=channels) for _ in range(kernels)]
        )
        self.route = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, kernels, 1))
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Route features across dynamic kernel experts."""
        weights = self.route(x).softmax(1)
        update = sum(expert(x) * weights[:, i : i + 1] for i, expert in enumerate(self.experts))
        return self._residual(x, self.project(update))


@register_module("se_block")
class SEBlock(_ResidualFeatureBlock):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__(channels)
        hidden = max(channels // reduction, 1)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        """Reweight channels of ``(B, C, H, W)``."""
        return self._residual(x, x * self.gate(x))


@register_module("cbam_block")
class CBAMBlock(_ResidualFeatureBlock):
    """Convolutional block attention module with channel and spatial gates."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__(channels)
        hidden = max(channels // reduction, 1)
        self.channel = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, channels, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x: Tensor) -> Tensor:
        """Apply sequential channel and spatial attention."""
        channel = torch.sigmoid(self.channel(F.adaptive_avg_pool2d(x, 1)))
        refined = x * channel
        spatial = torch.cat((refined.mean(1, keepdim=True), refined.amax(1, keepdim=True)), 1)
        return self._residual(x, refined * torch.sigmoid(self.spatial(spatial)))


@register_module("eca_block")
class ECABlock(_ResidualFeatureBlock):
    """Efficient channel attention using a local 1-D channel convolution."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__(channels)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd")
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply lightweight channel reweighting."""
        descriptor = F.adaptive_avg_pool2d(x, 1).flatten(1).unsqueeze(1)
        gate = torch.sigmoid(self.conv(descriptor)).transpose(1, 2).unsqueeze(-1)
        return self._residual(x, x * gate)


@register_module("coordinate_attention")
class CoordinateAttention(_ResidualFeatureBlock):
    """Coordinate attention with separate height and width pooling."""

    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__(channels)
        hidden = max(channels // reduction, 8)
        self.shared = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.BatchNorm2d(hidden), nn.Hardswish())
        self.height = nn.Conv2d(hidden, channels, 1)
        self.width = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply position-aware channel gates to ``(B, C, H, W)``."""
        h = x.mean(dim=3, keepdim=True)
        w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        shared = self.shared(torch.cat((h, w), 2))
        h_feat, w_feat = torch.split(shared, [x.shape[2], x.shape[3]], 2)
        w_feat = w_feat.transpose(2, 3)
        gate = torch.sigmoid(self.height(h_feat)) * torch.sigmoid(self.width(w_feat))
        return self._residual(x, x * gate)


@register_module("mfmsa_block")
class MFMSABlock(_ResidualFeatureBlock):
    """Multi-frequency, multi-scale refinement for domain and boundary shifts."""

    def __init__(self, channels: int, dilations: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__(channels)
        self.low = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.high = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.scales = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=d, dilation=d, groups=channels) for d in dilations])
        self.gate = nn.Conv2d(channels, 2, 1)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Fuse low/high-frequency and multi-scale responses."""
        low = self.low(F.avg_pool2d(x, 3, 1, 1))
        high = self.high(x - F.avg_pool2d(x, 3, 1, 1))
        multi = sum(branch(x) for branch in self.scales) / len(self.scales)
        weights = self.gate(x).softmax(1)
        update = weights[:, :1] * low + weights[:, 1:] * high + multi
        return self._residual(x, self.project(F.gelu(update)))


@register_module("cenet_block")
class CENetBlock(_ResidualFeatureBlock):
    """DSEB/CFAM-inspired boundary-aware multi-scale skip refinement."""

    def __init__(self, channels: int) -> None:
        super().__init__(channels)
        self.edge = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.context = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels)
        self.gate = nn.Conv2d(channels, channels, 1)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Refine skip features using edge and context branches."""
        edge = self.edge(x)
        context = self.context(x)
        return self._residual(x, self.project((edge + context) * torch.sigmoid(self.gate(x))))


@register_module("ad_gbc_block")
class ADGBCBlock(_ResidualFeatureBlock):
    """Prototype-based anisotropic region redistribution approximation."""

    def __init__(self, channels: int, prototypes: int = 4) -> None:
        super().__init__(channels)
        if prototypes <= 0:
            raise ValueError("prototypes must be positive")
        self.assignment = nn.Conv2d(channels, prototypes, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Aggregate global region prototypes and redistribute them per pixel."""
        assignment = self.assignment(x).flatten(2).softmax(-1)
        values = self.value(x).flatten(2).transpose(1, 2)
        prototypes = torch.bmm(assignment, values)
        context = torch.bmm(assignment.transpose(1, 2), prototypes)
        context = context.transpose(1, 2).reshape_as(x)
        return self._residual(x, self.project(context))


@register_module("mambalite_fusion")
class MambaLiteFusion(_ResidualFeatureBlock):
    """Compact local-global gated fusion inspired by MambaLiteUNet."""

    def __init__(self, channels: int) -> None:
        super().__init__(channels)
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.state = nn.Conv1d(channels, channels, 3, padding=1, groups=channels)
        self.gate = nn.Conv2d(channels, channels, 1)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Mix local features with a flattened spatial state scan."""
        b, c, h, w = x.shape
        tokens = x.flatten(2)
        state = self.state(tokens).reshape(b, c, h, w)
        return self._residual(x, self.project(self.local(x) + state * torch.sigmoid(self.gate(x))))


@register_module("dysample")
class DySample(_ResidualFeatureBlock):
    """Learned point-sampling refinement inspired by dynamic upsampling."""

    def __init__(self, channels: int, max_offset: float = 1.0) -> None:
        super().__init__(channels)
        self.max_offset = float(max_offset)
        self.offset = nn.Conv2d(channels, 2, 3, padding=1)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Use learned point offsets while preserving spatial shape."""
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype), indexing="ij"
        )
        grid = torch.stack((xx, yy), -1).unsqueeze(0).expand(b, -1, -1, -1)
        offsets = torch.tanh(self.offset(x)).permute(0, 2, 3, 1)
        offsets = offsets * torch.tensor([2 / max(w - 1, 1), 2 / max(h - 1, 1)], device=x.device, dtype=x.dtype) * self.max_offset
        return self._residual(x, self.project(F.grid_sample(x, grid + offsets, align_corners=True, padding_mode="border")))


@register_module("carafe_block")
class CARAFEBlock(_ResidualFeatureBlock):
    """Content-aware local feature reassembly approximation."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__(channels)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd")
        self.kernel_size = kernel_size
        self.kernel = nn.Conv2d(channels, kernel_size * kernel_size, 1)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Reassemble local patches with content-conditioned kernels."""
        b, c, h, w = x.shape
        weights = self.kernel(x).softmax(1)
        patches = F.unfold(x, self.kernel_size, padding=self.kernel_size // 2).reshape(b, c, -1, h, w)
        update = (patches * weights.unsqueeze(1)).sum(2)
        return self._residual(x, self.project(update))


@register_module("pointrend_refiner")
class PointRendRefiner(_ResidualFeatureBlock):
    """Uncertainty-gated point refinement for coarse segmentation features."""

    def __init__(self, channels: int) -> None:
        super().__init__(channels)
        self.uncertainty = nn.Conv2d(channels, 1, 1)
        self.refine = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(), nn.Conv2d(channels, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Refine features at high-uncertainty points."""
        gate = torch.sigmoid(self.uncertainty(x))
        return self._residual(x, gate * self.refine(x))


@register_module("maxvit_block")
class MaxViTBlock(_ResidualFeatureBlock):
    """Lightweight MBConv plus pooled global attention approximation."""

    def __init__(self, channels: int, pool_size: int = 4) -> None:
        super().__init__(channels)
        self.local = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels), nn.GELU(), nn.Conv2d(channels, channels, 1))
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.global_proj = nn.Conv2d(channels, channels, 1)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Combine local convolution and pooled global context."""
        global_context = F.interpolate(self.global_proj(self.pool(x)), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self._residual(x, self.project(self.local(x) + global_context))


@register_module("biformer_block")
class BiFormerBlock(_ResidualFeatureBlock):
    """Pooled top-k routed attention approximation for low-resolution features."""

    def __init__(self, channels: int, regions: int = 4, top_k: int = 2) -> None:
        super().__init__(channels)
        if regions <= 0 or top_k <= 0 or top_k > regions:
            raise ValueError("regions and top_k must satisfy 0 < top_k <= regions")
        self.regions = regions
        self.top_k = top_k
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Route each token to its strongest pooled regions."""
        pooled = F.adaptive_avg_pool2d(x, (self.regions, self.regions))
        query = self.query(x).flatten(2).transpose(1, 2)
        key = self.key(pooled).flatten(2)
        value = self.value(pooled).flatten(2).transpose(1, 2)
        scores = torch.bmm(query, key) / math.sqrt(x.shape[1])
        top_values, top_indices = scores.topk(self.top_k, dim=-1)
        weights = top_values.softmax(-1)
        selected = value[:, None].expand(-1, query.shape[1], -1, -1).gather(
            2, top_indices.unsqueeze(-1).expand(-1, -1, -1, x.shape[1])
        )
        update = (selected * weights.unsqueeze(-1)).sum(2).transpose(1, 2).reshape_as(x)
        return self._residual(x, self.project(update))


@register_module("tokkan_block")
class TokKANBlock(_ResidualFeatureBlock):
    """Learnable nonlinear basis mixer inspired by Tok-KAN/U-KAN."""

    def __init__(self, channels: int, basis: int = 4) -> None:
        super().__init__(channels)
        if basis <= 0:
            raise ValueError("basis must be positive")
        self.coefficients = nn.Conv2d(channels, channels * basis, 1)
        self.project = nn.Conv2d(channels * basis, channels, 1)
        self.scales = nn.Parameter(torch.arange(1, basis + 1, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate sinusoidal nonlinear bases per spatial token."""
        coefficients = self.coefficients(x).reshape(x.shape[0], self.scales.numel(), self.channels, *x.shape[-2:])
        bases = torch.stack([torch.sin(scale * x) for scale in self.scales], dim=1)
        return self._residual(x, self.project((coefficients * bases).reshape_as(coefficients.flatten(1, 2))))


@register_module("bilateral_graph_block")
class BilateralGraphBlock(_ResidualFeatureBlock):
    """Semantic-boundary bilateral graph interaction approximation."""

    def __init__(self, channels: int) -> None:
        super().__init__(channels)
        self.semantic = nn.Conv2d(channels, channels, 1)
        self.boundary = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.graph = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Exchange global semantic and local boundary descriptors."""
        semantic = self.semantic(x)
        boundary = self.boundary(x)
        graph = torch.cat((semantic.mean((2, 3), keepdim=True).expand_as(x), boundary), 1)
        return self._residual(x, self.graph(graph))


@register_module("phase_aligned_fusion")
class PhaseAlignedFusion(_ResidualFeatureBlock):
    """2-D phase-calibration component separated from CROWn's 3-D design."""

    def __init__(self, channels: int) -> None:
        super().__init__(channels)
        self.shift = nn.Conv2d(channels, 2, 3, padding=1)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Calibrate local sampling phase with learned offsets."""
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype), torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype), indexing="ij")
        grid = torch.stack((xx, yy), -1).unsqueeze(0).expand(b, -1, -1, -1)
        shift = torch.tanh(self.shift(x)).permute(0, 2, 3, 1)
        shift = shift * torch.tensor([2 / max(w - 1, 1), 2 / max(h - 1, 1)], device=x.device, dtype=x.dtype)
        aligned = F.grid_sample(x, grid + shift, align_corners=True, padding_mode="border")
        return self._residual(x, self.project(aligned))


__all__ = [
    "ADGBCBlock", "BilateralGraphBlock", "BiFormerBlock", "CARAFEBlock", "CBAMBlock",
    "ConvNeXtV2Block", "CoordinateAttention", "CENetBlock", "DCNv4Block", "DLKABlock",
    "DySample", "ECABlock", "MFMSABlock", "MambaLiteFusion", "MaxViTBlock", "MedNeXtBlock",
    "MSCABlock", "ODConvBlock", "PhaseAlignedFusion", "PointRendRefiner", "SEBlock", "TokKANBlock",
]
