"""Adapters for canonical segmentation baselines used in the common benchmark.

The DeepLabV3+ and TransUNet adapters call the public reference sources saved
under ``third_party``. SegFormer-B0 is a dependency-free port of NVIDIA's
MiT-B0 encoder and all-MLP decoder, because the original project requires an
obsolete MMCV stack that is incompatible with the pinned PyTorch runtime.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _import_reference(package_root: Path, module: str):
    """Import a vendored reference package without making it a runtime dependency."""
    root = str(package_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return __import__(module, fromlist=["*"])


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ ResNet-50 adapter from the VainF public implementation."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        output_stride: int = 16,
        backbone: str = "resnet50",
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("The reference DeepLabV3+ ResNet backbone requires three channels")
        if backbone not in {"resnet50", "resnet101"}:
            raise ValueError("DeepLabV3+ benchmark supports resnet50 or resnet101")
        module = _import_reference(_PROJECT_ROOT / "third_party" / "DeepLabV3Plus-Pytorch", "network.modeling")
        build = getattr(module, f"deeplabv3plus_{backbone}")
        self.model = build(num_classes=num_classes, output_stride=output_stride, pretrained_backbone=False)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Segment images.

        Args:
            x: (B, 3, H, W) legacy three-channel image representation.

        Returns:
            Dictionary with ``logits`` of shape (B, K, H, W).
        """
        return {"logits": self.model(x)}


class TransUNet(nn.Module):
    """TransUNet adapter using the Beckschen public implementation."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        image_size: int = 256,
        variant: str = "R50-ViT-B_16",
        n_skip: int = 3,
    ) -> None:
        super().__init__()
        if in_channels not in {1, 3}:
            raise ValueError("TransUNet supports one grayscale or three-channel input")
        module = _import_reference(_PROJECT_ROOT / "third_party" / "TransUNet", "networks.vit_seg_modeling")
        if variant not in module.CONFIGS:
            raise ValueError(f"Unknown TransUNet variant {variant!r}")
        config = copy.deepcopy(module.CONFIGS[variant])
        config.n_classes = num_classes
        config.n_skip = n_skip
        self.model = module.VisionTransformer(config, img_size=image_size, num_classes=num_classes)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Segment images with the reference transformer decoder.

        Args:
            x: (B, C, H, W), C in {1, 3}.

        Returns:
            Dictionary with ``logits`` of shape (B, K, H, W).
        """
        return {"logits": self.model(x)}


class _DropPath(nn.Module):
    """Per-sample stochastic depth used by MiT."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class _DWConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        batch, _, channels = x.shape
        image = x.transpose(1, 2).reshape(batch, channels, height, width)
        return self.conv(image).flatten(2).transpose(1, 2)


class _MiTMlp(nn.Module):
    def __init__(self, channels: int, ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(channels * ratio)
        self.fc1 = nn.Linear(channels, hidden)
        self.dwconv = _DWConv(hidden)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        return self.fc2(self.activation(self.dwconv(self.fc1(x), height, width)))


class _MiTAttention(nn.Module):
    def __init__(self, channels: int, heads: int, sr_ratio: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("MiT channels must be divisible by the number of heads")
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.query = nn.Linear(channels, channels, bias=True)
        self.key_value = nn.Linear(channels, channels * 2, bias=True)
        self.projection = nn.Linear(channels, channels)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.spatial_reduction = nn.Conv2d(channels, channels, sr_ratio, stride=sr_ratio)
            self.normalization = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        batch, tokens, channels = x.shape
        query = self.query(x).reshape(batch, tokens, self.heads, channels // self.heads).transpose(1, 2)
        reduced = x
        if self.sr_ratio > 1:
            reduced = x.transpose(1, 2).reshape(batch, channels, height, width)
            reduced = self.spatial_reduction(reduced).flatten(2).transpose(1, 2)
            reduced = self.normalization(reduced)
        key, value = self.key_value(reduced).reshape(
            batch, -1, 2, self.heads, channels // self.heads
        ).permute(2, 0, 3, 1, 4)
        attention = (query @ key.transpose(-2, -1) * self.scale).softmax(dim=-1)
        return self.projection((attention @ value).transpose(1, 2).reshape(batch, tokens, channels))


class _MiTBlock(nn.Module):
    def __init__(self, channels: int, heads: int, sr_ratio: int, drop_path: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.attention = _MiTAttention(channels, heads, sr_ratio)
        self.drop_path = _DropPath(drop_path)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = _MiTMlp(channels)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x), height, width))
        return x + self.drop_path(self.mlp(self.norm2(x), height, width))


class _OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2)
        self.normalization = nn.LayerNorm(out_channels, eps=1e-6)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x = self.projection(x)
        height, width = x.shape[-2:]
        return self.normalization(x.flatten(2).transpose(1, 2)), height, width


class _MiTEncoder(nn.Module):
    """Exact MiT-B0 stage dimensions and depth schedule from NVIDIA SegFormer."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        dims = (32, 64, 160, 256)
        depths = (2, 2, 2, 2)
        heads = (1, 2, 5, 8)
        ratios = (8, 4, 2, 1)
        inputs = (in_channels,) + dims[:-1]
        kernels = (7, 3, 3, 3)
        strides = (4, 2, 2, 2)
        self.embeddings = nn.ModuleList(
            _OverlapPatchEmbed(cin, cout, kernel, stride)
            for cin, cout, kernel, stride in zip(inputs, dims, kernels, strides)
        )
        probabilities = torch.linspace(0, 0.1, sum(depths)).tolist()
        offset = 0
        stages: list[nn.Module] = []
        for channels, depth, count, ratio in zip(dims, depths, heads, ratios):
            stages.append(nn.ModuleList(
                _MiTBlock(channels, count, ratio, probabilities[offset + index])
                for index in range(depth)
            ))
            offset += depth
        self.stages = nn.ModuleList(stages)
        self.normalizations = nn.ModuleList(nn.LayerNorm(dim, eps=1e-6) for dim in dims)

    def forward(self, x: Tensor) -> list[Tensor]:
        features: list[Tensor] = []
        for embedding, blocks, normalization in zip(self.embeddings, self.stages, self.normalizations):
            tokens, height, width = embedding(x)
            for block in blocks:
                tokens = block(tokens, height, width)
            x = normalization(tokens).transpose(1, 2).reshape(tokens.shape[0], -1, height, width)
            features.append(x)
        return features


class SegFormerB0(nn.Module):
    """SegFormer-B0 with NVIDIA MiT encoder and all-MLP decoder."""

    def __init__(self, in_channels: int = 3, num_classes: int = 5, decoder_channels: int = 256) -> None:
        super().__init__()
        self.encoder = _MiTEncoder(in_channels)
        dimensions = (32, 64, 160, 256)
        self.projections = nn.ModuleList(nn.Linear(channels, decoder_channels) for channels in dimensions)
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * 4, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=False),
        )
        self.classifier = nn.Conv2d(decoder_channels, num_classes, 1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Segment images.

        Args:
            x: (B, C, H, W) input image.

        Returns:
            Dictionary with ``logits`` of shape (B, K, H, W).
        """
        output_size = x.shape[-2:]
        features = self.encoder(x)
        target_size = features[0].shape[-2:]
        decoded = []
        for feature, projection in zip(features, self.projections):
            batch, _, height, width = feature.shape
            embedded = projection(feature.flatten(2).transpose(1, 2)).transpose(1, 2)
            embedded = embedded.reshape(batch, -1, height, width)
            decoded.append(F.interpolate(embedded, target_size, mode="bilinear", align_corners=False))
        logits = self.classifier(self.fuse(torch.cat(decoded[::-1], dim=1)))
        return {"logits": F.interpolate(logits, output_size, mode="bilinear", align_corners=False)}
