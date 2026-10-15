"""Mask Transfiner-lite: sparse refinement of incoherent regions with a small transformer.

Mask Transfiner (Ke et al., 2022) detects error-prone "incoherent" regions,
represents them as quadtree nodes across feature levels and refines all nodes
jointly with a transformer. This lite adaptation for semantic segmentation keeps
that structure with hard caps to avoid over-fitting on a small dataset: a fixed
number of nodes per decoder level is chosen by an incoherence score (the pooled
top-1/top-2 uncertainty of the dense prediction), every node becomes a token
(level feature + coarse logits + position + level embedding), a two-layer
transformer encoder refines the tokens, and the full-resolution node predictions
are written back into the dense logits. It is *inspired*, not a faithful
reproduction (no instance ROIs, no learned incoherence detector loss).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_module
from .point_rend import calculate_uncertainty, get_uncertain_point_coords_on_grid, point_sample


def sample_by_score(
    score: torch.Tensor, num_points: int, oversample_ratio: float, importance_ratio: float
) -> torch.Tensor:
    """Biased random sampling of ``num_points`` coordinates from a (B, 1, h, w) score map."""
    batch = score.shape[0]
    candidates = torch.rand(batch, int(num_points * oversample_ratio), 2, device=score.device)
    values = point_sample(score, candidates)[:, 0]
    num_important = int(importance_ratio * num_points)
    top = values.topk(num_important, dim=1).indices
    chosen = torch.gather(candidates, 1, top.unsqueeze(-1).expand(-1, -1, 2))
    remaining = num_points - num_important
    if remaining > 0:
        chosen = torch.cat((chosen, torch.rand(batch, remaining, 2, device=score.device)), dim=1)
    return chosen


@register_module("mask_transfiner_lite")
class MaskTransfinerLite(nn.Module):
    """Quadtree-style node refinement over decoder levels with a small transformer.

    Args:
        level_channels: Decoder channels from finest to coarsest, e.g. ``(c, 2c, 4c)``.
        num_classes: K.
        nodes_per_level: Nodes selected at each level (finest first).
        d_model / nhead / num_layers / dim_feedforward / dropout: Transformer size (kept small).
        classes_of_interest: Channels defining the incoherence score.
        oversample_ratio / importance_ratio: Training-time biased sampling.

    Input: decoder features ``[d1, d2, d3]`` and dense logits (B, K, H, W).
    Training output: node logits (B, K, N_total) and node coordinates (B, N_total, 2).
    Inference output: refined dense logits (B, K, H, W).
    """

    def __init__(
        self,
        level_channels: Sequence[int],
        num_classes: int,
        nodes_per_level: Sequence[int] = (1024, 512, 256),
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        classes_of_interest: Sequence[int] = (1, 2, 4),
        oversample_ratio: float = 3.0,
        importance_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        channels = [int(c) for c in level_channels]
        nodes = [int(n) for n in nodes_per_level]
        if len(channels) != len(nodes) or not channels:
            raise ValueError(
                "level_channels and nodes_per_level must have the same non-zero length"
            )
        if any(n < 1 for n in nodes):
            raise ValueError("nodes_per_level entries must be >= 1")
        if d_model % nhead:
            raise ValueError(f"d_model {d_model} must be divisible by nhead {nhead}")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        classes = tuple(int(c) for c in classes_of_interest)
        if len(classes) < 2 or any(c < 0 or c >= num_classes for c in classes):
            raise ValueError(f"classes_of_interest must hold >= 2 valid class ids, got {classes}")
        self.nodes_per_level = nodes
        self.classes_of_interest = classes
        self.oversample_ratio = oversample_ratio
        self.importance_ratio = importance_ratio
        self.feature_projections = nn.ModuleList([nn.Linear(c, d_model) for c in channels])
        self.logit_projection = nn.Linear(num_classes, d_model)
        self.position_projection = nn.Linear(2, d_model)
        self.level_embedding = nn.Embedding(len(channels), d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def _incoherence(self, logits: torch.Tensor, level: int) -> torch.Tensor:
        """(B, 1, H/2^level, W/2^level) pooled uncertainty (higher = more incoherent)."""
        uncertainty = calculate_uncertainty(logits, self.classes_of_interest)
        if level == 0:
            return uncertainty
        return F.avg_pool2d(uncertainty, 2**level)

    def _tokens(
        self, features: Sequence[torch.Tensor], logits: torch.Tensor, coords: list[torch.Tensor]
    ) -> torch.Tensor:
        tokens = []
        for level, level_coords in enumerate(coords):
            feat = point_sample(features[level], level_coords).transpose(1, 2)
            token = (
                self.feature_projections[level](feat)
                + self.logit_projection(point_sample(logits, level_coords).transpose(1, 2))
                + self.position_projection(level_coords)
                + self.level_embedding.weight[level]
            )
            tokens.append(token)
        return torch.cat(tokens, dim=1)

    def _predict(
        self, features: Sequence[torch.Tensor], logits: torch.Tensor, coords: list[torch.Tensor]
    ) -> torch.Tensor:
        """(B, K, N_total) node logits, finest level first."""
        return self.head(self.encoder(self._tokens(features, logits, coords))).transpose(1, 2)

    def forward_train(
        self, features: Sequence[torch.Tensor], logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = [
            sample_by_score(
                self._incoherence(logits.detach(), level),
                count,
                self.oversample_ratio,
                self.importance_ratio,
            )
            for level, count in enumerate(self.nodes_per_level)
        ]
        return self._predict(features, logits, coords), torch.cat(coords, dim=1)

    def forward_inference(
        self, features: Sequence[torch.Tensor], logits: torch.Tensor
    ) -> torch.Tensor:
        coords, fine_indices = [], None
        for level, count in enumerate(self.nodes_per_level):
            indices, level_coords = get_uncertain_point_coords_on_grid(
                self._incoherence(logits, level), count
            )
            coords.append(level_coords)
            if level == 0:
                fine_indices = indices
        node_logits = self._predict(features, logits, coords)
        fine_count = coords[0].shape[1]
        batch, classes, height, width = logits.shape
        flat = logits.reshape(batch, classes, -1).clone()
        flat.scatter_(
            2, fine_indices.unsqueeze(1).expand(-1, classes, -1), node_logits[:, :, :fine_count]
        )
        return flat.view(batch, classes, height, width)
