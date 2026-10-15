from __future__ import annotations

from collections.abc import Sequence

import torch.nn.functional as F
import torch
from torch import Tensor


def soft_anatomy_relation_loss(
    logits: Tensor,
    target: Tensor,
    relation_pairs: Sequence[tuple[int, int]],
    kernel_size: int = 5,
) -> Tensor:
    """Match soft local class relations without enforcing hard nesting.

    Args:
        logits: (B, K, H, W) segmentation logits.
        target: (B, H, W) integer labels in the fixed five-class contract.
        relation_pairs: (outer, inner) class-index pairs to supervise softly.
        kernel_size: Odd local context window used for relation maps.

    Returns:
        Scalar relation loss. Zero when no relation pairs are configured.

    The target relation is local class co-occurrence, so hatching and
    zona-ablated examples are representable instead of being forced into a
    global containment constraint.
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if not relation_pairs:
        return logits.new_zeros(())

    probabilities = logits.softmax(dim=1)
    num_classes = logits.shape[1]
    one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).to(logits.dtype)
    padding = kernel_size // 2
    losses: list[Tensor] = []
    for outer, inner in relation_pairs:
        if not 0 <= outer < num_classes or not 0 <= inner < num_classes:
            raise ValueError(f"relation pair {(outer, inner)} is outside K={num_classes}")
        predicted_relation = probabilities[:, inner : inner + 1] * F.avg_pool2d(
            probabilities[:, outer : outer + 1], kernel_size, stride=1, padding=padding
        )
        target_relation = one_hot[:, inner : inner + 1] * F.avg_pool2d(
            one_hot[:, outer : outer + 1], kernel_size, stride=1, padding=padding
        )
        losses.append(F.smooth_l1_loss(predicted_relation, target_relation))
    return torch.stack(losses).mean()
