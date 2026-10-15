"""Losses specified by the requested paper baselines."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class WeightedSoftJaccardLoss(nn.Module):
    """Weighted soft Jaccard objective used by BTFormer.

    The paper adds weights 1.0 and 0.8 to the two classes with the lowest
    soft-Jaccard scores in each batch.  Background remains in the objective,
    matching the paper's five-channel training formulation.
    """

    def __init__(
        self, minimum_class_weights: tuple[float, float] = (1.0, 0.8), eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.minimum_class_weights = minimum_class_weights
        self.eps = eps

    def forward(self, output: dict[str, Tensor] | Tensor, target: Tensor) -> dict[str, Tensor]:
        """Compute batch soft Jaccard loss.

        Args:
            output: Logits or a dict with ``logits`` shaped (B, K, H, W).
            target: (B, H, W) integer class labels.

        Returns:
            Dictionary containing scalar ``total_loss``.
        """
        logits = output["logits"] if isinstance(output, dict) else output
        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).to(logits.dtype)
        intersection = (probabilities * one_hot).sum((0, 2, 3))
        union = (probabilities + one_hot - probabilities * one_hot).sum((0, 2, 3))
        scores = (intersection + self.eps) / (union + self.eps)
        original_scores = scores
        weights = torch.ones_like(scores)
        available = torch.ones_like(scores, dtype=torch.bool)
        for rank, weight in enumerate(self.minimum_class_weights):
            ranked_scores = original_scores.detach().masked_fill(~available, float("inf"))
            index = torch.argmin(ranked_scores)
            weights[index] += weight
            available[index] = False
        return {"total_loss": 1.0 - (weights * original_scores).sum() / weights.sum()}


class I2CNetCrossEntropyLoss(nn.Module):
    """Cross-entropy at the five I2CNet encoder scales."""

    def forward(
        self, output: dict[str, Tensor | list[Tensor]], target: Tensor
    ) -> dict[str, Tensor]:
        """Compute average deep-supervision cross entropy.

        Args:
            output: Dict with ``deep_logits`` list of (B, K, h, w) tensors.
            target: (B, H, W) integer class labels.

        Returns:
            Dictionary containing scalar ``total_loss``.
        """
        deep_logits = output.get("deep_logits")
        if not isinstance(deep_logits, list) or not deep_logits:
            raise ValueError("I2CNet output must contain a non-empty deep_logits list")
        losses = []
        for logits in deep_logits:
            scaled_target = (
                F.interpolate(target.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest")
                .squeeze(1)
                .long()
            )
            losses.append(F.cross_entropy(logits, scaled_target))
        final_logits = output.get("logits")
        if not isinstance(final_logits, Tensor):
            raise ValueError("I2CNet output must contain final logits")
        losses.append(F.cross_entropy(final_logits, target))
        return {
            "total_loss": torch.stack(losses).mean(),
            "deep_ce": torch.stack(losses).mean().detach(),
        }


class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice loss used by MiMics-Net.

    Class weights are inverse squared target volumes, including background as
    described by the standard generalized-Dice formulation.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, output: dict[str, Tensor] | Tensor, target: Tensor) -> dict[str, Tensor]:
        """Compute generalized Dice loss.

        Args:
            output: Logits or a dict with ``logits`` shaped (B, K, H, W).
            target: (B, H, W) integer class labels.

        Returns:
            Dictionary containing scalar ``total_loss``.
        """
        logits = output["logits"] if isinstance(output, dict) else output
        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).to(logits.dtype)
        target_volume = one_hot.sum((0, 2, 3))
        class_weights = 1.0 / target_volume.square().clamp_min(self.eps)
        numerator = 2.0 * (probabilities * one_hot).sum((0, 2, 3))
        denominator = (probabilities + one_hot).sum((0, 2, 3))
        score = (class_weights * numerator).sum() / (class_weights * denominator).sum().clamp_min(
            self.eps
        )
        return {"total_loss": 1.0 - score}
