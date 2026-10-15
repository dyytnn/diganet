"""Exponential moving average of model weights."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEMA:
    """Keep an exponential moving average copy of a model.

    Float parameters *and* float buffers (e.g. BatchNorm running statistics) are
    averaged; non-float buffers are copied. With ``warmup_steps > 0`` the decay
    ramps up as ``min(decay, (1 + n) / (warmup_steps + n))`` so early averages are
    not dominated by the random initialisation.

    Args:
        model: The live training model (must already be on its device).
        decay: Asymptotic EMA decay.
        warmup_steps: Length of the decay ramp; 0 disables it.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 0) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"ema decay must be in [0, 1), got {decay}")
        if warmup_steps < 0:
            raise ValueError(f"ema warmup_steps must be >= 0, got {warmup_steps}")
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.updates = 0
        self.module = copy.deepcopy(model)
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    def current_decay(self) -> float:
        """Decay used by the next update (accounts for warm-up)."""
        if self.warmup_steps == 0:
            return self.decay
        return min(self.decay, (1.0 + self.updates) / (self.warmup_steps + self.updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the live ``model`` weights into the shadow copy."""
        decay = self.current_decay()
        live = model.state_dict()
        for name, shadow in self.module.state_dict().items():
            value = live[name]
            if shadow.dtype.is_floating_point:
                shadow.mul_(decay).add_(value.to(shadow.device), alpha=1.0 - decay)
            else:
                shadow.copy_(value)
        self.updates += 1

    def state_dict(self) -> dict:
        return {"decay": self.decay, "warmup_steps": self.warmup_steps, "updates": self.updates}
