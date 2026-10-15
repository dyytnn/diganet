"""nnU-Net-style augmentation policy for single images and label maps.

Reimplements the default nnU-Net v2 training augmentations with plain ``torch``
ops so that no new dependency is introduced and every random draw follows the
seeded torch generator of the calling DataLoader worker:

rotation (any angle; embryos have no canonical orientation), isotropic scaling,
optional elastic deformation, mirroring, Gaussian noise, Gaussian blur,
multiplicative brightness, contrast, low-resolution simulation and gamma
(including the inverted-gamma variant).

All hyperparameters live in :data:`NNUNET_AUG_DEFAULTS` and can be overridden
from ``data.augmentation_params``; unknown keys raise.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from ..config import assert_known_keys

NNUNET_AUG_DEFAULTS: dict[str, Any] = {
    "p_rotation": 0.2,
    "rotation_deg": 180.0,
    "p_scale": 0.2,
    "scale_range": (0.7, 1.4),
    "p_elastic": 0.0,
    "elastic_alpha_px": (0.0, 8.0),
    "elastic_grid": 4,
    "p_mirror_x": 0.5,
    "p_mirror_y": 0.5,
    "p_gaussian_noise": 0.1,
    "noise_std_range": (0.0, 0.05),
    "p_gaussian_blur": 0.2,
    "blur_sigma_range": (0.5, 1.0),
    "p_brightness": 0.15,
    "brightness_range": (0.75, 1.25),
    "p_contrast": 0.15,
    "contrast_range": (0.75, 1.25),
    "p_lowres": 0.25,
    "lowres_zoom_range": (0.5, 1.0),
    "p_gamma": 0.3,
    "p_gamma_inverted": 0.1,
    "gamma_range": (0.7, 1.5),
}


def resolve_augmentation_params(
    policy: str, params: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Merge user overrides into the defaults for the ``nnunet`` policy.

    Args:
        policy: Augmentation policy name.
        params: Overrides from ``data.augmentation_params`` (may be ``None``).

    Returns:
        The full parameter dict for ``nnunet``, ``None`` for every other policy.

    Raises:
        ValueError: If overrides are given for a policy that takes none, or if
            an override key is unknown.
    """
    if policy != "nnunet":
        if params:
            raise ValueError(
                f"data.augmentation_params is only valid with augmentation 'nnunet', got {policy!r}"
            )
        return None
    params = dict(params or {})
    assert_known_keys(params, NNUNET_AUG_DEFAULTS, "data.augmentation_params")
    resolved = {**NNUNET_AUG_DEFAULTS, **params}
    for key, value in resolved.items():
        if isinstance(value, list):
            resolved[key] = tuple(value)
    return resolved


def _uniform(low_high: tuple[float, float]) -> float:
    low, high = low_high
    return float(torch.empty(()).uniform_(float(low), float(high)).item())


def _chance(probability: float) -> bool:
    return bool(torch.rand(()) < probability)


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a ``(C, H, W)`` tensor with reflect padding."""
    radius = int(math.ceil(3.0 * sigma))
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype)
    kernel = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    channels = x.shape[0]
    pad = (radius, radius, radius, radius)
    padded = F.pad(x.unsqueeze(0), pad, mode="reflect")
    horizontal = F.conv2d(
        padded, kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels
    )
    vertical = F.conv2d(
        horizontal, kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels
    )
    return vertical.squeeze(0)


def _geometric(
    image: torch.Tensor, label: torch.Tensor, params: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation, scaling and elastic deformation through a single ``grid_sample``."""
    rotate = _chance(params["p_rotation"])
    rescale = _chance(params["p_scale"])
    elastic = _chance(params["p_elastic"])
    if not (rotate or rescale or elastic):
        return image, label
    height, width = label.shape[-2:]
    angle = _uniform((-params["rotation_deg"], params["rotation_deg"])) if rotate else 0.0
    scale = _uniform(params["scale_range"]) if rescale else 1.0
    radians = angle * math.pi / 180.0
    theta = image.new_tensor(
        [
            [math.cos(radians) / scale, -math.sin(radians) / scale, 0.0],
            [math.sin(radians) / scale, math.cos(radians) / scale, 0.0],
        ]
    ).unsqueeze(0)
    grid = F.affine_grid(theta, (1, image.shape[0], height, width), align_corners=False)
    if elastic:
        alpha_px = _uniform(params["elastic_alpha_px"])
        size = int(params["elastic_grid"])
        coarse = torch.randn(1, 2, size, size, dtype=image.dtype)
        displacement = F.interpolate(
            coarse, size=(height, width), mode="bilinear", align_corners=False
        )
        # Normalised grid units: 2 px per full extent -> alpha_px * 2 / extent.
        displacement = displacement.permute(0, 2, 3, 1) * image.new_tensor(
            [2.0 * alpha_px / width, 2.0 * alpha_px / height]
        )
        grid = grid + displacement
    warped_image = F.grid_sample(
        image.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False
    ).squeeze(0)
    warped_label = (
        F.grid_sample(
            label.to(image.dtype).unsqueeze(0).unsqueeze(0),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
        .long()
    )
    return warped_image, warped_label


def _photometric(raw: torch.Tensor, params: Mapping[str, Any]) -> torch.Tensor:
    """Intensity augmentations on raw channels in ``[0, 1]``; returns ``(C, H, W)``."""
    if _chance(params["p_gaussian_noise"]):
        raw = raw + torch.randn_like(raw) * _uniform(params["noise_std_range"])
    if _chance(params["p_gaussian_blur"]):
        raw = _gaussian_blur(raw, _uniform(params["blur_sigma_range"]))
    if _chance(params["p_brightness"]):
        raw = raw * _uniform(params["brightness_range"])
    if _chance(params["p_contrast"]):
        mean = raw.mean()
        raw = (raw - mean) * _uniform(params["contrast_range"]) + mean
    if _chance(params["p_lowres"]):
        zoom = _uniform(params["lowres_zoom_range"])
        height, width = raw.shape[-2:]
        small = (max(1, int(round(height * zoom))), max(1, int(round(width * zoom))))
        raw = F.interpolate(
            F.interpolate(raw.unsqueeze(0), size=small, mode="nearest"),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    raw = raw.clamp(0.0, 1.0)
    if _chance(params["p_gamma_inverted"]):
        raw = 1.0 - _gamma(1.0 - raw, _uniform(params["gamma_range"]))
    if _chance(params["p_gamma"]):
        raw = _gamma(raw, _uniform(params["gamma_range"]))
    return raw.clamp(0.0, 1.0)


def _gamma(raw: torch.Tensor, gamma: float) -> torch.Tensor:
    """Gamma transform on ``[0, 1]`` data that preserves the input mean and std."""
    mean, std = raw.mean(), raw.std()
    low, high = raw.min(), raw.max()
    normalised = (raw - low) / (high - low + 1e-7)
    transformed = normalised.pow(gamma) * (high - low + 1e-7) + low
    transformed = (transformed - transformed.mean()) / (transformed.std() + 1e-7) * std + mean
    return transformed


def rederive_representation(
    image: torch.Tensor, raw: torch.Tensor, input_representation: str
) -> torch.Tensor:
    """Rebuild the channel recipe after the raw intensity channel changed.

    Args:
        image: (C, H, W) augmented tensor (used for its channel layout).
        raw: (1, H, W) or (C, H, W) photometrically augmented raw channel(s).
        input_representation: The dataset's representation name.

    Returns:
        (C, H, W) tensor consistent with how ``_read_rgb`` builds inputs.
    """
    from .dataset import (
        _global_l2_normalize,
        _intensity_zscore_dualraw,
        _intensity_zscore_gradient,
    )

    gray = raw[:1]
    if image.shape[0] == 1:
        return _global_l2_normalize(gray) if input_representation == "global_l2" else gray
    if input_representation == "intensity_zscore_gradient":
        return _intensity_zscore_gradient(gray)
    if input_representation == "intensity_zscore_dualraw":
        return _intensity_zscore_dualraw(gray)
    # Legacy contract: raw, global z-score, second raw channel.
    second = raw[1:2] if raw.shape[0] > 1 else gray
    return torch.cat((gray, (gray - gray.mean()) / (gray.std() + 1e-6), second), dim=0)


def nnunet_augment(
    image: torch.Tensor,
    label: torch.Tensor,
    params: Mapping[str, Any],
    input_representation: str = "legacy",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the nnU-Net-style policy to one sample.

    Args:
        image: (C, H, W) input as produced by ``_read_rgb`` (C in {1, 3}).
        label: (H, W) integer labels.
        params: Resolved parameters, see :func:`resolve_augmentation_params`.
        input_representation: Dataset representation, used to rebuild channels.

    Returns:
        Augmented ``(image, label)`` with unchanged shapes and dtypes.
    """
    image, label = _geometric(image, label, params)
    if _chance(params["p_mirror_x"]):
        image, label = image.flip(-1), label.flip(-1)
    if _chance(params["p_mirror_y"]):
        image, label = image.flip(-2), label.flip(-2)
    if image.shape[0] == 1:
        raw = image
    elif input_representation == "legacy":
        raw = torch.cat((image[:1], image[2:3]), dim=0)
    else:
        raw = image[:1]
    raw = _photometric(raw, params)
    return rederive_representation(image, raw, input_representation), label
