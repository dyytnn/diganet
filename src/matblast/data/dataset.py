from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from ..config import assert_known_keys
from .augment_nnunet import nnunet_augment, rederive_representation, resolve_augmentation_params
from .stage_metadata import load_stage_metadata


EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RESIZE_MODES = ("resize", "pad")
# "True" is what ``str(True)`` yields when a config sets ``augment: true``.
KNOWN_POLICIES = ("default", "True", "paper", "mimics", "massnet", "nnunet")
OVERSAMPLE_KEYS = ("enabled", "key", "power", "num_samples", "eps")


def seed_worker(worker_id: int) -> None:
    """Seed numpy/random inside a DataLoader worker from its torch seed."""
    del worker_id
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _target_hw(image_size: int | Sequence[int]) -> tuple[int, int]:
    """Normalise ``image_size`` (int or ``[H, W]``) to an ``(H, W)`` tuple."""
    if isinstance(image_size, int):
        return image_size, image_size
    if isinstance(image_size, Sequence) and len(image_size) == 2:
        height, width = (int(v) for v in image_size)
        return height, width
    raise ValueError(f"image_size must be an int or [H, W], got {image_size!r}")


def _resize_pil(
    image: Image.Image,
    size_hw: tuple[int, int],
    resize_mode: str,
    resample: Image.Resampling,
    fill: int = 0,
) -> Image.Image:
    """Bring a PIL image to ``size_hw``.

    ``resize`` stretches to the target (the historical behaviour). ``pad`` keeps the
    aspect ratio: scale so the image fits inside the target, then centre it on a
    canvas filled with ``fill`` (0 = black for images, background for masks).
    """
    height, width = size_hw
    if resize_mode == "resize":
        return image.resize((width, height), resample)
    if resize_mode == "pad":
        src_width, src_height = image.size
        scale = min(height / src_height, width / src_width)
        new_size = (max(1, round(src_width * scale)), max(1, round(src_height * scale)))
        resized = image.resize(new_size, resample)
        canvas = Image.new(image.mode, (width, height), fill)
        canvas.paste(resized, ((width - new_size[0]) // 2, (height - new_size[1]) // 2))
        return canvas
    raise ValueError(f"resize_mode must be one of {RESIZE_MODES}, got {resize_mode!r}")


def _intensity_zscore_gradient(gray: torch.Tensor) -> torch.Tensor:
    """Build raw-intensity, z-score, and gradient channels from ``(1, H, W)`` input."""
    norm = (gray - gray.mean()) / (gray.std() + 1e-6)
    grad_x = torch.zeros_like(gray)
    grad_y = torch.zeros_like(gray)
    grad_x[:, :, 1:] = gray[:, :, 1:] - gray[:, :, :-1]
    grad_y[:, 1:, :] = gray[:, 1:, :] - gray[:, :-1, :]
    gradient = torch.sqrt(grad_x.square() + grad_y.square())
    gradient = (gradient - gradient.mean()) / (gradient.std() + 1e-6)
    return torch.cat((gray, norm, gradient), dim=0)


def _intensity_zscore_dualraw(gray: torch.Tensor) -> torch.Tensor:
    """Build coherent raw-intensity, z-score, raw-intensity channels from ``(1, H, W)``."""
    norm = (gray - gray.mean()) / (gray.std() + 1e-6)
    return torch.cat((gray, norm, gray), dim=0)


def _global_l2_normalize(gray: torch.Tensor) -> torch.Tensor:
    """Globally L2-normalize one grayscale image, as used by MASS-Net."""
    return gray / gray.square().sum().sqrt().clamp_min(1e-6)


def _read_rgb(
    path: Path,
    size: int | Sequence[int],
    in_channels: int = 3,
    input_representation: str = "legacy",
    resize_mode: str = "resize",
) -> torch.Tensor:
    """Read one image into the requested input representation.

    Args:
        path: Image file.
        size: Target size, an int (square) or ``[H, W]``.
        in_channels: 1 or 3 output channels.
        input_representation: Channel recipe, see module docstring.
        resize_mode: ``"resize"`` (stretch) or ``"pad"`` (aspect-preserving letterbox).

    Returns:
        Tensor of shape (C, H, W), where C is ``in_channels``.
    """
    image = _resize_pil(
        Image.open(path).convert("RGB"), _target_hw(size), resize_mode, Image.Resampling.BILINEAR
    )
    x = torch.from_numpy(np.array(image, dtype=np.float32, copy=True)).permute(2, 0, 1) / 255.0
    if in_channels == 1:
        gray = x.mean(0, keepdim=True)
        if input_representation == "global_l2":
            return _global_l2_normalize(gray)
        if input_representation != "legacy":
            raise ValueError(
                f"Unsupported one-channel input representation: {input_representation}"
            )
        return gray
    if in_channels != 3:
        raise ValueError(f"Only one- and three-channel inputs are supported, got {in_channels}")
    gray = x.mean(0, keepdim=True)
    norm = (gray - gray.mean()) / (gray.std() + 1e-6)
    if input_representation == "intensity_zscore_gradient":
        return _intensity_zscore_gradient(gray)
    if input_representation == "intensity_zscore_dualraw":
        return _intensity_zscore_dualraw(gray)
    if input_representation != "legacy":
        raise ValueError(f"Unsupported input representation: {input_representation}")
    # Legacy contract: raw, global z-score, and a second raw grayscale channel.
    return torch.cat((x[:1], norm, x[1:2]), dim=0)


def _paper_augment(image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the paper-reported scale/rotation/crop/flip augmentation."""
    height, width = label.shape
    angle = float(torch.empty(()).uniform_(-15.0, 15.0).item())
    scale = float(torch.empty(()).uniform_(0.5, 2.0).item())
    radians = angle * np.pi / 180.0
    theta = image.new_tensor(
        [
            [np.cos(radians) / scale, -np.sin(radians) / scale, 0.0],
            [np.sin(radians) / scale, np.cos(radians) / scale, 0.0],
        ]
    ).unsqueeze(0)
    grid = torch.nn.functional.affine_grid(
        theta, (1, image.shape[0], height, width), align_corners=False
    )
    image = torch.nn.functional.grid_sample(
        image.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False
    ).squeeze(0)
    label = (
        torch.nn.functional.grid_sample(
            label.float().unsqueeze(0).unsqueeze(0),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
        .long()
    )
    if torch.rand(()) < 0.5:
        image, label = image.flip(-1), label.flip(-1)
    return image, label


def _augment(
    image: torch.Tensor,
    label: torch.Tensor,
    policy: str = "default",
    variant: int = 0,
    input_representation: str = "legacy",
    params: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply label-preserving augmentations to one image and mask."""
    if policy not in KNOWN_POLICIES:
        raise ValueError(f"unknown augmentation policy {policy!r}; choose from {KNOWN_POLICIES}")
    if policy == "nnunet":
        resolved = params if params is not None else resolve_augmentation_params(policy, None)
        return nnunet_augment(image, label, resolved, input_representation)
    if params:
        raise ValueError(f"augmentation params are only valid with policy 'nnunet', got {policy!r}")
    if policy == "paper":
        return _paper_augment(image, label)
    if policy == "mimics":
        rotations = variant % 4
        horizontal = (variant // 4) % 2
        vertical = (variant // 8) % 2
        if horizontal:
            image, label = image.flip(-1), label.flip(-1)
        if vertical:
            image, label = image.flip(-2), label.flip(-2)
        if rotations:
            image, label = (
                torch.rot90(image, rotations, (-2, -1)),
                torch.rot90(label, rotations, (-2, -1)),
            )
        return image, label
    if policy == "massnet":
        return _paper_augment(image, label)
    if torch.rand(()) < 0.5:
        image, label = image.flip(-1), label.flip(-1)
    if torch.rand(()) < 0.5:
        image, label = image.flip(-2), label.flip(-2)
    rotations = int(torch.randint(0, 4, ()).item())
    if label.shape[-2] != label.shape[-1]:
        # Rectangular inputs cannot be rotated by 90 degrees without changing shape.
        rotations = 2 * (rotations // 2)
    if rotations:
        image, label = (
            torch.rot90(image, rotations, (-2, -1)),
            torch.rot90(label, rotations, (-2, -1)),
        )
    if torch.rand(()) < 0.7:
        scale = torch.empty(()).uniform_(0.85, 1.15)
        offset = torch.empty(()).uniform_(-0.08, 0.08)
        image[:1] = (image[:1] * scale + offset).clamp(0, 1)
        gray = image[:1]
        image[1:2] = (gray - gray.mean()) / (gray.std() + 1e-6)
    if torch.rand(()) < 0.25:
        image[:1] = (image[:1] + torch.randn_like(image[:1]) * 0.025).clamp(0, 1)
    if input_representation == "intensity_zscore_gradient":
        image = _intensity_zscore_gradient(image[:1])
    elif input_representation == "intensity_zscore_dualraw":
        image = _intensity_zscore_dualraw(image[:1])
    return image, label


class BlastocystDataset(Dataset):
    """Configurable source/target dataset with an explicit multiclass contract."""

    def __init__(
        self,
        image_dir: str,
        mask_root: str | None = None,
        image_size: int | Sequence[int] = 256,
        mask_classes: list[str] | tuple[str, ...] = ("icm", "te", "zp"),
        indices: list[str] | None = None,
        augment: bool | str = False,
        in_channels: int = 3,
        augmentation_repeats: int = 1,
        input_representation: str = "legacy",
        stage_metadata: dict[str, int] | None = None,
        resize_mode: str = "resize",
        augmentation_params: dict[str, Any] | None = None,
        train_noise_sigma_max: float = 0.0,
    ):
        if train_noise_sigma_max < 0:
            raise ValueError("train_noise_sigma_max must be >= 0")
        self.train_noise_sigma_max = float(train_noise_sigma_max)
        self.image_dir, self.mask_root, self.image_size = (
            Path(image_dir),
            Path(mask_root) if mask_root else None,
            image_size,
        )
        self.size_hw = _target_hw(image_size)
        if resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {RESIZE_MODES}, got {resize_mode!r}")
        self.resize_mode = resize_mode
        self.mask_classes = tuple(mask_classes)
        self.augment = augment
        policy = str(augment) if augment else None
        if policy is not None and policy not in KNOWN_POLICIES:
            raise ValueError(
                f"unknown augmentation policy {policy!r}; choose from {KNOWN_POLICIES}"
            )
        self.augmentation_params = resolve_augmentation_params(
            policy or "default", augmentation_params
        )
        self.in_channels = in_channels
        self.input_representation = input_representation
        self.stage_metadata = stage_metadata
        self.augmentation_repeats = (
            max(1, int(augmentation_repeats)) if augment in {"mimics", "massnet"} else 1
        )
        if not self.image_dir.exists():
            raise FileNotFoundError(self.image_dir)
        paths = sorted(p for p in self.image_dir.rglob("*") if p.suffix.lower() in EXTS)
        selected = set(indices) if indices is not None else None
        base_paths = [p for p in paths if selected is None or p.name in selected]
        self.paths = base_paths * self.augmentation_repeats
        if not self.paths:
            raise ValueError(f"No images found in {self.image_dir}")
        if self.stage_metadata is not None:
            missing = sorted({path.name for path in self.paths} - self.stage_metadata.keys())
            if missing:
                raise ValueError(
                    f"Stage metadata is missing {len(missing)} image(s), first: {missing[0]}"
                )

    def __len__(self):
        return len(self.paths)

    def class_pixel_fraction(self, key: str) -> torch.Tensor:
        """Fraction of pixels of mask class ``key`` per item, at native mask resolution.

        Items without a mask file get 0. Results are cached per file, so repeated
        (augmentation-repeat) items are cheap.

        Returns:
            (N,) float tensor aligned with ``self.paths``.
        """
        if self.mask_root is None:
            raise ValueError("class_pixel_fraction requires a mask_root")
        cache: dict[str, float] = getattr(self, "_fraction_cache", {}).get(key, {})
        fractions = []
        for path in self.paths:
            if path.name not in cache:
                mask_path = self.mask_root / key / path.name
                if mask_path.exists():
                    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
                    cache[path.name] = float(mask.mean())
                else:
                    cache[path.name] = 0.0
            fractions.append(cache[path.name])
        self._fraction_cache = {**getattr(self, "_fraction_cache", {}), key: cache}
        return torch.tensor(fractions, dtype=torch.float64)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_index = (
            index % len(self.paths)
            if self.augmentation_repeats == 1
            else index % (len(self.paths) // self.augmentation_repeats)
        )
        path = self.paths[index] if self.augmentation_repeats == 1 else self.paths[base_index]
        image = _read_rgb(
            path, self.size_hw, self.in_channels, self.input_representation, self.resize_mode
        )
        label = torch.zeros(self.size_hw, dtype=torch.long)
        if self.mask_root is not None and self.mask_root.exists():
            # Priority assigns specific structures over coarse overlapping masks.
            for class_id, name in ((i + 1, n) for i, n in enumerate(self.mask_classes)):
                mask_path = self.mask_root / name / path.name
                if mask_path.exists():
                    m = _resize_pil(
                        Image.open(mask_path).convert("L"),
                        self.size_hw,
                        self.resize_mode,
                        Image.Resampling.NEAREST,
                    )
                    label[torch.from_numpy(np.array(m, copy=True)) > 0] = class_id
        if self.augment:
            variant = index // max(1, len(self.paths) // self.augmentation_repeats)
            image, label = _augment(
                image,
                label,
                str(self.augment),
                variant,
                self.input_representation,
                self.augmentation_params,
            )
            if self.train_noise_sigma_max > 0:
                # additive Gaussian noise on the raw intensity, then rebuild the channel recipe so the z-score
                # channel sees the noisy image too (the model is otherwise photometrically invariant)
                sigma = float(torch.rand(()) * self.train_noise_sigma_max)
                noisy = (image[:1] + sigma * torch.randn_like(image[:1])).clamp(0.0, 1.0)
                image = rederive_representation(image, noisy, self.input_representation)
        sample: dict[str, Any] = {"image": image, "mask": label, "image_id": path.name}
        if self.stage_metadata is not None:
            sample["stage"] = torch.tensor(self.stage_metadata[path.name], dtype=torch.long)
        return sample


def build_sample_weights(
    dataset: Dataset, key: str = "icm", power: float = 1.0, eps: float = 1e-4
) -> torch.Tensor:
    """Inverse-class-fraction sampling weights (``(fraction + eps) ** -power``).

    nnU-Net oversamples patches that contain foreground; every image in this
    dataset already contains the embryo, so the analogue is to draw images with a
    small ``key`` (ICM) area more often. ``power = 0`` gives uniform weights.
    Works for a :class:`BlastocystDataset` or a ``Subset`` of one.

    Returns:
        (N,) float tensor summing to 1, aligned with ``dataset``.
    """
    if isinstance(dataset, Subset):
        base = build_sample_weights(dataset.dataset, key, power, eps)
        weights = base[torch.as_tensor(list(dataset.indices))]
        return weights / weights.sum()
    if not isinstance(dataset, BlastocystDataset):
        raise TypeError(f"cannot build sample weights for {type(dataset).__name__}")
    fractions = dataset.class_pixel_fraction(key)
    weights = (fractions + eps) ** (-float(power))
    return weights / weights.sum()


def build_dataloaders(
    cfg: dict[str, Any], smoke_limit: int | None = None
) -> tuple[DataLoader, DataLoader]:
    """Build train/val loaders from a config.

    Args:
        cfg: Loaded configuration.
        smoke_limit: If given, restrict both splits to their first ``smoke_limit``
            items and use ``num_workers=0`` (smoke tests); samplers are preserved.
    """
    d, t = cfg["data"], cfg["training"]
    oversample = dict(d.get("oversample") or {})
    assert_known_keys(oversample, OVERSAMPLE_KEYS, "data.oversample")
    split_path = Path(d.get("split_json", ""))
    if not split_path.exists() and not cfg.get("allow_random_split", False):
        raise FileNotFoundError(
            f"Frozen split not found: {split_path}. Run scripts/split_data.py first, or explicitly set allow_random_split: true for a smoke test."
        )
    split = json.loads(split_path.read_text()) if split_path.exists() else {}
    fold = d.get("fold")
    if fold is not None:
        folds = split.get("folds", {})
        if str(fold) not in folds:
            raise ValueError(f"data.fold={fold!r} not in split folds {sorted(folds)}")
        train_ids, val_ids = folds[str(fold)]["train"], folds[str(fold)]["val"]
    else:
        train_ids = split.get("train", split.get("folds", {}).get("0", {}).get("train", []))
        val_ids = split.get("val", split.get("folds", {}).get("0", {}).get("val", []))
    input_channels = int(d.get("in_channels", cfg.get("model", {}).get("in_channels", 3)))
    input_representation = str(d.get("input_representation", "legacy"))
    stage_metadata = None
    if cfg.get("model", {}).get("stage_conditioned_graph", False):
        metadata_path = d.get("stage_metadata_csv")
        if not metadata_path:
            raise ValueError(
                "stage_metadata_csv is required when stage_conditioned_graph is enabled"
            )
        stage_metadata = load_stage_metadata(metadata_path)
    augmentation = d.get(
        "augmentation", "default" if d.get("augment", False) is True else d.get("augment", False)
    )
    repeats = int(d.get("augmentation_repeats", 16 if augmentation in {"mimics", "massnet"} else 1))
    resize_mode = str(d.get("resize_mode", "resize"))
    augmentation_params = d.get("augmentation_params")
    train_noise_sigma_max = float(d.get("train_noise_sigma_max", 0.0))
    full = BlastocystDataset(
        d["image_dir"],
        d.get("mask_root"),
        d.get("image_size", 256),
        d.get("mask_classes", ["icm", "te", "zp"]),
        in_channels=input_channels,
        input_representation=input_representation,
        stage_metadata=stage_metadata,
        resize_mode=resize_mode,
    )
    if train_ids and val_ids:
        train = BlastocystDataset(
            d["image_dir"],
            d.get("mask_root"),
            d.get("image_size", 256),
            d.get("mask_classes", ["icm", "te", "zp"]),
            train_ids,
            augmentation,
            input_channels,
            repeats,
            input_representation,
            stage_metadata,
            resize_mode=resize_mode,
            augmentation_params=augmentation_params,
            train_noise_sigma_max=train_noise_sigma_max,
        )
        val = BlastocystDataset(
            d["image_dir"],
            d.get("mask_root"),
            d.get("image_size", 256),
            d.get("mask_classes", ["icm", "te", "zp"]),
            val_ids,
            False,
            input_channels,
            1,
            input_representation,
            stage_metadata,
            resize_mode=resize_mode,
        )
    else:
        n = max(1, int(len(full) * 0.8))
        train, val = torch.utils.data.random_split(
            full,
            [n, len(full) - n],
            generator=torch.Generator().manual_seed(cfg.get("seed", 0)),
        )
    num_workers = t.get("num_workers", 0)
    if smoke_limit is not None:
        train = Subset(train, range(min(int(smoke_limit), len(train))))
        val = Subset(val, range(min(int(smoke_limit), len(val))))
        num_workers = 0
    kw = dict(
        batch_size=t.get("batch_size", 8),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    if oversample.get("enabled", False):
        weights = build_sample_weights(
            train,
            str(oversample.get("key", "icm")),
            float(oversample.get("power", 1.0)),
            float(oversample.get("eps", 1e-4)),
        )
        sampler = WeightedRandomSampler(
            weights,
            int(oversample.get("num_samples") or len(train)),
            replacement=True,
            generator=torch.Generator().manual_seed(int(cfg.get("seed", 0))),
        )
        train_loader = DataLoader(train, sampler=sampler, drop_last=t.get("drop_last", False), **kw)
    else:
        train_loader = DataLoader(train, shuffle=True, drop_last=t.get("drop_last", False), **kw)
    return train_loader, DataLoader(val, shuffle=False, **kw)
