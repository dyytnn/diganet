#!/usr/bin/env python3
"""Compare two protocols over aligned training seeds with hierarchical CIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from matblast.config import load_config
from matblast.data import build_dataloaders
from matblast.eval import multiseed_paired_summary, per_image_iou
from matblast.models import build_model


def _predict(
    model: torch.nn.Module, image: torch.Tensor, tta: bool, stage: torch.Tensor | None = None
) -> torch.Tensor:
    """Predict logits for a batch, optionally with rotation/reflection TTA."""
    if not tta:
        output = model(image, stage=stage) if stage is not None else model(image)
        return output["logits"]
    augmented = []
    for turns in range(4):
        rotated = torch.rot90(image, turns, (-2, -1))
        output = model(rotated, stage=stage) if stage is not None else model(rotated)
        augmented.append(torch.rot90(output["logits"], -turns, (-2, -1)))
        reflected = torch.rot90(image.flip(-1), turns, (-2, -1))
        output = model(reflected, stage=stage) if stage is not None else model(reflected)
        restored = torch.rot90(output["logits"], -turns, (-2, -1)).flip(-1)
        augmented.append(restored)
    return torch.stack(augmented).mean(0)


def _collect(
    config: dict[str, Any], checkpoint_path: str, tta: bool, workers: int
) -> tuple[list[str], np.ndarray]:
    """Evaluate one independently trained seed on the fixed validation loader."""
    config["training"]["num_workers"] = workers
    _, loader = build_dataloaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    image_ids: list[str] = []
    values: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            stage = batch.get("stage")
            if stage is not None:
                stage = stage.to(device)
            values.append(per_image_iou(_predict(model, image, tta, stage).cpu(), batch["mask"]))
            image_ids.extend(batch["image_id"])
    return image_ids, np.concatenate(values, axis=0)


def main() -> None:
    """Run seed- and image-aware paired evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-a", required=True)
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--checkpoints-a", nargs="+", required=True)
    parser.add_argument("--checkpoints-b", nargs="+", required=True)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if len(args.checkpoints_a) != len(args.checkpoints_b):
        raise ValueError("both protocols require the same number of aligned seeds")

    config_a, config_b = load_config(args.config_a), load_config(args.config_b)
    values_a: list[np.ndarray] = []
    values_b: list[np.ndarray] = []
    reference_ids: list[str] | None = None
    for checkpoint_a, checkpoint_b in zip(args.checkpoints_a, args.checkpoints_b, strict=True):
        ids_a, seed_a = _collect(config_a.copy(), checkpoint_a, args.tta, args.num_workers)
        ids_b, seed_b = _collect(config_b.copy(), checkpoint_b, args.tta, args.num_workers)
        if ids_a != ids_b or (reference_ids is not None and reference_ids != ids_a):
            raise ValueError("protocols do not evaluate the same validation image order")
        reference_ids = ids_a
        values_a.append(seed_a)
        values_b.append(seed_b)

    scores_a, scores_b = np.stack(values_a), np.stack(values_b)
    names = ["BG", *config_a["data"].get("mask_classes", [])]
    if len(names) != scores_a.shape[2] or scores_a.shape != scores_b.shape:
        raise ValueError("protocols have incompatible class contracts")
    foreground = np.arange(1, len(names))
    macro_a = np.nanmean(scores_a[:, :, foreground], axis=2)
    macro_b = np.nanmean(scores_b[:, :, foreground], axis=2)
    report: dict[str, Any] = {
        "comparison": "B_minus_A",
        "n_seeds": len(values_a),
        "n_images": len(reference_ids or []),
        "tta": args.tta,
        "per_class": {
            name: multiseed_paired_summary(
                scores_a[:, :, index], scores_b[:, :, index], args.bootstrap_samples, args.seed
            )
            for index, name in enumerate(names)
        },
        "macro_fg": multiseed_paired_summary(macro_a, macro_b, args.bootstrap_samples, args.seed),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
