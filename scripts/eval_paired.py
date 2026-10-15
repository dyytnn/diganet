#!/usr/bin/env python3
"""Compare two segmentation protocols with paired per-image statistics."""

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
from matblast.eval import paired_summary, per_image_iou
from matblast.models import build_model


def _load_models(
    config: dict[str, Any], checkpoints: list[str], device: torch.device
) -> list[torch.nn.Module]:
    models = []
    for checkpoint_path in checkpoints:
        model = build_model(config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        models.append(model.to(device).eval())
    return models


def _predict(
    models: list[torch.nn.Module],
    image: torch.Tensor,
    tta: bool,
    stage: torch.Tensor | None = None,
) -> torch.Tensor:
    predictions = []
    for model in models:
        if not tta:
            output = model(image, stage=stage) if stage is not None else model(image)
            predictions.append(output["logits"])
            continue
        augmented = []
        for turns in range(4):
            rotated = torch.rot90(image, turns, (-2, -1))
            output = model(rotated, stage=stage) if stage is not None else model(rotated)
            augmented.append(torch.rot90(output["logits"], -turns, (-2, -1)))
            reflected = torch.rot90(image.flip(-1), turns, (-2, -1))
            output = model(reflected, stage=stage) if stage is not None else model(reflected)
            restored = torch.rot90(output["logits"], -turns, (-2, -1)).flip(-1)
            augmented.append(restored)
        predictions.append(torch.stack(augmented).mean(0))
    return torch.stack(predictions).mean(0)


def _collect(
    config: dict[str, Any], checkpoints: list[str], tta: bool, workers: int
) -> tuple[list[str], np.ndarray]:
    config["training"]["num_workers"] = workers
    _, loader = build_dataloaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = _load_models(config, checkpoints, device)
    image_ids: list[str] = []
    values: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            stage = batch.get("stage")
            if stage is not None:
                stage = stage.to(device)
            logits = _predict(models, image, tta, stage)
            values.append(per_image_iou(logits.cpu(), batch["mask"]))
            image_ids.extend(batch["image_id"])
    return image_ids, np.concatenate(values, axis=0)


def main() -> None:
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
    config_a, config_b = load_config(args.config_a), load_config(args.config_b)
    ids_a, values_a = _collect(config_a, args.checkpoints_a, args.tta, args.num_workers)
    ids_b, values_b = _collect(config_b, args.checkpoints_b, args.tta, args.num_workers)
    if ids_a != ids_b:
        raise ValueError("protocols do not evaluate the same validation image order")
    names = ["BG", *config_a["data"].get("mask_classes", [])]
    if len(names) != values_a.shape[1] or values_a.shape != values_b.shape:
        raise ValueError("protocols have incompatible class contracts")
    report: dict[str, Any] = {"n_images": len(ids_a), "tta": args.tta, "comparison": "B_minus_A"}
    report["per_class"] = {
        name: paired_summary(
            values_a[:, index], values_b[:, index], args.bootstrap_samples, args.seed
        )
        for index, name in enumerate(names)
    }
    foreground = np.arange(1, len(names))
    report["macro_fg"] = paired_summary(
        np.nanmean(values_a[:, foreground], axis=1),
        np.nanmean(values_b[:, foreground], axis=1),
        args.bootstrap_samples,
        args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
