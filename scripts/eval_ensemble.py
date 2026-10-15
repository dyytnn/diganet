#!/usr/bin/env python3
"""Evaluate an ensemble of checkpoints with optional rotation/fliplr TTA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from matblast.config import load_config
from matblast.data import build_dataloaders
from matblast.metrics import per_image_scores, summarize_scores
from matblast.models import build_model


def predict_tta(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Average logits over four rotations and their horizontal reflections."""
    predictions = []
    for turns in range(4):
        rotated = torch.rot90(image, turns, (-2, -1))
        predictions.append(torch.rot90(model(rotated)["logits"], -turns, (-2, -1)))
        reflected = torch.rot90(image.flip(-1), turns, (-2, -1))
        restored = torch.rot90(model(reflected)["logits"], -turns, (-2, -1)).flip(-1)
        predictions.append(restored)
    return torch.stack(predictions).mean(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--include-background", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["training"]["num_workers"] = args.num_workers
    _, loader = build_dataloaders(cfg)
    models = []
    for checkpoint_path in args.checkpoints:
        model = build_model(cfg)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)
    dice_scores, iou_scores = [], []
    with torch.no_grad():
        for batch in loader:
            logits = []
            for model in models:
                logits.append(
                    predict_tta(model, batch["image"])
                    if args.tta
                    else model(batch["image"])["logits"]
                )
            dice, iou = per_image_scores(torch.stack(logits).mean(0), batch["mask"])
            dice_scores.append(dice)
            iou_scores.append(iou)
    report = summarize_scores(
        torch.cat(dice_scores), torch.cat(iou_scores), args.include_background
    )
    names = cfg["data"].get("mask_classes", [])
    for class_id, class_name in enumerate(names, 1):
        report[f"dice_{class_name}"] = report.pop(f"dice_class_{class_id}")
        report[f"iou_{class_name}"] = report.pop(f"iou_class_{class_id}")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
