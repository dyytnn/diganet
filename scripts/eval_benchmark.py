#!/usr/bin/env python3
"""Evaluate all completed baseline seeds with one common metric protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from matblast.config import load_config
from matblast.data import build_dataloaders
from matblast.metrics import per_image_scores, summarize_scores
from matblast.models import build_model
from matblast.engine import seed_everything
from matblast.provenance import current_commit


def _predict_tta(
    model: torch.nn.Module, image: torch.Tensor, stage: torch.Tensor | None = None
) -> torch.Tensor:
    """Average logits over rotations and horizontal reflections.

    Args:
        model: Segmentation model accepting ``(B, C, H, W)``.
        image: (B, C, H, W) normalized input.

    Returns:
        (B, K, H, W) averaged logits.
    """
    predictions: list[torch.Tensor] = []
    for turns in range(4):
        rotated = torch.rot90(image, turns, (-2, -1))
        output = model(rotated, stage=stage) if stage is not None else model(rotated)
        predictions.append(torch.rot90(output["logits"], -turns, (-2, -1)))
        reflected = torch.rot90(image.flip(-1), turns, (-2, -1))
        output = model(reflected, stage=stage) if stage is not None else model(reflected)
        restored = torch.rot90(output["logits"], -turns, (-2, -1)).flip(-1)
        predictions.append(restored)
    return torch.stack(predictions).mean(0)


def _evaluate_checkpoint(
    config_path: str, checkpoint_path: str, device: torch.device, tta: bool, workers: int
) -> tuple[dict[str, float], np.ndarray, list[str]]:
    """Evaluate one checkpoint on the config's fixed validation split."""
    config = load_config(config_path)
    config["training"]["num_workers"] = workers
    _, loader = build_dataloaders(config)
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in checkpoint:
        raise ValueError(f"Checkpoint has no model state: {checkpoint_path}")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dice_values: list[torch.Tensor] = []
    iou_values: list[torch.Tensor] = []
    image_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            stage = batch.get("stage")
            if stage is not None:
                stage = stage.to(device)
            logits = (
                _predict_tta(model, image, stage)
                if tta
                else (model(image, stage=stage) if stage is not None else model(image))["logits"]
            )
            dice, iou = per_image_scores(logits.cpu(), batch["mask"])
            dice_values.append(dice)
            iou_values.append(iou)
            image_ids.extend(batch["image_id"])
    report = summarize_scores(torch.cat(dice_values), torch.cat(iou_values), False)
    class_names = ["BG", *config["data"].get("mask_classes", [])]
    if len(class_names) != dice_values[0].shape[1]:
        raise ValueError(f"Class contract mismatch for {config_path}")
    renamed: dict[str, float] = {}
    for index, name in enumerate(class_names):
        key = name.lower()
        renamed[f"dice_{key}"] = report["dice_bg"] if index == 0 else report[f"dice_class_{index}"]
        renamed[f"iou_{key}"] = report["iou_bg"] if index == 0 else report[f"iou_class_{index}"]
    renamed["mdice_fg"] = report["mdice_fg"]
    renamed["miou_fg"] = report["miou_fg"]
    return renamed, torch.cat(iou_values).numpy(), image_ids


def _aggregate(model_name: str, seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate completed seed reports while preserving NaN semantics."""
    metric_names = sorted(key for key in seed_reports[0] if key.startswith(("dice_", "iou_")))
    metric_names += ["mdice_fg", "miou_fg"]
    result: dict[str, Any] = {"model": model_name, "n_seeds": len(seed_reports)}
    for metric in metric_names:
        values = np.asarray([report[metric] for report in seed_reports], dtype=float)
        result[f"{metric}_mean"] = float(np.nanmean(values))
        result[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    result["seeds"] = seed_reports
    return result


def _parse_spec(value: str) -> tuple[str, str, str]:
    """Parse ``name=config:checkpoint0,checkpoint1`` benchmark specification."""
    try:
        name, remainder = value.split("=", 1)
        config, checkpoints = remainder.split(":", 1)
    except ValueError as exc:
        raise ValueError("spec must be name=config_path:checkpoint0,checkpoint1,...") from exc
    paths = [path for path in checkpoints.split(",") if path]
    if not name or not config or not paths:
        raise ValueError(f"invalid benchmark spec: {value}")
    return name, config, ",".join(paths)


def main() -> None:
    """Evaluate each requested model and write JSON/CSV aggregate reports."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        help="name=config_path:checkpoint0,checkpoint1,... (one checkpoint per seed)",
    )
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--csv-out", required=True, help="output CSV path")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aggregate_reports: list[dict[str, Any]] = []
    per_image_iou: dict[str, list[list[float | None]]] = {}
    validation_image_ids: list[str] | None = None
    for raw_spec in args.spec:
        name, config_path, checkpoint_csv = _parse_spec(raw_spec)
        checkpoints = checkpoint_csv.split(",")
        seed_reports: list[dict[str, Any]] = []
        for seed, checkpoint in enumerate(checkpoints):
            report, iou, image_ids = _evaluate_checkpoint(
                config_path, checkpoint, device, args.tta, args.num_workers
            )
            if validation_image_ids is None:
                validation_image_ids = image_ids
            elif validation_image_ids != image_ids:
                raise ValueError("all model specs must use the same validation image order")
            seed_reports.append(report)
            per_image_iou[f"{name}:seed{seed}"] = [
                [None if not np.isfinite(value) else float(value) for value in row] for row in iou
            ]
        aggregate_reports.append(_aggregate(name, seed_reports))

    output = {
        "protocol": "fixed grouped validation split; foreground macro excludes BG",
        "tta": args.tta,
        "device": str(device),
        "commit": current_commit(Path(__file__).resolve().parents[1]),
        "specs": args.spec,
        "validation_image_ids": validation_image_ids or [],
        "per_image_iou": per_image_iou,
        "models": aggregate_reports,
    }
    output_path = Path(args.out)
    csv_path = Path(args.csv_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    rows = [
        {key: value for key, value in report.items() if key != "seeds"}
        for report in aggregate_reports
    ]
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
