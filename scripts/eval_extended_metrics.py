#!/usr/bin/env python3
"""Evaluate geometry, topology, morphology, and calibration metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from matblast.config import load_config
from matblast.data import build_dataloaders
from matblast.engine import seed_everything
from matblast.metrics import betti_error, boundary_f, ece, hd95, morphometry
from matblast.models import build_model
from matblast.provenance import current_commit


def _parse_spec(value: str) -> tuple[str, str, list[str]]:
    """Parse ``name=config:checkpoint0,checkpoint1``."""
    try:
        name, remainder = value.split("=", 1)
        config, checkpoints = remainder.split(":", 1)
    except ValueError as exc:
        raise ValueError("spec must be name=config:checkpoint0,checkpoint1,...") from exc
    paths = [path for path in checkpoints.split(",") if path]
    if not name or not config or not paths:
        raise ValueError(f"invalid spec: {value}")
    return name, config, paths


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return NaN-safe mean and sample standard deviation."""
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)), float(np.nanstd(array, ddof=1)) if len(array) > 1 else 0.0


def _evaluate_checkpoint(
    config_path: str, checkpoint_path: str, device: torch.device
) -> dict[str, float]:
    """Evaluate one checkpoint on the fixed validation split."""
    config = load_config(config_path)
    config["training"]["num_workers"] = 0
    _, loader = build_dataloaders(config)
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    class_names = ["BG", *config["data"].get("mask_classes", [])]
    per_class_hd95: list[list[float]] = [[] for _ in class_names]
    per_class_boundary: list[list[float]] = [[] for _ in class_names]
    topology: list[float] = []
    morphometry_values: dict[str, list[float]] = {name: [] for name in (
        "bl_area", "zp_thickness", "icm_area", "icm_te_ratio", "te_continuity"
    )}
    calibration: list[float] = []
    start = time.perf_counter()
    image_count = 0
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["image"].to(device))
            probabilities = output["logits"].softmax(1).cpu().numpy()
            predictions = probabilities.argmax(1)
            targets = batch["mask"].numpy()
            for prediction, target, probability in zip(predictions, targets, probabilities):
                image_count += 1
                topology_value = 0
                for class_id in range(1, len(class_names)):
                    predicted_mask = prediction == class_id
                    target_mask = target == class_id
                    if target_mask.any():
                        per_class_hd95[class_id].append(hd95(predicted_mask, target_mask))
                        per_class_boundary[class_id].append(boundary_f(predicted_mask, target_mask))
                    else:
                        # Match the project-wide convention: absent GT classes
                        # are NaN and excluded by nanmean, never treated as zero.
                        per_class_hd95[class_id].append(float("nan"))
                        per_class_boundary[class_id].append(float("nan"))
                    topology_value += betti_error(predicted_mask, target_mask)["total"]
                topology.append(float(topology_value))
                measures = morphometry(prediction)
                for name in morphometry_values:
                    morphometry_values[name].append(measures[name])
                calibration.append(ece(probability, target))
    elapsed = time.perf_counter() - start
    report: dict[str, float] = {
        "params_m": sum(parameter.numel() for parameter in model.parameters()) / 1e6,
        "fps_cpu": image_count / max(elapsed, 1e-9),
        "betti_error_mean": _mean_std(topology)[0],
        "ece_mean": _mean_std(calibration)[0],
    }
    for class_id, class_name in enumerate(class_names[1:], 1):
        report[f"hd95_{class_name.lower()}_mean"] = _mean_std(per_class_hd95[class_id])[0]
        report[f"boundary_f_{class_name.lower()}_mean"] = _mean_std(per_class_boundary[class_id])[0]
    for name, values in morphometry_values.items():
        report[f"{name}_mean"] = _mean_std(values)[0]
    return report


def main() -> None:
    """Evaluate all requested specs and write JSON/CSV artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    seed_everything(args.seed)
    resolved_device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if resolved_device == "auto":
        resolved_device = "cpu"
    device = torch.device(resolved_device)
    models: list[dict[str, Any]] = []
    for raw_spec in args.spec:
        name, config, checkpoints = _parse_spec(raw_spec)
        seed_reports = [
            _evaluate_checkpoint(config, checkpoint, device) for checkpoint in checkpoints
        ]
        keys = sorted(seed_reports[0])
        aggregate: dict[str, Any] = {"model": name, "n_seeds": len(seed_reports), "seeds": seed_reports}
        for key in keys:
            # Per-seed reports use a ``*_mean`` suffix.  Strip it before
            # adding the across-seed mean/std suffixes to avoid names such as
            # ``hd95_te_mean_mean`` in the generated artifact.
            base_key = key.removesuffix("_mean")
            aggregate[f"{base_key}_mean"], aggregate[f"{base_key}_std"] = _mean_std(
                [report[key] for report in seed_reports]
            )
        models.append(aggregate)
    payload = {
        "protocol": "fixed grouped validation split; no test data; BG excluded from class summaries",
        "commit": current_commit(Path(__file__).resolve().parents[1]),
        "specs": args.spec,
        "models": models,
    }
    output_path = Path(args.out)
    csv_path = Path(args.csv_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    rows = [{key: value for key, value in model.items() if key != "seeds"} for model in models]
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"models": [model["model"] for model in models], "n_models": len(models)}))


if __name__ == "__main__":
    main()
