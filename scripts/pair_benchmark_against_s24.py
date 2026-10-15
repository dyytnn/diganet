#!/usr/bin/env python3
"""Create paired IoU comparisons for every benchmark model against S24."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from matblast.eval import multiseed_paired_summary  # noqa: E402
from matblast.provenance import current_commit  # noqa: E402


CLASS_NAMES = ("BG", "ZP", "TE", "BL", "ICM")


def _load_scores(payload: dict[str, Any], model: str) -> np.ndarray:
    """Load one model's ``(seeds, images, classes)`` IoU array."""
    rows = []
    for seed in range(3):
        key = f"{model}:seed{seed}"
        if key not in payload["per_image_iou"]:
            raise KeyError(f"missing three-seed scores for {model}: {key}")
        rows.append(payload["per_image_iou"][key])
    return np.asarray(rows, dtype=np.float64)


def _compare(
    candidate: np.ndarray, reference: np.ndarray, samples: int, seed: int
) -> dict[str, Any]:
    """Compare candidate minus S24 for each class and foreground macro IoU."""
    if candidate.shape != reference.shape or candidate.ndim != 3:
        raise ValueError("candidate and S24 scores must both have shape (3, N, 5)")
    macro_candidate = np.nanmean(candidate[:, :, 1:], axis=2)
    macro_reference = np.nanmean(reference[:, :, 1:], axis=2)
    return {
        "comparison": "candidate_minus_S24",
        "n_seeds": int(candidate.shape[0]),
        "n_images": int(candidate.shape[1]),
        "per_class_iou": {
            name: multiseed_paired_summary(
                reference[:, :, index], candidate[:, :, index], samples, seed
            )
            for index, name in enumerate(CLASS_NAMES[1:], start=1)
        },
        "macro_fg_iou": multiseed_paired_summary(macro_reference, macro_candidate, samples, seed),
    }


def main() -> None:
    """Write generated paired JSON and CSV reports."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--reference", default="S24")
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(Path(args.benchmark).read_text())
    ids = payload.get("validation_image_ids", [])
    reference = _load_scores(payload, args.reference)
    model_names = [report["model"] for report in payload["models"]]
    if args.reference not in model_names:
        raise ValueError(f"reference model {args.reference} is not present")
    comparisons = {
        model: _compare(_load_scores(payload, model), reference, args.bootstrap_samples, args.seed)
        for model in model_names
        if model != args.reference
    }
    output = {
        "protocol": payload["protocol"],
        "commit": current_commit(Path(__file__).resolve().parents[1]),
        "benchmark": args.benchmark,
        "reference": args.reference,
        "n_images": len(ids),
        "n_seeds": 3,
        "comparisons": comparisons,
    }
    output_path, csv_path = Path(args.out), Path(args.csv_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    rows: list[dict[str, Any]] = []
    for model, comparison in comparisons.items():
        for class_name, summary in comparison["per_class_iou"].items():
            rows.append({"model": model, "metric": f"iou_{class_name.lower()}", **summary})
        rows.append({"model": model, "metric": "miou_fg", **comparison["macro_fg_iou"]})
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
