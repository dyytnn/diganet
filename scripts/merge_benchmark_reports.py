#!/usr/bin/env python3
"""Merge generic, nnU-Net, and optional reproduced-baseline reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    """Merge generated report artifacts after validating the common protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--generic", required=True)
    parser.add_argument("--nnunet", required=True)
    parser.add_argument(
        "--additional",
        action="append",
        default=[],
        help="Optional no-TTA report from eval_benchmark.py; may be supplied repeatedly.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv-out", required=True)
    args = parser.parse_args()

    generic = json.loads(Path(args.generic).read_text())
    nnunet = json.loads(Path(args.nnunet).read_text())
    if generic["validation_image_ids"] != nnunet["validation_image_ids"]:
        raise ValueError("generic and nnU-Net validation image order differs")
    if generic["tta"]:
        raise ValueError("expected no-TTA generic report")
    if nnunet["aggregate"]["n_seeds"] != 3:
        raise ValueError("nnU-Net report must contain three seeds")

    additional_reports = [json.loads(Path(path).read_text()) for path in args.additional]
    for path, report in zip(args.additional, additional_reports, strict=True):
        if report["validation_image_ids"] != generic["validation_image_ids"]:
            raise ValueError(f"additional report validation image order differs: {path}")
        if report["tta"]:
            raise ValueError(f"additional report must disable TTA: {path}")
        for model in report["models"]:
            if model["n_seeds"] != 3:
                raise ValueError(
                    f"additional report model must contain three seeds: {model['model']}"
                )

    aggregate = [*generic["models"], nnunet["aggregate"]]
    for report in additional_reports:
        aggregate.extend(report["models"])
    per_image_iou = dict(generic["per_image_iou"])
    for seed, rows in nnunet["per_image_iou"].items():
        per_image_iou[f"B9_nnunet:seed{seed}"] = rows
    for report in additional_reports:
        duplicate_keys = per_image_iou.keys() & report["per_image_iou"].keys()
        if duplicate_keys:
            raise ValueError(f"duplicate per-image benchmark keys: {sorted(duplicate_keys)}")
        per_image_iou.update(report["per_image_iou"])
    source_commits = {
        "generic": generic["commit"],
        "nnunet": nnunet["commit"],
        "additional": [report["commit"] for report in additional_reports],
    }
    output: dict[str, Any] = {
        "protocol": f"{generic['protocol']}; official nnU-Net, mirroring disabled",
        "tta": False,
        "device": f"{generic['device']} + {nnunet['device']}",
        "commit": "mixed",
        "commits": source_commits,
        "specs": [
            *generic["specs"],
            "B9_nnunet=official_nnUNet",
            *(spec for report in additional_reports for spec in report["specs"]),
        ],
        "validation_image_ids": generic["validation_image_ids"],
        "per_image_iou": per_image_iou,
        "models": aggregate,
        "sources": {
            "generic": args.generic,
            "nnunet": args.nnunet,
            "additional": args.additional,
        },
    }
    out_path, csv_path = Path(args.out), Path(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    rows = [{key: value for key, value in item.items() if key != "seeds"} for item in aggregate]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "models": [item["model"] for item in aggregate],
                "n_images": len(output["validation_image_ids"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
