#!/usr/bin/env python3
"""Evaluate official nnU-Net checkpoints on the common MAT-BLAST split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from matblast.metrics import per_image_scores, summarize_scores  # noqa: E402
from matblast.provenance import current_commit  # noqa: E402


CLASS_NAMES = ("BG", "ZP", "TE", "BL", "ICM")


def _read_mask(mask_root: Path, image_name: str, image_size: tuple[int, int]) -> np.ndarray:
    """Compose one source mask at the original image geometry."""
    from PIL import Image

    width, height = image_size
    target = np.zeros((height, width), dtype=np.uint8)
    for class_id, class_name in enumerate(("zp", "te", "bl", "icm"), start=1):
        path = mask_root / class_name / image_name
        if path.exists():
            mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            if mask.shape != target.shape:
                raise ValueError(f"Mask geometry mismatch: {path}")
            target[mask > 0] = class_id
    return target


def _score_prediction(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[dict[str, float], np.ndarray]:
    """Score one label map and return named aggregate metrics and per-class IoU."""
    labels = torch.as_tensor(prediction, dtype=torch.long).unsqueeze(0)
    target_tensor = torch.as_tensor(target, dtype=torch.long).unsqueeze(0)
    logits = torch.nn.functional.one_hot(labels, num_classes=len(CLASS_NAMES))
    logits = logits.permute(0, 3, 1, 2).float()
    dice, iou = per_image_scores(logits, target_tensor)
    raw = summarize_scores(dice, iou, include_background=False)
    report = {
        "dice_bg": raw["dice_bg"],
        "iou_bg": raw["iou_bg"],
        "mdice_fg": raw["mdice_fg"],
        "miou_fg": raw["miou_fg"],
    }
    for class_id, class_name in enumerate(CLASS_NAMES[1:], start=1):
        report[f"dice_{class_name.lower()}"] = raw[f"dice_class_{class_id}"]
        report[f"iou_{class_name.lower()}"] = raw[f"iou_class_{class_id}"]
    return report, iou[0].numpy()


def _parse_folders(values: list[str]) -> list[tuple[int, Path]]:
    """Parse ``seed=trained_model_folder`` arguments."""
    parsed: list[tuple[int, Path]] = []
    for value in values:
        try:
            seed_text, folder = value.split("=", 1)
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError("model folders must use seed=folder syntax") from exc
        parsed.append((seed, Path(folder)))
    if not parsed:
        raise ValueError("at least one nnU-Net model folder is required")
    return parsed


def _evaluate_seed(
    seed: int,
    model_folder: Path,
    image_names: list[str],
    image_dir: Path,
    mask_root: Path,
    device: torch.device,
    fold: int = 0,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate one official checkpoint with nnU-Net's preprocessing and inference."""
    sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "nnUNet"))
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from PIL import Image

    if not (model_folder / f"fold_{fold}" / "checkpoint_best.pth").exists():
        raise FileNotFoundError(model_folder / f"fold_{fold}" / "checkpoint_best.pth")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder), (fold,), checkpoint_name="checkpoint_best.pth"
    )
    dice_values: list[torch.Tensor] = []
    iou_values: list[torch.Tensor] = []
    iou_rows: list[np.ndarray] = []
    for image_name in image_names:
        image_path = image_dir / image_name
        # Training converts the RGB source BMP to one grayscale modality.
        # Keep the singleton z-axis because NaturalImage2DIO represents 2D
        # cases as (C, 1, H, W), which is what nnU-Net's 2D preprocessor uses.
        grayscale = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        image = grayscale[None, None]
        properties = {"spacing": (999, 1, 1)}
        prediction = predictor.predict_single_npy_array(image, properties)
        prediction = np.asarray(prediction).squeeze()
        target = _read_mask(mask_root, image_name, (image.shape[-1], image.shape[-2]))
        report, iou = _score_prediction(prediction, target)
        iou_rows.append(iou)
        dice_values.append(torch.tensor([report[f"dice_{name.lower()}"] for name in CLASS_NAMES]))
        iou_values.append(torch.tensor([report[f"iou_{name.lower()}"] for name in CLASS_NAMES]))
    del predictor
    raw_scores = summarize_scores(torch.stack(dice_values), torch.stack(iou_values), False)
    scores: dict[str, float] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        key = class_name.lower()
        scores[f"dice_{key}"] = (
            raw_scores["dice_bg"] if class_id == 0 else raw_scores[f"dice_class_{class_id}"]
        )
        scores[f"iou_{key}"] = (
            raw_scores["iou_bg"] if class_id == 0 else raw_scores[f"iou_class_{class_id}"]
        )
    scores["mdice_fg"] = raw_scores["mdice_fg"]
    scores["miou_fg"] = raw_scores["miou_fg"]
    return scores, np.stack(iou_rows)


def _aggregate(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed reports while preserving NaN semantics."""
    metrics = [
        *(f"dice_{name.lower()}" for name in CLASS_NAMES),
        *(f"iou_{name.lower()}" for name in CLASS_NAMES),
        "mdice_fg",
        "miou_fg",
    ]
    result: dict[str, Any] = {"model": "B9_nnunet", "n_seeds": len(seed_reports)}
    for metric in metrics:
        values = np.asarray([report[metric] for report in seed_reports], dtype=float)
        result[f"{metric}_mean"] = float(np.nanmean(values))
        result[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    return result


def main() -> None:
    """Evaluate nnU-Net seeds and write JSON/CSV generated artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-folder", action="append", required=True, help="seed=nnU-Net model folder"
    )
    parser.add_argument("--image-dir", default="dataset/source/images")
    parser.add_argument("--mask-root", default="dataset/source/masks")
    parser.add_argument("--split", default="data/splits/source_grouped_v1.json")
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="nnU-Net fold index and, for k-fold splits, the split fold to evaluate",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    split = json.loads(Path(args.split).read_text())
    image_names = (
        list(split["folds"][str(args.fold)]["val"])
        if "folds" in split and "val" not in split
        else list(split["val"])
    )
    folders = _parse_folders(args.model_folder)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    seed_reports: list[dict[str, Any]] = []
    per_seed_iou: dict[str, list[list[float | None]]] = {}
    for seed, folder in folders:
        report, iou = _evaluate_seed(
            seed,
            folder,
            image_names,
            Path(args.image_dir),
            Path(args.mask_root),
            device,
            fold=args.fold,
        )
        seed_reports.append({"seed": seed, **report})
        per_seed_iou[str(seed)] = [
            [None if not np.isfinite(value) else float(value) for value in row] for row in iou
        ]

    output: dict[str, Any] = {
        "protocol": "fixed grouped validation split; foreground macro excludes BG; nnU-Net mirroring disabled",
        "commit": current_commit(PROJECT_ROOT),
        "split": args.split,
        "device": str(device),
        "validation_image_ids": image_names,
        "aggregate": _aggregate(seed_reports),
        "seeds": seed_reports,
        "per_image_iou": per_seed_iou,
    }
    output_path, csv_path = Path(args.out), Path(args.csv_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(output["aggregate"]))
        writer.writeheader()
        writer.writerow(output["aggregate"])
    print(json.dumps(output["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
