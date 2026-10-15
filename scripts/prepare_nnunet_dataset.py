#!/usr/bin/env python3
"""Create an nnU-Net-v2 2D dataset from the frozen MAT-BLAST split.

The source split is read only. This converter derives nnU-Net case identifiers
and writes its required ``splits_final.json`` outside ``data/splits``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = ("zp", "te", "bl", "icm")


def _make_label(mask_root: Path, image_name: str, size: tuple[int, int]) -> np.ndarray:
    """Compose the fixed five-class target in the repository priority order.

    Args:
        mask_root: Root holding one binary mask directory per foreground class.
        image_name: Basename of the source image.
        size: ``(width, height)`` of the source image.

    Returns:
        ``(H, W)`` uint8 label map with BG=0, ZP=1, TE=2, BL=3, ICM=4.
    """
    target = np.zeros((size[1], size[0]), dtype=np.uint8)
    for class_id, class_name in enumerate(CLASS_NAMES, start=1):
        path = mask_root / class_name / image_name
        if path.exists():
            mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            if mask.shape != target.shape:
                raise ValueError(f"Mask geometry mismatch: {path}")
            target[mask > 0] = class_id
    return target


def main() -> None:
    """Convert every source image and derive the fold-0 nnU-Net split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="dataset/source/images")
    parser.add_argument("--mask-root", default="dataset/source/masks")
    parser.add_argument("--split", default="data/splits/source_grouped_v1.json")
    parser.add_argument("--raw-root", default="nnunet/raw")
    parser.add_argument("--dataset-name", default="Dataset601_MATBLAST")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    mask_root = Path(args.mask_root)
    split = json.loads(Path(args.split).read_text())
    image_ids = sorted(set(split["train"]) | set(split["val"]))
    dataset_dir = Path(args.raw_root) / args.dataset_name
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, str] = {}
    for index, image_name in enumerate(image_ids):
        source_path = image_dir / image_name
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        case_id = f"blast_{index:04d}"
        mapping[image_name] = case_id
        # nnU-Net receives one brightfield intensity modality. This is separate
        # from the legacy three-channel representation used by MAT-BLAST.
        image = Image.open(source_path).convert("L")
        image.save(images_dir / f"{case_id}_0000.png")
        Image.fromarray(_make_label(mask_root, image_name, image.size)).save(labels_dir / f"{case_id}.png")

    dataset_json = {
        "channel_names": {"0": "Brightfield"},
        "labels": {"background": 0, "ZP": 1, "TE": 2, "BL": 3, "ICM": 4},
        "numTraining": len(mapping),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n")
    (dataset_dir / "case_map.json").write_text(json.dumps(mapping, indent=2) + "\n")
    derived_split = [{
        "train": [mapping[name] for name in split["train"]],
        "val": [mapping[name] for name in split["val"]],
    }]
    output = Path(args.raw_root) / f"{args.dataset_name}_fold0_from_source_grouped_v1.json"
    output.write_text(json.dumps(derived_split, indent=2) + "\n")
    print(f"Prepared {len(mapping)} cases at {dataset_dir}")


if __name__ == "__main__":
    main()
