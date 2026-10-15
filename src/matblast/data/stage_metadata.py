"""Validation and loading of developmental-stage annotations."""

from __future__ import annotations

import csv
from pathlib import Path


STAGES = ("early", "expanded", "collapsed", "hatching", "hatched")
STAGE_TO_ID = {name: index for index, name in enumerate(STAGES)}


def load_stage_metadata(path: str | Path) -> dict[str, int]:
    """Load a complete image-to-stage CSV mapping.

    Args:
        path: CSV with ``image_id`` or ``image_path`` and ``developmental_stage``.

    Returns:
        Mapping from image filename to integer stage ID.

    Raises:
        ValueError: If the schema or stage values are invalid.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Stage metadata is required for stage-conditioned experiments: {csv_path}"
        )
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        key = (
            "image_id" if "image_id" in fields else "image_path" if "image_path" in fields else None
        )
        if key is None or "developmental_stage" not in fields:
            raise ValueError(
                "Stage metadata must contain image_id (or image_path) and developmental_stage"
            )
        mapping: dict[str, int] = {}
        for row_number, row in enumerate(reader, start=2):
            image_value = Path((row.get(key) or "").strip()).name
            stage = (row.get("developmental_stage") or "").strip().lower()
            if not image_value:
                raise ValueError(f"Missing image identifier at CSV row {row_number}")
            if stage not in STAGE_TO_ID:
                raise ValueError(
                    f"Invalid developmental_stage {stage!r} at CSV row {row_number}; "
                    f"expected one of {list(STAGES)}"
                )
            if image_value in mapping:
                raise ValueError(f"Duplicate image identifier {image_value!r}")
            mapping[image_value] = STAGE_TO_ID[stage]
    if not mapping:
        raise ValueError(f"Stage metadata contains no rows: {csv_path}")
    return mapping
