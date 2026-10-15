"""Verify draft mask rendering without altering the recovered annotation geometry."""

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

_SPEC = importlib.util.spec_from_file_location(
    "hand_overlays",
    Path(__file__).resolve().parents[1] / "scripts/build_hand_annotation_overlays.py",
)
overlay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(overlay)


def test_filled_overlay_preserves_holes_and_background() -> None:
    original = np.full((30, 30), 100, np.uint8)
    ring = np.zeros_like(original, dtype=bool)
    ring[5:25, 5:25] = True
    ring[10:20, 10:20] = False
    layers = {"ZP": ring, "TE": None, "ICM": np.zeros_like(ring)}
    records = {name: {} for name in layers}
    result = np.asarray(
        overlay.render_overlay(original, layers, records, mode="filled", alpha=0.5)
    )[-30:]
    np.testing.assert_array_equal(result[15, 15], [100, 100, 100])
    np.testing.assert_array_equal(result[0, 0], [100, 100, 100])
    np.testing.assert_array_equal(result[6, 6], [177, 155, 50])
    assert ring.sum() == 300


def test_overlap_averages_class_colors_without_priority() -> None:
    original = np.full((30, 30), 100, np.uint8)
    mask = np.ones_like(original, dtype=bool)
    layers = {"ZP": mask, "TE": mask, "ICM": None}
    records = {name: {} for name in layers}
    result = np.asarray(overlay.render_overlay(original, layers, records, mode="filled", alpha=1))[
        -30:
    ]
    expected = ((overlay.COLORS["ZP"] + overlay.COLORS["TE"]) / 2).astype(np.uint8)
    np.testing.assert_array_equal(result[15, 15], expected)


def test_native_candidate_removes_padding_without_completing_region(tmp_path) -> None:
    raster = np.zeros((16, 16), np.uint8)
    raster[4:8, 4:8] = 255
    Image.fromarray(raster).save(tmp_path / "TE_candidate_x2.png")
    record = {"structure": "TE", "supersampled_padding_px": 2, "native_shape_hw": [6, 6]}
    result = overlay.native_layer(tmp_path, record, "candidate")
    expected = np.zeros((6, 6), bool)
    expected[1:3, 1:3] = True
    np.testing.assert_array_equal(result, expected)
    record["structure"] = "ICM"
    assert overlay.native_layer(tmp_path, record, "candidate") is None


def test_zero_opacity_preserves_original_and_masks() -> None:
    original = np.full((30, 30), 123, np.uint8)
    layers = {name: np.ones_like(original, dtype=bool) for name in overlay.STRUCTURES}
    records = {name: {} for name in layers}
    result = np.asarray(overlay.render_overlay(original, layers, records, mode="filled", alpha=0))[
        -30:
    ]
    np.testing.assert_array_equal(result, np.repeat(original[..., None], 3, axis=2))
    assert all(mask.all() for mask in layers.values())
