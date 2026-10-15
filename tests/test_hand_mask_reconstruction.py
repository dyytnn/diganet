"""Synthetic reconstruction tests independent of dataset masks or model outputs."""

import cv2
import numpy as np
import pytest
from skimage.morphology import skeletonize

from matblast.data.hand_mask_reconstruction import ReconstructionConfig, reconstruct_mask


def run(skeleton: np.ndarray, structure: str = "ICM", **kwargs):
    """Reconstruct a native-resolution synthetic contour."""
    return reconstruct_mask(
        skeleton,
        skeleton,
        structure,
        skeleton.shape,
        0,
        1,
        ReconstructionConfig(allow_border_closure=False, **kwargs),
    )


def test_short_aligned_gap_closed_and_exactly_recorded() -> None:
    sk = np.zeros((100, 100), np.uint8)
    cv2.rectangle(sk, (20, 20), (80, 80), 1, 1)
    sk[20, 45:53] = 0
    sk = sk.astype(bool)
    before = sk.copy()
    rasters, qc = run(sk)
    assert rasters["candidate"][50, 50]
    assert not rasters["candidate"][0, 0]
    assert len(qc["repairs"]) == 1
    assert qc["repairs"][0]["type"] == "tangent_gap"
    assert qc["repairs"][0]["length_native_px"] <= 12
    np.testing.assert_array_equal(rasters["repair_added"], rasters["skeleton"] & ~sk)
    np.testing.assert_array_equal(sk, before)
    again, again_qc = run(sk)
    np.testing.assert_array_equal(rasters["candidate"], again["candidate"])
    assert qc == again_qc


def test_long_interior_gap_is_not_invented() -> None:
    sk = np.zeros((100, 100), np.uint8)
    cv2.rectangle(sk, (20, 20), (80, 80), 1, 1)
    sk[20, 40:62] = 0
    rasters, qc = run(sk.astype(bool))
    assert not rasters["candidate"].any()
    assert qc["repairs"] == []


def test_ring_cavity_preserved_after_outer_gap_repair() -> None:
    sk = np.zeros((150, 150), np.uint8)
    cv2.rectangle(sk, (20, 20), (130, 130), 1, 1)
    cv2.rectangle(sk, (45, 45), (105, 105), 1, 1)
    sk[20, 70:78] = 0
    rasters, _ = run(sk.astype(bool), "ZP")
    assert rasters["candidate"][30, 30]
    assert not rasters["candidate"][75, 75]


def test_single_loop_does_not_become_ring() -> None:
    sk = np.zeros((100, 100), np.uint8)
    cv2.circle(sk, (50, 50), 30, 1, 1)
    rasters, qc = run(sk.astype(bool), "TE")
    assert not rasters["candidate"].any()
    assert "only_one_closed_region_ring_ambiguous" in qc["issues"]


def test_field_edge_closure_is_flagged_and_preserves_inner_hole() -> None:
    sk = np.zeros((120, 120), np.uint8)
    cv2.rectangle(sk, (10, 10), (109, 119), 1, 1)
    cv2.rectangle(sk, (35, 35), (85, 90), 1, 1)
    sk[119, 35:85] = 0
    rasters, qc = reconstruct_mask(
        sk.astype(bool), sk.astype(bool), "ZP", sk.shape, 0, 1, ReconstructionConfig()
    )
    # Without padding, a boundary at the array edge is exterior. Test with margin.
    padded = np.pad(sk.astype(bool), 10)
    rasters, qc = reconstruct_mask(padded, padded, "ZP", sk.shape, 10, 1, ReconstructionConfig())
    assert rasters["candidate"][30, 30]
    assert not rasters["candidate"][70, 70]
    assert "field_edge_closure_requires_review" in qc["issues"]
    assert not qc["approved"]


def test_invalid_config_and_input_rejected() -> None:
    with pytest.raises(TypeError):
        ReconstructionConfig(unknown=3)
    with pytest.raises(ValueError):
        ReconstructionConfig(max_gap=-1)
    with pytest.raises(ValueError):
        reconstruct_mask(
            np.zeros((3, 3)), np.zeros((3, 3)), "ZP", (3, 3), 0, 1, ReconstructionConfig()
        )


def test_close_parallel_strokes_not_joined_sideways() -> None:
    sk = np.zeros((100, 100), bool)
    sk[20:80, 40] = True
    sk[20:80, 46] = True
    rasters, qc = run(sk)
    assert not rasters["candidate"].any()
    assert not qc["repairs"]


def test_radial_fallback_recovers_broken_ring_without_reference_labels() -> None:
    """A pair of broken pen circles should produce one auditable ring region."""
    scale, padding = 3, 12
    native_shape = (96, 96)
    shape = tuple(v * scale + 2 * padding for v in native_shape)
    ink_u8 = np.zeros(shape, np.uint8)
    center = (shape[1] // 2, shape[0] // 2)
    for radius in (30 * scale, 39 * scale):
        cv2.ellipse(ink_u8, center, (radius, radius), 0, 12, 347, 1, 4)
    ink = ink_u8.astype(bool)
    rasters, qc = reconstruct_mask(
        ink,
        skeletonize(ink),
        "TE",
        native_shape,
        padding,
        scale,
        ReconstructionConfig(
            allow_border_closure=False,
            max_gap=2.0,
            radial_min_valid_fraction=0.35,
        ),
    )

    assert rasters["candidate"].any()
    assert qc["radial_ring_reconstruction"] is not None
    assert "radial_ring_reconstruction_requires_review" in qc["issues"]
    assert qc["approved"] is False
    expected_area = np.pi * (39**2 - 30**2)
    assert qc["candidate_area_native_px"] == pytest.approx(expected_area, rel=0.2)
