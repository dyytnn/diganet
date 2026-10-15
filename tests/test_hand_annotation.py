"""Synthetic checks for scanner geometry and contour recovery, without GT labels."""

import cv2
import numpy as np
import pytest

from matblast.data.hand_annotation import (
    ExtractionConfig,
    detect_red,
    extract_regions,
    register_panel,
)


def test_red_ink_excludes_grayscale() -> None:
    img = np.array([[[10, 10, 10], [230, 230, 230], [150, 45, 80]]], np.uint8)
    assert detect_red(img, ExtractionConfig()).tolist() == [[False, False, True]]


def test_faint_ink_needs_connection_to_strong_ink() -> None:
    img = np.full((8, 12, 3), 140, np.uint8)
    img[3, 2] = [150, 45, 80]
    img[3, 3:6] = [65, 57, 63]
    img[3, 9] = [65, 57, 63]
    ink = detect_red(img, ExtractionConfig())
    assert ink[3, 2:6].all()
    assert not ink[3, 9]


def test_ring_is_not_filled_disk_and_open_loop_is_flagged() -> None:
    img = np.full((240, 240, 3), 150, np.uint8)
    cv2.circle(img, (120, 120), 90, (210, 20, 60), 3)
    cv2.circle(img, (120, 120), 60, (210, 20, 60), 3)
    config = ExtractionConfig(scale=1, close_radius=0)
    rasters, qc = extract_regions(img, "ZP", config)
    assert rasters["candidate"][120, 195]
    assert not rasters["candidate"][120, 120]
    assert len(qc["closed_face_areas_native_px"]) == 2
    img[110:130, 200:220] = 150
    _, qc = extract_regions(img, "ZP", config)
    assert "only_one_closed_region_ring_ambiguous" in qc["issues"]
    assert "unresolved_contour_with_endpoints" in qc["issues"]


def test_single_icm_and_blank_unknown() -> None:
    img = np.full((240, 240, 3), 150, np.uint8)
    config = ExtractionConfig(scale=1, close_radius=0)
    _, qc = extract_regions(img, "ICM", config)
    assert "no_ink_recovered" in qc["issues"]
    cv2.circle(img, (100, 110), 50, (210, 20, 60), 3)
    rasters, qc = extract_regions(img, "ICM", config)
    assert rasters["candidate"][110, 100]
    assert not rasters["candidate"][10, 10]
    assert qc["status"] == "needs_visual_approval"


def test_registration_recovers_known_transform() -> None:
    rng = np.random.default_rng(3)
    original = cv2.GaussianBlur(rng.integers(0, 255, (260, 300), dtype=np.uint8), (3, 3), 0)
    expected = np.array([[2.1, 0.04, 25], [-0.03, 2.1, 35], [0.00002, -0.00003, 1.0]])
    warped = cv2.warpPerspective(original, expected, (720, 650))
    panel = np.repeat(warped[..., None], 3, axis=2)
    cv2.circle(panel, (320, 290), 80, (210, 20, 60), 4)
    recovered, qc = register_panel(original, panel, ExtractionConfig())
    points = np.float32([[[40, 40]], [[250, 200]], [[150, 130]]])
    err = np.linalg.norm(
        cv2.perspectiveTransform(points, recovered) - cv2.perspectiveTransform(points, expected),
        axis=2,
    )
    assert err.max() < 2
    assert qc["registration_pass"]
    again, _ = register_panel(original, panel, ExtractionConfig())
    np.testing.assert_allclose(recovered, again)


def test_unknown_config_key_rejected() -> None:
    with pytest.raises(TypeError):
        ExtractionConfig(unknown=2)
    with pytest.raises(ValueError):
        ExtractionConfig(close_radius=-1)


def test_only_one_ring_boundary_is_not_invented() -> None:
    img = np.full((240, 240, 3), 150, np.uint8)
    cv2.circle(img, (120, 120), 70, (210, 20, 60), 3)
    rasters, qc = extract_regions(img, "TE", ExtractionConfig(scale=1))
    assert not rasters["candidate"].any()
    assert "only_one_closed_region_ring_ambiguous" in qc["issues"]


def test_short_gap_is_recorded_and_long_gap_is_not_closed() -> None:
    img = np.full((240, 240, 3), 150, np.uint8)
    cv2.rectangle(img, (60, 60), (170, 170), (210, 20, 60), 1)
    img[60, 110:112] = 150
    rasters, qc = extract_regions(img, "ICM", ExtractionConfig(scale=1, close_radius=0))
    assert rasters["candidate"][100, 100]
    assert rasters["gap_bridges"].any()
    assert len(qc["automatic_gap_bridges"]) == 1
    img[60, 100:125] = 150
    rasters, qc = extract_regions(img, "ICM", ExtractionConfig(scale=1, close_radius=0))
    assert not rasters["candidate"].any()
    assert "no_closed_region" in qc["issues"]


def test_closed_loop_with_pen_tail_is_not_called_open() -> None:
    img = np.full((240, 240, 3), 150, np.uint8)
    cv2.rectangle(img, (60, 60), (170, 170), (210, 20, 60), 1)
    cv2.line(img, (110, 60), (110, 50), (210, 20, 60), 1)
    rasters, qc = extract_regions(img, "ICM", ExtractionConfig(scale=1, close_radius=0))
    assert rasters["candidate"][100, 100]
    assert qc["endpoints"] > 0
    assert qc["observations"] == ["pen_endpoints_present"]
    assert qc["status"] == "needs_visual_approval"


def test_affine_fallback_recovers_geometry_without_weakening_qc(monkeypatch) -> None:
    rng = np.random.default_rng(13)
    original = cv2.GaussianBlur(rng.integers(0, 255, (260, 300), dtype=np.uint8), (3, 3), 0)
    expected = np.array([[2.1, 0.04, 25], [-0.03, 2.1, 35], [0, 0, 1.0]])
    warped = cv2.warpPerspective(original, expected, (720, 650))
    panel = np.repeat(warped[..., None], 3, axis=2)
    monkeypatch.setattr(cv2, "findHomography", lambda *args: (None, None))
    with pytest.raises(ValueError, match="Registration fit failed"):
        register_panel(original, panel, ExtractionConfig())
    config = ExtractionConfig(fallback_to_affine=True)
    transform, qc = register_panel(original, panel, config)
    assert qc["method"] == "affine"
    assert qc["fallback_reasons"] == ["Registration fit failed"]
    assert qc["registration_pass"]
    corners = np.float32([[[0, 0]], [[299, 259]]])
    np.testing.assert_allclose(
        cv2.perspectiveTransform(corners, transform),
        cv2.perspectiveTransform(corners, expected),
        atol=2,
    )
    # Insufficient held-out accuracy must remain failed even after recovery.
    strict = ExtractionConfig(fallback_to_affine=True, max_holdout_p95=0.000001)
    _, strict_qc = register_panel(original, panel, strict)
    assert not strict_qc["registration_pass"]


def test_rescaled_fallback_preserves_failed_qc(monkeypatch) -> None:
    import matblast.data.hand_annotation as module

    attempts = []

    def fake_fit(original, panel, config, *, affine):
        attempts.append((affine, config.registration_long_edge, config.feature_scan_sigma))
        if len(attempts) < 3:
            raise ValueError("Insufficient supported matches")
        return np.eye(3), {"inliers": 14, "inlier_coverage": 0.2, "registration_pass": False}

    monkeypatch.setattr(module, "_register_panel_once", fake_fit)
    _, qc = register_panel(
        np.zeros((2, 2)), np.zeros((2, 2, 3)), ExtractionConfig(fallback_to_affine=True)
    )
    assert attempts == [(False, 650, 1.0), (True, 650, 1.0), (False, 500, 0.6)]
    assert len(qc["fallback_reasons"]) == 2
    assert not qc["registration_pass"]
