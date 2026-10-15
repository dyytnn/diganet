"""Recover pen contours from scanned annotation sheets without reference labels."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class ExtractionConfig:
    """Parameters expressed in native image pixels unless stated otherwise."""

    seed: int = 7
    scale: int = 3
    padding: int = 16
    red_green: int = 18
    red_blue: int = 8
    weak_red_green: int = 6
    weak_red_blue: int = 0
    min_ink_component: float = 1.0
    close_radius: float = 1.0
    max_gap: float = 4.0
    correlation_sigma: float = 1.5
    min_face_area: float = 30.0
    min_ring_area: float = 120.0
    sift_features: int = 5000
    registration_long_edge: int = 650
    feature_scan_sigma: float = 1.0
    feature_original_sigma: float = 0.6
    match_ratio: float = 0.75
    ransac_threshold: float = 2.0
    min_inliers: int = 12
    min_coverage: float = 0.15
    max_holdout_p95: float = 2.5
    min_correlation: float = 0.55
    fallback_to_affine: bool = False
    fallback_long_edge: int = 500
    fallback_scan_sigma: float = 0.6

    def __post_init__(self) -> None:
        """Reject invalid extraction or registration settings."""
        if not 1 <= self.scale <= 8 or self.sift_features < 100:
            raise ValueError("Invalid raster scale or feature count")
        if not 0 < self.match_ratio < 1 or not 0 < self.min_coverage <= 1:
            raise ValueError("Invalid match ratio or coverage")
        if self.min_inliers < 4 or self.close_radius < 0:
            raise ValueError("Invalid minimum inliers or closing radius")
        if self.padding < 0 or self.max_gap < 0 or self.correlation_sigma < 0:
            raise ValueError("Padding, gap length and smoothing must be nonnegative")
        if (
            self.registration_long_edge < 100
            or self.fallback_long_edge < 100
            or min(self.feature_scan_sigma, self.feature_original_sigma, self.fallback_scan_sigma)
            < 0
        ):
            raise ValueError("Invalid feature smoothing or registration size")
        if (
            not 0 < self.weak_red_green <= self.red_green
            or not 0 <= self.weak_red_blue <= self.red_blue
        ):
            raise ValueError("Weak pen thresholds must not exceed strong thresholds")
        if (
            min(
                self.red_green,
                self.red_blue,
                self.min_ink_component,
                self.min_face_area,
                self.min_ring_area,
                self.ransac_threshold,
                self.max_holdout_p95,
            )
            <= 0
        ):
            raise ValueError("Thresholds must be positive")
        if not -1 <= self.min_correlation <= 1:
            raise ValueError("Correlation must be between -1 and 1")


def detect_red(rgb: np.ndarray, config: ExtractionConfig) -> np.ndarray:
    """Return Boolean pen pixels for RGB scan of shape (H, W, 3)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB array (H, W, 3)")
    r, g, b = np.moveaxis(rgb.astype(np.int16), -1, 0)
    strong = (r - g >= config.red_green) & (r - b >= config.red_blue)
    weak = (r - g >= config.weak_red_green) & (r - b >= config.weak_red_blue)
    # Include faint ink only when connected to clear red ink; isolated paper
    # colour noise is not evidence of a contour. No geometric gap is added here.
    return ndi.binary_propagation(strong, structure=np.ones((3, 3)), mask=weak)


def register_panel(
    original: np.ndarray, panel: np.ndarray, config: ExtractionConfig
) -> tuple[np.ndarray, dict]:
    """Map native gray (H, W) to RGB scan panel (P, Q, 3) using image features.

    Returns a (3, 3) native-to-panel homography and geometric/photometric QC.
    Neither original labels nor model predictions are inputs.
    """
    try:
        return _register_panel_once(original, panel, config, affine=False)
    except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
        if not config.fallback_to_affine:
            raise
        failures = [str(error)]
        try:
            transform, metrics = _register_panel_once(original, panel, config, affine=True)
        except (ValueError, cv2.error, np.linalg.LinAlgError) as affine_error:
            failures.append(str(affine_error))
            fallback = replace(
                config,
                registration_long_edge=config.fallback_long_edge,
                feature_scan_sigma=config.fallback_scan_sigma,
            )
            transform, metrics = _register_panel_once(original, panel, fallback, affine=False)
            if (
                metrics["inliers"] < config.min_inliers
                or metrics["inlier_coverage"] < config.min_coverage
            ):
                raise ValueError("Rescaled fallback lacks sufficient spatial texture support")
        metrics["fallback_reasons"] = failures
        return transform, metrics


def _register_panel_once(
    original: np.ndarray, panel: np.ndarray, config: ExtractionConfig, *, affine: bool
) -> tuple[np.ndarray, dict]:
    """Fit one texture-only transform; affine fallback retains explicit failed QC."""
    cv2.setRNGSeed(config.seed)
    gray = cv2.cvtColor(panel, cv2.COLOR_RGB2GRAY)
    ratio = min(1.0, config.registration_long_edge / max(gray.shape))
    small = cv2.resize(gray, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    pen = cv2.resize(
        detect_red(panel, config).astype(np.uint8),
        (small.shape[1], small.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    mask = (1 - cv2.dilate(pen, np.ones((7, 7), np.uint8))) * 255
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    sift = cv2.SIFT_create(nfeatures=config.sift_features, contrastThreshold=0.015)
    reference = (
        cv2.GaussianBlur(original, (0, 0), config.feature_original_sigma)
        if config.feature_original_sigma
        else original
    )
    descreened = (
        cv2.GaussianBlur(small, (0, 0), config.feature_scan_sigma)
        if config.feature_scan_sigma
        else small
    )
    k1, d1 = sift.detectAndCompute(clahe.apply(reference), None)
    k2, d2 = sift.detectAndCompute(clahe.apply(descreened), mask)
    if d1 is None or d2 is None or len(d2) < 2:
        raise ValueError("Insufficient texture to register panel")
    matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for a, b in matches if a.distance < config.match_ratio * b.distance]
    # Reject many-to-one matches; repeated structures must not inflate support.
    unique = {}
    for m in sorted(good, key=lambda m: m.distance):
        unique.setdefault(m.trainIdx, m)
    good = list(unique.values())
    if len(good) < config.min_inliers + 5:
        raise ValueError(f"Only {len(good)} distinct texture matches")
    a = np.float32([k1[m.queryIdx].pt for m in good])
    b = np.float32([k2[m.trainIdx].pt for m in good])
    ids = np.random.default_rng(config.seed).permutation(len(a))
    held = ids[: max(5, len(a) // 5)]
    train = ids[max(5, len(a) // 5) :]

    def fit(source: np.ndarray, target: np.ndarray) -> tuple:
        if not affine:
            return cv2.findHomography(source, target, cv2.RANSAC, config.ransac_threshold)
        matrix, support = cv2.estimateAffine2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=config.ransac_threshold
        )
        return (np.vstack([matrix, [0, 0, 1]]) if matrix is not None else None), support

    initial, _ = fit(b[train], a[train])
    if initial is None:
        raise ValueError("Registration fit failed")
    errors = np.linalg.norm(cv2.perspectiveTransform(b[:, None], initial)[:, 0] - a, axis=1)
    # Count unmatched/outlier held-out descriptors separately, never hide them.
    held_valid = errors[held][errors[held] < config.ransac_threshold * 3]
    if len(held_valid) < 4 and not affine:
        raise ValueError("Fewer than four geometrically supported held-out matches")
    inverse_small, inliers = fit(b, a)
    if inverse_small is None or inliers is None:
        raise ValueError("Final registration fit failed")
    keep = inliers.ravel().astype(bool)
    coverage = cv2.contourArea(cv2.convexHull(a[keep])) / original.size
    if affine and (keep.sum() < config.min_inliers or coverage < config.min_coverage):
        raise ValueError("Affine fallback lacks sufficient spatial texture support")
    h_native_panel = np.diag([1 / ratio, 1 / ratio, 1]) @ np.linalg.inv(inverse_small)
    corners = np.float32(
        [
            [0, 0],
            [original.shape[1] - 1, 0],
            [original.shape[1] - 1, original.shape[0] - 1],
            [0, original.shape[0] - 1],
        ]
    )
    projected = cv2.perspectiveTransform(corners[:, None], h_native_panel)[:, 0]
    if affine and (
        not np.isfinite(projected).all()
        or not cv2.isContourConvex(projected)
        or (projected < 0).any()
        or (projected[:, 0] >= panel.shape[1]).any()
        or (projected[:, 1] >= panel.shape[0]).any()
        or np.linalg.det(h_native_panel[:2, :2]) <= 0
    ):
        raise ValueError("Affine fallback maps outside the printed panel or reflects it")
    aligned = cv2.warpPerspective(
        panel, np.linalg.inv(h_native_panel), (original.shape[1], original.shape[0])
    )
    ag = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
    valid = ~cv2.dilate(
        detect_red(aligned, config).astype(np.uint8), np.ones((5, 5), np.uint8)
    ).astype(bool)
    valid[:5] = valid[-5:] = False
    valid[:, :5] = valid[:, -5:] = False
    raw_corr = float(np.corrcoef(original[valid], ag[valid])[0, 1])
    if config.correlation_sigma:
        filtered_original = cv2.GaussianBlur(original, (0, 0), config.correlation_sigma)
        filtered_aligned = cv2.GaussianBlur(ag, (0, 0), config.correlation_sigma)
    else:
        filtered_original, filtered_aligned = original, ag
    corr = float(np.corrcoef(filtered_original[valid], filtered_aligned[valid])[0, 1])
    metrics = {
        "method": "affine" if affine else "homography",
        "feature_long_edge": config.registration_long_edge,
        "feature_scan_sigma": config.feature_scan_sigma,
        "matches": len(good),
        "inliers": int(keep.sum()),
        "inlier_coverage": float(coverage),
        "holdout_total": len(held),
        "holdout_supported": len(held_valid),
        "holdout_supported_p95_native_px": (
            float(np.percentile(held_valid, 95)) if len(held_valid) else None
        ),
        "holdout_all_median_native_px": float(np.median(errors[held])),
        "image_correlation": corr,
        "raw_image_correlation": raw_corr,
    }
    metrics["registration_pass"] = bool(
        keep.sum() >= config.min_inliers
        and coverage >= config.min_coverage
        and len(held_valid) >= 4
        and len(held_valid) >= len(held) / 2
        and metrics["holdout_supported_p95_native_px"] <= config.max_holdout_p95
        and np.isfinite(corr)
        and corr >= config.min_correlation
    )
    return h_native_panel, metrics


def extract_regions(
    aligned: np.ndarray, structure: str, config: ExtractionConfig
) -> tuple[dict[str, np.ndarray], dict]:
    """Extract draft regions from an aligned RGB (H*scale, W*scale, 3) panel.

    ZP/TE: use bounded faces other than the deepest (the cavity). ICM: fill
    bounded faces. Open or ambiguous contours are flagged, never convex-hulled.
    Returns Boolean rasters and QC. All candidate masks need visual approval.
    """
    if structure not in {"ZP", "TE", "ICM"}:
        raise ValueError("Unknown structure")
    ink = detect_red(aligned, config)
    lab, _ = ndi.label(ink, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    keep = sizes >= config.min_ink_component * config.scale**2
    keep[0] = False
    cleaned = keep[lab]
    radius = round(config.close_radius * config.scale)
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
        closed = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    else:
        closed = cleaned.copy()
    skeleton = skeletonize(closed)
    neighbours = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8))
    points = np.argwhere(skeleton & (neighbours == 2))
    bridges = np.zeros(skeleton.shape, np.uint8)
    pairs = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = float(np.linalg.norm(points[i] - points[j]))
            if distance <= config.max_gap * config.scale:
                pairs.append((distance, i, j))
    used = set()
    bridge_records = []
    for distance, i, j in sorted(pairs):
        if i in used or j in used:
            continue
        p, q = points[i], points[j]
        cv2.line(bridges, tuple(p[::-1]), tuple(q[::-1]), 1, 1)
        bridge_records.append(
            {
                "from_yx": p.tolist(),
                "to_yx": q.tolist(),
                "length_native_px": distance / config.scale,
            }
        )
        used.update((i, j))
    skeleton = skeletonize(skeleton | bridges.astype(bool))
    neighbours = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8))
    endpoints = skeleton & (neighbours == 2)
    # Use 4-connected free space against 8-connected pen strokes.
    regions, _ = ndi.label(~skeleton)
    border = np.unique(np.concatenate([regions[0], regions[-1], regions[:, 0], regions[:, -1]]))
    area = np.bincount(regions.ravel())
    faces = [
        i
        for i in np.argsort(area)[::-1]
        if i and i not in border and area[i] >= config.min_face_area * config.scale**2
    ]
    candidate = np.zeros_like(ink)
    issues = []
    if not faces:
        issues.append("no_closed_region")
    elif structure == "ICM":
        candidate = np.isin(regions, faces)
        if len(faces) != 1:
            issues.append("multiple_icm_regions")
    elif len(faces) < 2:
        issues.append("only_one_closed_region_ring_ambiguous")
    else:
        # Area alone can mistake a thick ring for the central hole. The cavity
        # has the largest inscribed disk; still a draft, never a topology prior.
        depths = {i: float(ndi.distance_transform_edt(regions == i).max()) for i in faces}
        cavity = max(faces, key=lambda i: depths[i])
        candidate = np.isin(regions, [i for i in faces if i != cavity])
        if candidate.sum() < config.min_ring_area * config.scale**2:
            issues.append("ring_area_too_small")
    # Endpoints can be pen tails on an otherwise closed loop. They are an
    # observation, not evidence of absent annotation or an unclosed boundary.
    observations = ["pen_endpoints_present"] if endpoints.any() else []
    if endpoints.any() and not candidate.any():
        issues.append("unresolved_contour_with_endpoints")
    margin = 2 * config.scale
    edge = ink.copy()
    edge[margin:-margin, margin:-margin] = False
    if edge.any():
        issues.append("pen_at_image_edge")
    if not cleaned.any():
        issues.append("no_ink_recovered")
    if (
        ink.any()
        and (ink & ~ndi.binary_dilation(candidate, iterations=5 * config.scale)).sum() / ink.sum()
        > 0.15
    ):
        issues.append("substantial_ink_outside_candidate")
    return {
        "ink": ink,
        "cleaned_ink": cleaned,
        "skeleton": skeleton,
        "closing_added": closed & ~cleaned,
        "gap_bridges": bridges.astype(bool),
        "candidate": candidate,
    }, {
        "closed_face_areas_native_px": [float(area[i] / config.scale**2) for i in faces],
        "endpoints": int(endpoints.sum()),
        "automatic_gap_bridges": bridge_records,
        "pen_pixels_native": float(ink.sum() / config.scale**2),
        "candidate_area_native_px": float(candidate.sum() / config.scale**2),
        "issues": issues,
        "observations": observations,
        "status": "needs_correction" if issues else "needs_visual_approval",
    }
