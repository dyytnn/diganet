"""Conservative, auditable filled-region repair from frozen extracted pen rasters."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class ReconstructionConfig:
    """Repair distances in native pixels; no image labels or predictions used."""

    max_gap: float = 12.0
    tangent_length: float = 6.0
    min_alignment: float = 0.2
    max_rounds: int = 3
    border_tolerance: float = 8.0
    allow_border_closure: bool = True
    min_face_area: float = 30.0
    support_distance: float = 5.0
    max_unsupported_fraction: float = 0.15
    radial_ring_fallback: bool = True
    radial_angle_bins: int = 720
    radial_min_separation: float = 6.0
    radial_min_valid_fraction: float = 0.45
    radial_smoothing_bins: int = 9

    def __post_init__(self) -> None:
        """Reject unsafe/invalid geometry settings and unknown dataclass keys."""
        if min(self.max_gap, self.tangent_length, self.min_face_area, self.support_distance) <= 0:
            raise ValueError("Distances and area must be positive")
        if self.border_tolerance < 0 or self.max_rounds < 1:
            raise ValueError("Invalid border tolerance or number of rounds")
        if not 0 <= self.min_alignment <= 1 or not 0 <= self.max_unsupported_fraction <= 1:
            raise ValueError("Invalid angular/support threshold")
        if self.radial_angle_bins < 90 or self.radial_smoothing_bins < 1:
            raise ValueError("Invalid radial reconstruction resolution")
        if self.radial_min_separation <= 0 or not 0 < self.radial_min_valid_fraction <= 1:
            raise ValueError("Invalid radial reconstruction threshold")


def _cyclic_interpolate(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Interpolate a cyclic one-dimensional contour signal."""
    indices = np.flatnonzero(valid)
    if indices.size < 2:
        raise ValueError("At least two contour observations are required")
    n = len(values)
    extended_x = np.concatenate((indices - n, indices, indices + n))
    extended_y = np.tile(values[indices], 3)
    return np.interp(np.arange(n), extended_x, extended_y)


def _radial_ring(
    ink: np.ndarray,
    native_shape: tuple[int, int],
    padding: int,
    scale: int,
    config: ReconstructionConfig,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Trace two mostly star-shaped pen contours and fill the region between them.

    This fallback is only for ZP/TE panels where scan gaps break an otherwise
    visible pair of rings. It uses pen pixels, never images, labels, or model
    predictions. ``padding`` is part of the recorded geometry and retained in
    the signature to make that provenance explicit.
    """
    del padding
    ys, xs = np.nonzero(ink)
    if len(xs) < config.radial_angle_bins:
        return None
    center_y = 0.5 * (float(ys.min()) + float(ys.max()))
    center_x = 0.5 * (float(xs.min()) + float(xs.max()))
    radius = np.hypot(ys - center_y, xs - center_x)
    theta = np.mod(np.arctan2(ys - center_y, xs - center_x), 2 * np.pi)
    bins = config.radial_angle_bins
    bin_ids = np.floor(theta * bins / (2 * np.pi)).astype(int) % bins
    inner = np.full(bins, np.nan)
    outer = np.full(bins, np.nan)
    neighbourhood = max(1, bins // 360)
    for index in range(bins):
        delta = np.minimum((bin_ids - index) % bins, (index - bin_ids) % bins)
        observed = radius[delta <= neighbourhood]
        if observed.size < 4:
            continue
        lo, hi = np.percentile(observed, (15, 85))
        if hi - lo >= config.radial_min_separation * scale:
            inner[index], outer[index] = lo, hi
    valid = np.isfinite(inner) & np.isfinite(outer)
    valid_fraction = float(valid.mean())
    if valid_fraction < config.radial_min_valid_fraction:
        return None
    inner = _cyclic_interpolate(inner, valid)
    outer = _cyclic_interpolate(outer, valid)
    smooth = config.radial_smoothing_bins
    if smooth > 1:
        if smooth % 2 == 0:
            smooth += 1
        inner = ndi.median_filter(inner, size=smooth, mode="wrap")
        outer = ndi.median_filter(outer, size=smooth, mode="wrap")
    angles = 2 * np.pi * np.arange(bins) / bins

    def polygon(radial: np.ndarray) -> np.ndarray:
        x = np.rint(center_x + radial * np.cos(angles)).astype(np.int32)
        y = np.rint(center_y + radial * np.sin(angles)).astype(np.int32)
        return np.column_stack((x, y))

    candidate_u8 = np.zeros(ink.shape, np.uint8)
    cv2.fillPoly(candidate_u8, [polygon(outer)], 1)
    cv2.fillPoly(candidate_u8, [polygon(inner)], 0)
    candidate = candidate_u8.astype(bool)
    boundary = candidate ^ ndi.binary_erosion(candidate)
    distance = ndi.distance_transform_edt(~boundary)
    unsupported = float(
        (ink & (distance > config.support_distance * scale)).sum() / max(1, ink.sum())
    )
    content_area = native_shape[0] * native_shape[1] * scale**2
    area_fraction = float(candidate.sum() / content_area)
    if not 0.01 <= area_fraction <= 0.65:
        return None
    return (
        candidate,
        boundary,
        {
            "center_yx": [center_y, center_x],
            "valid_angle_fraction": valid_fraction,
            "unsupported_ink_fraction": unsupported,
            "candidate_area_native_px": float(candidate.sum() / scale**2),
        },
    )


def _endpoints(skeleton: np.ndarray) -> np.ndarray:
    return np.argwhere(
        skeleton & (ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8)) == 2)
    )


def _direction(skeleton: np.ndarray, point: np.ndarray, length: int) -> np.ndarray | None:
    """Estimate outward tangent by tracing only connected skeleton neighbours."""
    start = tuple(point)
    queue = deque([(start, 0)])
    seen = {start}
    far = []
    while queue:
        (y, x), depth = queue.popleft()
        if depth >= length // 2:
            far.append((y, x))
        if depth >= length:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (y + dy, x + dx)
                if (
                    0 <= q[0] < skeleton.shape[0]
                    and 0 <= q[1] < skeleton.shape[1]
                    and q not in seen
                    and skeleton[q]
                ):
                    seen.add(q)
                    queue.append((q, depth + 1))
    if not far:
        return None
    vector = point - np.mean(far, axis=0)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else None


def _line(shape: tuple[int, int], points: list[np.ndarray]) -> np.ndarray:
    canvas = np.zeros(shape, np.uint8)
    cv2.polylines(canvas, [np.asarray(points, np.int32)[:, ::-1]], False, 1, 1)
    return canvas.astype(bool)


def reconstruct_mask(
    ink: np.ndarray,
    skeleton: np.ndarray,
    structure: str,
    native_shape: tuple[int, int],
    padding: int,
    scale: int,
    config: ReconstructionConfig,
) -> tuple[dict[str, np.ndarray], dict]:
    """Repair binary (s*(H+2p), s*(W+2p)) ink/skeleton into draft regions.

    Straight bridges require two compatible endpoint tangents and bounded gaps.
    Same-component contour ends near the same field edge may be closed along
    that edge, explicitly flagged as a clipping assumption. Never fills a disk
    for a ring lacking a second contour. Returns rasters and repair provenance.
    """
    if structure not in {"ZP", "TE", "ICM"} or scale < 1 or padding < 0:
        raise ValueError("Invalid structure or geometry")
    expected = tuple(v * scale + 2 * padding for v in native_shape)
    if (
        ink.shape != expected
        or skeleton.shape != expected
        or ink.dtype != bool
        or skeleton.dtype != bool
    ):
        raise ValueError("Expected Boolean rasters with the recorded padded shape")
    repaired = skeleton.copy()
    added = np.zeros_like(ink)
    records = []
    for _ in range(config.max_rounds):
        endpoints = _endpoints(repaired)
        directions = [
            _direction(repaired, p, max(4, round(config.tangent_length * scale))) for p in endpoints
        ]
        proposals = []
        for i, p in enumerate(endpoints):
            for j in range(i + 1, len(endpoints)):
                q = endpoints[j]
                distance = float(np.linalg.norm(q - p))
                if (
                    distance == 0
                    or distance > config.max_gap * scale
                    or directions[i] is None
                    or directions[j] is None
                ):
                    continue
                forward = (q - p) / distance
                alignment = min(float(directions[i] @ forward), float(directions[j] @ -forward))
                if alignment < config.min_alignment:
                    continue
                bridge = _line(ink.shape, [p, q])
                ys, xs = np.nonzero(bridge)
                away = ((ys - p[0]) ** 2 + (xs - p[1]) ** 2 > (2 * scale) ** 2) & (
                    (ys - q[0]) ** 2 + (xs - q[1]) ** 2 > (2 * scale) ** 2
                )
                if repaired[ys[away], xs[away]].any():
                    continue
                proposals.append((distance / max(alignment, 0.1), i, j, distance, alignment))
        used = set()
        changed = False
        for _, i, j, distance, alignment in sorted(proposals):
            if i in used or j in used:
                continue
            p, q = endpoints[i], endpoints[j]
            bridge = _line(ink.shape, [p, q])
            added |= bridge & ~repaired
            repaired |= bridge
            used.update((i, j))
            changed = True
            records.append(
                {
                    "type": "tangent_gap",
                    "from_yx": p.tolist(),
                    "to_yx": q.tolist(),
                    "length_native_px": distance / scale,
                    "alignment": alignment,
                }
            )
        if not changed:
            break
        repaired = skeletonize(repaired)

    if config.allow_border_closure:
        endpoints = _endpoints(repaired)
        components, _ = ndi.label(repaired, np.ones((3, 3)))
        used = set()
        # Closure follows the observed field edge, not an invented curved arc.
        for axis, edge in (
            (0, padding),
            (0, padding + native_shape[0] * scale - 1),
            (1, padding),
            (1, padding + native_shape[1] * scale - 1),
        ):
            near = [
                i
                for i, p in enumerate(endpoints)
                if abs(p[axis] - edge) <= config.border_tolerance * scale
            ]
            border_pairs = []
            directions = {
                i: _direction(repaired, endpoints[i], max(4, round(config.tangent_length * scale)))
                for i in near
            }
            for i in near:
                for j in near:
                    if j <= i or directions[i] is None or directions[j] is None:
                        continue
                    delta = endpoints[j] - endpoints[i]
                    delta[axis] = 0
                    length = float(np.linalg.norm(delta))
                    if not length:
                        continue
                    alignment = min(
                        float(directions[i] @ (delta / length)),
                        float(directions[j] @ (-delta / length)),
                    )
                    if alignment >= config.min_alignment:
                        border_pairs.append((length / alignment, i, j))
            for _, i, j in sorted(border_pairs):
                if i in used:
                    continue
                if j in used:
                    continue
                p, q = endpoints[i], endpoints[j]
                a, b = p.copy(), q.copy()
                a[axis] = b[axis] = edge
                bridge = _line(ink.shape, [p, a, b, q])
                # Do not connect across a different contour on the frame edge.
                hits = components[bridge & repaired]
                if np.any(
                    (hits != 0) & (hits != components[tuple(p)]) & (hits != components[tuple(q)])
                ):
                    continue
                repaired |= bridge
                added |= bridge & ~skeleton
                used.update((i, j))
                records.append(
                    {
                        "type": "field_edge_closure",
                        "from_yx": p.tolist(),
                        "to_yx": q.tolist(),
                        "via_yx": [a.tolist(), b.tolist()],
                        "length_native_px": float(
                            (np.abs(p - a).sum() + np.abs(a - b).sum() + np.abs(b - q).sum())
                            / scale
                        ),
                    }
                )

    repaired = skeletonize(repaired)
    labels, _ = ndi.label(~repaired)
    outside = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    areas = np.bincount(labels.ravel())
    faces = [
        i
        for i in range(1, len(areas))
        if i not in outside and areas[i] >= config.min_face_area * scale**2
    ]
    candidate = np.zeros_like(ink)
    issues = []
    if not faces:
        issues.append("no_closed_region")
    elif structure == "ICM":
        candidate = np.isin(labels, faces)
        if len(faces) != 1:
            issues.append("multiple_icm_regions")
    elif len(faces) < 2:
        issues.append("only_one_closed_region_ring_ambiguous")
    else:
        cavity = max(faces, key=lambda i: ndi.distance_transform_edt(labels == i).max())
        candidate = np.isin(labels, [i for i in faces if i != cavity])
    distance_to_candidate = (
        ndi.distance_transform_edt(~candidate) if candidate.any() else np.full(ink.shape, np.inf)
    )
    unsupported = float(
        (ink & (distance_to_candidate > config.support_distance * scale)).sum() / max(1, ink.sum())
    )
    radial_qc = None
    if (
        structure in {"ZP", "TE"}
        and config.radial_ring_fallback
        and (unsupported > config.max_unsupported_fraction or len(faces) > 3)
    ):
        radial = _radial_ring(ink, native_shape, padding, scale, config)
        if radial is not None and radial[2]["unsupported_ink_fraction"] < unsupported - 0.01:
            candidate, radial_boundary, radial_qc = radial
            repaired = skeletonize(repaired | radial_boundary)
            added |= radial_boundary & ~skeleton
            unsupported = radial_qc["unsupported_ink_fraction"]
            issues = [
                issue
                for issue in issues
                if issue not in {"no_closed_region", "only_one_closed_region_ring_ambiguous"}
            ]
    if unsupported > config.max_unsupported_fraction:
        issues.append("substantial_ink_outside_candidate")
    if radial_qc is not None:
        issues.append("radial_ring_reconstruction_requires_review")
    if any(r["type"] == "field_edge_closure" for r in records):
        issues.append("field_edge_closure_requires_review")
    return {"candidate": candidate, "skeleton": repaired, "repair_added": added}, {
        "issues": issues,
        "status": "needs_correction" if issues else "needs_visual_approval",
        "repairs": records,
        "unsupported_ink_fraction": unsupported,
        "candidate_area_native_px": float(candidate.sum() / scale**2),
        "closed_face_areas_native_px": [float(areas[i] / scale**2) for i in faces],
        "radial_ring_reconstruction": radial_qc,
        "approved": False,
    }
