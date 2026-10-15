#!/usr/bin/env python3
"""Register scanned red-pen annotation panels and export auditable draft masks.

Run from the repository root. Reads images and sheet manifest only, never labels.
Requires opencv-python and scikit-image in addition to project dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import logging
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from matblast.data.hand_annotation import (  # noqa: E402
    ExtractionConfig,
    detect_red,
    extract_regions,
    register_panel,
)
from matblast.engine.trainer import seed_everything  # noqa: E402

LOG = logging.getLogger(__name__)
STRUCTURES = ("ZP", "TE", "ICM")


def digest(path: Path) -> str:
    """Hash a source file without loading it all into memory."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def panel_box(shape: tuple, row: int, col: int) -> tuple[int, int, int, int]:
    """Return a padded scan crop for the documented 4-row/3-column sheet layout."""
    h, w = shape[:2]
    # Headers and margins are included; feature registration locates the exact image.
    x = 0.05 + col * 0.314
    y = 0.09 + (row - 1) * 0.23
    return (
        max(0, round((x - 0.025) * w)),
        max(0, round((y - 0.02) * h)),
        min(w, round((x + 0.312) * w)),
        min(h, round((y + 0.205) * h)),
    )


def make_preview(original: np.ndarray, panels: list[dict], out: Path) -> None:
    """Save a 3-row comparison of native image, aligned scan, pen, and draft mask."""
    thumb_w, thumb_h = 390, 370
    canvas = Image.new("RGB", (thumb_w * 4, (thumb_h + 52) * 3), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    for row, item in enumerate(panels):
        y = row * (thumb_h + 52)
        title = f"{item['structure']}: {item['status']} | " + ", ".join(item.get("issues", []))
        draw.text((8, y + 4), title[:200], fill="black")
        rgb = np.repeat(original[..., None], 3, axis=2)
        scan = item.get("aligned_native", rgb)
        ink_view = rgb.copy()
        ink_view[item.get("ink_native", np.zeros(original.shape, bool))] = [230, 0, 65]
        mask_view = rgb.copy()
        cand = item.get("candidate_native", np.zeros(original.shape, bool))
        mask_view[cand] = (0.45 * mask_view[cand] + 0.55 * np.array([0, 210, 200])).astype(np.uint8)
        for col, (name, array) in enumerate(
            zip(
                ("Original", "Aligned scan", "Recovered red pen", "DRAFT region (unapproved)"),
                (rgb, scan, ink_view, mask_view),
                strict=True,
            )
        ):
            draw.text((col * thumb_w + 8, y + 23), name, fill="black")
            im = Image.fromarray(array)
            im.thumbnail((thumb_w - 8, thumb_h))
            canvas.paste(im, (col * thumb_w + (thumb_w - im.width) // 2, y + 45))
    canvas.save(out, quality=90)


def write_review(out: Path, records: list[dict], cases: list[dict]) -> None:
    """Write a portable HTML review index, machine-readable QC and provenance."""
    (out / "manifest.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    fields = (
        "image",
        "structure",
        "scan",
        "row",
        "status",
        "registration_pass",
        "image_correlation",
        "inliers",
        "endpoints",
        "candidate_area_native_px",
        "issues",
    )
    with (out / "qc.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            data = {**rec, **rec.get("registration", {}), **rec.get("extraction", {})}
            writer.writerow(
                {k: ";".join(data.get(k, [])) if k == "issues" else data.get(k, "") for k in fields}
            )
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Hand annotation extraction review</title>",
        "<style>body{font:16px system-ui;max-width:1580px;margin:24px auto;padding:16px;background:#f4f5f6}img{width:100%}section{background:white;padding:16px;margin:20px 0}code{overflow-wrap:anywhere}.flag{color:#a12620}</style>",
        "<h1>Hand annotation extraction — DRAFT</h1>",
        "<p>50-image historical validation annotation study. Red = recovered ink; turquoise = proposed filled region. Every mask is unapproved. Blank masks mean extraction is unresolved, not an absent structure. Inspect scan identity, alignment and every pen stroke before accepting a mask. Original dataset labels and model predictions were not used.</p>",
        "<p>ZP and TE candidates use two boundaries drawn on their respective copies. The original instructions specify TE outer boundary from the ZP inner boundary; this discrepancy requires annotator confirmation before agreement scoring. The extraction does not enforce nesting or combine overlapping classes.</p>",
        "<p><a href='qc.csv'>QC table</a> · <a href='manifest.json'>Full provenance and transforms</a> · <a href='README.md'>Procedure and limitations</a></p>",
    ]
    for case in cases:
        parts.append(
            f"<section><h2>{html.escape(case['image'])}</h2><p>{html.escape(case['scan'])} — row {case['row']}</p>"
        )
        for rec in [r for r in records if r["image"] == case["image"]]:
            issues = rec.get("extraction", {}).get("issues", [rec.get("error", "")])
            parts.append(
                f"<p class='flag'>{rec['structure']}: {rec['status']} — {html.escape(', '.join(issues))}</p>"
            )
        parts.append(
            f"<a href='{case['preview']}'><img loading='lazy' src='{case['preview']}' alt='{html.escape(case['image'])}'></a></section>"
        )
    (out / "review.html").write_text("\n".join(parts))


def main() -> None:
    """Extract all mapped scans into a new directory; refuse accidental overwrite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans", type=Path, default=Path("dataset/Label Public Data"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("paper/annotation_sheets/manifest.csv")
    )
    parser.add_argument("--images", type=Path, default=Path("dataset/source/images"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--pages", nargs="+", type=int, help="Optional subset for smoke testing")
    parser.add_argument(
        "--annotation-presence-confirmed-on",
        type=date.fromisoformat,
        help="Date of user confirmation that all sheet images contain all three annotations; not mask approval",
    )
    args = parser.parse_args()
    config = (
        ExtractionConfig(**json.loads(args.config.read_text()))
        if args.config
        else ExtractionConfig()
    )
    if args.out.exists():
        raise FileExistsError(f"Output already exists: {args.out}; choose a new version")
    with args.manifest.open(newline="") as stream:
        entries = list(csv.DictReader(stream))
    if len({e["image"] for e in entries}) != len(entries):
        raise ValueError("Duplicate image identifiers in sheet manifest")
    scans = {}
    for path in sorted(args.scans.glob("*.jpg")):
        match = re.search(r"_(\d+)\.jpg$", path.name, re.I)
        if not match or int(match[1]) in scans:
            raise ValueError(f"Ambiguous scan page number: {path}")
        scans[int(match[1])] = path
    required = {int(re.match(r"page(\d+)", e["pdf"])[1]) for e in entries}
    if required != set(scans):
        raise ValueError(f"Manifest/scan page mismatch: {required ^ set(scans)}")
    if args.pages and not set(args.pages) <= required:
        raise ValueError("Requested page not in manifest")
    if not all((args.images / e["image"]).is_file() for e in entries):
        raise FileNotFoundError("A manifest original image is missing")
    args.out.mkdir(parents=True)
    (args.out / "cases").mkdir()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(config.seed)
    seed_everything(config.seed)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    metadata = {
        "parameters": asdict(config),
        "manifest_sha256": digest(args.manifest),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {
            str(p): digest(p)
            for p in (Path(__file__), Path("src/matblast/data/hand_annotation.py"))
        },
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "page_assignment": "Scan numeric suffix matched to manifest page; verify printed IDs in review",
        "uses_original_labels": False,
        "uses_model_predictions": False,
        "approved_for_agreement": False,
        "annotation_presence": {
            "all_three_classes_user_confirmed": bool(args.annotation_presence_confirmed_on),
            "confirmation_date": (
                args.annotation_presence_confirmed_on.isoformat()
                if args.annotation_presence_confirmed_on
                else None
            ),
            "scope": "annotation presence only; extracted geometry remains unapproved",
        },
    }
    (args.out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.out / "parameters.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (args.out / "source_snapshot").mkdir()
    for source in (Path(__file__), Path("src/matblast/data/hand_annotation.py")):
        (args.out / "source_snapshot" / source.name).write_bytes(source.read_bytes())
    records, cases = [], []
    for page in sorted(required):
        if args.pages and page not in args.pages:
            continue
        scan_path = scans[page]
        scan = np.asarray(Image.open(scan_path).convert("RGB"))
        scan_hash = digest(scan_path)
        page_rows = [e for e in entries if int(re.match(r"page(\d+)", e["pdf"])[1]) == page]
        for entry in page_rows:
            native = np.asarray(Image.open(args.images / entry["image"]).convert("L"))
            h, w = native.shape
            case_dir = args.out / "cases" / entry["code"]
            case_dir.mkdir()
            Image.fromarray(native).save(case_dir / "original.png")
            views = []
            for col, structure in enumerate(STRUCTURES):
                x0, y0, x1, y1 = panel_box(scan.shape, int(entry["row"]), col)
                panel = scan[y0:y1, x0:x1]
                # Keep raw scan evidence even when registration cannot be estimated.
                Image.fromarray(panel).save(case_dir / f"{structure}_raw_crop.jpg", quality=95)
                Image.fromarray(detect_red(panel, config).astype(np.uint8) * 255).save(
                    case_dir / f"{structure}_raw_ink.png"
                )
                rec = {
                    "image": entry["image"],
                    "structure": structure,
                    "class_id": {"ZP": 1, "TE": 2, "ICM": 4}[structure],
                    "page": page,
                    "row": int(entry["row"]),
                    "scan": str(scan_path),
                    "scan_sha256": scan_hash,
                    "original_sha256": digest(args.images / entry["image"]),
                    "scan_crop_xyxy": [x0, y0, x1, y1],
                    "native_shape_hw": [h, w],
                    "approved": False,
                    "annotation_presence": (
                        "user_confirmed" if args.annotation_presence_confirmed_on else "unreviewed"
                    ),
                }
                try:
                    transform, qc = register_panel(native, panel, config)
                    rec["registration"] = qc
                    rec["native_to_scan"] = (
                        np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]]) @ transform
                    ).tolist()
                    padding = config.padding * config.scale
                    translation = np.array(
                        [[1, 0, config.padding], [0, 1, config.padding], [0, 0, 1]]
                    )
                    target = (
                        np.diag([config.scale, config.scale, 1])
                        @ translation
                        @ np.linalg.inv(transform)
                    )
                    rec["supersampled_padding_px"] = padding
                    aligned = cv2.warpPerspective(
                        panel,
                        target,
                        (
                            (w + 2 * config.padding) * config.scale,
                            (h + 2 * config.padding) * config.scale,
                        ),
                        flags=cv2.INTER_LINEAR,
                        borderValue=(255, 255, 255),
                    )
                    Image.fromarray(aligned).save(
                        case_dir / f"{structure}_aligned_scan.jpg", quality=95
                    )
                    rasters, extraction = extract_regions(aligned, structure, config)
                    rec["extraction"] = extraction
                    rec["status"] = (
                        extraction["status"]
                        if qc["registration_pass"]
                        else "registration_qc_failed"
                    )
                    for name, arr in rasters.items():
                        Image.fromarray(arr.astype(np.uint8) * 255).save(
                            case_dir / f"{structure}_{name}_x{config.scale}.png"
                        )
                    roi = (
                        slice(padding, padding + h * config.scale),
                        slice(padding, padding + w * config.scale),
                    )
                    ink_native = (
                        cv2.resize(
                            rasters["ink"][roi].astype(np.float32),
                            (w, h),
                            interpolation=cv2.INTER_AREA,
                        )
                        > 0.15
                    )
                    candidate = (
                        cv2.resize(
                            rasters["candidate"][roi].astype(np.float32),
                            (w, h),
                            interpolation=cv2.INTER_AREA,
                        )
                        >= 0.5
                    )
                    if candidate.any() and qc["registration_pass"]:
                        path = case_dir / f"{structure}_candidate_native.png"
                        Image.fromarray(candidate.astype(np.uint8) * 255).save(path)
                        rec["candidate_file"] = str(path.relative_to(args.out))
                    rec["review_file"] = str((case_dir / "review.jpg").relative_to(args.out))
                    views.append(
                        {
                            "structure": structure,
                            "status": rec["status"],
                            "issues": extraction["issues"],
                            "ink_native": ink_native,
                            "aligned_native": cv2.resize(aligned[roi], (w, h)),
                            "candidate_native": candidate,
                        }
                    )
                except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
                    rec.update(status="registration_not_recovered", error=str(error))
                    views.append(
                        {"structure": structure, "status": rec["status"], "issues": [str(error)]}
                    )
                records.append(rec)
                LOG.info("page %02d row %s %s: %s", page, entry["row"], structure, rec["status"])
            make_preview(native, views, case_dir / "review.jpg")
            cases.append(
                {
                    "image": entry["image"],
                    "scan": scan_path.name,
                    "row": entry["row"],
                    "preview": str((case_dir / "review.jpg").relative_to(args.out)),
                }
            )
        write_review(args.out, records, cases)
    counts = {
        status: sum(r["status"] == status for r in records)
        for status in sorted({r["status"] for r in records})
    }
    summary = {
        "images": len(cases),
        "panels": len(records),
        "status_counts": counts,
        "native_candidate_files": sum("candidate_file" in r for r in records),
        "approved_masks": 0,
        "annotation_presence": metadata["annotation_presence"],
        "estimated_transforms": sum("native_to_scan" in r for r in records),
        "registration_qc_pass": sum(
            r.get("registration", {}).get("registration_pass", False) for r in records
        ),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "README.md").write_text(
        "# Scanned annotation recovery (DRAFT)\n\n" + json.dumps(summary, indent=2) + "\n\n"
        "Open review.html. All files are unapproved extraction candidates, not new ground truth.\n"
        "Each case contains the original, three aligned scans, raw red-pen rasters, skeletons,\n"
        "explicit closing-added pixels, gap bridges and available native binary masks (0/255).\n"
        "Missing native masks denote unresolved extraction, never an absent class.\n"
        "User confirmation of annotation presence is recorded separately from mask approval.\n"
        "Pen endpoints alone may be harmless tails, not open boundaries.\n"
        "Supersampled files include padding on all sides: see supersampled_padding_px in manifest.json.\n"
        "Native masks crop away the padding and retain the original image dimensions.\n"
        "The supersampled files use the scale in provenance.json. No BL or combined 5-class mask is inferred.\n\n"
        "Registration uses original image texture only (SIFT/RANSAC), independently per column.\n"
        "QC includes held-out feature residuals in native pixels, match coverage, and image correlation.\n"
        "Inspect registration at image corners and at the drawn boundaries; scan deformation can remain.\n"
        "Red dominance with connected weak-ink recovery separates ink. Small components are filtered, limited closing and short endpoint bridges are recorded,\n"
        "and bounded faces of the skeleton propose filled regions. ZP/TE require a cavity and ring faces.\n"
        "This rule can fail on touching loops, open contours, scribbles, question marks, or clipped edges.\n"
        "Inspect ALL candidates. Obtain annotator confirmation for unclear lines and for TE's two\n"
        "drawn contours versus the original instruction to reuse the inner ZP boundary.\n"
        "Keep binary class masks independent; overlapping annotations must be resolved explicitly.\n"
        "Do not tune extraction thresholds against the original labels or model performance.\n"
        "Before agreement: confirm reviewer/annotator identity, independence, date and experience;\n"
        "record per-class accepted/uncertain/missing status and preserve every correction.\n"
        "Inter-annotator agreement measures observer variability; it is not by itself a hard model ceiling.\n"
    )
    LOG.info("Completed: %s", summary)


if __name__ == "__main__":
    main()
