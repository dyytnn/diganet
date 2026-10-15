#!/usr/bin/env python3
"""Rebuild filled draft masks from frozen pen extraction into a separate version."""

from __future__ import annotations

import argparse
import copy
import html
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from matblast.data.hand_mask_reconstruction import ReconstructionConfig, reconstruct_mask  # noqa: E402
from matblast.engine.trainer import seed_everything  # noqa: E402
from extract_hand_annotations import digest, make_preview, write_review  # noqa: E402


def load_mask(path: Path) -> np.ndarray:
    """Load a binary (H, W) extracted raster, without relabeling."""
    return np.asarray(Image.open(path).convert("L")) > 0


def native(mask: np.ndarray, record: dict, scale: int, threshold: float = 0.5) -> np.ndarray:
    """Crop padding from a supersampled Boolean mask and area-resample to (H, W)."""
    h, w = record["native_shape_hw"]
    p = record["supersampled_padding_px"]
    return (
        cv2.resize(
            mask[p : p + h * scale, p : p + w * scale].astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_AREA,
        )
        >= threshold
    )


def repair_preview(
    original: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    ink: np.ndarray,
    added: np.ndarray,
    title: str,
    out: Path,
) -> None:
    """Show old/new filled masks and added bridge pixels on native (H, W) image."""
    rgb = np.repeat(original[..., None], 3, axis=2)
    views = []
    for mask in (old, new):
        view = rgb.copy()
        view[mask] = (0.5 * view[mask] + 0.5 * np.array([0, 210, 200])).astype(np.uint8)
        views.append(view)
    evidence = rgb.copy()
    evidence[ink] = [240, 0, 70]
    evidence[cv2.dilate(added.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = [0, 255, 70]
    views.append(evidence)
    canvas = Image.new("RGB", (1200, 470), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill="black")
    for i, (view, name) in enumerate(
        zip(
            views,
            ("BEFORE draft fill", "AFTER draft fill", "Red: original ink; GREEN: added bridge"),
            strict=True,
        )
    ):
        draw.text((i * 400 + 8, 30), name, fill="black")
        image = Image.fromarray(view)
        image.thumbnail((390, 410))
        canvas.paste(image, (i * 400, 55))
    canvas.save(out, quality=93)


def main() -> None:
    """Copy evidence, repair only flagged panels, and preserve every QC decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("dataset/annotation_extraction_v3"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = (
        ReconstructionConfig(**json.loads(args.config.read_text()))
        if args.config
        else ReconstructionConfig()
    )
    if args.out.exists():
        raise FileExistsError("Output exists; choose a new version")
    records = json.loads((args.source / "manifest.json").read_text())
    provenance = json.loads((args.source / "provenance.json").read_text())
    scale = provenance["parameters"]["scale"]
    seed_everything(provenance["parameters"]["seed"])
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True)
    # Copies, not hard links: new draft writes cannot modify the frozen source.
    for source in (args.source / "cases").rglob("*"):
        target = args.out / "cases" / source.relative_to(args.source / "cases")
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    provenance["reconstruction"] = {
        "parameters": asdict(config),
        "source": str(args.source.resolve()),
        "manifest_sha256": digest(args.source / "manifest.json"),
        "source_hashes": {
            str(p): digest(p)
            for p in (Path(__file__), Path("src/matblast/data/hand_mask_reconstruction.py"))
        },
        "registration_changed": False,
        "selection": "repair flagged panels only; accept nonempty proposals only when ink coverage improves",
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.out / "source_snapshot").mkdir()
    for path in provenance["reconstruction"]["source_hashes"]:
        shutil.copyfile(path, args.out / "source_snapshot" / Path(path).name)
    (args.out / "parameters.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    grouped = defaultdict(list)
    repair_cards = []
    for record in records:
        grouped[record["image"]].append(record)
        if not record["extraction"]["issues"]:
            record["reconstruction"] = {
                "attempted": False,
                "reason": "no blocking extraction flags",
            }
            continue
        case = args.out / "cases" / Path(record["image"]).stem
        prefix = record["structure"]
        ink = load_mask(case / f"{prefix}_ink_x{scale}.png")
        skeleton = load_mask(case / f"{prefix}_skeleton_x{scale}.png")
        old = load_mask(case / f"{prefix}_candidate_x{scale}.png")
        rasters, qc = reconstruct_mask(
            ink,
            skeleton,
            prefix,
            tuple(record["native_shape_hw"]),
            record["supersampled_padding_px"],
            scale,
            config,
        )
        old_distance = ndi.distance_transform_edt(~old) if old.any() else np.full(old.shape, np.inf)
        old_unsupported = float(
            (ink & (old_distance > config.support_distance * scale)).sum() / max(1, ink.sum())
        )
        accepted = bool(
            rasters["candidate"].any() and qc["unsupported_ink_fraction"] < old_unsupported - 0.01
        )
        changed = bool(accepted and not np.array_equal(old, rasters["candidate"]))
        record["reconstruction"] = {
            "attempted": True,
            "accepted_as_draft": accepted,
            "mask_changed": changed,
            "old_unsupported_ink_fraction": old_unsupported,
            "proposal": qc,
        }
        # Keep proposals even when rejected, so repair decisions can be audited.
        for name, mask in rasters.items():
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                case / f"{prefix}_proposal_{name}_x{scale}.png"
            )
        record["previous_extraction"] = copy.deepcopy(record["extraction"])
        if accepted:
            record["extraction"] = qc
            record["status"] = (
                qc["status"]
                if record["registration"]["registration_pass"]
                else "registration_qc_failed"
            )
            for name in ("candidate", "skeleton"):
                Image.fromarray(rasters[name].astype(np.uint8) * 255).save(
                    case / f"{prefix}_{name}_x{scale}.png"
                )
            if record["registration"]["registration_pass"]:
                path = case / f"{prefix}_candidate_native.png"
                Image.fromarray(
                    native(rasters["candidate"], record, scale).astype(np.uint8) * 255
                ).save(path)
                record["candidate_file"] = str(path.relative_to(args.out))
        original = np.asarray(Image.open(case / "original.png").convert("L"))
        preview = case / f"{prefix}_repair_review.jpg"
        repair_preview(
            original,
            native(old, record, scale),
            native(rasters["candidate"] if accepted else old, record, scale),
            native(ink, record, scale),
            native(rasters["repair_added"], record, scale, threshold=0.01),
            f"{record['image']} {prefix} | accepted as DRAFT: {accepted} | proposal additions shown even if rejected",
            preview,
        )
        repair_cards.append(
            f"<h2>{html.escape(record['image'])} {prefix}</h2><p>{html.escape(str(record['reconstruction']))}</p><img loading='lazy' src='{html.escape(str(preview.relative_to(args.out)))}'>"
        )
        print(
            record["image"],
            prefix,
            "changed",
            changed,
            "issues",
            record["extraction"]["issues"],
            flush=True,
        )
    cases = []
    for image, rows in grouped.items():
        case = args.out / "cases" / Path(image).stem
        original = np.asarray(Image.open(case / "original.png").convert("L"))
        views = []
        for record in rows:
            prefix = record["structure"]
            p = record["supersampled_padding_px"]
            h, w = record["native_shape_hw"]
            aligned = np.asarray(Image.open(case / f"{prefix}_aligned_scan.jpg"))
            views.append(
                {
                    "structure": prefix,
                    "status": record["status"],
                    "issues": record["extraction"]["issues"],
                    "aligned_native": cv2.resize(
                        aligned[p : p + h * scale, p : p + w * scale], (w, h)
                    ),
                    "ink_native": native(
                        load_mask(case / f"{prefix}_ink_x{scale}.png"), record, scale
                    ),
                    "candidate_native": native(
                        load_mask(case / f"{prefix}_candidate_x{scale}.png"), record, scale
                    ),
                }
            )
        make_preview(original, views, case / "review.jpg")
        cases.append(
            {
                "image": image,
                "scan": rows[0]["scan"],
                "row": rows[0]["row"],
                "preview": str((case / "review.jpg").relative_to(args.out)),
            }
        )
    write_review(args.out, records, cases)
    summary = {
        "images": len(cases),
        "panels": len(records),
        "status_counts": dict(Counter(r["status"] for r in records)),
        "changed_draft_masks": sum(r["reconstruction"].get("mask_changed", False) for r in records),
        "native_candidate_files": sum("candidate_file" in r for r in records),
        "approved_masks": 0,
        "annotation_presence": provenance["annotation_presence"],
        "registration_changed": False,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "repairs.html").write_text(
        "<meta charset='utf-8'><title>Mask repair audit</title><style>body{font:14px system-ui;margin:24px}img{max-width:100%}p{overflow-wrap:anywhere}</style><h1>Draft repairs: before / after / added pixels</h1><p>Green = algorithm-added pixels, NOT expert ink. All repairs require review. Failed registration QC remains failed.</p>"
        + "\n".join(repair_cards)
    )
    (args.out / "README.md").write_text(
        "# Filled-mask reconstruction (DRAFT)\n\nOpen repairs.html for before/after and every added bridge; review.html for all panels.\nRegistration, original ink and scans are unchanged. Only flagged panels were considered.\nNative candidates remain unapproved; border closures are assumptions requiring review.\nNo labels, predictions or agreement scores used.\n\n"
        + json.dumps(summary, indent=2)
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
