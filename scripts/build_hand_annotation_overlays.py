#!/usr/bin/env python3
"""Build compact three-class overlays from recovered hand-drawn pen contours."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

STRUCTURES = ("ZP", "TE", "ICM")
COLORS = {
    "ZP": np.array([255, 210, 0], dtype=np.float32),
    "TE": np.array([0, 205, 255], dtype=np.float32),
    "ICM": np.array([255, 40, 130], dtype=np.float32),
}


def safe_name(name: str) -> str:
    """Return a stable filename while preserving the recognizable image ID."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("_")


def native_ink(case_dir: Path, record: dict) -> np.ndarray | None:
    """Load recovered ink and map padded supersampled pixels to native shape."""
    return native_layer(case_dir, record, "ink")


def native_layer(case_dir: Path, record: dict, kind: str) -> np.ndarray | None:
    """Read padded (s*H, s*W) draft raster as native (H, W), without filling gaps."""
    if kind not in {"ink", "candidate"}:
        raise ValueError("Unknown raster kind")
    paths = list(case_dir.glob(f"{record['structure']}_{kind}_x*.png"))
    if not paths or "supersampled_padding_px" not in record:
        return None
    if len(paths) != 1:
        raise ValueError(f"Ambiguous raster scales in {case_dir}")
    path = paths[0]
    ink = np.asarray(Image.open(path).convert("L")) > 0
    padding = int(record["supersampled_padding_px"])
    height, width = map(int, record["native_shape_hw"])
    scale_y = (ink.shape[0] - 2 * padding) / height
    scale_x = (ink.shape[1] - 2 * padding) / width
    if abs(scale_x - scale_y) > 1e-6 or scale_x <= 0:
        raise ValueError(f"Invalid padded ink geometry: {path}")
    y1, x1 = padding + round(height * scale_y), padding + round(width * scale_x)
    crop = ink[padding:y1, padding:x1].astype(np.float32)
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
    return resized >= 0.5 if kind == "candidate" else resized > 0.12


def render_overlay(
    original: np.ndarray,
    layers: dict[str, np.ndarray | None],
    records: dict,
    *,
    mode: str = "boundary",
    alpha: float = 0.45,
) -> Image.Image:
    """Render (H, W) image with pen or filled draft masks, preserving empty regions."""
    if mode not in {"boundary", "filled"} or not 0 <= alpha <= 1:
        raise ValueError("Invalid overlay mode or alpha")
    rgb = np.repeat(original[..., None], 3, axis=2).astype(np.float32)
    color_sum = np.zeros_like(rgb)
    count = np.zeros(original.shape, np.uint8)
    for structure in STRUCTURES:
        ink = layers[structure]
        if ink is None:
            continue
        if ink.shape != original.shape:
            raise ValueError("Layer shape differs from original")
        if mode == "filled":
            color_sum[ink] += COLORS[structure]
            count[ink] += 1
        else:
            visible = cv2.dilate(ink.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
            rgb[visible] = 0.15 * rgb[visible] + 0.85 * COLORS[structure]
    if mode == "filled":
        visible = count > 0
        # Average colours at overlaps instead of silently giving a class priority.
        rgb[visible] = (1 - alpha) * rgb[visible] + alpha * color_sum[visible] / count[
            visible, None
        ]

    height, width = original.shape
    header = 86
    canvas = Image.new("RGB", (width, height + header), "white")
    canvas.paste(Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)), (0, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (8, 5),
        "DRAFT filled masks (unapproved)"
        if mode == "filled"
        else "DRAFT pen overlay (no original labels/model predictions)",
        fill="black",
        font=font,
    )
    for index, structure in enumerate(STRUCTURES):
        rec = records[structure]
        available = layers[structure] is not None and layers[structure].any()
        passed = bool(rec.get("registration", {}).get("registration_pass", False))
        state = (
            "alignment QC pass"
            if passed
            else ("alignment needs review" if available else "extraction not recovered")
        )
        if mode == "filled":
            if not available:
                state = "no filled region recovered"
            elif not passed:
                state = "DRAFT - alignment needs review"
            elif rec.get("extraction", {}).get("issues"):
                state = "DRAFT - extraction needs review"
            else:
                state = "DRAFT - visual review required"
        color = tuple(COLORS[structure].astype(np.uint8)) if available else (130, 130, 130)
        y = 27 + index * 18
        draw.rectangle((8, y, 21, y + 13), fill=color)
        draw.text((26, y), f"{structure}: {state}", fill="black", font=font)
    return canvas


def main() -> None:
    """Create one three-class overlay per image plus an HTML review index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", type=Path, default=Path("dataset/annotation_extraction_v3"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mode", choices=("boundary", "filled"), default="boundary")
    parser.add_argument("--alpha", type=float, default=0.45, help="Filled-mask opacity (0..1)")
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    out = args.out or args.extraction / (
        "overlay_3class_filled" if args.mode == "filled" else "overlay_3class"
    )
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}; choose a new folder")
    out.mkdir(parents=True)

    records = json.loads((args.extraction / "manifest.json").read_text())
    provenance = json.loads((args.extraction / "provenance.json").read_text())
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["image"]].append(record)
    if len(grouped) != 50 or any(
        {r["structure"] for r in rows} != set(STRUCTURES) for rows in grouped.values()
    ):
        raise ValueError("Expected exactly 50 images with ZP, TE and ICM records")

    rows = []
    for image, image_records in sorted(
        grouped.items(), key=lambda item: (item[1][0]["page"], item[1][0]["row"])
    ):
        by_structure = {record["structure"]: record for record in image_records}
        case_dir = args.extraction / "cases" / Path(image).stem
        original = np.asarray(Image.open(case_dir / "original.png").convert("L"))
        layers = {
            structure: native_layer(
                case_dir, by_structure[structure], "candidate" if args.mode == "filled" else "ink"
            )
            for structure in STRUCTURES
        }
        available = [
            structure
            for structure in STRUCTURES
            if layers[structure] is not None and layers[structure].any()
        ]
        review = [
            structure
            for structure in STRUCTURES
            if (
                structure not in available
                or by_structure[structure].get("extraction", {}).get("issues")
            )
        ]
        coverage = sum(
            (layer.astype(np.uint8) for layer in layers.values() if layer is not None),
            start=np.zeros(original.shape, np.uint8),
        )
        passed = [
            structure
            for structure in STRUCTURES
            if by_structure[structure].get("registration", {}).get("registration_pass", False)
        ]
        page, row = int(image_records[0]["page"]), int(image_records[0]["row"])
        filename = f"p{page:02d}_r{row}__{safe_name(image)}.png"
        render_overlay(original, layers, by_structure, mode=args.mode, alpha=args.alpha).save(
            out / filename
        )
        rows.append(
            {
                "page": page,
                "row": row,
                "image": image,
                "overlay": filename,
                "available_layers": ";".join(available),
                "registration_qc_pass": ";".join(passed),
                "all_layers_available": len(available) == 3,
                "all_registration_qc_pass": len(passed) == 3,
                "approved": False,
                "extraction_review": ";".join(review),
                "overlap_pixels": int((coverage > 1).sum()),
                "mode": args.mode,
                "all_annotations_user_confirmed": all(
                    r.get("annotation_presence") == "user_confirmed" for r in image_records
                ),
            }
        )

    with (out / "index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    cards = []
    for row in rows:
        cls = (
            "pass"
            if row["all_registration_qc_pass"]
            and (args.mode != "filled" or not row["extraction_review"])
            else "review"
        )
        cards.append(
            f"<figure class='{cls}'><a href='{html.escape(row['overlay'])}'>"
            f"<img loading='lazy' src='{html.escape(row['overlay'])}'></a>"
            f"<figcaption>{html.escape(row['image'])}<br>"
            f"layers: {html.escape(row['available_layers'])}; QC pass: "
            f"{html.escape(row['registration_qc_pass'])}<br>"
            f"Extraction review: {html.escape(row['extraction_review']) or 'visual approval only'}"
            f"</figcaption></figure>"
        )
    document = (
        """<!doctype html><meta charset="utf-8"><title>Three-class annotation overlays</title>
<style>body{font:14px system-ui;margin:20px;background:#f3f4f5}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}figure{margin:0;background:white;padding:9px;border:3px solid #c97900}figure.pass{border-color:#198754}img{width:100%}figcaption{overflow-wrap:anywhere;margin-top:6px}.legend{position:sticky;top:0;background:white;padding:10px;z-index:2}</style>
<div class="legend"><h1>Draft three-class overlays</h1><p><b style="color:#c99f00">ZP yellow</b> · <b style="color:#009dc4">TE cyan</b> · <b style="color:#df1468">ICM magenta</b>. Green frame: all three registrations pass automated QC; orange: inspect alignment or unresolved extraction. An unrecovered layer does NOT mean the expert omitted a class. These are recovered pen strokes, not approved masks.</p></div><div class="grid">"""
        + "\n".join(cards)
        + "</div>"
    )
    if args.mode == "filled":
        document = document.replace("Draft three-class overlays", "Draft filled-mask overlays")
        document = document.replace(
            "These are recovered pen strokes, not approved masks.",
            f"Filled candidate masks at {args.alpha:.0%} opacity, including drafts with failed alignment QC. "
            "Empty or partial fills mean unresolved extraction, NOT absent annotation. "
            "Overlapping classes use averaged colours; masks remain independent. "
            "Green frames require passing alignment and no blocking extraction flags, but are still unapproved.",
        )
        if "reconstruction" in provenance:
            document = document.replace(
                '<div class="grid">',
                "<p><b>Reconstructed drafts:</b> some regions include algorithm-added bridges. "
                '<a href="../repairs.html">Review before / after and added pixels</a>. '
                'Field-edge closures are assumptions, not expert-drawn lines.</p><div class="grid">',
            )
    (out / "index.html").write_text(document)
    summary = {
        "images": len(rows),
        "all_three_layers_available": sum(row["all_layers_available"] for row in rows),
        "all_three_registration_qc_pass": sum(row["all_registration_qc_pass"] for row in rows),
        "approved": 0,
        "mode": args.mode,
        "alpha": args.alpha if args.mode == "filled" else None,
        "annotation_presence": provenance.get("annotation_presence", {}),
        "source_manifest": str(args.extraction / "manifest.json"),
        "content": (
            "filled draft candidate regions, including failed-QC drafts; renderer does not modify mask geometry"
            if args.mode == "filled"
            else "registered recovered pen contours only"
        )
        + "; no reference labels or predictions",
    }
    summary["source_has_algorithmic_repairs"] = "reconstruction" in provenance
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
