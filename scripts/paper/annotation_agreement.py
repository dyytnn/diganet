#!/usr/bin/env python3
"""Run A3 descriptive DRAFT annotation agreement without approving extracted masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from matblast.engine.trainer import seed_everything  # noqa: E402
from matblast.eval.annotation_agreement import annotation_scores, grouped_descriptive_summary  # noqa: E402

CLASSES = {"ZP": 1, "TE": 2, "ICM": 4}
METRICS = ("iou", "dice", "boundary_f1", "boundary_f2", "boundary_f3", "hd95")


def digest(path: Path) -> str:
    """Return SHA256 of an input/artifact file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_json(value):
    """Represent unavailable numbers as null, infinite distances explicitly as strings."""
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def save_json(path: Path, value) -> None:
    """Write standards-compliant JSON preserving explicit missingness."""
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")


def main() -> None:
    """Freeze cohorts using extraction QC before accessing reference mask values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extraction", type=Path, default=Path("dataset/annotation_reconstruction_v4")
    )
    parser.add_argument("--reference", type=Path, default=Path("dataset/source/masks"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/source_grouped_v1.json"))
    parser.add_argument(
        "--technical-review",
        type=Path,
        help="Frozen scan-only visual-review record; does not imply human/expert approval",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--draft",
        action="store_true",
        required=True,
        help="Required acknowledgement: extraction masks remain unapproved",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--evaluation-size",
        type=int,
        help="Optional square evaluation size; omit to retain native resolution",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Output exists; choose a new analysis version")
    if args.bootstrap_samples < 1:
        raise ValueError("Bootstrap count must be positive")
    if args.evaluation_size is not None and args.evaluation_size < 8:
        raise ValueError("Evaluation size must be at least 8 pixels")
    seed_everything(args.seed)
    cv2.setNumThreads(1)
    split = json.loads(args.split.read_text())
    records = json.loads((args.extraction / "manifest.json").read_text())
    provenance = json.loads((args.extraction / "provenance.json").read_text())
    if len(records) != 150 or {r["image"] for r in records} != set(split["val"]):
        raise ValueError("Expected exactly the 50 historical validation images / 150 panels")
    if len({(r["image"], r["structure"]) for r in records}) != 150:
        raise ValueError("Duplicate image/class records")
    technical_review = None
    if args.technical_review:
        technical_review = json.loads(args.technical_review.read_text())
        allowed_keys = {
            "schema_version",
            "source_extraction",
            "source_manifest_sha256",
            "review_date",
            "reviewer",
            "scope",
            "decision",
            "expected_images",
            "expected_panels",
            "accepted_structures",
            "review_evidence",
            "reference_masks_viewed_during_geometry_review",
            "aggregate_v4_draft_scores_known",
            "human_approved",
            "annotator_metadata_complete",
            "te_boundary_interpretation_confirmed_by_annotator",
            "notes",
        }
        if set(technical_review) != allowed_keys:
            raise ValueError("Unknown or missing technical-review key")
        expected = {
            "schema_version": 1,
            "source_manifest_sha256": digest(args.extraction / "manifest.json"),
            "decision": "technical_accept_all",
            "expected_images": 50,
            "expected_panels": 150,
            "accepted_structures": ["ZP", "TE", "ICM"],
            "human_approved": False,
        }
        for key, value in expected.items():
            if technical_review.get(key) != value:
                raise ValueError(f"Technical-review mismatch for {key}")
    groups = {}
    for group, names in split["groups"].items():
        for name in names:
            if name in groups:
                raise ValueError("Duplicate group membership")
            groups[name] = group
    rows, masks = [], {}
    for r in records:
        c, name = r["structure"], r["image"]
        if c not in CLASSES or r["class_id"] != CLASSES[c] or name not in groups:
            raise ValueError("Invalid class ID or missing frozen group")
        paths = list((args.extraction / "cases" / Path(name).stem).glob(f"{c}_candidate_x*.png"))
        if len(paths) != 1:
            raise ValueError(f"Ambiguous candidate: {name} {c}")
        path = paths[0]
        raw = np.asarray(Image.open(path).convert("L"))
        if not set(np.unique(raw)) <= {0, 255}:
            raise ValueError("Non-binary extraction raster")
        h, w = r["native_shape_hw"]
        padding = r["supersampled_padding_px"]
        scale = provenance["parameters"]["scale"]
        if raw.shape != (h * scale + 2 * padding, w * scale + 2 * padding):
            raise ValueError("Padded geometry mismatch")
        candidate = (
            cv2.resize(
                raw[padding : padding + h * scale, padding : padding + w * scale].astype(np.float32)
                / 255,
                (w, h),
                interpolation=cv2.INTER_AREA,
            )
            >= 0.5
        )
        available = bool(candidate.any())
        passed = bool(r["registration"]["registration_pass"])
        issues = list(r["extraction"]["issues"])
        # Inclusion is decided without reading reference masks or comparison scores.
        screened = available and passed and not issues
        reasons = (
            ([] if available else ["unresolved_empty_extraction"])
            + ([] if passed else ["registration_qc_failed"])
            + issues
        )
        rows.append(
            {
                "image": name,
                "group": groups[name],
                "class": c,
                "class_id": CLASSES[c],
                "candidate_file": str(path),
                "candidate_sha256": digest(path),
                "height": h,
                "width": w,
                "candidate_pixels": int(candidate.sum()),
                "all_nonempty_draft": available,
                "qc_screened_draft": screened,
                "technical_visual_review": bool(technical_review) and available,
                "approved": bool(r["approved"]),
                "repaired": r.get("reconstruction", {}).get("mask_changed", False),
                "exclusion_reasons": ";".join(reasons),
            }
        )
        masks[name, c] = candidate
    if technical_review and sum(r["technical_visual_review"] for r in rows) != 150:
        raise ValueError("Technical accept-all review requires 150 nonempty candidates")
    args.out.mkdir(parents=True)
    save_json(args.out / "cohorts_before_scoring.json", rows)
    metadata = {
        "analysis": "A3",
        "status": (
            "PROVISIONAL_TECHNICALLY_REVIEWED"
            if technical_review
            else "PROVISIONAL_UNAPPROVED_EXTRACTION"
        ),
        "split": str(args.split),
        "split_sha256": digest(args.split),
        "extraction": str(args.extraction),
        "manifest_sha256": digest(args.extraction / "manifest.json"),
        "extraction_provenance_sha256": digest(args.extraction / "provenance.json"),
        "reference": str(args.reference),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {
            str(p): digest(p)
            for p in (
                Path(__file__),
                ROOT / "src/matblast/eval/annotation_agreement.py",
                ROOT / "src/matblast/metrics/segmentation.py",
            )
        },
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "n_seeds": 0,
        "std_definition": "between-image sample SD, not model seed SD",
        "bootstrap_unit": "frozen filename group; not verified patient identity",
        "resolution": (
            f"{args.evaluation_size}x{args.evaluation_size}; nearest-neighbour mask resize"
            if args.evaluation_size
            else "native"
        )
        + "; no test-set inference or training",
        "classes": CLASSES,
        "include_background": False,
        "include_bl": False,
        "mask_comparison": "independent binary classes; no argmax/overlap priority",
        "hd95": (
            "95th percentile of pooled bidirectional boundary distances; "
            + (
                f"{args.evaluation_size}x{args.evaluation_size} evaluation-grid pixels"
                if args.evaluation_size
                else "native pixels"
            )
            + "; spacing (1,1)"
        ),
        "boundary_f": "Euclidean tolerance 1/2/3 native pixels; primary tolerance 2",
        "missing_policy": "empty unresolved extraction excluded; absent reference class -> NaN",
        "annotator_identity": None,
        "annotator_experience": None,
        "independence_confirmed": None,
        "te_boundary_interpretation": "two contours on TE panel; not substituted from ZP; confirmation pending",
        "official_interannotator_agreement_ready": False,
        "technical_review": (
            {
                "file": str(args.technical_review),
                "sha256": digest(args.technical_review),
                "reviewer": technical_review["reviewer"],
                "human_approved": False,
            }
            if technical_review
            else None
        ),
        "limitations": [
            "extraction/repair masks not human-approved",
            "scan registration and geometric repair error included",
            "QC filtering may select easier images",
            "annotator metadata/independence unverified",
            "no annotation ceiling or SOTA claim; not comparable to model metrics at different resolution",
        ],
    }
    save_json(args.out / "meta.json", metadata)
    (args.out / "source_snapshot").mkdir()
    for i, path in enumerate(metadata["source_hashes"]):
        (args.out / "source_snapshot" / f"{i}_{Path(path).name}").write_bytes(
            Path(path).read_bytes()
        )
    for row in rows:
        ref_path = args.reference / row["class"].lower() / row["image"]
        raw = np.asarray(Image.open(ref_path).convert("L"))
        if raw.shape != (row["height"], row["width"]) or not set(np.unique(raw)) <= {0, 255}:
            raise ValueError(f"Reference shape/value mismatch: {ref_path}")
        reference = raw > 0
        candidate = masks[row["image"], row["class"]]
        if args.evaluation_size:
            output_shape = (args.evaluation_size, args.evaluation_size)
            candidate = cv2.resize(
                candidate.astype(np.uint8), output_shape, interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            reference = cv2.resize(
                reference.astype(np.uint8), output_shape, interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        row.update(
            reference_file=str(ref_path),
            reference_sha256=digest(ref_path),
            reference_pixels=int(reference.sum()),
        )
        row["reference_present"] = bool(reference.any())
        row.update(
            annotation_scores(candidate, reference)
            if row["all_nonempty_draft"]
            else {key: float("nan") for key in METRICS}
        )
    summaries = []
    cohorts = ["qc_screened_draft", "all_nonempty_draft"]
    if technical_review:
        cohorts.insert(0, "technical_visual_review")
    for cohort in cohorts:
        selected = [r for r in rows if r[cohort]]
        for c in CLASSES:
            items = [r for r in selected if r["class"] == c]
            for metric in METRICS:
                summaries.append(
                    {
                        "cohort": cohort,
                        "class": c,
                        "metric": metric,
                        "split": str(args.split),
                        "n_selected": len(items),
                        **grouped_descriptive_summary(
                            np.array([r[metric] for r in items]),
                            [r["group"] for r in items],
                            seed=args.seed,
                            bootstrap_samples=args.bootstrap_samples,
                        ),
                    }
                )
        # Macro across exactly three classes only on images eligible for all three.
        grouped = defaultdict(list)
        for row in selected:
            grouped[row["image"]].append(row)
        complete = [
            items
            for items in grouped.values()
            if len(items) == 3 and all(r["reference_present"] for r in items)
        ]
        for metric in METRICS:
            summaries.append(
                {
                    "cohort": cohort,
                    "class": "macro3_complete_cases",
                    "metric": metric,
                    "split": str(args.split),
                    "n_selected": len(complete),
                    **grouped_descriptive_summary(
                        np.array([np.mean([r[metric] for r in items]) for items in complete]),
                        [items[0]["group"] for items in complete],
                        seed=args.seed,
                        bootstrap_samples=args.bootstrap_samples,
                    ),
                }
            )
    save_json(args.out / "per_image.json", rows)
    save_json(args.out / "summary.json", summaries)
    for filename, data in (("per_image.csv", rows), ("summary.csv", summaries)):
        with (args.out / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    report = [
        "# A3 — provisional annotation comparison",
        "",
        "**Not final inter-annotator agreement. All extraction masks remain unapproved.**",
        "",
        f"Split: `{args.split}` (historical validation, 50 images, {len({r['group'] for r in rows})} filename groups). Commit: `{metadata['git_commit']}`.",
        f"Resolution: {metadata['resolution']}. Independent ZP/TE/ICM masks; no BG or derived BL. No model predictions or external test used.",
        "Means and SD are across images (not training seeds; n_seeds=0). Brackets show 95% filename-group bootstrap CI.",
        "IoU/Dice/Boundary-F reported as percentages; HD95 uses evaluation-grid pixels. Boundary-F uses a 2-pixel tolerance on that grid.",
        "",
    ]
    for cohort in cohorts:
        report += [
            f"## {cohort}",
            "",
            "| Class | n images / groups | IoU mean ± SD [95% CI] | Dice mean | Boundary-F@2 mean | HD95 mean |",
            "|---|---:|---|---:|---:|---:|",
        ]
        for c in (*CLASSES, "macro3_complete_cases"):
            data = {r["metric"]: r for r in summaries if r["cohort"] == cohort and r["class"] == c}
            i = data["iou"]
            report.append(
                f"| {c} | {i['n_images']} / {i['n_groups']} | {100 * i['mean']:.2f} ± {100 * i['std']:.2f} [{100 * i['ci95_low']:.2f}, {100 * i['ci95_high']:.2f}] | {100 * data['dice']['mean']:.2f} | {100 * data['boundary_f2']['mean']:.2f} | {data['hd95']['mean']:.2f} |"
            )
        report.append("")
    report += [
        "## Coverage and limitations",
        "",
        f"Scored nonempty masks: {sum(r['all_nonempty_draft'] for r in rows)}/150. QC-screened: {sum(r['qc_screened_draft'] for r in rows)}/150. Technically reviewed: {sum(r['technical_visual_review'] for r in rows)}/150. Human-approved: {sum(r['approved'] for r in rows)}/150.",
        "Technical review checks digitization fidelity only. It is not approval by the original annotator or a clinical/domain expert.",
        "The automated-QC table excludes failed registration and extraction flags before reading reference masks and can be biased toward easier cases. The all-nonempty table is sensitivity-only. When present, technical_visual_review is the primary provisional cohort because every panel was visually checked and none was excluded by score.",
        "The macro3 row uses common complete cases, so it must not be compared with a mean using different image subsets.",
        "Unknown: annotator identity/count, expertise, independence and TE boundary interpretation. User confirmed annotation presence, not extracted-mask accuracy.",
        "Scores conflate observer disagreement with scan/registration/repair error. They do not establish a performance ceiling, clinical validity or a SOTA claim.",
        "Do not tune reconstruction against these agreement scores. Future corrections must follow scan evidence independently of reference masks.",
        "The 5000-replicate (configurable) group CI omits annotator sampling, systematic extraction error and protocol uncertainty. Frozen groups are filename-derived, not verified patient IDs.",
        "",
        "## Unscored image-class pairs",
        "",
    ]
    for row in rows:
        if not row["all_nonempty_draft"]:
            report.append(
                f"- {row['image']} / {row['class']}: unresolved empty extraction; NOT interpreted as absent annotation."
            )
    (args.out / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report[:25]))
    print("Output:", args.out)


if __name__ == "__main__":
    main()
