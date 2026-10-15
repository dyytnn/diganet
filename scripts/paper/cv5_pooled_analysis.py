#!/usr/bin/env python3
"""Pool the 5-fold CV results of the frozen shortlist and quantify selection bias.

Inputs: ``results/cv5/fold{k}_validation_no_tta.json`` (eval_benchmark; models named
``CV5_<stem>_fold{k}``, three seeds) and optionally ``results/cv5/fold{k}_nnunet_no_tta.json``
(eval_nnunet; one seed). Fold 0 is the historical validation set on which every model was
selected ("selection fold"); folds 1-4 are the historical training images, never used for any
model choice ("clean"). The primary estimate is the pooled clean set; the fold0 - clean gap is a
direct measurement of the winner's curse. nnU-Net has a single seed per fold, so in paired
comparisons its scores are broadcast against the three DiGA-Net+ seeds (the CI therefore omits
nnU-Net's initialisation variance; this is stated in the output).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from matblast.eval.paired import multiseed_paired_summary  # noqa: E402

CLASSES = ("BG", "ZP", "TE", "BL", "ICM")
SHORT = {
    "S24_pyramid_context": "S24",
    "S43_matblast_point_rend": "S43",
    "S54_matblast_bmask_boundary": "S54",
}
NNUNET = "B9_nnunet"
GROUPS = {"clean_folds_1_4": (1, 2, 3, 4), "selection_fold_0": (0,), "all_folds": (0, 1, 2, 3, 4)}


def load_fold(results_dir: Path, fold: int) -> tuple[list[str], dict[str, np.ndarray]]:
    """Return (image ids, {model: (seeds, N, 5) IoU}) for one fold."""
    payload = json.loads((results_dir / f"fold{fold}_validation_no_tta.json").read_text())
    ids = list(payload["validation_image_ids"])
    scores: dict[str, list[list[list[float | None]]]] = {}
    for key, rows in payload["per_image_iou"].items():
        name, seed = key.split(":seed")
        match = re.fullmatch(r"CV5_(.+)_fold(\d)", name)
        if not match or int(match.group(2)) != fold:
            raise ValueError(f"unexpected model key {key!r} in fold {fold}")
        scores.setdefault(SHORT[match.group(1)], []).append(rows)
    nn_path = results_dir / f"fold{fold}_nnunet_no_tta.json"
    if nn_path.exists():
        nn = json.loads(nn_path.read_text())
        if list(nn["validation_image_ids"]) != ids:
            raise ValueError(f"nnU-Net image ids differ from DiGA-Net+ ids in fold {fold}")
        scores[NNUNET] = [nn["per_image_iou"][seed] for seed in sorted(nn["per_image_iou"])]
    arrays = {}
    for model, rows in scores.items():
        arr = np.asarray(rows, dtype=np.float64)
        if arr.shape[1:] != (len(ids), len(CLASSES)):
            raise ValueError(f"{model} fold {fold}: shape {arr.shape} != (S, {len(ids)}, 5)")
        arrays[model] = arr
    return ids, arrays


def macro_fg(arr: np.ndarray) -> np.ndarray:
    """(S, N, 5) -> (S, N) foreground macro IoU (BG excluded)."""
    return np.nanmean(arr[:, :, 1:], axis=2)


def describe(arr: np.ndarray) -> dict[str, Any]:
    """Seed-level mean/std of per-class IoU and mIoU-FG."""
    per_seed = np.nanmean(macro_fg(arr), axis=1)
    out = {
        "n_seeds": int(arr.shape[0]),
        "n_images": int(arr.shape[1]),
        "miou_fg_mean": float(per_seed.mean()),
        "miou_fg_seed_std": float(per_seed.std(ddof=1)) if arr.shape[0] > 1 else 0.0,
    }
    for index, name in enumerate(CLASSES):
        out[f"iou_{name.lower()}"] = float(np.nanmean(arr[:, :, index]))
    return out


def broadcast(ref: np.ndarray, cand: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Align seed axes; a single-seed model is repeated against the other's seeds."""
    if ref.shape[0] == cand.shape[0]:
        return ref, cand, False
    if ref.shape[0] == 1:
        return np.repeat(ref, cand.shape[0], axis=0), cand, True
    if cand.shape[0] == 1:
        return ref, np.repeat(cand, ref.shape[0], axis=0), True
    raise ValueError("seed counts differ and neither model is single-seed")


def compare(cand: np.ndarray, ref: np.ndarray, samples: int, seed: int) -> dict[str, Any]:
    """Paired candidate - reference summary per class and for mIoU-FG."""
    ref_b, cand_b, broadcast_used = broadcast(ref, cand)
    return {
        "seed_broadcast": broadcast_used,
        "macro_fg_iou": multiseed_paired_summary(macro_fg(ref_b), macro_fg(cand_b), samples, seed),
        "per_class_iou": {
            name: multiseed_paired_summary(ref_b[:, :, i], cand_b[:, :, i], samples, seed)
            for i, name in enumerate(CLASSES[1:], start=1)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/cv5")
    parser.add_argument("--split", default="data/splits/source_grouped_5fold_v1.json")
    parser.add_argument("--out-dir", default="results/cv5")
    parser.add_argument("--doc", default=None, help="optional markdown summary path")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    results_dir, out_dir = Path(args.results_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text())

    folds: dict[int, tuple[list[str], dict[str, np.ndarray]]] = {}
    for k in range(5):
        if (results_dir / f"fold{k}_validation_no_tta.json").exists():
            ids, arrays = load_fold(results_dir, k)
            if sorted(ids) != sorted(split["folds"][str(k)]["val"]):
                raise ValueError(f"fold {k}: evaluated ids differ from the frozen split")
            folds[k] = (ids, arrays)
    if not folds:
        raise SystemExit("no fold results found")
    all_ids = [i for k in sorted(folds) for i in folds[k][0]]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("an image appears in more than one fold")

    output: dict[str, Any] = {
        "protocol": (
            "5-fold patient-grouped CV of the frozen shortlist; fold 0 = historical validation set "
            "(selection fold), folds 1-4 = historical training images (never used for model "
            "selection). DiGA-Net+ models: 3 seeds/fold; nnU-Net: 1 seed/fold, broadcast in paired "
            "comparisons. mIoU-FG excludes BG. No TTA."
        ),
        "folds_present": sorted(folds),
        "n_images_total": len(all_ids),
        "per_fold": {},
        "groups": {},
    }
    for k, (ids, arrays) in sorted(folds.items()):
        output["per_fold"][str(k)] = {m: describe(a) for m, a in arrays.items()}

    models_common = set.intersection(*(set(a) for _, a in folds.values()))
    for group, members in GROUPS.items():
        present = [k for k in members if k in folds]
        if not present:
            continue
        pooled = {
            m: np.concatenate([folds[k][1][m] for k in present], axis=1) for m in models_common
        }
        entry: dict[str, Any] = {
            "folds": present,
            "n_images": int(sum(len(folds[k][0]) for k in present)),
            "models": {m: describe(a) for m, a in pooled.items()},
            "paired": {},
        }
        for ref in ("S24", NNUNET):
            if ref not in pooled:
                continue
            entry["paired"][f"vs_{ref}"] = {
                m: compare(a, pooled[ref], args.bootstrap_samples, args.seed)
                for m, a in pooled.items()
                if m != ref
            }
        output["groups"][group] = entry

    # Winner's curse: selection fold minus clean folds, per model and for the S43-S24 contrast.
    if "selection_fold_0" in output["groups"] and "clean_folds_1_4" in output["groups"]:
        sel, clean = output["groups"]["selection_fold_0"], output["groups"]["clean_folds_1_4"]
        curse = {
            m: sel["models"][m]["miou_fg_mean"] - clean["models"][m]["miou_fg_mean"]
            for m in sel["models"]
            if m in clean["models"]
        }
        contrast = {}
        for ref, block in sel["paired"].items():
            for m, comp in block.items():
                if m in clean["paired"].get(ref, {}):
                    contrast[f"{m}_{ref}"] = (
                        comp["macro_fg_iou"]["delta"]
                        - clean["paired"][ref][m]["macro_fg_iou"]["delta"]
                    )
        output["winners_curse_miou_fg"] = {
            "selection_minus_clean_per_model": curse,
            "selection_minus_clean_paired_delta": contrast,
        }

    (out_dir / "cv5_summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    rows = []
    for group, entry in output["groups"].items():
        for m, d in entry["models"].items():
            rows.append({"group": group, "model": m, **d})
    with (out_dir / "cv5_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    if args.doc:
        Path(args.doc).write_text(render_markdown(output))
    print(
        json.dumps(
            {k: v for k, v in output.items() if k in ("groups", "winners_curse_miou_fg")},
            indent=1,
            default=float,
        )[:4000]
    )


def render_markdown(output: dict[str, Any]) -> str:
    lines = [
        "# 5-fold CV of the frozen shortlist (selection-bias check)",
        "",
        output["protocol"],
        "",
    ]
    for group, entry in output["groups"].items():
        lines += [f"## {group} ({entry['n_images']} images, folds {entry['folds']})", ""]
        lines += [
            "| model | seeds | mIoU-FG | seed std | ZP | TE | BL | ICM |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for m, d in sorted(entry["models"].items()):
            lines.append(
                f"| {m} | {d['n_seeds']} | {d['miou_fg_mean']:.4f} | {d['miou_fg_seed_std']:.4f} | "
                f"{d['iou_zp']:.4f} | {d['iou_te']:.4f} | {d['iou_bl']:.4f} | {d['iou_icm']:.4f} |"
            )
        for ref, block in entry["paired"].items():
            lines += [
                "",
                f"Paired mIoU-FG, candidate minus {ref[3:]} (hierarchical bootstrap CI; image-level one-sided Wilcoxon):",
                "",
                "| candidate | Δ | 95% CI | p(image) | Δ TE | Δ ICM | seed broadcast |",
                "|---|---|---|---|---|---|---|",
            ]
            for m, comp in sorted(block.items()):
                s = comp["macro_fg_iou"]
                lo, hi = s["ci95_hierarchical"]
                lines.append(
                    f"| {m} | {s['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {s['p_image_wilcoxon_greater']:.3g} | "
                    f"{comp['per_class_iou']['TE']['delta']:+.4f} | {comp['per_class_iou']['ICM']['delta']:+.4f} | {comp['seed_broadcast']} |"
                )
        lines.append("")
    if "winners_curse_miou_fg" in output:
        wc = output["winners_curse_miou_fg"]
        lines += ["## Winner's curse: selection fold minus clean folds (mIoU-FG)", ""]
        lines += [
            f"- {m}: {v:+.4f}" for m, v in sorted(wc["selection_minus_clean_per_model"].items())
        ]
        lines += [
            f"- paired Δ {m}: {v:+.4f}"
            for m, v in sorted(wc["selection_minus_clean_paired_delta"].items())
        ]
        lines.append("")
    lines += [
        "## Per fold (mIoU-FG)",
        "",
        "| fold | " + " | ".join(sorted(next(iter(output["per_fold"].values())))) + " |",
        "|---|" + "---|" * len(next(iter(output["per_fold"].values()))),
    ]
    for k, d in output["per_fold"].items():
        lines.append(
            f"| {k} | "
            + " | ".join(
                f"{d[m]['miou_fg_mean']:.4f}" if m in d else "—"
                for m in sorted(next(iter(output["per_fold"].values())))
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
