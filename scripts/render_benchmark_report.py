#!/usr/bin/env python3
"""Render the generated benchmark and paired comparison artifacts as Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    """Format a numeric value for a compact scientific report."""
    if value is None:
        return "—"
    return f"{float(value):.4f}"


def main() -> None:
    """Render a report without editing experimental result artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--paired", required=True)
    parser.add_argument("--extended", default=None)
    parser.add_argument("--literature", default=None)
    parser.add_argument("--literature-paired", default=None)
    parser.add_argument("--shortlist-extended", action="append", default=None)
    parser.add_argument("--custom-paired", action="append", default=None)
    parser.add_argument("--out", default="REPORT.md")
    args = parser.parse_args()

    benchmark = json.loads(Path(args.benchmark).read_text())
    paired = json.loads(Path(args.paired).read_text())
    extended = json.loads(Path(args.extended).read_text()) if args.extended else None
    literature = json.loads(Path(args.literature).read_text()) if args.literature else None
    literature_paired = (
        json.loads(Path(args.literature_paired).read_text())
        if args.literature_paired
        else None
    )
    shortlist_extended_reports = [
        json.loads(Path(path).read_text()) for path in (args.shortlist_extended or [])
    ]
    custom_paired_reports = [
        json.loads(Path(path).read_text()) for path in (args.custom_paired or [])
    ]
    lines = [
        "# MAT-BLAST benchmark report",
        "",
        f"- Protocol: {benchmark['protocol']}",
        f"- Split: `data/splits/source_grouped_v1.json`; validation images: {len(benchmark['validation_image_ids'])}",
        f"- Commit: `{benchmark['commit']}`",
        "- Test set: not used; this report is validation-only.",
        "- S35 stage-conditioned graph: not evaluated due to missing developmental-stage metadata; no stage labels were fabricated.",
        "",
        "## Validation metrics",
        "",
        "| Model | Seeds | mDice-FG | mIoU-FG | IoU ZP | IoU TE | IoU BL | IoU ICM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if benchmark.get("commits"):
        commits = benchmark["commits"]
        additional = ", ".join(commits.get("additional", [])) or "none"
        lines.insert(
            4,
            f"- Source commits: generic=`{commits.get('generic')}`, nnU-Net=`{commits.get('nnunet')}`, additional=`{additional}`",
        )
    for model in benchmark["models"]:
        lines.append(
            "| {model} | {n} | {dice} ± {dice_std} | {iou} ± {iou_std} | {zp} | {te} | {bl} | {icm} |".format(
                model=model["model"],
                n=model["n_seeds"],
                dice=_fmt(model["mdice_fg_mean"]),
                dice_std=_fmt(model["mdice_fg_std"]),
                iou=_fmt(model["miou_fg_mean"]),
                iou_std=_fmt(model["miou_fg_std"]),
                zp=_fmt(model["iou_zp_mean"]),
                te=_fmt(model["iou_te_mean"]),
                bl=_fmt(model["iou_bl_mean"]),
                icm=_fmt(model["iou_icm_mean"]),
            )
        )
    if extended is not None:
        lines.extend(
            [
                "",
                "## Extended validation metrics",
                "",
                "| Model | Seeds | Params (M) mean ± std | FPS CPU mean ± std | HD95 TE mean ± std | HD95 ICM mean ± std | Boundary-F TE mean ± std | Boundary-F ICM mean ± std | Betti error mean ± std | ECE mean ± std | BL area mean ± std | ZP thickness mean ± std | ICM area mean ± std | ICM/TE ratio mean ± std | TE continuity mean ± std |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for model in extended["models"]:
            def value(key: str) -> str:
                return f"{_fmt(model.get(f'{key}_mean'))} ± {_fmt(model.get(f'{key}_std'))}"

            lines.append(
                "| {model} | {seeds} | {params} | {fps} | {hd_te} | {hd_icm} | {bf_te} | {bf_icm} | {betti} | {ece} | {bl} | {zp} | {icm} | {ratio} | {continuity} |".format(
                    model=model["model"],
                    seeds=model["n_seeds"],
                    params=value("params_m"),
                    fps=value("fps_cpu"),
                    hd_te=value("hd95_te"),
                    hd_icm=value("hd95_icm"),
                    bf_te=value("boundary_f_te"),
                    bf_icm=value("boundary_f_icm"),
                    betti=value("betti_error"),
                    ece=value("ece"),
                    bl=value("bl_area"),
                    zp=value("zp_thickness"),
                    icm=value("icm_area"),
                    ratio=value("icm_te_ratio"),
                    continuity=value("te_continuity"),
                )
            )
        lines.append("")
    if literature is not None:
        lines.extend(
            [
                "## Literature-block validation metrics",
                "",
                "All 23 additional blocks were trained with three seeds and evaluated on the same fixed grouped validation split; BG is excluded from foreground macro averages and TTA is disabled.",
                "",
                "| Model | Seeds | mDice-FG mean ± std | mIoU-FG mean ± std | IoU ZP mean ± std | IoU TE mean ± std | IoU BL mean ± std | IoU ICM mean ± std |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for model in sorted(literature["models"], key=lambda item: item["miou_fg_mean"], reverse=True):
            def pair(key: str) -> str:
                return f"{_fmt(model.get(f'{key}_mean'))} ± {_fmt(model.get(f'{key}_std'))}"

            lines.append(
                f"| {model['model']} | {model['n_seeds']} | {pair('mdice_fg')} | {pair('miou_fg')} | {pair('iou_zp')} | {pair('iou_te')} | {pair('iou_bl')} | {pair('iou_icm')} |"
            )
        lines.append("")
    if literature_paired is not None:
        lines.extend(
            [
                "## Literature-block paired comparison versus S24",
                "",
                "Candidate minus S24 on the same 50 validation images, with three seeds. The hierarchical CI and image-level Wilcoxon p-value are reported; no test set was used.",
                "",
                "| Model | mIoU Δ | mIoU 95% CI | p | ICM IoU Δ | ICM 95% CI | p | TE IoU Δ | TE 95% CI | p |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        comparisons = literature_paired["comparisons"]
        for name, comparison in sorted(
            comparisons.items(),
            key=lambda item: item[1]["macro_fg_iou"]["delta"],
            reverse=True,
        ):
            macro = comparison["macro_fg_iou"]
            icm = comparison["per_class_iou"]["ICM"]
            te = comparison["per_class_iou"]["TE"]
            lines.append(
                "| {name} | {md} | {mci} | {mp} | {id} | {ici} | {ip} | {td} | {tci} | {tp} |".format(
                    name=name,
                    md=_fmt(macro["delta"]),
                    mci=f"[{_fmt(macro['ci95_hierarchical'][0])}, {_fmt(macro['ci95_hierarchical'][1])}]",
                    mp=_fmt(macro["p_image_wilcoxon_greater"]),
                    id=_fmt(icm["delta"]),
                    ici=f"[{_fmt(icm['ci95_hierarchical'][0])}, {_fmt(icm['ci95_hierarchical'][1])}]",
                    ip=_fmt(icm["p_image_wilcoxon_greater"]),
                    td=_fmt(te["delta"]),
                    tci=f"[{_fmt(te['ci95_hierarchical'][0])}, {_fmt(te['ci95_hierarchical'][1])}]",
                    tp=_fmt(te["p_image_wilcoxon_greater"]),
                )
            )
        lines.append("")
    for shortlist_extended in shortlist_extended_reports:
        names = ", ".join(model["model"] for model in shortlist_extended["models"])
        lines.extend(
            [
                f"## Shortlist extended diagnostics ({names})",
                "",
                "Extended validation diagnostics for the shortlisted modules. Values are mean ± std across three seeds; HD95 is in pixels and foreground classes only.",
                "",
                "| Model | Seeds | HD95 TE | HD95 ICM | Boundary-F TE | Boundary-F ICM | Betti error | ECE | BL area | ZP thickness | ICM area | ICM/TE ratio | TE continuity |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for model in shortlist_extended["models"]:
            def pair(key: str) -> str:
                return f"{_fmt(model.get(f'{key}_mean'))} ± {_fmt(model.get(f'{key}_std'))}"

            lines.append(
                "| {model} | {seeds} | {hd_te} | {hd_icm} | {bf_te} | {bf_icm} | {betti} | {ece} | {bl} | {zp} | {icm} | {ratio} | {continuity} |".format(
                    model=model["model"],
                    seeds=model["n_seeds"],
                    hd_te=pair("hd95_te"),
                    hd_icm=pair("hd95_icm"),
                    bf_te=pair("boundary_f_te"),
                    bf_icm=pair("boundary_f_icm"),
                    betti=pair("betti_error"),
                    ece=pair("ece"),
                    bl=pair("bl_area"),
                    zp=pair("zp_thickness"),
                    icm=pair("icm_area"),
                    ratio=pair("icm_te_ratio"),
                    continuity=pair("te_continuity"),
                )
            )
        lines.append("")
    # Group custom paired reports by reference model; the first report that
    # contains a (reference, candidate) pair wins so re-evaluated references are
    # not listed twice.
    grouped_custom: dict[str, dict[str, Any]] = {}
    grouped_n_images: dict[str, Any] = {}
    for custom_paired in custom_paired_reports:
        reference = custom_paired.get("reference", "S24")
        bucket = grouped_custom.setdefault(reference, {})
        grouped_n_images.setdefault(reference, custom_paired.get("n_images", 50))
        for name, comparison in custom_paired["comparisons"].items():
            if name != reference:
                bucket.setdefault(name, comparison)
    for reference, comparisons in grouped_custom.items():
        lines.extend(
            [
                f"## MAT-BLAST finalist paired comparison versus {reference}",
                "",
                f"Candidate minus {reference} on the same {grouped_n_images[reference]} validation images, with three seeds. The hierarchical CI and image-level Wilcoxon p-value are reported; no test set was used.",
                "",
                "| Model | mIoU Δ | mIoU 95% CI | p | ICM IoU Δ | ICM 95% CI | p | TE IoU Δ | TE 95% CI | p |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, comparison in comparisons.items():
            macro = comparison["macro_fg_iou"]
            icm = comparison["per_class_iou"]["ICM"]
            te = comparison["per_class_iou"]["TE"]
            lines.append(
                "| {name} | {md} | {mci} | {mp} | {id} | {ici} | {ip} | {td} | {tci} | {tp} |".format(
                    name=name,
                    md=_fmt(macro["delta"]),
                    mci=f"[{_fmt(macro['ci95_hierarchical'][0])}, {_fmt(macro['ci95_hierarchical'][1])}]",
                    mp=_fmt(macro["p_image_wilcoxon_greater"]),
                    id=_fmt(icm["delta"]),
                    ici=f"[{_fmt(icm['ci95_hierarchical'][0])}, {_fmt(icm['ci95_hierarchical'][1])}]",
                    ip=_fmt(icm["p_image_wilcoxon_greater"]),
                    td=_fmt(te["delta"]),
                    tci=f"[{_fmt(te['ci95_hierarchical'][0])}, {_fmt(te['ci95_hierarchical'][1])}]",
                    tp=_fmt(te["p_image_wilcoxon_greater"]),
                )
            )
        lines.append("")
    lines.extend(
        [
            "",
            "## Paired IoU versus S24",
            "",
            "| Model | Metric | Δ mean | 95% hierarchical CI | p (image) | p (seed) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for model, comparison in paired["comparisons"].items():
        for metric, summary in [
            ("mIoU-FG", comparison["macro_fg_iou"]),
            *[
                (f"IoU {name}", comparison["per_class_iou"][name])
                for name in ("ICM", "TE", "ZP", "BL")
            ],
        ]:
            lines.append(
                f"| {model} | {metric} | {_fmt(summary['delta'])} | [{_fmt(summary['ci95_hierarchical'][0])}, {_fmt(summary['ci95_hierarchical'][1])}] | {_fmt(summary['p_image_wilcoxon_greater'])} | {_fmt(summary['p_seed_wilcoxon_greater'])} |"
            )
    lines.extend(
        [
            "",
            "The image-level paired p-value is computed after averaging the three independent seeds per validation image. The seed-level p-value is reported separately; with three seeds it cannot itself reach 0.05. S24 provenance is retained as the existing legacy reference run; all candidate models use the fixed grouped split and three seeds.",
            "",
        ]
    )
    conclusion_reports = [
        (paired.get("reference", "S24"), paired.get("comparisons", {})),
        *grouped_custom.items(),
    ]
    if any(comparisons for _, comparisons in conclusion_reports):
        lines.extend(["## Ablation conclusion", ""])
    for reference, comparisons in conclusion_reports:
        for model, comparison in comparisons.items():
            icm = comparison["per_class_iou"].get("ICM")
            te = comparison["per_class_iou"].get("TE")
            if icm is None or te is None:
                continue
            gate_pass = (
                icm["delta"] >= 0.0
                and te["delta"] >= 0.0
                and icm["p_image_wilcoxon_greater"] < 0.05
                and te["p_image_wilcoxon_greater"] < 0.05
                and icm["ci95_hierarchical"][0] > 0.0
                and te["ci95_hierarchical"][0] > 0.0
            )
            conclusion = "passes" if gate_pass else "does not pass"
            verdict = (
                "This result passes the ICM/TE gate on this validation split only; no SOTA claim is made without an external test."
                if gate_pass
                else "This result is recorded as a negative result; no improvement or SOTA claim is made."
            )
            lines.append(
                f"{model} {conclusion} the pre-registered improvement gate versus {reference}: ICM Δ={_fmt(icm['delta'])}, 95% CI [{_fmt(icm['ci95_hierarchical'][0])}, {_fmt(icm['ci95_hierarchical'][1])}], p(image)={_fmt(icm['p_image_wilcoxon_greater'])}; TE Δ={_fmt(te['delta'])}, 95% CI [{_fmt(te['ci95_hierarchical'][0])}, {_fmt(te['ci95_hierarchical'][1])}], p(image)={_fmt(te['p_image_wilcoxon_greater'])}. {verdict}")
    if any(comparisons for _, comparisons in conclusion_reports):
        lines.append("")
    Path(args.out).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
