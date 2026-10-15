#!/usr/bin/env python3
"""Compose the two merged display items of the Nature Methods version from the existing panel PNGs.

Nature Methods Analyses allow six display items, so two pairs of figures are stacked into one:
  N3 = F2_agreement (model-model vs model-annotation) over F4_error_location (where the error lives)
  N5 = F6_pareto (accuracy against cost) over S2_robustness (accuracy under corruption)
Stacking the rendered panels, rather than re-plotting, guarantees the merged figure carries exactly the
numbers of the Elsevier version. Panel letters are drawn on top; the source panels keep their own internal
labels, which are stripped by cropping nothing -- they were generated without titles (figure-style rule).

    python scripts/paper/nature_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "figures/paper"
OUT = ROOT / "paper_nature/figures"
MERGES = {
    "N3_agreement_and_errors.png": ["F2_agreement.png", "F4_error_location.png"],
    "N5_cost_and_robustness.png": ["F6_pareto.png", "S2_robustness.png"],
}
GAP = 40
PAD_LEFT = 70


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def stack(paths: list[Path], out: Path) -> None:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    width = max(im.width for im in imgs) + PAD_LEFT
    height = sum(im.height for im in imgs) + GAP * (len(imgs) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(max(28, imgs[0].width // 45))
    y = 0
    for i, im in enumerate(imgs):
        canvas.paste(im, (PAD_LEFT, y))
        draw.text((14, y + 6), "abcdefg"[i], fill=(11, 11, 11), font=font)
        y += im.height + GAP
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, optimize=True)
    print(f"{out.name}: {' + '.join(p.name for p in paths)} -> {canvas.width}x{canvas.height}")


def main() -> None:
    missing = [n for names in MERGES.values() for n in names if not (SRC / n).exists()]
    if missing:
        sys.exit(f"missing source panels: {missing}; run scripts/reproduce_paper.sh --figures first")
    for out_name, names in MERGES.items():
        stack([SRC / n for n in names], OUT / out_name)
    # the four single-panel display items and every Extended Data figure are copied unchanged
    for name in sorted(p.name for p in SRC.glob("*.png")):
        (OUT / name).write_bytes((SRC / name).read_bytes())
    (OUT / "S7_architecture.pdf").write_bytes((SRC / "S7_architecture.pdf").read_bytes())
    print(f"copied {len(list(OUT.glob('*.png')))} panels into {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
