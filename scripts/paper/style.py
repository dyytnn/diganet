"""Shared plotting style for the paper figures (dataviz reference palette, light mode)."""

from __future__ import annotations

import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# categorical slots in validated order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIVERGING = (BLUE, "#f0efec", RED)

FAMILY_COLORS = {
    "nnU-Net": BLUE,
    "DiGA-Net family": ORANGE,
    "Literature block": AQUA,
    "Blastocyst paper": VIOLET,
    "Generic baseline": MUTED,
    "Boundary stage": RED,
    "Recipe rung": YELLOW,
}
FAMILY_MARKERS = {
    "nnU-Net": "*", "DiGA-Net family": "o", "Literature block": "s", "Blastocyst paper": "D",
    "Generic baseline": "^", "Boundary stage": "P", "Recipe rung": "X",
}
CLASS_NAMES = ["BG", "ZP", "TE", "BL", "ICM"]


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.labelcolor": INK2, "text.color": INK, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.grid": True, "axes.grid.axis": "y", "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2, "lines.markersize": 6, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


# ---- reader-facing experiment names (internal config codes never appear in the paper) ----
_DISPLAY_EXACT = {
    "S43": "DiGA-Net+", "S24": "DiGA-Net", "S54": "DiGA-Net + boundary branch",
    "B9_nnunet": "nnU-Net", "B0": "U-Net", "B1": "Attention U-Net", "B2": "U-Net++", "B3": "BTFormer (repr.)",
    "B4": "I2CNet (repr.)", "B5": "MiMics-Net (repr.)", "B6": "DeepLabV3+", "B7": "TransUNet", "B8": "SegFormer-B0",
    "B10_MASSNet": "MASS-Net (repr.)", "B11_BLASTNet_documented_recon": "BLAST-NET (repr.)", "B12_BLASTO_Net": "Blasto-Net (repr.)",
    "S33": "base + adaptive skip fusion", "S34": "base + frequency multi-scale", "S36": "base + ConvNeXt block",
    "S37": "base + residual encoder", "S38": "base + adaptive receptive field", "S40": "base + anti-aliased pooling",
    "S41": "base + CNN-SSM block", "S42": "base + token MoE", "S44": "base + MedNeXt block", "S45": "base + D-LKA block",
    "S46": "PointRend (faithful)", "S47": "BPR proxy", "S48": "DeepStrip", "S49": "RefineMask-lite", "S50": "Mask Transfiner-lite",
    "S51": "SharpContour", "S52": "SegFix", "S53": "CascadePSP-lite", "S55": "Deep Snake", "S56": "PolyTransform-lite",
    "A1": "base + dynamic snake conv",
    "S57": "base + entropy-gated refinement (d1)", "S58": "DiGA-Net+ with texture gates and noise",
    "L01": "base + MedNeXt", "L02": "base + ConvNeXt V2", "L03": "base + MSCA", "L04": "base + D-LKA", "L05": "base + DCNv4",
    "L06": "base + ODConv", "L07": "base + SE", "L08": "base + CBAM", "L09": "base + ECA", "L10": "base + coordinate attention",
    "L11": "base + MFMSA", "L12": "base + CE-Net blocks", "L13": "base + AD-GBC", "L14": "base + Mamba-lite", "L15": "base + DySample",
    "L16": "base + CARAFE", "L17": "base + PointRend-style refiner", "L18": "base + MaxViT", "L19": "base + BiFormer",
    "L20": "base + KAN token mixer", "L21": "base + bilateral graph", "L22": "base + phase-aligned fusion",
    "V2_0": "recipe control", "V2_1": "control + residual backbone", "V2_2": "control + nnU-Net augmentation",
    "V2_3": "control + EMA", "V2_4": "control + ICM oversampling", "V2_5": "control + per-image Dice", "V2_6": "control + 384x448 padding",
}


def display_name(code: str) -> str:
    """Map an internal run/config code (e.g. 'S43', 'L01_mednext_grouped_v1_metricfixed', 'V2_2_nnunet_...') to its paper name."""
    if code in _DISPLAY_EXACT:
        return _DISPLAY_EXACT[code]
    import re as _re
    m = _re.match(r"^(V2_\d|S\d\d|L\d\d|A1|B\d+)(?:_|$)", code)
    if m and m.group(1) in _DISPLAY_EXACT:
        return _DISPLAY_EXACT[m.group(1)]
    return code
