#!/usr/bin/env python3
"""S7: schematic of DiGA-Net+ S43 (matblast_unet, base 24, pyramid + global context, gated point refinement)."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "figures/paper/S7_architecture.png"
style.apply()
fig, ax = plt.subplots(figsize=(12.5, 4.8))
ax.set_xlim(0, 25); ax.set_ylim(0, 9.4); ax.axis("off"); ax.grid(False)

def box(x, y, w, h, text, fc, ec=style.AXIS, fs=7.5, tc=style.INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)

def arrow(x0, y0, x1, y1, color=style.INK2, ls="-", lw=1.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, linestyle=ls, shrinkA=1, shrinkB=1))

ENC, DEC, BOT, HEAD = "#dbe8fa", "#fde4d6", "#fff1cc", "#dff3ea"
# encoder (left, descending) and decoder (right, ascending); y by resolution level
lv = {256: 7.6, 128: 5.7, 64: 3.8, 32: 1.9}
box(0.3, lv[256], 2.6, 1.0, "input 3$\\times$256$^2$\nraw, z-score, raw", "#f0efec")
box(3.4, lv[256], 3.0, 1.0, "stem + enc1 @256$^2$\n2$\\times$conv3$\\times$3-BN-ReLU, 24 ch", ENC, fs=6.2)
box(3.4, lv[128], 3.0, 1.0, "enc2 @128$^2$\n48 ch", ENC)
box(3.4, lv[64], 3.0, 1.0, "enc3 @64$^2$\n96 ch", ENC)
box(3.4, lv[32], 3.0, 1.0, "bottleneck @32$^2$\n192 ch, dropout 0.2", ENC)
for a, b in [(256, 128), (128, 64), (64, 32)]:
    arrow(4.9, lv[a], 4.9, lv[b] + 1.0)
    ax.text(5.05, (lv[a] + lv[b] + 1.0) / 2, "maxpool 2", fontsize=6.5, color=style.MUTED, va="center")
# bottleneck modules (S24 + S43)
box(7.0, lv[32] - 0.15, 3.9, 1.3, "pyramid context\ndepth-wise dilated 1,2,4,8\n+ local branch, 1/4 width", BOT)
box(11.3, lv[32] - 0.15, 3.9, 1.3, "pooled global context\n8$\\times$8 pool, 4-head attention,\nexpand + residual", BOT)
box(15.6, lv[32] - 0.15, 4.1, 1.3, "gated point refinement (S43)\n$b + \\sigma(u(b))\\cdot r(b)$\nuncertainty-gated residual", BOT, ec=style.ORANGE)
arrow(6.4, lv[32] + 0.5, 7.0, lv[32] + 0.5); arrow(10.9, lv[32] + 0.5, 11.3, lv[32] + 0.5); arrow(15.2, lv[32] + 0.5, 15.6, lv[32] + 0.5)
# decoder
box(20.4, lv[64], 3.0, 1.0, "up 2$\\times$ + concat e3\ndec3, 96 ch", DEC)
box(20.4, lv[128], 3.0, 1.0, "up 2$\\times$ + concat e2\ndec2, 48 ch", DEC)
box(20.4, lv[256], 3.0, 1.0, "up 2$\\times$ + concat e1\ndec1, 24 ch", DEC)
arrow(19.7, lv[32] + 0.5, 21.9, lv[64], color=style.INK2)
arrow(21.9, lv[64] + 1.0, 21.9, lv[128]); arrow(21.9, lv[128] + 1.0, 21.9, lv[256])
# skips
for l in (256, 128, 64):
    arrow(6.4, lv[l] + 0.5, 20.4, lv[l] + 0.5, color=style.MUTED, ls="--", lw=0.8)
ax.text(13.4, lv[256] + 0.62, "skip connections", fontsize=6.5, color=style.MUTED, ha="center")
# heads
box(20.4, 8.85, 3.0, 0.5, "1$\\times$1 conv $\\to$ 5-class logits", HEAD, fs=7)
arrow(21.9, lv[256] + 1.0, 21.9, 8.85)
box(23.7, lv[128] + 0.1, 0.75, 0.8, "aux\n1/2", HEAD, fs=6); box(23.7, lv[64] + 0.1, 0.75, 0.8, "aux\n1/4", HEAD, fs=6)
arrow(23.4, lv[128] + 0.5, 23.7, lv[128] + 0.5); arrow(23.4, lv[64] + 0.5, 23.7, lv[64] + 0.5)
ax.text(0.3, 0.55, "S24 = everything except the orange block (1.19 M parameters). S43 = S24 + gated point refinement at the bottleneck (1.56 M).\n"
        "Loss: soft Jaccard over 5 classes (main) + 0.15 $\\times$ same loss on the two auxiliary heads. Input 256$\\times$256; 3 poolings; BatchNorm; no attention in the encoder/decoder.",
        fontsize=7, color=style.INK2, va="bottom")
fig.savefig(OUT); print("saved", OUT)
