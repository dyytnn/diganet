#!/usr/bin/env python3
"""S2: robustness of the shortlist to image corruptions on the 50 validation images (no re-training, no TTA).

Corruptions are applied to the grayscale image in [0, 1] after the 256x256 resize and before the legacy
3-channel contract (raw, z-score, raw), so every model sees exactly the same corrupted input:
  blur      Gaussian sigma in {0.5, 1, 2, 3} px
  noise     additive Gaussian sigma in {0.02, 0.05, 0.10}
  brightness multiplicative gain in {0.7, 0.85, 1.15, 1.3}
  gamma     {0.6, 0.8, 1.25, 1.6}
  contrast  {0.6, 0.8, 1.25}
Outputs results/paper/robustness.csv (model, seed, family, severity, mIoU-FG, per-class IoU) and
figures/paper/S2_robustness.png (mIoU-FG vs severity, one panel per corruption, models as lines, legend outside).

    PYTHONPATH=src python scripts/paper/robustness_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
from matblast.metrics import per_image_scores  # noqa: E402
from matblast.models import build_model  # noqa: E402

MODELS = {
    "S43": ("configs/S43_matblast_point_rend_grouped_v1.yaml", "runs/matblast_shortlist_queue/S43_matblast_point_rend_grouped_v1/S43_matblast_point_rend_grouped_v1_seed{s}/best.pt", style.ORANGE),
    "S24": ("configs/S24_pyramid_context_grouped_v1_metricfixed.yaml", "runs/S24_pyramid_context_grouped_v1_metricfixed/S24_pyramid_context_grouped_v1_metricfixed_seed{s}/best.pt", style.AQUA),
    "S54": ("configs/S54_matblast_bmask_boundary_grouped_v1.yaml", "runs/boundary_stage_queue/S54_matblast_bmask_boundary_grouped_v1/S54_matblast_bmask_boundary_grouped_v1_seed{s}/best.pt", style.RED),
    "S58": ("configs/S58_matblast_multiscale_texture_gate_noise_grouped_v1.yaml", "runs/gate_variants_queue/S58_matblast_multiscale_texture_gate_noise_grouped_v1/S58_matblast_multiscale_texture_gate_noise_grouped_v1_seed{s}/best.pt", style.GREEN),
    "U-Net": ("configs/B0_unet_grouped_v2_metricfixed.yaml", "runs/B0_unet_grouped_v2_metricfixed_seed{s}/best.pt", style.MUTED),
}
CORRUPTIONS = {
    "blur (sigma px)": [0.5, 1.0, 2.0, 3.0],
    "noise (sigma)": [0.02, 0.05, 0.10],
    "brightness (gain)": [0.7, 0.85, 1.15, 1.3],
    "gamma": [0.6, 0.8, 1.25, 1.6],
    "contrast (gain)": [0.6, 0.8, 1.25],
}


def corrupt(x: np.ndarray, family: str, s: float, rng: np.random.Generator) -> np.ndarray:
    if family.startswith("blur"):
        return ndimage.gaussian_filter(x, s)
    if family.startswith("noise"):
        return np.clip(x + rng.normal(0, s, x.shape), 0, 1)
    if family.startswith("brightness"):
        return np.clip(x * s, 0, 1)
    if family.startswith("gamma"):
        return np.clip(x, 0, 1) ** s
    if family.startswith("contrast"):
        m = x.mean()
        return np.clip((x - m) * s + m, 0, 1)
    raise ValueError(family)


def legacy(x: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(x.astype(np.float32))[None]
    z = (t - t.mean()) / (t.std() + 1e-6)
    return torch.cat((t, z, t), 0)


def find_ckpt(pattern: str, s: int) -> Path:
    p = ROOT / pattern.format(s=s)
    if p.exists():
        return p
    hits = list(ROOT.glob(f"runs/**/{Path(pattern.format(s=s)).parent.name}/best.pt"))
    if not hits:
        raise FileNotFoundError(pattern.format(s=s))
    return hits[0]


@torch.no_grad()
def main() -> None:
    style.apply()
    if "--plot-only" in sys.argv:
        plot(pd.read_csv(ROOT / "results/paper/robustness.csv"))
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = json.load(open(ROOT / "results/paper/preds/image_ids.json"))
    gt = torch.from_numpy(np.load(ROOT / "results/paper/preds/gt.npy").astype(np.int64))
    base = [np.array(Image.open(ROOT / "dataset/source/images" / n).convert("L").resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0 for n in ids]
    rows = []
    for name, (cfg_path, pat, _) in MODELS.items():
        cfg = yaml.safe_load((ROOT / cfg_path).read_text())
        for s in range(3):
            model = build_model(cfg)
            state = torch.load(find_ckpt(pat, s), map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"] if "model" in state else state)
            model.to(device).eval()
            for family, sevs in [("clean", [0.0])] + list(CORRUPTIONS.items()):
                for sev in sevs:
                    rng = np.random.default_rng(1234)
                    xs = torch.stack([legacy(corrupt(b, family, sev, rng) if family != "clean" else b) for b in base])
                    ious = []
                    for i in range(0, len(xs), 10):
                        out = model(xs[i:i + 10].to(device))
                        logits = out["logits"] if isinstance(out, dict) else out
                        _, iou = per_image_scores(logits.cpu(), gt[i:i + 10])
                        ious.append(iou)
                    iou = torch.cat(ious).numpy()
                    fg = np.nanmean(iou[:, 1:], axis=1)
                    rows.append({"model": name, "seed": s, "family": family, "severity": sev, "miou_fg": float(np.nanmean(fg)),
                                 **{f"iou_{c}": float(np.nanmean(iou[:, k])) for k, c in enumerate(["bg", "zp", "te", "bl", "icm"])}})
            print(name, "seed", s, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/paper/robustness.csv", index=False)
    plot(df)


def plot(df: pd.DataFrame) -> None:
    fams = list(CORRUPTIONS)
    fig, axes = plt.subplots(1, len(fams), figsize=(3.0 * len(fams), 3.0), sharey=True)
    for ax, fam in zip(axes, fams):
        for name, (_, _, col) in MODELS.items():
            d = df[(df.model == name)]
            clean = d[d.family == "clean"].groupby("seed").miou_fg.mean().mean()
            sub = d[d.family == fam].groupby("severity").miou_fg.agg(["mean", "std"]).reset_index()
            xs_ = [0] + sub.severity.tolist() if not fam.startswith(("brightness", "gamma", "contrast")) else sub.severity.tolist()
            ys = ([clean] if len(xs_) > len(sub) else []) + sub["mean"].tolist()
            ax.plot(xs_, ys, marker="o", color=col, label=style.display_name(name) if name != "U-Net" else name, lw=1.8, ms=4)
            if fam.startswith(("brightness", "gamma", "contrast")):
                ax.axhline(clean, color=col, lw=0.8, ls=":", alpha=0.6)
        ax.set_xlabel(fam)
        ax.grid(True, axis="y")
    axes[0].set_ylabel("mIoU-FG (50 val images)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/paper/S2_robustness.png")
    print("wrote results/paper/robustness.csv and figures/paper/S2_robustness.png")


if __name__ == "__main__":
    main()
