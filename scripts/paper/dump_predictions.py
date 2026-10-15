#!/usr/bin/env python3
"""Dump argmax predictions (256x256) of selected models on the 50 validation images.

Checkpoint paths are taken from the ``specs`` recorded in the evaluation reports, so the
exact evaluated weights are used. nnU-Net predictions are read from its fold_0/validation
PNGs (native resolution, resized to 256 with nearest neighbour) via nnunet case_map.json.
Output: results/paper/preds/<model>_seed<k>.npy (int8, 50x256x256), gt.npy, image_ids.json.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from matblast.config import load_config
from matblast.data.dataset import _read_rgb
from matblast.models import build_model

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "paper" / "preds"
ORDER = ["zp", "te", "bl", "icm"]


def gt_stack(ids):
    out = []
    for n in ids:
        lab = np.zeros((256, 256), np.int8)
        for i, c in enumerate(ORDER, 1):
            m = Image.open(ROOT / "dataset/source/masks" / c / n).convert("L").resize((256, 256), Image.NEAREST)
            lab[np.array(m) > 0] = i
        out.append(lab)
    return np.stack(out)


def collect_specs():
    specs = {}
    for path in glob.glob(str(ROOT / "results" / "*_validation_no_tta.json")):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        for s in d.get("specs", []):
            if "=" not in s or ":" not in s:
                continue
            name, rest = s.split("=", 1)
            cfg, ckpts = rest.split(":", 1)
            specs.setdefault(name, (cfg, ckpts.split(",")))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["S43", "S24", "S54_matblast_bmask_boundary_grouped_v1", "S37_residual_encoder",
                                                    "L01_mednext_grouped_v1_metricfixed", "B0", "B3", "B4", "B12_BLASTO_Net",
                                                    "S46_matblast_pointrend_faithful_grouped_v1", "S49_matblast_refine_mask_grouped_v1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = json.load(open(ROOT / "results/benchmark_all_validation_no_tta.json"))["validation_image_ids"]
    json.dump(ids, open(OUT / "image_ids.json", "w"))
    np.save(OUT / "gt.npy", gt_stack(ids))
    specs = collect_specs()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    image_cache: dict[tuple, torch.Tensor] = {}
    for name in args.models:
        if name not in specs:
            print("no spec for", name); continue
        cfg_path, ckpts = specs[name]
        cfg = load_config(ROOT / cfg_path)
        in_ch = int(cfg["data"].get("in_channels", cfg["model"].get("in_channels", 3)))
        rep = str(cfg["data"].get("input_representation", "legacy"))
        size = cfg["data"].get("image_size", 256)
        key = (in_ch, rep, str(size))
        if key not in image_cache:
            image_cache[key] = torch.stack([_read_rgb(ROOT / "dataset/source/images" / n, size, in_ch, rep) for n in ids])
        images = image_cache[key]
        for seed in args.seeds:
            target = OUT / f"{name}_seed{seed}.npy"
            if target.exists() or seed >= len(ckpts):
                continue
            model = build_model(cfg)
            ck = torch.load(ROOT / ckpts[seed], map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"]); model = model.to(dev).eval()
            preds = []
            with torch.no_grad():
                for i in range(0, len(ids), 10):
                    logits = model(images[i:i + 10].to(dev))["logits"]
                    if logits.shape[-2:] != (256, 256):
                        logits = torch.nn.functional.interpolate(logits, size=(256, 256), mode="bilinear", align_corners=False)
                    preds.append(logits.argmax(1).cpu().numpy().astype(np.int8))
            np.save(target, np.concatenate(preds)); print("saved", target.name)
    # nnU-Net from validation PNGs
    case_map = json.load(open(ROOT / "nnunet/raw/Dataset601_MATBLAST/case_map.json"))
    for seed in args.seeds:
        target = OUT / f"B9_nnunet_seed{seed}.npy"
        vdir = ROOT / f"runs/nnunet/seed{seed}/Dataset601_MATBLAST/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation"
        if target.exists() or not vdir.exists():
            continue
        preds = []
        for n in ids:
            p = Image.open(vdir / f"{case_map[n]}.png").resize((256, 256), Image.NEAREST)
            preds.append(np.array(p).astype(np.int8))
        np.save(target, np.stack(preds)); print("saved", target.name, "(nnU-Net label ids as stored)")
    print("nnU-Net label ids in dataset.json:", json.load(open(ROOT / "nnunet/raw/Dataset601_MATBLAST/dataset.json"))["labels"])


if __name__ == "__main__":
    main()
