#!/usr/bin/env python3
"""Run one deterministic nnU-Net-v2 seed through its public Python API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random

import numpy as np
import torch
from matblast.provenance import write_run_metadata


def main() -> None:
    """Set every RNG before constructing the official nnU-Net trainer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", default="Dataset601_MATBLAST")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--trainer", default="nnUNetTrainer")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["MATBLAST_NNUNET_SEED"] = str(args.seed)
    os.environ["MATBLAST_DETERMINISTIC"] = "1"
    write_run_metadata(
        os.environ["nnUNet_results"],
        project_root,
        project_root / "configs/B9_nnunet_grouped_v1_metricfixed.yaml",
        args.seed,
    )
    from nnunetv2.run.run_training import run_training

    run_training(
        args.dataset,
        args.configuration,
        args.fold,
        trainer_class_name=args.trainer,
        device=torch.device("cuda"),
    )


if __name__ == "__main__":
    main()
