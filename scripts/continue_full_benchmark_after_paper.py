#!/usr/bin/env python3
"""Smoke-test and train canonical baselines after paper baselines complete."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


CANONICAL = (
    ("configs/B0_unet_grouped_v2_metricfixed.yaml", "runs/B0_unet_grouped_v2_metricfixed"),
    ("configs/B1_attention_unet_grouped_v2_metricfixed.yaml", "runs/B1_attention_unet_grouped_v2_metricfixed"),
    ("configs/B6_deeplabv3plus_grouped_v1_metricfixed.yaml", "runs/B6_deeplabv3plus_grouped_v1_metricfixed"),
    ("configs/B7_transunet_grouped_v1_metricfixed.yaml", "runs/B7_transunet_grouped_v1_metricfixed"),
    ("configs/B8_segformer_b0_grouped_v1_metricfixed.yaml", "runs/B8_segformer_b0_grouped_v1_metricfixed"),
)


def _complete(run_root: Path, config_stem: str, seed: int, final_epoch: int) -> bool:
    """Return whether one run reaches its expected final epoch."""
    path = run_root / f"{config_stem}_seed{seed}" / "history.json"
    if not path.exists():
        return False
    history = json.loads(path.read_text())
    return bool(history) and int(history[-1]["epoch"]) >= final_epoch


def _wait_for_paper_queue() -> None:
    """Wait for all MiMics-Net seeds, which prove the preceding queue succeeded."""
    root = Path("runs/B5_mimics_net_grouped_v3_metricfixed")
    stem = "B5_mimics_net_grouped_v3_metricfixed"
    while True:
        complete = [_complete(root, stem, seed, 63) for seed in (0, 1, 2)]
        if all(complete):
            print("Paper baseline queue complete; starting canonical models.", flush=True)
            return
        print(f"Waiting for MiMics-Net artifacts: {complete}", flush=True)
        time.sleep(30)


def _smoke(config: str) -> None:
    """Require a two-epoch smoke run to succeed before a full queue starts."""
    smoke_dir = Path("runs/smoke") / Path(config).stem
    subprocess.run(
        [sys.executable, "scripts/train.py", "--config", config, "--smoke", "--run-dir", str(smoke_dir)],
        check=True,
    )


def _train_queue(config: str, run_root: str) -> None:
    """Run one three-seed canonical queue on the two visible GPUs."""
    subprocess.run(
        [
            sys.executable,
            "scripts/train_queue.py",
            "--config",
            config,
            "--seeds",
            "0",
            "1",
            "2",
            "--gpus",
            "0",
            "1",
            "--run-root",
            run_root,
        ],
        check=True,
    )


def _train_nnunet_seed(seed: int, gpu: str, trainer: str = "nnUNetTrainer") -> subprocess.Popen[bytes]:
    """Start one official nnU-Net seed with isolated output state."""
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["nnUNet_raw"] = "/workspace/nnunet/raw"
    environment["nnUNet_preprocessed"] = "/workspace/nnunet/preprocessed"
    output_name = f"smoke_seed{seed}" if trainer != "nnUNetTrainer" else f"seed{seed}"
    environment["nnUNet_results"] = f"/workspace/runs/nnunet/{output_name}"
    return subprocess.Popen(
        [sys.executable, "scripts/train_nnunet_seed.py", "--seed", str(seed), "--trainer", trainer],
        env=environment,
    )


def _train_nnunet() -> None:
    """Smoke-test, then run the official 2D nnU-Net baseline for three seeds."""
    smoke = _train_nnunet_seed(0, "0", trainer="nnUNetTrainerBenchmark_5epochs")
    if smoke.wait() != 0:
        raise RuntimeError("nnU-Net five-epoch smoke test failed")
    active = [_train_nnunet_seed(0, "0"), _train_nnunet_seed(1, "1")]
    for process in active:
        if process.wait() != 0:
            raise RuntimeError("An nnU-Net seed failed")
    final = _train_nnunet_seed(2, "0")
    if final.wait() != 0:
        raise RuntimeError("nnU-Net seed 2 failed")


def main() -> None:
    """Run all remaining reproducible baselines in a resource-safe order."""
    parser = argparse.ArgumentParser()
    parser.parse_args()
    _wait_for_paper_queue()
    for config, run_root in CANONICAL:
        print(f"Smoke testing {config}", flush=True)
        _smoke(config)
        print(f"Training {config}", flush=True)
        _train_queue(config, run_root)
    print("Training official nnU-Net baseline.", flush=True)
    _train_nnunet()
    print("All baseline queues completed.", flush=True)


if __name__ == "__main__":
    main()
