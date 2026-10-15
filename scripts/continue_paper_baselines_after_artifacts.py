#!/usr/bin/env python3
"""Run follow-up paper baselines only after BTFormer artifacts are complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def _completed(run_root: Path, config_stem: str, seed: int, final_epoch: int) -> bool:
    """Return whether one queue run recorded its expected final epoch."""
    history_path = run_root / f"{config_stem}_seed{seed}" / "history.json"
    if not history_path.exists():
        return False
    history = json.loads(history_path.read_text())
    return bool(history) and int(history[-1]["epoch"]) >= final_epoch


def _wait_for_btformer(run_root: Path, config_stem: str, final_epoch: int) -> None:
    """Poll artifacts, never treating a merely idle GPU as completed training."""
    while True:
        complete = [_completed(run_root, config_stem, seed, final_epoch) for seed in (0, 1, 2)]
        if all(complete):
            print("BTFormer artifacts complete; starting I2CNet.", flush=True)
            return
        print(f"Waiting for BTFormer artifacts: {complete}", flush=True)
        time.sleep(30)


def _run_queue(config: str, run_root: str) -> None:
    """Run one three-seed queue on two visible GPUs and propagate failures."""
    command = [
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
    ]
    subprocess.run(command, check=True)


def main() -> None:
    """Wait for B3, then run B4 and B5 in order."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--b3-root", default="runs/B3_btformer_grouped_v2_metricfixed")
    parser.add_argument("--b3-config-stem", default="B3_btformer_grouped_v2_metricfixed")
    parser.add_argument("--b3-final-epoch", type=int, default=1999)
    args = parser.parse_args()
    _wait_for_btformer(Path(args.b3_root), args.b3_config_stem, args.b3_final_epoch)
    _run_queue("configs/B4_i2cnet_grouped_v2_metricfixed.yaml", "runs/B4_i2cnet_grouped_v2_metricfixed")
    _run_queue("configs/B5_mimics_net_grouped_v3_metricfixed.yaml", "runs/B5_mimics_net_grouped_v3_metricfixed")


if __name__ == "__main__":
    main()
