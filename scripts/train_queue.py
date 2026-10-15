#!/usr/bin/env python3
"""Run independent training seeds concurrently across visible GPUs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/S0_5class_candidate.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--run-root", default="runs/queue_regularized")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not args.gpus and not args.allow_cpu:
        raise RuntimeError("No GPU was specified; use --allow-cpu for a CPU queue.")

    run_root = project_root / args.run_root
    gpu_ids = args.gpus or ["cpu"]
    commands = []
    for seed in args.seeds:
        run_dir = run_root / f"{config_path.stem}_seed{seed}"
        commands.append((seed, gpu_ids[len(commands) % len(gpu_ids)], run_dir))

    for seed, gpu_id, run_dir in commands:
        command = [
            sys.executable,
            str(project_root / "scripts/train.py"),
            "--config",
            str(config_path),
            "--seed",
            str(seed),
            "--run-dir",
            str(run_dir),
        ]
        if args.resume_existing and (run_dir / "best.pt").exists():
            command.extend(["--resume", str(run_dir / "best.pt")])
        print(f"seed={seed} gpu={gpu_id} run_dir={run_dir}")
        if args.dry_run:
            print(" ".join(command))

    if args.dry_run:
        return 0

    if not args.allow_cpu:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < len(args.gpus):
            raise RuntimeError(
                f"Requested {len(args.gpus)} GPU(s), but only {torch.cuda.device_count()} visible."
            )

    pending = list(commands)
    active: list[tuple[int, str, Path, subprocess.Popen[bytes]]] = []
    failed = False
    while pending or active:
        while pending and len(active) < len(gpu_ids):
            seed, gpu_id, run_dir = pending.pop(0)
            run_dir.mkdir(parents=True, exist_ok=True)
            log_file = (run_dir / "train.log").open("ab")
            environment = os.environ.copy()
            if gpu_id != "cpu":
                environment["CUDA_VISIBLE_DEVICES"] = gpu_id
            process_command = [
                sys.executable,
                str(project_root / "scripts/train.py"),
                "--config",
                str(config_path),
                "--seed",
                str(seed),
                "--run-dir",
                str(run_dir),
            ]
            if args.resume_existing and (run_dir / "best.pt").exists():
                process_command.extend(["--resume", str(run_dir / "best.pt")])
            process = subprocess.Popen(
                process_command,
                cwd=project_root,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            log_file.close()
            active.append((seed, gpu_id, run_dir, process))
        time.sleep(2)
        still_active = []
        for seed, gpu_id, run_dir, process in active:
            return_code = process.poll()
            if return_code is None:
                still_active.append((seed, gpu_id, run_dir, process))
            else:
                print(f"seed={seed} gpu={gpu_id} exit={return_code} run_dir={run_dir}")
                if return_code != 0:
                    failed = True
        active = still_active
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
