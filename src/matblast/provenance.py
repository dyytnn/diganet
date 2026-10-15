"""Run provenance helpers required by the MAT-BLAST experiment protocol."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any


def current_commit(repo_root: str | Path) -> str:
    """Return the current Git commit hash.

    Args:
        repo_root: Path to the repository worktree.

    Returns:
        Full commit hash for the current checkout.

    Raises:
        RuntimeError: If the worktree is not a committed Git repository.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("A committed Git worktree is required before training") from exc
    return result.stdout.strip()


def write_run_metadata(
    run_dir: str | Path,
    repo_root: str | Path,
    config_path: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Write immutable provenance metadata for one training run.

    Args:
        run_dir: Directory where the run artifacts are written.
        repo_root: Repository worktree used for the run.
        config_path: YAML configuration used for the run.
        seed: Random seed used by the run.

    Returns:
        Metadata dictionary written to ``meta.json``.
    """
    config_file = Path(config_path)
    metadata: dict[str, Any] = {
        "commit": current_commit(repo_root),
        "config_path": str(config_file),
        "config_text": config_file.read_text(),
        "seed": int(seed),
    }
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
