#!/usr/bin/env python3
"""Write per-fold copies of frozen configs for the 5-fold cross-validation (no other change)."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "data/splits/source_grouped_5fold_v1.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["configs/S24_pyramid_context_grouped_v1_metricfixed.yaml", "configs/S43_matblast_point_rend_grouped_v1.yaml", "configs/S54_matblast_bmask_boundary_grouped_v1.yaml"])
    args = ap.parse_args()
    stems = []
    for cfg_path in args.configs:
        text = Path(ROOT / cfg_path).read_text()
        header = "".join(line for line in text.splitlines(keepends=True) if line.startswith("#"))
        cfg = yaml.safe_load(text)
        stem = Path(cfg_path).stem.split("_grouped")[0]
        for k in range(5):
            c = dict(cfg); c["data"] = dict(cfg["data"]); c["data"]["split_json"] = SPLIT; c["data"]["fold"] = k
            name = f"CV5_{stem}_fold{k}"; c["run_dir"] = f"runs/{name}"
            out = ROOT / "configs" / f"{name}.yaml"
            out.write_text(f"# {name}: frozen copy of {Path(cfg_path).name}; only data.split_json/data.fold changed for 5-fold CV.\n" + header + yaml.safe_dump(c, sort_keys=False, default_flow_style=None, width=100))
            stems.append(name)
    print(" ".join(stems))


if __name__ == "__main__":
    main()
