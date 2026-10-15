#!/usr/bin/env python3
"""Create a lightweight image manifest for split generation.

Metadata can be supplied as JSON (a list of records) to preserve patient and
embryo grouping. Without metadata, the filename prefix before the first `_`
is used as a conservative grouping fallback and must be audited manually.
"""
import argparse, json
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--image-dir", required=True); p.add_argument("--output", required=True); args = p.parse_args()
    root = Path(args.image_dir)
    records = [{"image": x.name, "patient_id": x.stem.split("_")[0], "image_path": str(x)} for x in sorted(root.rglob("*")) if x.suffix.lower() in EXTS]
    Path(args.output).write_text(json.dumps(records, indent=2)); print(f"wrote {len(records)} records to {args.output}")

if __name__ == "__main__": main()
