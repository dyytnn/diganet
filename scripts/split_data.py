#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from matblast.data.splits import save_split

def main():
    p=argparse.ArgumentParser(); p.add_argument("--image-dir",required=True); p.add_argument("--output",required=True); p.add_argument("--metadata"); p.add_argument("--n-splits",type=int,default=5); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
    if a.metadata: records=json.loads(Path(a.metadata).read_text())
    else: records=[{"image":x.name,"patient_id":x.stem.split("_")[0]} for x in sorted(Path(a.image_dir).glob("*")) if x.is_file()]
    save_split(records,a.output,a.n_splits,a.seed); print(f"wrote {a.output} ({len(records)} images)")
if __name__=="__main__": main()
