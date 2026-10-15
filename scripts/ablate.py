#!/usr/bin/env python3
"""Run a list of config files sequentially, preserving one run directory per config."""
import argparse, subprocess, sys

def main():
    p=argparse.ArgumentParser(); p.add_argument("configs",nargs="+"); a=p.parse_args()
    for config in a.configs: subprocess.run([sys.executable,"scripts/train.py","--config",config],check=True)
if __name__=="__main__": main()
