#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from matblast.config import load_config
from matblast.data import build_dataloaders
from matblast.models import build_model
from matblast.metrics import per_image_scores, summarize_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--include-background", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["training"]["num_workers"] = args.num_workers
    _, loader = build_dataloaders(cfg)
    model = build_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dice_scores, iou_scores = [], []
    with torch.no_grad():
        for batch in loader:
            stage = batch.get("stage")
            output = (
                model(batch["image"], stage=stage) if stage is not None else model(batch["image"])
            )
            dice, iou = per_image_scores(output["logits"], batch["mask"])
            dice_scores.append(dice)
            iou_scores.append(iou)
    report = summarize_scores(
        torch.cat(dice_scores), torch.cat(iou_scores), args.include_background
    )
    class_names = cfg["data"].get("mask_classes", ["icm", "te", "zp"])
    for class_id, class_name in enumerate(class_names, 1):
        report[f"dice_{class_name}"] = report.pop(f"dice_class_{class_id}")
        report[f"iou_{class_name}"] = report.pop(f"iou_class_{class_id}")
    print(report)


if __name__ == "__main__":
    main()
