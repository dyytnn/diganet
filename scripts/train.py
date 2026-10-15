#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from matblast.config import load_config
from matblast.data import build_dataloaders
from matblast.models import build_model
from matblast.provenance import write_run_metadata
from matblast.losses import (
    CombinedLoss,
    GeneralizedDiceLoss,
    I2CNetCrossEntropyLoss,
    WeightedSoftJaccardLoss,
)
from matblast.engine import Trainer, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.run_dir is not None:
        cfg["run_dir"] = args.run_dir
    project_root = Path(__file__).resolve().parents[1]
    write_run_metadata(
        cfg.get("run_dir", "runs/default"), project_root, args.config, cfg.get("seed", 0)
    )
    seed_everything(cfg.get("seed", 0))
    train_loader, val_loader = build_dataloaders(cfg, smoke_limit=20 if args.smoke else None)
    if args.smoke:
        cfg["training"]["epochs"] = min(2, cfg["training"].get("epochs", 2))
    model = build_model(cfg)
    # Place the model before constructing/loading the optimizer so resumed
    # optimizer moments are restored onto the same device as parameters.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_name = cfg["loss"].get("name", "combined")
    loss_builders = {
        "combined": lambda: CombinedLoss(
            **{key: value for key, value in cfg["loss"].items() if key != "name"}
        ),
        "btformer_wsja": WeightedSoftJaccardLoss,
        "i2cnet_ce": I2CNetCrossEntropyLoss,
        "mimics_generalized_dice": GeneralizedDiceLoss,
    }
    if loss_name not in loss_builders:
        raise ValueError(f"Unknown loss {loss_name!r}; choose from {sorted(loss_builders)}")
    loss = loss_builders[loss_name]()
    loss = loss.to(device)
    optimizer_name = cfg["training"].get("optimizer", "adamw").lower()
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError(f"Unknown optimizer {optimizer_name!r}; choose from ['adam', 'adamw']")
    optimizer = optimizer_class(
        model.parameters(),
        lr=cfg["training"].get("lr", 3e-4),
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
        eps=cfg["training"].get("optimizer_eps", 1e-8),
    )
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    start_epoch = 0
    best_score = None
    ema_init_state = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        # EMA checkpoints store the shadow under "model" and the live weights
        # under "model_raw"; older/non-EMA checkpoints only have "model".
        model.load_state_dict(checkpoint.get("model_raw", checkpoint["model"]))
        if "model_raw" in checkpoint:
            ema_init_state = checkpoint["model"]
        elif cfg["training"].get("ema_enabled", False):
            print("resume: checkpoint has no EMA shadow; EMA re-initialised from loaded weights")
        if "optimizer" in checkpoint:
            # Checkpoints produced by older runs may contain optimizer
            # moments tied to CPU parameter objects. Reusing them after a
            # CUDA resume can mix CPU moments with CUDA gradients, so resume
            # model/epoch state and intentionally start a fresh optimizer.
            print("resume: optimizer state skipped; initialized fresh on target device")
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_score = checkpoint.get("monitor_value")
    Trainer(
        model,
        loss,
        optimizer,
        run_dir=cfg.get("run_dir", "runs/default"),
        patience=cfg["training"].get("patience", 30),
        monitor=cfg["training"].get("monitor", "mdice_fg"),
        mode=cfg["training"].get("monitor_mode", "max"),
        ema_enabled=cfg["training"].get("ema_enabled", False),
        ema_decay=cfg["training"].get("ema_decay", 0.999),
        ema_warmup_steps=cfg["training"].get("ema_warmup_steps", 0),
        ema_init_state=ema_init_state,
    ).fit(
        train_loader,
        val_loader,
        cfg["training"].get("epochs", 200),
        cfg["training"].get("lr_policy", "constant"),
        cfg["training"].get("poly_power", 0.9),
        start_epoch=start_epoch,
        best_score=best_score,
    )


if __name__ == "__main__":
    main()
