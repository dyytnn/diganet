from __future__ import annotations
from pathlib import Path
import json
import random
import numpy as np
import torch
from ..metrics import per_image_scores, summarize_scores
from .ema import ModelEMA


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        device="auto",
        run_dir="runs/default",
        patience=30,
        monitor="mdice_fg",
        mode="max",
        ema_enabled=False,
        ema_decay=0.999,
        ema_warmup_steps=0,
        ema_init_state=None,
    ):
        """Supervised trainer with early stopping on a validation metric.

        Args:
            ema_enabled: Keep an EMA shadow of the weights; validation and the saved
                ``best.pt["model"]`` then use the shadow, the live weights are stored
                under ``best.pt["model_raw"]``.
            ema_decay / ema_warmup_steps: See :class:`ModelEMA`.
            ema_init_state: Optional state dict to initialise the shadow from (resume).
        """
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optimizer
        self._move_optimizer_state_to_device()
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.ema = None
        if ema_enabled:
            self.ema = ModelEMA(self.model, ema_decay, ema_warmup_steps)
            if ema_init_state is not None:
                self.ema.module.load_state_dict(ema_init_state)

    def _move_optimizer_state_to_device(self) -> None:
        """Move tensor-valued optimizer state to the trainer device.

        Resumed checkpoints are loaded on CPU before the model is moved to
        CUDA. Adam's running moments must follow their corresponding model
        parameters or the first optimizer step fails with a device mismatch.
        """
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

    def _gradient_diagnostics(self):
        layer_norms = {}
        total_squared = 0.0
        max_abs = 0.0
        with_grad = 0
        without_grad = 0
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                without_grad += 1
                continue
            with_grad += 1
            norm = float(parameter.grad.detach().norm().item())
            layer = name.split(".")[0]
            layer_norms[layer] = layer_norms.get(layer, 0.0) + norm * norm
            total_squared += norm * norm
            max_abs = max(max_abs, float(parameter.grad.detach().abs().max().item()))
        return {
            "grad_norm": total_squared**0.5,
            "grad_max_abs": max_abs,
            "params_with_grad": with_grad,
            "params_without_grad": without_grad,
            "layer_grad_norms": {name: value**0.5 for name, value in layer_norms.items()},
        }

    def _epoch(self, loader, train):
        self.model.train(train)
        network = self.model if train or self.ema is None else self.ema.module
        total = 0.0
        dice_scores = []
        iou_scores = []
        diagnostics = []
        for batch in loader:
            x, y = batch["image"].to(self.device), batch["mask"].to(self.device)
            stage = batch.get("stage")
            if stage is not None:
                stage = stage.to(self.device)
            with torch.set_grad_enabled(train):
                out = network(x, stage=stage) if stage is not None else network(x)
                loss = self.loss_fn(out, y)["total_loss"]
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    diagnostics.append(self._gradient_diagnostics())
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    if self.ema is not None:
                        self.ema.update(self.model)
            total += loss.item()
            dice, iou = per_image_scores(out["logits"].detach(), y)
            dice_scores.append(dice.cpu())
            iou_scores.append(iou.cpu())
        mean = (
            summarize_scores(torch.cat(dice_scores), torch.cat(iou_scores)) if dice_scores else {}
        )
        if diagnostics:
            mean["grad_norm"] = float(np.mean([item["grad_norm"] for item in diagnostics]))
            mean["grad_max_abs"] = float(np.max([item["grad_max_abs"] for item in diagnostics]))
            mean["params_without_grad"] = max(item["params_without_grad"] for item in diagnostics)
            layer_names = sorted(
                {name for item in diagnostics for name in item["layer_grad_norms"]}
            )
            mean["layer_grad_norms"] = {
                name: float(
                    np.mean([item["layer_grad_norms"].get(name, 0.0) for item in diagnostics])
                )
                for name in layer_names
            }
        return total / max(1, len(loader)), mean

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        lr_policy="constant",
        poly_power=0.9,
        start_epoch=0,
        best_score=None,
    ):
        best = (
            (-(float("inf")) if self.mode == "max" else float("inf"))
            if best_score is None
            else best_score
        )
        stale = 0
        history = []
        for epoch in range(start_epoch, epochs):
            if lr_policy == "poly":
                factor = max(0.0, 1.0 - epoch / max(1, epochs)) ** poly_power
                for group in self.optimizer.param_groups:
                    group["lr"] = group.get("initial_lr", group["lr"]) * factor
            tr, tm = self._epoch(train_loader, True)
            va, vm = self._epoch(val_loader, False)
            score = vm.get(self.monitor, float("nan"))
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": tr,
                    "val_loss": va,
                    "train_metrics": tm,
                    "val_metrics": vm,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
            )
            print(
                f"epoch {epoch + 1}/{epochs} train={tr:.4f} val={va:.4f} monitor={self.monitor}={score:.4f} grad={tm.get('grad_norm', float('nan')):.4f}"
            )
            improved = score > best if self.mode == "max" else score < best
            if not np.isfinite(score):
                improved = False
            if improved:
                best = score
                stale = 0
                checkpoint = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": vm,
                    "monitor": self.monitor,
                    "monitor_value": score,
                }
                if self.ema is not None:
                    # Evaluation scripts read "model": give them the EMA weights that
                    # produced the validation score; keep the live weights for resume.
                    checkpoint["model"] = self.ema.module.state_dict()
                    checkpoint["model_raw"] = self.model.state_dict()
                    checkpoint["ema"] = self.ema.state_dict()
                torch.save(checkpoint, self.run_dir / "best.pt")
            else:
                stale += 1
            if stale >= self.patience:
                break
        (self.run_dir / "history.json").write_text(json.dumps(history, indent=2))
        return history
