from __future__ import annotations

import json
import math
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import AppConfig
from .losses import HeatmapLoss
from .sliding_window import optimizer_steps_per_epoch


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: AppConfig,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = resolve_device(config.training.device)
        self.model = model.to(self.device)
        self.criterion = HeatmapLoss(
            pos_weight=config.training.pos_weight,
            bce_weight=config.training.bce_weight,
            focal_weight=config.training.focal_weight,
            dice_weight=config.training.dice_weight,
            focal_alpha=config.training.focal_alpha,
            focal_gamma=config.training.focal_gamma,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        steps_per_epoch = optimizer_steps_per_epoch(
            len(train_loader), config.training.accumulation_steps
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.training.learning_rate,
            total_steps=config.training.epochs * steps_per_epoch,
        )
        self.amp_enabled = config.training.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.output_dir = config.training.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = 1
        self.global_step = 0
        self.best_val_loss = math.inf
        self.best_epoch = 0

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "config": asdict(self.config),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

    def save_checkpoint(self, epoch: int, *, best: bool = False) -> Path:
        path = self.output_dir / ("best.pt" if best else "last.pt")
        _atomic_torch_save(self._checkpoint_payload(epoch), path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint["global_step"])
        self.best_val_loss = float(checkpoint["best_val_loss"])
        self.best_epoch = int(checkpoint["best_epoch"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        self.model.to(self.device)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

    def _run_training_epoch(self) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        samples = 0
        accumulation = self.config.training.accumulation_steps
        for batch_index, batch in enumerate(self.train_loader):
            volume = batch["volume"].to(self.device, non_blocking=True)
            heatmap = batch["heatmap"].to(self.device, non_blocking=True)
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(
                self.train_loader
            )
            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                logits = self.model(volume)
                loss, components = self.criterion(logits, heatmap)
                group_start = (batch_index // accumulation) * accumulation
                group_size = min(accumulation, len(self.train_loader) - group_start)
                scaled_loss = loss / group_size
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at training batch {batch_index}")
            self.scaler.scale(scaled_loss).backward()
            if should_step:
                self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip_norm
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"non-finite gradient at training batch {batch_index}")
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1
            batch_size = volume.shape[0]
            samples += batch_size
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
        return {name: value / samples for name, value in totals.items()}

    @torch.no_grad()
    def _run_validation(self) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        samples = 0
        for batch in self.val_loader:
            volume = batch["volume"].to(self.device, non_blocking=True)
            heatmap = batch["heatmap"].to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                logits = self.model(volume)
                loss, components = self.criterion(logits, heatmap)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            batch_size = volume.shape[0]
            samples += batch_size
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
        return {name: value / samples for name, value in totals.items()}

    def fit(self, *, resume: str | Path | None = None) -> dict[str, float | int]:
        seed_everything(self.config.training.seed)
        if resume is not None:
            self.load_checkpoint(resume)
        history_path = self.output_dir / "metrics.jsonl"
        stale_validations = 0
        for epoch in range(self.start_epoch, self.config.training.epochs + 1):
            train_metrics = self._run_training_epoch()
            val_metrics = self._run_validation()
            improved = val_metrics["total"] < self.best_val_loss
            if improved:
                self.best_val_loss = val_metrics["total"]
                self.best_epoch = epoch
                stale_validations = 0
            else:
                stale_validations += 1
            row = {
                "epoch": epoch,
                "global_step": self.global_step,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            self.save_checkpoint(epoch)
            if improved:
                self.save_checkpoint(epoch, best=True)
            patience = self.config.training.early_stopping_patience
            if patience and stale_validations >= patience:
                break
        return {"best_val_loss": self.best_val_loss, "best_epoch": self.best_epoch}
