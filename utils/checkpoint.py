from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Checkpoint:
    epoch: int
    best_metric: float

def save_checkpoint(path: Path, model, optimizer, epoch: int, best_metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": float(best_metric),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        str(path),
    )

def load_checkpoint(path: Path, model, optimizer=None, map_location="cpu") -> Checkpoint:
    ckpt = torch.load(str(path), map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return Checkpoint(epoch=int(ckpt.get("epoch", 0)), best_metric=float(ckpt.get("best_metric", 0.0)))
