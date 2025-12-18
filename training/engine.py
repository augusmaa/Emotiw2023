from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader

@dataclass
class EpochResult:
    loss: float
    acc: float

def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

def train_one_epoch(model, loader: DataLoader, optimizer, criterion, device: torch.device, mode: str) -> EpochResult:
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        if mode == "audio":
            spec, y = batch
            spec, y = spec.to(device), y.to(device)
            logits = model(spec)
        elif mode == "video":
            img, _, y = batch
            img, y = img.to(device), y.to(device)
            logits = model(img)
        else:
            img, spec, y = batch
            img, spec, y = img.to(device), spec.to(device), y.to(device)
            logits = model(img, spec)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += _accuracy(logits.detach(), y) * bs
        n += bs

    return EpochResult(loss=total_loss / max(n, 1), acc=total_acc / max(n, 1))

@torch.no_grad()
def validate(model, loader: DataLoader, criterion, device: torch.device, mode: str) -> EpochResult:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for batch in loader:
        if mode == "audio":
            spec, y = batch
            spec, y = spec.to(device), y.to(device)
            logits = model(spec)
        elif mode == "video":
            img, _, y = batch
            img, y = img.to(device), y.to(device)
            logits = model(img)
        else:
            img, spec, y = batch
            img, spec, y = img.to(device), spec.to(device), y.to(device)
            logits = model(img, spec)

        loss = criterion(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += _accuracy(logits, y) * bs
        n += bs

    return EpochResult(loss=total_loss / max(n, 1), acc=total_acc / max(n, 1))
