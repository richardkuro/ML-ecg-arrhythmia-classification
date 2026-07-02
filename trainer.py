"""
trainer.py
----------
Training loop, evaluation, and metric collection.
Saves per-epoch history and best model checkpoints.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize


# ── Device setup ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[CUDA] Using GPU: {torch.cuda.get_device_name(0)}")
        # Enable TF32 for Ampere+ GPUs (3050Ti is Ampere) → faster matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True   # auto-tune kernels
    else:
        dev = torch.device("cpu")
        print("[CPU] CUDA not available, using CPU.")
    return dev


# ── Loss: Focal loss for class imbalance ─────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for handling extreme class imbalance (MIT-BIH).
    alpha: per-class weights (optional)
    gamma: focusing parameter (2.0 standard)
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt   = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()


import torch.nn.functional as F


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray, num_classes: int) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1w  = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # ROC-AUC (one-vs-rest, macro)
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            auc   = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    return {"accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w, "auc_roc": auc}


# ── Train / eval steps ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, n = 0.0, 0

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:                         # AMP
            with torch.autocast(device_type="cuda"):
                logits = model(X)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * len(y)
        n          += len(y)

    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss, n = 0.0, 0

    for X, y in loader:
        X, y   = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        loss   = criterion(logits, y)

        probs  = torch.softmax(logits, dim=-1)
        preds  = probs.argmax(dim=-1)

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        total_loss += loss.item() * len(y)
        n          += len(y)

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes)
    metrics["loss"] = total_loss / n
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_prob"] = y_prob
    return metrics


# ── Main training function ────────────────────────────────────────────────────

def train_model(model_name: str,
                model: nn.Module,
                train_loader,
                test_loader,
                num_classes: int,
                device: torch.device,
                epochs: int = 30,
                lr: float = 1e-3,
                weight_decay: float = 1e-4,
                save_dir: str = "outputs/checkpoints",
                use_focal: bool = True):
    """
    Full training loop with:
    - AdamW optimizer
    - Cosine annealing LR schedule
    - Focal loss (optional)
    - Automatic Mixed Precision (CUDA only)
    - Best-model checkpointing
    - Per-epoch history logging
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    criterion = FocalLoss(gamma=2.0) if use_focal else nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # AMP only on CUDA
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = []
    best_f1  = -1.0
    best_ckpt = os.path.join(save_dir, f"{model_name}_best.pt")

    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}  |  {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")
    print(f"  Device: {device}  |  Epochs: {epochs}  |  LR: {lr}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        test_mets  = eval_epoch(model, test_loader, criterion, device, num_classes)
        scheduler.step()

        elapsed = time.time() - t0
        rec = {
            "epoch":       epoch,
            "train_loss":  round(train_loss,             4),
            "test_loss":   round(test_mets["loss"],      4),
            "accuracy":    round(test_mets["accuracy"],  4),
            "f1_macro":    round(test_mets["f1_macro"],  4),
            "f1_weighted": round(test_mets["f1_weighted"], 4),
            "auc_roc":     round(test_mets["auc_roc"],   4),
            "lr":          round(scheduler.get_last_lr()[0], 6),
            "time_s":      round(elapsed, 1),
        }
        history.append(rec)

        # Checkpoint on best macro-F1
        if rec["f1_macro"] > best_f1:
            best_f1 = rec["f1_macro"]
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "metrics":    rec,
            }, best_ckpt)
            star = "*"
        else:
            star = ""

        print(f"  Ep {epoch:02d}/{epochs} | "
              f"loss {rec['train_loss']:.4f}->{rec['test_loss']:.4f} | "
              f"acc {rec['accuracy']:.4f} | "
              f"f1_mac {rec['f1_macro']:.4f} | "
              f"auc {rec['auc_roc']:.4f} | "
              f"{elapsed:.1f}s {star}")

    # Load best weights for final eval
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    final_mets = eval_epoch(model, test_loader, criterion, device, num_classes)

    report = classification_report(
        final_mets["y_true"], final_mets["y_pred"], zero_division=0
    )
    cm = confusion_matrix(final_mets["y_true"], final_mets["y_pred"])

    print(f"\n[{model_name}] Best epoch {ckpt['epoch']}  F1-macro={best_f1:.4f}")
    print(report)

    return {
        "model_name":   model_name,
        "history":      history,
        "best_metrics": ckpt["metrics"],
        "final_metrics": {k: v for k, v in final_mets.items()
                          if k not in ("y_true", "y_pred", "y_prob")},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "y_true": final_mets["y_true"],
        "y_pred": final_mets["y_pred"],
        "y_prob": final_mets["y_prob"],
    }
