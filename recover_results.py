"""
recover_results.py
------------------
Loads all saved checkpoints, runs full evaluation on the test set,
and regenerates every figure and results JSON.

Place this file in your project folder (alongside models.py etc.) and run:
    python recover_results.py

Make sure your checkpoints are in:
    outputs/checkpoints/mitbih/   (or change CKPT_DIR below)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import get_loaders
from models      import get_model, count_params
from trainer     import get_device, eval_epoch, FocalLoss
from plot_results import generate_all_plots

# ── Config ────────────────────────────────────────────────────────────────────

CKPT_DIR    = "outputs/checkpoints/mitbih"   # folder with *_best.pt files
RESULTS_DIR = "outputs/results"
FIGURES_DIR = "outputs/figures"
DATA_DIR    = "data"
BATCH_SIZE  = 256
DATASET     = "mitbih"

# Models to recover — edit this list if you're missing any checkpoint
MODEL_NAMES = ["resnet1d", "tcn", "transformer", "kan", "convkan", "s4", "bigru"]

# ── BiGRU definition (add here so models.py doesn't need editing) ─────────────
# If you already added BiGRU to models.py, this block is harmlessly redundant.
try:
    from models import BiGRU
except ImportError:
    class BiGRU(nn.Module):
        def __init__(self, num_classes=5, input_size=1,
                     hidden_size=128, num_layers=2, dropout=0.3):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if num_layers > 1 else 0.0)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size * 2),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, num_classes),
            )
        def forward(self, x):
            x = x.permute(0, 2, 1)
            out, _ = self.gru(x)
            return self.head(out.mean(dim=1))

    # Patch into get_model registry
    import models as _models
    _models.BiGRU = BiGRU
    _orig_get = _models.get_model
    def _patched_get(name, num_classes, **kwargs):
        if name == "bigru":
            return BiGRU(num_classes=num_classes, **kwargs)
        return _orig_get(name, num_classes, **kwargs)
    _models.get_model = _patched_get

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = get_device()

    # Load test data
    _, test_loader, num_classes = get_loaders(
        dataset     = DATASET,
        data_dir    = DATA_DIR,
        batch_size  = BATCH_SIZE,
        balance     = False,   # no need to balance for eval
        num_workers = 2,
    )

    criterion = FocalLoss(gamma=2.0)

    all_results  = []
    param_counts = {}

    for model_name in MODEL_NAMES:
        ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] No checkpoint found for {model_name} at {ckpt_path}")
            continue

        print(f"\n[Evaluating] {model_name} ...")

        # Build model and load weights
        from models import get_model
        model = get_model(model_name, num_classes=num_classes)
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device)

        params = count_params(model)
        param_counts[model_name] = params

        # Full evaluation
        metrics = eval_epoch(model, test_loader, criterion, device, num_classes)

        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(metrics["y_true"], metrics["y_pred"], zero_division=0)
        cm     = confusion_matrix(metrics["y_true"], metrics["y_pred"])

        print(f"  acc={metrics['accuracy']:.4f}  f1_mac={metrics['f1_macro']:.4f}  "
              f"auc={metrics['auc_roc']:.4f}  params={params:,}")
        print(report)

        # Reconstruct history from checkpoint (single point — best epoch)
        best_ep = ckpt.get("epoch", 1)
        bm      = ckpt.get("metrics", {})
        # Build a minimal history so plots don't break
        # (only best-epoch point available without rerunning)
        history = [{
            "epoch":       ep,
            "train_loss":  bm.get("train_loss", 0.0) if ep == best_ep else None,
            "test_loss":   bm.get("test_loss",  metrics["loss"]),
            "accuracy":    bm.get("accuracy",   metrics["accuracy"]),
            "f1_macro":    bm.get("f1_macro",   metrics["f1_macro"]),
            "f1_weighted": bm.get("f1_weighted",metrics["f1_weighted"]),
            "auc_roc":     bm.get("auc_roc",    metrics["auc_roc"]),
            "lr":          0.001,
            "time_s":      0.0,
        } for ep in range(1, best_ep + 1)]

        result = {
            "model_name":            model_name,
            "history":               history,
            "best_metrics":          bm,
            "final_metrics":         {k: v for k, v in metrics.items()
                                      if k not in ("y_true", "y_pred", "y_prob")},
            "confusion_matrix":      cm.tolist(),
            "classification_report": report,
            "y_true":                metrics["y_true"],
            "y_pred":                metrics["y_pred"],
            "y_prob":                metrics["y_prob"],
        }
        all_results.append(result)

    # ── Save JSON ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    serialisable = [{k: v for k, v in r.items()
                     if k not in ("y_true", "y_pred", "y_prob")}
                    for r in all_results]
    json_path = os.path.join(RESULTS_DIR, f"results_{DATASET}.json")
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n[Saved] {json_path}")

    # ── Print summary table ──
    metrics_list = ["accuracy", "f1_macro", "f1_weighted", "auc_roc"]
    header = f"{'Model':<14} | " + " | ".join(f"{m:>12}" for m in metrics_list)
    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS — {DATASET.upper()}")
    print(f"{'='*65}")
    print(header)
    print("-" * len(header))
    for r in all_results:
        row = f"{r['model_name']:<14} | "
        row += " | ".join(f"{r['final_metrics'].get(m, 0):>12.4f}"
                          for m in metrics_list)
        print(row)
    print("-" * len(header))

    # ── Regenerate all figures ──
    generate_all_plots(
        results      = all_results,
        dataset      = DATASET,
        num_classes  = num_classes,
        param_counts = param_counts,
        out_dir      = FIGURES_DIR,
    )

    print("\n[Done] All figures regenerated in", FIGURES_DIR)


if __name__ == "__main__":
    main()
