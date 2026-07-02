"""
run_experiments.py
------------------
Main entry point. Trains all models on MIT-BIH and/or PTB-DB,
saves results, checkpoints, and generates all paper figures.

Usage
-----
# Full run (both datasets, all models, 30 epochs):
  python run_experiments.py

# Quick smoke-test (3 epochs, 2 models):
  python run_experiments.py --epochs 3 --models resnet1d kan --dataset mitbih

# Only PTB, skip figure generation:
  python run_experiments.py --dataset ptbdb --no-plots

# Reproduce with exact seed:
  python run_experiments.py --seed 42
"""

import os
import json
import argparse
import random
import numpy as np
import torch

from data_loader import get_loaders
from models      import get_model, count_params
from trainer     import train_model, get_device
from plot_results import generate_all_plots


# ── All models to benchmark ───────────────────────────────────────────────────

ALL_MODELS = ["resnet1d", "tcn", "transformer", "kan", "convkan", "s4", "bigru"]


# ── Seed ─────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Save results ─────────────────────────────────────────────────────────────

def save_results(all_results: list, dataset: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"results_{dataset}.json")

    # JSON-serialisable subset (exclude np arrays)
    serialisable = []
    for r in all_results:
        serialisable.append({
            "model_name":            r["model_name"],
            "history":               r["history"],
            "best_metrics":          r["best_metrics"],
            "final_metrics":         r["final_metrics"],
            "confusion_matrix":      r["confusion_matrix"],
            "classification_report": r["classification_report"],
        })

    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n[Results] Saved to {path}")


def print_summary_table(all_results: list, dataset: str):
    metrics = ["accuracy", "f1_macro", "f1_weighted", "auc_roc"]
    header  = f"{'Model':<14} | " + " | ".join(f"{m:>12}" for m in metrics)
    sep     = "-" * len(header)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — {dataset.upper()}")
    print(f"{'='*60}")
    print(header)
    print(sep)

    for r in all_results:
        row = f"{r['model_name']:<14} | "
        row += " | ".join(
            f"{r['final_metrics'].get(m, 0):>12.4f}" for m in metrics
        )
        print(row)

    print(sep)

    # Best model per metric
    for m in metrics:
        best = max(all_results, key=lambda r: r["final_metrics"].get(m, 0))
        print(f"  Best {m:<14}: {best['model_name']}  ({best['final_metrics'].get(m, 0):.4f})")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ECG Deep Learning Benchmark")
    p.add_argument("--dataset",  type=str, default="both",
                   choices=["mitbih", "ptbdb", "both"],
                   help="Which dataset to use")
    p.add_argument("--models",   type=str, nargs="+", default=ALL_MODELS,
                   choices=ALL_MODELS, metavar="MODEL",
                   help=f"Models to train. Choices: {ALL_MODELS}")
    p.add_argument("--epochs",   type=int,   default=30)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--batch",    type=int,   default=256)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--data-dir", type=str,   default="data")
    p.add_argument("--out-dir",  type=str,   default="outputs")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip figure generation")
    p.add_argument("--balance",  action="store_true", default=True,
                   help="Use weighted sampler to balance classes")
    p.add_argument("--workers",  type=int, default=2,
                   help="DataLoader num_workers")
    return p.parse_args()


# ── Per-dataset experiment ────────────────────────────────────────────────────

def run_dataset(dataset: str, args, device: torch.device):
    print(f"\n{'#'*60}")
    print(f"#  Dataset: {dataset.upper()}")
    print(f"{'#'*60}")

    train_loader, test_loader, num_classes = get_loaders(
        dataset    = dataset,
        data_dir   = args.data_dir,
        batch_size = args.batch,
        balance    = args.balance,
        num_workers= args.workers,
    )

    all_results  = []
    param_counts = {}
    ckpt_dir     = os.path.join(args.out_dir, "checkpoints", dataset)

    for model_name in args.models:
        set_seed(args.seed)   # reset seed per model for fair comparison

        model  = get_model(model_name, num_classes=num_classes)
        params = count_params(model)
        param_counts[model_name] = params

        result = train_model(
            model_name   = model_name,
            model        = model,
            train_loader = train_loader,
            test_loader  = test_loader,
            num_classes  = num_classes,
            device       = device,
            epochs       = args.epochs,
            lr           = args.lr,
            save_dir     = ckpt_dir,
            use_focal    = True,
        )
        all_results.append(result)

    # ── Outputs ──
    results_dir = os.path.join(args.out_dir, "results")
    figures_dir = os.path.join(args.out_dir, "figures")

    save_results(all_results, dataset, results_dir)
    print_summary_table(all_results, dataset)

    if not args.no_plots:
        generate_all_plots(
            results      = all_results,
            dataset      = dataset,
            num_classes  = num_classes,
            param_counts = param_counts,
            out_dir      = figures_dir,
        )

    return all_results


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device()

    # Print GPU memory info
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  VRAM: {props.total_memory / 1e9:.1f} GB  |  "
              f"SM count: {props.multi_processor_count}  |  "
              f"Compute: {props.major}.{props.minor}")
        print(f"  AMP (mixed precision) ENABLED — faster training on Ampere")

    datasets = ["mitbih", "ptbdb"] if args.dataset == "both" else [args.dataset]

    for ds in datasets:
        run_dataset(ds, args, device)

    print("\n[Done] All experiments complete.")
    print(f"  Results : {args.out_dir}/results/")
    print(f"  Figures : {args.out_dir}/figures/")
    print(f"  Ckpts   : {args.out_dir}/checkpoints/")


if __name__ == "__main__":
    main()
