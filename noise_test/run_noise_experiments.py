"""
run_noise_experiments.py
------------------------
Runs the same training pipeline as run_experiments.py, but with hardware
noise applied to the MIT-BIH training AND test sets to simulate two real
deployment scenarios:

  ad8232_esp32  — AD8232 AFE + ESP32-C3 internal 12-bit SAR ADC (noisy)
  ads1115       — AD8232 AFE + ADS1115 external 16-bit ADC       (cleaner)

Results are saved alongside the existing clean baseline:
  outputs/results/results_mitbih_ad8232.json
  outputs/results/results_mitbih_ads1115.json

Comparison plots (all three conditions) are written to:
  outputs/figures_noise/

Usage
-----
# Full retrain under both noise profiles (mirrors run_experiments.py):
  python run_noise_experiments.py --profile both --epochs 30

# Quick smoke-test (two models, 3 epochs):
  python run_noise_experiments.py --profile ad8232_esp32 --epochs 3 --models kan convkan

# Zero-shot eval: load clean checkpoints, test on noisy data (fast, no retrain):
  python run_noise_experiments.py --eval-only --profile both

# Custom seed / batch:
  python run_noise_experiments.py --profile both --epochs 30 --seed 0 --batch 256
"""

import os
import sys
import io
import json
import argparse
import random
import time

# Force UTF-8 output on Windows (avoids charmap UnicodeEncodeError)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from data_loader      import load_mitbih, ECGDataset, make_weighted_sampler
from models           import get_model, count_params
from trainer          import train_model, eval_epoch, FocalLoss, get_device
from plot_results     import generate_all_plots
from noise_simulator  import apply_noise, snr_db


# ── Constants ─────────────────────────────────────────────────────────────────

ALL_MODELS = ["resnet1d", "tcn", "transformer", "kan", "convkan", "s4", "bigru"]

PROFILE_LABEL = {
    "ad8232_esp32": "AD8232 + ESP32 ADC",
    "ads1115":      "AD8232 + ADS1115",
}
PROFILE_TAG = {
    "ad8232_esp32": "ad8232",
    "ads1115":      "ads1115",
}


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Noisy data helpers ────────────────────────────────────────────────────────

def make_noisy_loaders(X_train, X_test, y_train, y_test,
                       profile: str, seed: int,
                       batch_size: int, balance: bool, num_workers: int):
    """Apply noise offline then build DataLoaders."""
    print(f"  Applying '{profile}' noise to train ({len(X_train):,}) "
          f"and test ({len(X_test):,}) sets ...")

    X_tr_n = apply_noise(X_train, profile=profile, seed=seed)
    X_te_n = apply_noise(X_test,  profile=profile, seed=seed + 9999)

    snr_tr = snr_db(X_train, X_tr_n)
    snr_te = snr_db(X_test,  X_te_n)
    print(f"  Train SNR: {snr_tr:.1f} dB  |  Test SNR: {snr_te:.1f} dB")

    train_ds = ECGDataset(X_tr_n, y_train)
    test_ds  = ECGDataset(X_te_n, y_test)

    sampler = make_weighted_sampler(y_train) if balance else None
    shuffle = not balance

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, test_loader, snr_tr, snr_te


# ── Serialise & pretty-print results ─────────────────────────────────────────

def save_results(all_results: list, tag: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"results_mitbih_{tag}.json")
    serial = []
    for r in all_results:
        serial.append({
            "model_name":            r["model_name"],
            "history":               r["history"],
            "best_metrics":          r["best_metrics"],
            "final_metrics":         r["final_metrics"],
            "confusion_matrix":      r["confusion_matrix"],
            "classification_report": r["classification_report"],
        })
    with open(path, "w") as f:
        json.dump(serial, f, indent=2)
    print(f"[Results] Saved -> {path}")


def print_summary(all_results: list, tag: str):
    metrics = ["accuracy", "f1_macro", "f1_weighted", "auc_roc"]
    header  = f"{'Model':<14} | " + " | ".join(f"{m:>12}" for m in metrics)
    sep     = "-" * len(header)
    print(f"\n{'='*60}\n  RESULTS — mitbih_{tag.upper()}\n{'='*60}")
    print(header); print(sep)
    for r in all_results:
        row = f"{r['model_name']:<14} | "
        row += " | ".join(f"{r['final_metrics'].get(m, 0):>12.4f}" for m in metrics)
        print(row)
    print(sep)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ECG Noise Robustness Benchmark")
    p.add_argument("--profile",   type=str, default="both",
                   choices=["ad8232_esp32", "ads1115", "both"],
                   help="Noise profile(s) to test")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training: load clean checkpoints, eval on noisy test")
    p.add_argument("--models",    type=str, nargs="+", default=ALL_MODELS,
                   choices=ALL_MODELS, metavar="MODEL")
    p.add_argument("--epochs",    type=int,   default=30)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--batch",     type=int,   default=256)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--data-dir",  type=str,   default="data")
    p.add_argument("--out-dir",   type=str,   default="outputs")
    p.add_argument("--no-plots",  action="store_true")
    p.add_argument("--balance",   action="store_true", default=True)
    p.add_argument("--workers",   type=int,   default=2)
    return p.parse_args()


# ── Eval-only mode (zero-shot robustness of clean-trained models) ─────────────

def eval_only_run(profile: str, args, device: torch.device,
                  X_train, X_test, y_train, y_test):
    tag = PROFILE_TAG[profile]
    print(f"\n{'#'*60}\n#  EVAL-ONLY  |  profile: {profile}\n{'#'*60}")

    _, test_loader, snr_tr, snr_te = make_noisy_loaders(
        X_train, X_test, y_train, y_test,
        profile, args.seed, args.batch, args.balance, args.workers)

    num_classes  = 5
    criterion    = FocalLoss(gamma=2.0).to(device)
    clean_ckpt   = os.path.join(args.out_dir, "checkpoints", "mitbih")
    all_results  = []
    param_counts = {}

    for mname in args.models:
        ckpt_path = os.path.join(clean_ckpt, f"{mname}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] No checkpoint for '{mname}' at {ckpt_path}")
            continue

        model = get_model(mname, num_classes=num_classes).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        param_counts[mname] = count_params(model)

        print(f"  Evaluating {mname} on {PROFILE_LABEL[profile]} noise ...")
        metrics = eval_epoch(model, test_loader, criterion, device, num_classes)

        result = {
            "model_name":   mname,
            "history":      [],   # no training history in eval-only
            "best_metrics": ckpt.get("metrics", {}),
            "final_metrics": {k: v for k, v in metrics.items()
                              if k not in ("y_true", "y_pred", "y_prob")},
            "confusion_matrix":      [],
            "classification_report": "",
            "y_true": metrics["y_true"],
            "y_pred": metrics["y_pred"],
            "y_prob": metrics["y_prob"],
        }
        all_results.append(result)
        print(f"    acc={metrics['accuracy']:.4f}  "
              f"f1_macro={metrics['f1_macro']:.4f}  "
              f"auc={metrics['auc_roc']:.4f}")

    results_dir = os.path.join(args.out_dir, "results")
    save_results(all_results, tag, results_dir)
    print_summary(all_results, tag)
    return all_results, param_counts


# ── Full training run ─────────────────────────────────────────────────────────

def train_run(profile: str, args, device: torch.device,
              X_train, X_test, y_train, y_test):
    tag = PROFILE_TAG[profile]
    print(f"\n{'#'*60}\n#  TRAIN  |  profile: {profile}\n{'#'*60}")

    train_loader, test_loader, _, _ = make_noisy_loaders(
        X_train, X_test, y_train, y_test,
        profile, args.seed, args.batch, args.balance, args.workers)

    num_classes  = 5
    all_results  = []
    param_counts = {}
    ckpt_dir     = os.path.join(args.out_dir, f"checkpoints_noise", f"mitbih_{tag}")

    for mname in args.models:
        set_seed(args.seed)
        model  = get_model(mname, num_classes=num_classes)
        param_counts[mname] = count_params(model)

        result = train_model(
            model_name   = mname,
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

    results_dir = os.path.join(args.out_dir, "results")
    save_results(all_results, tag, results_dir)
    print_summary(all_results, tag)

    if not args.no_plots:
        figures_dir = os.path.join(args.out_dir, "figures_noise")
        generate_all_plots(
            results     = all_results,
            dataset     = f"mitbih_{tag}",
            num_classes = num_classes,
            param_counts= param_counts,
            out_dir     = figures_dir,
        )

    return all_results, param_counts


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    set_seed(args.seed)
    device = get_device()

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {props.total_memory / 1e9:.1f} GB  |  "
              f"Compute: {props.major}.{props.minor}")

    # Load raw (clean) data once -- noise is applied per profile
    print("\n[Data] Loading MIT-BIH (clean) ...")
    X_train, X_test, y_train, y_test = load_mitbih(args.data_dir)
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    profiles = (["ad8232_esp32", "ads1115"]
                if args.profile == "both"
                else [args.profile])

    all_profile_results = {}
    for prof in profiles:
        if args.eval_only:
            res, params = eval_only_run(
                prof, args, device, X_train, X_test, y_train, y_test)
        else:
            res, params = train_run(
                prof, args, device, X_train, X_test, y_train, y_test)
        all_profile_results[prof] = (res, params)

    # ── Generate 3-way comparison plots if both were run ──────────────────────
    if not args.no_plots and len(all_profile_results) == 2:
        from plot_noise_comparison import generate_comparison_plots
        clean_path = os.path.join(args.out_dir, "results", "results_mitbih.json")
        ad_path    = os.path.join(args.out_dir, "results", "results_mitbih_ad8232.json")
        ads_path   = os.path.join(args.out_dir, "results", "results_mitbih_ads1115.json")
        out_fig    = os.path.join(args.out_dir, "figures_noise")
        for p in [clean_path, ad_path, ads_path]:
            if not os.path.exists(p):
                print(f"[Comparison] Skipping — missing file: {p}")
                break
        else:
            generate_comparison_plots(clean_path, ad_path, ads_path, out_fig)

    print("\n[Done] Noise experiments complete.")
    print(f"  Results : {args.out_dir}/results/")
    print(f"  Figures : {args.out_dir}/figures_noise/")


if __name__ == "__main__":
    main()
