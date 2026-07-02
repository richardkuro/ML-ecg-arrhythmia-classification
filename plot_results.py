"""
plot_results.py
---------------
Generates all paper-ready figures from experiment results.

Outputs (saved to outputs/figures/):
  1. training_curves_{dataset}.png  — loss & F1 per epoch per model
  2. comparison_bar_{dataset}.png   — final metric comparison bar chart
  3. confusion_matrices_{dataset}.png — grid of confusion matrices
  4. roc_curves_{dataset}.png       — ROC curves (binary) or per-class (multi)
  5. param_vs_f1_{dataset}.png      — efficiency scatter: params vs F1
  6. radar_chart_{dataset}.png      — multi-metric radar per model
"""

import os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
})

PALETTE = [
    "#2563EB",  # resnet1d   – blue
    "#16A34A",  # tcn        – green
    "#9333EA",  # transformer – purple
    "#F59E0B",  # kan        – amber
    "#EF4444",  # convkan    – red
    "#0891B2",  # s4         – cyan
    "#EC4899",  # liquidnet  – pink
]


def _savefig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 1. Training curves ────────────────────────────────────────────────────────

def plot_training_curves(results: list, dataset: str, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f"Training Curves — {dataset.upper()}", fontweight="bold", fontsize=13)

    for i, res in enumerate(results):
        name    = res["model_name"]
        history = res["history"]
        epochs  = [h["epoch"] for h in history]
        color   = PALETTE[i % len(PALETTE)]

        tl = [h["train_loss"] for h in history]
        vl = [h["test_loss"]  for h in history]
        f1 = [h["f1_macro"]   for h in history]

        axes[0].plot(epochs, tl, linestyle="--", color=color, alpha=0.5, linewidth=1.2)
        axes[0].plot(epochs, vl, color=color, linewidth=1.8, label=name)
        axes[1].plot(epochs, f1, color=color, linewidth=1.8, label=name)

    for ax, ylabel, title in zip(
        axes,
        ["Cross-Entropy Loss", "Macro F1"],
        ["Loss (dashed=train, solid=test)", "Macro F1 (test)"]
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(ncol=2, framealpha=0.4)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, f"training_curves_{dataset}.png"))


# ── 2. Comparison bar chart ───────────────────────────────────────────────────

def plot_comparison_bar(results: list, dataset: str, out_dir: str):
    metrics  = ["accuracy", "f1_macro", "f1_weighted", "auc_roc"]
    labels   = ["Accuracy", "F1 Macro", "F1 Weighted", "AUC-ROC"]
    names    = [r["model_name"] for r in results]
    n_models = len(names)
    n_met    = len(metrics)

    fig, axes = plt.subplots(1, n_met, figsize=(3.5 * n_met, 5))
    fig.suptitle(f"Model Comparison — {dataset.upper()}", fontweight="bold", fontsize=13)

    for j, (met, lab) in enumerate(zip(metrics, labels)):
        values = [r["final_metrics"].get(met, 0) for r in results]
        bars   = axes[j].bar(names, values, color=PALETTE[:n_models], width=0.55,
                             edgecolor="white", linewidth=0.8)
        axes[j].set_ylim(max(0, min(values) - 0.05), 1.01)
        axes[j].set_title(lab)
        axes[j].set_xticks(range(n_models))
        axes[j].set_xticklabels(names, rotation=35, ha="right")
        axes[j].grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            axes[j].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.003,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, f"comparison_bar_{dataset}.png"))


# ── 3. Confusion matrices ─────────────────────────────────────────────────────

def plot_confusion_matrices(results: list, dataset: str, num_classes: int, out_dir: str):
    n = len(results)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle(f"Confusion Matrices — {dataset.upper()}", fontweight="bold", fontsize=13)

    for i, res in enumerate(results):
        cm  = np.array(res["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

        im = axes[i].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        axes[i].set_title(res["model_name"], fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
        axes[i].set_xticks(range(num_classes))
        axes[i].set_yticks(range(num_classes))

        for r in range(num_classes):
            for c in range(num_classes):
                color = "white" if cm_norm[r, c] > 0.6 else "black"
                axes[i].text(c, r, f"{cm_norm[r, c]:.2f}", ha="center", va="center",
                             fontsize=8, color=color)

        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    for j in range(len(results), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, f"confusion_matrices_{dataset}.png"))


# ── 4. ROC curves ─────────────────────────────────────────────────────────────

def plot_roc_curves(results: list, dataset: str, num_classes: int, out_dir: str):
    if num_classes == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title(f"ROC Curves — {dataset.upper()}", fontweight="bold")

        for i, res in enumerate(results):
            y_true = res["y_true"]
            y_prob = res["y_prob"][:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=PALETTE[i], linewidth=1.8,
                    label=f"{res['model_name']} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(loc="lower right"); ax.grid(alpha=0.3)
        _savefig(fig, os.path.join(out_dir, f"roc_curves_{dataset}.png"))

    else:
        # One subplot per model, all classes overlaid
        n      = len(results)
        ncols  = min(4, n)
        nrows  = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
        axes   = np.array(axes).flatten()
        fig.suptitle(f"Per-class ROC — {dataset.upper()}", fontweight="bold")

        class_colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

        for i, res in enumerate(results):
            y_true = res["y_true"]
            y_prob = res["y_prob"]
            y_bin  = label_binarize(y_true, classes=list(range(num_classes)))
            axes[i].set_title(res["model_name"], fontweight="bold")

            for c in range(num_classes):
                try:
                    fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
                    roc_auc     = auc(fpr, tpr)
                    axes[i].plot(fpr, tpr, color=class_colors[c], linewidth=1.4,
                                 label=f"C{c} ({roc_auc:.2f})")
                except Exception:
                    pass

            axes[i].plot([0, 1], [0, 1], "k--", linewidth=0.7)
            axes[i].set_xlabel("FPR"); axes[i].set_ylabel("TPR")
            axes[i].legend(loc="lower right", fontsize=7)
            axes[i].grid(alpha=0.3)

        for j in range(len(results), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        _savefig(fig, os.path.join(out_dir, f"roc_curves_{dataset}.png"))


# ── 5. Params vs F1 scatter ───────────────────────────────────────────────────

def plot_param_efficiency(results: list, param_counts: dict, dataset: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(f"Parameter Efficiency — {dataset.upper()}", fontweight="bold")

    for i, res in enumerate(results):
        name   = res["model_name"]
        params = param_counts.get(name, 0)
        f1     = res["final_metrics"].get("f1_macro", 0)

        ax.scatter(params / 1000, f1, color=PALETTE[i], s=120, zorder=5,
                   edgecolors="white", linewidths=1.5, label=name)
        ax.annotate(name, (params / 1000, f1),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Parameters (thousands)")
    ax.set_ylabel("Macro F1")
    ax.legend(ncol=2, framealpha=0.3, fontsize=8)
    ax.grid(alpha=0.3)
    _savefig(fig, os.path.join(out_dir, f"param_vs_f1_{dataset}.png"))


# ── 6. Radar chart ────────────────────────────────────────────────────────────

def plot_radar(results: list, dataset: str, out_dir: str):
    metrics = ["accuracy", "f1_macro", "f1_weighted", "auc_roc"]
    labels  = ["Accuracy", "F1 Macro", "F1 Weighted", "AUC-ROC"]
    N       = len(metrics)

    angles  = [2 * math.pi * i / N for i in range(N)] + [0]   # close loop
    xlabels = labels + [labels[0]]

    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, polar=True)
    ax.set_title(f"Multi-Metric Radar — {dataset.upper()}", fontweight="bold",
                 fontsize=13, pad=20)

    for i, res in enumerate(results):
        vals = [res["final_metrics"].get(m, 0) for m in metrics] + \
               [res["final_metrics"].get(metrics[0], 0)]
        ax.plot(angles, vals, color=PALETTE[i], linewidth=1.8,
                label=res["model_name"])
        ax.fill(angles, vals, color=PALETTE[i], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_tick_params(labelsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), framealpha=0.4)
    ax.grid(alpha=0.4)

    _savefig(fig, os.path.join(out_dir, f"radar_chart_{dataset}.png"))


# ── 7. Learning rate schedule visualisation ───────────────────────────────────

def plot_lr_schedule(results: list, dataset: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_title(f"Learning Rate Schedule — {dataset.upper()}", fontweight="bold")

    for i, res in enumerate(results[:1]):   # all models share the same schedule
        history = res["history"]
        epochs  = [h["epoch"] for h in history]
        lrs     = [h["lr"]    for h in history]
        ax.plot(epochs, lrs, color=PALETTE[i], linewidth=2)

    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.grid(alpha=0.3)
    _savefig(fig, os.path.join(out_dir, f"lr_schedule_{dataset}.png"))


# ── Master function ───────────────────────────────────────────────────────────

def generate_all_plots(results: list, dataset: str, num_classes: int,
                       param_counts: dict, out_dir: str = "outputs/figures"):
    print(f"\n[Plots] Generating figures for {dataset.upper()} ...")
    plot_training_curves(results, dataset, out_dir)
    plot_comparison_bar(results, dataset, out_dir)
    plot_confusion_matrices(results, dataset, num_classes, out_dir)
    plot_roc_curves(results, dataset, num_classes, out_dir)
    plot_param_efficiency(results, param_counts, dataset, out_dir)
    plot_radar(results, dataset, out_dir)
    plot_lr_schedule(results, dataset, out_dir)
    print(f"[Plots] Done. All figures in {out_dir}/")
