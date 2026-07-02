"""
plot_noise_comparison.py
------------------------
Generates publication-ready comparison figures across three hardware conditions:
  1. Clean (original MIT-BIH)
  2. AD8232 + ESP32-C3 internal ADC   (budget / noisy)
  3. AD8232 + ADS1115 external ADC    (improved / cleaner)

Outputs (saved to outputs/figures_noise/):
  noise_f1_comparison.png       — grouped bar chart F1-Macro per model × condition
  noise_accuracy_comparison.png — same for Accuracy
  noise_auc_comparison.png      — same for AUC-ROC
  noise_degradation_heatmap.png — % F1 drop from clean, model × condition
  noise_sample_beats.png        — visual of a synthetic ECG under each profile
  noise_robustness_scatter.png  — model size vs. F1 drop (who degrades least?)
  noise_confusion_ad8232.png    — confusion matrices under AD8232 noise
  noise_combined_summary.png    — all-in-one summary figure

Usage
-----
  python plot_noise_comparison.py \
      --clean  outputs/results/results_mitbih.json \
      --ad8232 outputs/results/results_mitbih_ad8232.json \
      --ads1115 outputs/results/results_mitbih_ads1115.json \
      --out    outputs/figures_noise
"""

import os
import json
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from noise_simulator import apply_noise, PROFILES, SEQ_LEN


# ── Style ─────────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
})

# Condition colours  (clean / ad8232 / ads1115)
COND_COLORS  = ["#2563EB", "#EF4444", "#16A34A"]
COND_LABELS  = ["Clean (MIT-BIH)", "AD8232 + ESP32 ADC", "AD8232 + ADS1115"]
COND_HATCHES = ["", "//", ".."]

# Model colours
MODEL_PALETTE = [
    "#2563EB", "#16A34A", "#9333EA", "#F59E0B",
    "#EF4444", "#0891B2", "#EC4899",
]


def _savefig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_json(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def get_metric(results: list, model: str, metric: str) -> float:
    for r in results:
        if r["model_name"] == model:
            return r["final_metrics"].get(metric, float("nan"))
    return float("nan")


# ── 1. Grouped bar chart (one metric) ─────────────────────────────────────────

def plot_metric_bar(clean, ad8232, ads1115, metric: str, ylabel: str,
                    out_path: str):
    models = [r["model_name"] for r in clean]
    M = len(models)
    x = np.arange(M)
    w = 0.25

    cond_data = [
        ("Clean",              clean,  COND_COLORS[0], COND_HATCHES[0]),
        ("AD8232 + ESP32 ADC", ad8232, COND_COLORS[1], COND_HATCHES[1]),
        ("AD8232 + ADS1115",   ads1115,COND_COLORS[2], COND_HATCHES[2]),
    ]

    fig, ax = plt.subplots(figsize=(max(9, M * 1.5), 5))
    ax.set_title(f"{ylabel} — Clean vs Hardware Noise", fontweight="bold")

    for k, (label, results, color, hatch) in enumerate(cond_data):
        vals = [get_metric(results, m, metric) for m in models]
        offset = (k - 1) * w
        bars = ax.bar(x + offset, vals, w, label=label,
                      color=color, hatch=hatch,
                      edgecolor="white", linewidth=0.6, alpha=0.9)
        for bar, val in zip(bars, vals):
            if not math.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    all_vals = [get_metric(r, m, metric)
                for r in [clean, ad8232, ads1115]
                for m in models
                if not math.isnan(get_metric(r, m, metric))]
    ax.set_ylim(max(0, min(all_vals) - 0.06), 1.01)
    ax.legend(loc="lower right", framealpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, out_path)


# ── 2. Degradation heatmap ────────────────────────────────────────────────────

def plot_degradation_heatmap(clean, ad8232, ads1115, out_path: str):
    models  = [r["model_name"] for r in clean]
    conds   = ["AD8232 + ESP32 ADC", "AD8232 + ADS1115"]
    results = [ad8232, ads1115]
    metric  = "f1_macro"

    data = np.zeros((len(models), 2))
    for i, m in enumerate(models):
        base = get_metric(clean, m, metric)
        for j, res in enumerate(results):
            val = get_metric(res, m, metric)
            data[i, j] = (base - val) * 100  # percentage-point drop

    cmap = LinearSegmentedColormap.from_list(
        "rg", ["#16A34A", "#FEFCE8", "#EF4444"])

    fig, ax = plt.subplots(figsize=(5, max(4, len(models) * 0.6)))
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=max(10, data.max()))

    ax.set_xticks([0, 1]); ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
    ax.set_title("F1-Macro Degradation (pp) vs Clean Baseline",
                 fontweight="bold")

    for i in range(len(models)):
        for j in range(2):
            text_col = "white" if data[i, j] > data.max() * 0.65 else "black"
            ax.text(j, i, f"{data[i, j]:.2f}pp", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_col)

    plt.colorbar(im, ax=ax, label="F1-Macro drop (percentage points)")
    plt.tight_layout()
    _savefig(fig, out_path)


# ── 3. Sample ECG beats under each hardware profile ───────────────────────────

def plot_sample_beats(out_path: str):
    t   = np.linspace(0, 1, SEQ_LEN)
    beat = (0.65 * np.exp(-((t - 0.50) ** 2) / 0.0012)
           + 0.12 * np.exp(-((t - 0.38) ** 2) / 0.006)
           + 0.18 * np.exp(-((t - 0.65) ** 2) / 0.005)
           + 0.04)
    beat = (beat - beat.min()) / (beat.ptp() + 1e-9)
    beat = beat.reshape(1, -1).astype(np.float32)

    titles = ["Clean (MIT-BIH)", "AD8232 + ESP32 ADC", "AD8232 + ADS1115"]
    profs  = ["clean", "ad8232_esp32", "ads1115"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    fig.suptitle("Synthetic ECG Beat Under Each Hardware Profile",
                 fontweight="bold", fontsize=13)

    for ax, prof, col, title in zip(axes, profs, COND_COLORS, titles):
        noisy = apply_noise(beat, profile=prof, seed=42)[0]
        ax.plot(noisy, color=col, linewidth=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-0.05, 1.15)
        ax.grid(alpha=0.3)
        # annotate SNR
        from noise_simulator import snr_db
        snr = snr_db(beat, apply_noise(beat, profile=prof, seed=42))
        ax.annotate(f"SNR ≈ {snr:.1f} dB", xy=(0.97, 0.85),
                    xycoords="axes fraction", ha="right", fontsize=9,
                    color=col, fontweight="bold")

    axes[-1].set_xlabel("Sample index (360 Hz)")
    plt.tight_layout()
    _savefig(fig, out_path)


# ── 4. Robustness scatter: params vs F1 drop ─────────────────────────────────

def plot_robustness_scatter(clean, ad8232, ads1115,
                            param_counts: dict, out_path: str):
    models = [r["model_name"] for r in clean]
    metric = "f1_macro"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Robustness: Params vs F1-Macro Drop Under Noise",
                 fontweight="bold")

    for ax, (res, label, col) in zip(axes, [
        (ad8232,  "AD8232 + ESP32 ADC", COND_COLORS[1]),
        (ads1115, "AD8232 + ADS1115",   COND_COLORS[2]),
    ]):
        for i, m in enumerate(models):
            base  = get_metric(clean,  m, metric)
            noisy = get_metric(res,    m, metric)
            drop  = (base - noisy) * 100
            params = param_counts.get(m, 0) / 1000
            ax.scatter(params, drop, color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
                       s=120, zorder=5, edgecolors="white", linewidths=1.5,
                       label=m)
            ax.annotate(m, (params, drop),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)

        ax.set_title(label)
        ax.set_xlabel("Parameters (thousands)")
        ax.set_ylabel("F1-Macro drop (pp)")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.grid(alpha=0.3)

    # Single legend from first axes
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4,
               framealpha=0.4, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    _savefig(fig, out_path)


# ── 5. Confusion matrices under AD8232 noise ──────────────────────────────────

def plot_confusion_ad8232(ad8232: list, out_path: str):
    results = [r for r in ad8232 if r.get("confusion_matrix")]
    if not results:
        print("  [Skip] No confusion matrices in AD8232 results "
              "(eval-only mode — re-run without --eval-only for full CMs).")
        return

    n     = len(results)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    num_classes = len(results[0]["confusion_matrix"])

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle("Confusion Matrices — AD8232 + ESP32 ADC Noise",
                 fontweight="bold", fontsize=13)

    for i, r in enumerate(results):
        cm      = np.array(r["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
        im = axes[i].imshow(cm_norm, cmap="Reds", vmin=0, vmax=1)
        axes[i].set_title(r["model_name"], fontweight="bold")
        axes[i].set_xlabel("Predicted"); axes[i].set_ylabel("True")
        axes[i].set_xticks(range(num_classes))
        axes[i].set_yticks(range(num_classes))
        for rr in range(num_classes):
            for cc in range(num_classes):
                col = "white" if cm_norm[rr, cc] > 0.6 else "black"
                axes[i].text(cc, rr, f"{cm_norm[rr, cc]:.2f}",
                             ha="center", va="center", fontsize=8, color=col)
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    for j in range(len(results), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _savefig(fig, out_path)


# ── 6. Combined summary (4-panel figure) ─────────────────────────────────────

def plot_combined_summary(clean, ad8232, ads1115, out_path: str):
    """One-page overview: F1 bar + accuracy bar + degradation + sample beats."""
    models = [r["model_name"] for r in clean]
    M      = len(models)
    x      = np.arange(M)
    w      = 0.25

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("ECG Model Performance: Clean vs Hardware Noise Profiles",
                 fontweight="bold", fontsize=14)

    for panel_idx, (metric, ylabel) in enumerate([("f1_macro", "F1 Macro"),
                                                   ("accuracy",  "Accuracy")]):
        ax = fig.add_subplot(gs[0, panel_idx])
        cond_data = [
            ("Clean",              clean,   COND_COLORS[0]),
            ("AD8232+ESP32",       ad8232,  COND_COLORS[1]),
            ("ADS1115",            ads1115, COND_COLORS[2]),
        ]
        for k, (label, res, col) in enumerate(cond_data):
            vals = [get_metric(res, m, metric) for m in models]
            ax.bar(x + (k - 1) * w, vals, w, label=label,
                   color=col, edgecolor="white", linewidth=0.5, alpha=0.9)
        ax.set_title(ylabel, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=35, ha="right",
                                               fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.4, 1.01)
        ax.legend(fontsize=7, framealpha=0.4)
        ax.grid(axis="y", alpha=0.3)

    # Degradation heatmap (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    conds   = ["AD8232+ESP32", "ADS1115"]
    results = [ad8232, ads1115]
    data    = np.zeros((M, 2))
    for i, m in enumerate(models):
        base = get_metric(clean, m, "f1_macro")
        for j, res in enumerate(results):
            data[i, j] = (base - get_metric(res, m, "f1_macro")) * 100

    cmap = LinearSegmentedColormap.from_list(
        "rg", ["#DCFCE7", "#FEF9C3", "#FEE2E2", "#EF4444"])
    im = ax3.imshow(data.T, cmap=cmap, aspect="auto",
                    vmin=0, vmax=max(10, data.max()))
    ax3.set_yticks([0, 1]); ax3.set_yticklabels(conds, fontsize=8)
    ax3.set_xticks(range(M)); ax3.set_xticklabels(models, rotation=35,
                                                    ha="right", fontsize=8)
    ax3.set_title("F1-Macro Degradation vs Clean (pp)", fontweight="bold")
    for i in range(M):
        for j in range(2):
            ax3.text(i, j, f"{data[i, j]:.1f}",
                     ha="center", va="center", fontsize=8, fontweight="bold")
    plt.colorbar(im, ax=ax3, label="pp drop")

    # Sample beats (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    t = np.linspace(0, 1, 187)
    beat = (0.65 * np.exp(-((t - 0.50) ** 2) / 0.0012)
           + 0.12 * np.exp(-((t - 0.38) ** 2) / 0.006)
           + 0.18 * np.exp(-((t - 0.65) ** 2) / 0.005)
           + 0.04)
    beat = (beat - beat.min()) / (beat.ptp() + 1e-9)
    beat = beat.reshape(1, -1).astype(np.float32)

    for prof, col, lbl in zip(PROFILES, COND_COLORS, COND_LABELS):
        noisy = apply_noise(beat, profile=prof, seed=42)[0]
        ax4.plot(noisy, color=col, linewidth=0.85, label=lbl, alpha=0.9)

    ax4.set_title("ECG Under Each Noise Profile", fontweight="bold")
    ax4.set_xlabel("Sample (360 Hz)"); ax4.set_ylabel("Amplitude")
    ax4.legend(fontsize=7, framealpha=0.4)
    ax4.grid(alpha=0.3)

    _savefig(fig, out_path)


# ── Master function ───────────────────────────────────────────────────────────

def generate_comparison_plots(clean_path: str, ad8232_path: str,
                               ads1115_path: str, out_dir: str,
                               param_counts: dict = None):
    """Load the three result JSONs and generate all comparison figures."""
    clean  = load_json(clean_path)
    ad8232 = load_json(ad8232_path)
    ads1115 = load_json(ads1115_path)

    if param_counts is None:
        param_counts = {}

    print(f"\n[NoiseComparison] Generating figures → {out_dir}")

    plot_metric_bar(clean, ad8232, ads1115, "f1_macro", "F1 Macro",
                    os.path.join(out_dir, "noise_f1_comparison.png"))

    plot_metric_bar(clean, ad8232, ads1115, "accuracy", "Accuracy",
                    os.path.join(out_dir, "noise_accuracy_comparison.png"))

    plot_metric_bar(clean, ad8232, ads1115, "auc_roc", "AUC-ROC",
                    os.path.join(out_dir, "noise_auc_comparison.png"))

    plot_degradation_heatmap(clean, ad8232, ads1115,
                             os.path.join(out_dir, "noise_degradation_heatmap.png"))

    plot_sample_beats(os.path.join(out_dir, "noise_sample_beats.png"))

    plot_robustness_scatter(clean, ad8232, ads1115, param_counts,
                            os.path.join(out_dir, "noise_robustness_scatter.png"))

    plot_confusion_ad8232(ad8232,
                          os.path.join(out_dir, "noise_confusion_ad8232.png"))

    plot_combined_summary(clean, ad8232, ads1115,
                          os.path.join(out_dir, "noise_combined_summary.png"))

    print(f"[NoiseComparison] Done — {out_dir}/")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate noise comparison figures")
    p.add_argument("--clean",   required=True,
                   help="Path to results_mitbih.json (clean baseline)")
    p.add_argument("--ad8232",  required=True,
                   help="Path to results_mitbih_ad8232.json")
    p.add_argument("--ads1115", required=True,
                   help="Path to results_mitbih_ads1115.json")
    p.add_argument("--out",     default="outputs/figures_noise",
                   help="Output directory for figures")
    args = p.parse_args()

    generate_comparison_plots(args.clean, args.ad8232, args.ads1115, args.out)
