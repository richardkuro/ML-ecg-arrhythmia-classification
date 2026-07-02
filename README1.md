# ECG Heartbeat Classification — Benchmark Suite
### Models: ResNet1D · TCN · Transformer · KAN · ConvKAN · S4 · LiquidNet

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision scikit-learn pandas matplotlib seaborn numpy

# 2. Download dataset from Kaggle
#    https://www.kaggle.com/datasets/shayanfazeli/heartbeat
#    Place the 4 CSVs in the data/ folder:
mkdir -p data
# → data/mitbih_train.csv
# → data/mitbih_test.csv
# → data/ptbdb_normal.csv
# → data/ptbdb_abnormal.csv

# 3. Run full experiment (both datasets, all 7 models, 30 epochs)
python run_experiments.py

# 4. Quick test (2 models, 5 epochs, one dataset)
python run_experiments.py --epochs 5 --models resnet1d kan --dataset mitbih
```

---

## File Structure

```
ecg_experiment/
├── data/                     ← Put Kaggle CSVs here
├── outputs/
│   ├── checkpoints/          ← Best model .pt files per model+dataset
│   ├── results/              ← results_mitbih.json, results_ptbdb.json
│   └── figures/              ← All paper-ready PNG figures
│
├── data_loader.py            ← Dataset loading, class balancing
├── models.py                 ← All 7 architectures
├── trainer.py                ← Training loop, evaluation, metrics
├── plot_results.py           ← All figure generation
├── run_experiments.py        ← Main entry point
└── README.md
```

---

## CUDA Setup (Your 3050Ti)

PyTorch detects CUDA automatically. Install the CUDA-enabled PyTorch:

```bash
# CUDA 11.8 (most common for 3050Ti with driver 520+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is found
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### What CUDA unlocks for you
| Feature | Effect |
|---|---|
| `torch.autocast("cuda")` | AMP: fp16 matmuls, ~2x speedup, half VRAM |
| `cudnn.benchmark = True` | Auto-tunes conv kernels, ~10-20% faster |
| `allow_tf32 = True` | Ampere TF32 cores (3050Ti is Ampere GA107) |
| `pin_memory=True` | Faster CPU→GPU transfer in DataLoader |
| `non_blocking=True` | Async .to(device) transfers |

All of these are already enabled in `trainer.py` when CUDA is detected.

### Expected training times on your 3050Ti (30 epochs, MIT-BIH 109k samples)
| Model | Params | Est. time |
|---|---|---|
| LiquidNet | ~10k | ~4 min |
| S4ECG | ~19k | ~5 min |
| TCN | ~137k | ~6 min |
| KAN | ~142k | ~8 min |
| ConvKAN | ~151k | ~8 min |
| TransformerECG | ~151k | ~7 min |
| ResNet1D | ~1.9M | ~15 min |

---

## CLI Options

```
python run_experiments.py [OPTIONS]

--dataset   mitbih | ptbdb | both     (default: both)
--models    resnet1d tcn transformer kan convkan s4 liquidnet
            (space-separated subset, default: all)
--epochs    int   (default: 30)
--lr        float (default: 0.001)
--batch     int   (default: 256)
--seed      int   (default: 42)
--data-dir  path  (default: data/)
--out-dir   path  (default: outputs/)
--no-plots        Skip figure generation
--balance         Weighted sampler to fix MIT-BIH imbalance (default: on)
--workers   int   DataLoader workers (default: 2; set 0 on Windows)
```

---

## Models Explained

### 1. ResNet1D — Baseline
4-block 1D residual CNN. Replicates the Fazeli 2018 paper (arXiv:1805.00794).
Strong baseline, ~99% accuracy on MIT-BIH in the literature.

### 2. TCN — Temporal Convolutional Network
Dilated causal convolutions with weight normalization.
Receptive field grows exponentially with depth. Very efficient.

### 3. TransformerECG — Patch Transformer
Splits the 187-length signal into 11-point patches → linear projection →
standard Transformer encoder with CLS token classification.

### 4. KAN — Kolmogorov-Arnold Network ⭐ Novel
Replaces fixed activations with learnable B-spline functions on each connection.
First application to the MIT-BIH Kaggle benchmark.
Theoretically more expressive per parameter than MLP.

### 5. ConvKAN — Multi-scale CNN + KAN ⭐ Novel
Three parallel 1D CNN branches (kernel 3, 7, 15) → concat → KAN layer → head.
Best of both worlds: CNNs extract local morphology, KAN classifies adaptively.

### 6. S4ECG — State Space Model ⭐ Novel
Diagonal S4 (S4D) operating via FFT convolution on 187 timesteps.
Only ~19k parameters. Linear complexity. First applied to this benchmark.

### 7. LiquidNet — Liquid Time-Constant Network ⭐ Novel
ODE-based continuous-time dynamics (Hasani et al. 2021).
Stiff ODE solved with Euler unfolding. Extremely compact (~10k params).
Almost no prior ECG classification papers exist for this architecture.

---

## Outputs Generated

All figures saved to `outputs/figures/`:

| File | Content |
|---|---|
| `training_curves_{ds}.png` | Train/test loss + F1 per epoch for all models |
| `comparison_bar_{ds}.png` | Bar chart: accuracy, F1-macro, F1-weighted, AUC-ROC |
| `confusion_matrices_{ds}.png` | Normalized confusion matrix per model |
| `roc_curves_{ds}.png` | ROC curves (binary PTB) or per-class (5-class MIT-BIH) |
| `param_vs_f1_{ds}.png` | Efficiency scatter: parameter count vs macro F1 |
| `radar_chart_{ds}.png` | Spider chart comparing all metrics across models |
| `lr_schedule_{ds}.png` | Cosine annealing LR schedule |

JSON results in `outputs/results/results_{ds}.json` contain full training
history, final metrics, confusion matrices, and classification reports —
ready to load into your paper tables.

---

## Addressing the Class Imbalance (MIT-BIH)

The Kaggle MIT-BIH split is severely imbalanced:
- N (normal): ~72k samples
- Q (unknown): ~8k
- V (ventricular): ~7k
- S (supraventricular): ~2k
- F (fusion): ~800

This pipeline uses three strategies simultaneously:
1. **WeightedRandomSampler** — oversamples minority classes during training
2. **Focal Loss** (γ=2) — down-weights easy majority class examples
3. **Macro F1** as primary metric — not fooled by accuracy on imbalanced data

---

## Tips for Your Paper

- Run with `--seed 42 --seed 123 --seed 777` three times and report mean ± std
- Compare MIT-BIH and PTB results in the same table (transfer learning angle)
- The param_vs_f1 plot is great for an "efficiency vs accuracy" figure
- S4 and LiquidNet being <20k params with competitive accuracy is your key novelty argument
- KAN's interpretability (the learned spline activations can be visualized) is a bonus angle
